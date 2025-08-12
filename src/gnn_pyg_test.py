import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import torchmetrics
from torch_geometric.utils import remove_self_loops, add_self_loops
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch_geometric.nn as geom_nn
from PIL import Image

from graph_datamodule import VideoDataModule

def extract_wav2vec_features(audio_segment, sr, processor, model, device):
    if audio_segment.size == 0:
        return np.zeros(768, dtype=np.float32)
    inputs = processor(audio_segment, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1)
    return features.squeeze().cpu().numpy()

def extract_vggface_features(img_paths, transform, model, device):
    imgs = [transform(Image.open(p).convert("RGB")) for p in img_paths if os.path.exists(p)]
    if not imgs:
        return np.zeros(512, dtype=np.float32)
    imgs_tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = model(imgs_tensor)
    return feats.mean(dim=0).cpu().numpy()

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
}

class GNNModel(nn.Module):
    def __init__(
    self,
    c_in: int = 1,
    c_hidden: int = 64,
    c_out: int = 2,
        num_layers=2,
        layer_name="GCN",
        dp_rate=0.1,
        **kwargs
    ):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dp_rate)
        in_channels = c_in
        for _ in range(num_layers - 1):
            self.convs.append(gnn_layer(in_channels, c_hidden, **kwargs))
            self.norms.append(geom_nn.GraphNorm(c_hidden))
            in_channels = c_hidden
        self.final_conv = gnn_layer(in_channels, c_out, **kwargs)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = self.relu(x)
            x = self.dropout(x)
            if residual.shape == x.shape:
                x = x + residual
        x = self.final_conv(x, edge_index)
        return x


class MLPModel(nn.Module):
    def __init__(self, c_in: int = 1, c_hidden: int = 64, c_out: int = 2, num_layers: int = 2, dp_rate: float = 0.1):
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.layers(x)

class VideoGNN(L.LightningModule):
    def __init__(self, model_name, num_classes=3, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.MSELoss()
        self.train_pearson = torchmetrics.PearsonCorrCoef(num_outputs=2)
        self.val_pearson = torchmetrics.PearsonCorrCoef(num_outputs=2)
        self.test_pearson = torchmetrics.PearsonCorrCoef(num_outputs=2)

        # --- Feature extractors ---
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        from facenet_pytorch import InceptionResnetV1
        from torchvision import transforms
        wav2vec_model_name = "facebook/wav2vec2-base-960h"
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        self.wav2vec_model.eval()
        self.vggface_model = InceptionResnetV1(pretrained='vggface2')
        self.vggface_model.eval()
        self.vggface_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ])
        # Feature extraction mini-batch size to cap peak memory
        self.feat_chunk_size = 8

    def extract_features(self, batch, device):
        """Batchified feature extraction on device with autocast; avoids CPU<->GPU thrash.

        Expects batch.x to contain equal-length audio windows (num_nodes, T).
        Uses one representative image (center frame) per window for VGGFace features.
        """
        # 1) Audio features (Wav2Vec2) — chunked batch process
        n = batch.x.shape[0]
        audio_list = [batch.x[i].detach().cpu().numpy() for i in range(n)]
        audio_feats = torch.empty(n, 768, device=device, dtype=torch.float32)
        for s in range(0, n, self.feat_chunk_size):
            e = min(n, s + self.feat_chunk_size)
            sub_list = audio_list[s:e]
            inputs = self.wav2vec_processor(sub_list, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                out = self.wav2vec_model(**inputs)
                hidden = out.last_hidden_state  # (B, T, 768)
                if "attention_mask" in inputs:
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    denom = mask.sum(dim=1).clamp(min=1.0)
                    pooled = (hidden * mask).sum(dim=1) / denom
                else:
                    pooled = hidden.mean(dim=1)
            audio_feats[s:e] = pooled.float()

        # 2) Image features (VGGFace) — select one center frame path per window robustly
        def _to_path_str(obj):
            if isinstance(obj, str):
                return obj
            if isinstance(obj, (list, tuple)):
                # Return first string found (handles nested lists)
                for el in obj:
                    if isinstance(el, str):
                        return el
            return None

        img_paths_per_node = None
        if hasattr(batch, "img_paths"):
            ip = batch.img_paths
            if isinstance(ip, list):
                if len(ip) == n and all(isinstance(e, (list, tuple)) for e in ip):
                    img_paths_per_node = ip
                elif len(ip) == 1 and isinstance(ip[0], list) and len(ip[0]) == n:
                    img_paths_per_node = ip[0]

        center_paths = [None] * n
        if img_paths_per_node is not None:
            for i in range(n):
                paths = img_paths_per_node[i]
                if isinstance(paths, (list, tuple)) and len(paths) > 0:
                    center = paths[len(paths) // 2]
                    center_paths[i] = _to_path_str(center)

        # Load and batch valid images
        imgs, valid_idx = [], []
        for i, p in enumerate(center_paths):
            p_str = _to_path_str(p)
            if p_str is not None and os.path.isfile(p_str):
                try:
                    img_t = self.vggface_transform(Image.open(p_str).convert("RGB"))
                    imgs.append(img_t)
                    valid_idx.append(i)
                except Exception:
                    # Skip unreadable/corrupt image
                    pass

        image_feats = torch.zeros(n, 512, device=device, dtype=torch.float32)
        if len(imgs) > 0:
            for s in range(0, len(imgs), self.feat_chunk_size):
                e = min(len(imgs), s + self.feat_chunk_size)
                sub_imgs = torch.stack(imgs[s:e]).to(device, non_blocking=True)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                    sub_feats = self.vggface_model(sub_imgs)  # (b,512)
                idx_tensor = torch.as_tensor(valid_idx[s:e], device=device)
                image_feats[idx_tensor] = sub_feats.float()

        # 3) Concatenate features on device
        x = torch.cat([audio_feats.float(), image_feats.float()], dim=1)  # (N, 1280)
        batch.x = x
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        if getattr(self, "_extractors_device", None) != device:
            self.wav2vec_model.to(device)  # type: ignore[arg-type]
            self.vggface_model.to(device)  # type: ignore[arg-type]
            self._extractors_device = device
        batch = self.extract_features(batch, device)
        batch = batch.to(device, non_blocking=True)
        return batch

    @staticmethod
    def _ccc(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = preds
        y = target
        x_mean, y_mean = x.mean(dim=0), y.mean(dim=0)
        x_var, y_var = x.var(dim=0, unbiased=False), y.var(dim=0, unbiased=False)
        cov = ((x - x_mean) * (y - y_mean)).mean(dim=0)
        ccc = 2 * cov / (x_var + y_var + (x_mean - y_mean).pow(2) + 1e-8)
        return ccc

    def sequential_video_predict(self, data, mode="test"):
        x, edge_index = data.x, data.edge_index
        if edge_index is None or edge_index.numel() == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        predictions = self.model(x, edge_index)
        y = data.y
        loss = self.loss_module(predictions, y)
        pearson = getattr(self, f"{mode}_pearson")(predictions, y)
        ccc = self._ccc(predictions, y)
        return loss, pearson, ccc, predictions
        
    def _shared_step(self, data, mode="train"):
        loss, pearson, ccc, _ = self.sequential_video_predict(data, mode)
        return loss, pearson, ccc

    def training_step(self, batch, batch_idx):
        loss, pearson, ccc = self._shared_step(batch, mode="train")
        bs = getattr(batch, "num_graphs", 1)
        self.log("train_loss", loss, on_epoch=True, on_step=False, batch_size=bs)
        self.log("train_pearson", pearson.mean(), on_epoch=True, on_step=False, batch_size=bs)
        self.log("train_ccc", ccc.mean(), on_epoch=True, on_step=False, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pearson, ccc = self._shared_step(batch, mode="val")
        bs = getattr(batch, "num_graphs", 1)
        self.log("val_loss", loss, on_epoch=True, on_step=False, batch_size=bs)
        self.log("val_pearson", pearson.mean(), on_epoch=True, on_step=False, batch_size=bs)
        self.log("val_ccc", ccc.mean(), on_epoch=True, on_step=False, batch_size=bs)

    def test_step(self, batch, batch_idx):
        loss, pearson, ccc = self._shared_step(batch, mode="test")
        bs = getattr(batch, "num_graphs", 1)
        self.log("test_loss", loss, on_epoch=True, on_step=False, batch_size=bs)
        self.log("test_pearson", pearson.mean(), on_epoch=True, on_step=False, batch_size=bs)
        self.log("test_ccc", ccc.mean(), on_epoch=True, on_step=False, batch_size=bs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        return optimizer

def plot_loss_and_acc(log_dir, loss_ylim=(0.0, 0.9), acc_ylim=(0.3, 1.0), save_loss=None, save_acc=None):
    metrics = pd.read_csv(f"{log_dir}/metrics.csv")
    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        mean_vals = dfg.mean(numeric_only=True).to_dict()
        mean_vals[agg_col] = i
        aggreg_metrics.append(mean_vals)
    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(grid=True, legend=True, xlabel="Epoch", ylabel="Loss")
    plt.ylim(loss_ylim)
    if save_loss is not None:
        plt.savefig(save_loss)
    corr_cols = [c for c in ["train_pearson", "val_pearson"] if c in df_metrics.columns]
    if len(corr_cols) == 2:
        df_metrics[corr_cols].plot(grid=True, legend=True, xlabel="Epoch", ylabel="Pearson")
    ccc_cols = [c for c in ["train_ccc", "val_ccc"] if c in df_metrics.columns]
    if len(ccc_cols) == 2:
        df_metrics[ccc_cols].plot(grid=True, legend=True, xlabel="Epoch", ylabel="CCC")
    plt.ylim(acc_ylim)
    if save_acc is not None:
        plt.savefig(save_acc)

if __name__ == "__main__":

    L.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Callbacks
    model_checkpoint = ModelCheckpoint(monitor="train_loss", mode="min", save_top_k=3)
    early_stopping_callback = EarlyStopping(monitor="train_loss", mode="min", patience=25)

    # Logger
    csv_logger = CSVLogger("logs", name="video_gnn_logs")

    # DataModule
    datamodule = VideoDataModule(
        video_dir="/home/user/liga-ia/datasets/affwild2/batch1",
        cropped_img_dir="/home/user/liga-ia/datasets/affwild2/batch1-cropped",
        annotation_root="/home/user/liga-ia/datasets/affwild2/Annotations/VA_Estimation_Challenge",
        time_window_sec=1.0,
    num_workers=8
    )
    datamodule.setup()

    try:
        print(f"[PreFit] Dataset sizes -> train: {len(datamodule.train_dataset)}, val: {len(datamodule.val_dataset)}, test: {len(datamodule.test_dataset)}", flush=True)
        train_loader = datamodule.train_dataloader()
        if len(train_loader) == 0:
            print("[PreFit] Train dataloader is empty.", flush=True)
        else:
            first_batch = next(iter(train_loader))
            raw_shape = tuple(first_batch.x.shape) if hasattr(first_batch, "x") else None
            print(f"[PreFit] Raw batch.x shape: {raw_shape}", flush=True)
    except Exception as e:
        print(f"[PreFit] Failed to inspect datasets/dataloader: {e}", flush=True)

    # Model
    video_model = VideoGNN(
        model_name="GCN", 
        c_in=768 + 512,  # Wav2Vec (768) + VGGFace (512) features
        c_hidden=64,
        c_out=2,
        num_classes=2,
        num_layers=3,
        layer_name="GCN", 
        dp_rate=0.2
    )

    # Trainer
    video_trainer = L.Trainer(
        callbacks=[model_checkpoint, early_stopping_callback],
        logger=csv_logger,
        accelerator="auto",
        max_epochs=50,
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=1,
    )

    # --- Training, Testing, and Plotting ---
    video_trainer.fit(video_model, datamodule=datamodule)
    
    log_dir = getattr(video_trainer.logger, "log_dir", ".") if video_trainer.logger is not None else "."
    plot_loss_and_acc(log_dir, save_loss="loss.png", save_acc="corr_ccc.png")
    plt.show()

    video_test_result = video_trainer.test(video_model, dataloaders=datamodule.test_dataloader())
    print(video_test_result)