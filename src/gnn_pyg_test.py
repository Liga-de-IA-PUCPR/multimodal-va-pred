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

# Legacy helpers removed in favor of in-module extractors (WavLM + ResNet50)

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
    def __init__(self, model_name, num_classes=3, loss_alpha: float = 0.5, img_pool_num: int = 3, use_pos_enc: bool = True, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.loss_alpha = float(loss_alpha)
        self.img_pool_num = int(max(1, img_pool_num))
        self.use_pos_enc = bool(use_pos_enc)
        self.loss_module = nn.MSELoss()
        self.train_pearson = torchmetrics.PearsonCorrCoef(num_outputs=2)
        self.val_pearson = torchmetrics.PearsonCorrCoef(num_outputs=2)
        self.test_pearson = torchmetrics.PearsonCorrCoef(num_outputs=2)

        # --- Feature extractors (Audio=WavLM, Image=ResNet50) ---
        # Audio: WavLM Base+ with feature extractor (no tokenizer)
        from transformers import WavLMModel, AutoFeatureExtractor
        self.wav_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.wav_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.wav_model.eval()
        self.audio_dim = 768  # WavLM Base+ hidden size

        # Image: ResNet50 (ImageNet) as feature extractor (pool -> 2048)
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        modules = list(backbone.children())[:-1]  # remove FC, keep avgpool
        self.vision_model = nn.Sequential(*modules)
        self.vision_model.eval()
        self.vision_transform = weights.transforms()  # includes resize+norm to 224
        self.vision_dim = 2048

        # Positional encoding dims (sin/cos with 2 frequencies)
        self.pos_enc_dim = 4

        # Feature extraction mini-batch size to cap peak memory
        self.feat_chunk_size = 8

        # Auto-wire model input dim and instantiate backbone
        expected_c_in = self.audio_dim + self.vision_dim + (self.pos_enc_dim if self.use_pos_enc else 0)
        # Respect explicit c_in if provided, otherwise inject auto c_in
        model_kwargs = dict(model_kwargs)
        model_kwargs.setdefault("c_in", expected_c_in)
        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)

    def _compute_positional_encoding(self, batch, device):
        if not self.use_pos_enc:
            return torch.zeros((batch.x.size(0), 0), device=device, dtype=torch.float32)
        # Determine graph boundaries
        n = batch.x.size(0)
        if hasattr(batch, "ptr") and batch.ptr is not None:
            ptr = batch.ptr.to(torch.long).tolist()
        else:
            ptr = [0, n]
        enc = torch.zeros((n, self.pos_enc_dim), device=device, dtype=torch.float32)
        for g in range(len(ptr) - 1):
            s, e = ptr[g], ptr[g + 1]
            L = max(1, e - s)
            if L == 1:
                pos = torch.zeros(1, device=device)
            else:
                pos = torch.linspace(0, 1, steps=L, device=device)
            # 2 frequencies
            pe = torch.stack([
                torch.sin(2 * torch.pi * pos),
                torch.cos(2 * torch.pi * pos),
                torch.sin(4 * torch.pi * pos),
                torch.cos(4 * torch.pi * pos),
            ], dim=1)
            enc[s:e] = pe
        return enc

    def extract_features(self, batch, device):
        """Batchified feature extraction on device with autocast; avoids CPU<->GPU thrash.

        Expects batch.x to contain equal-length audio windows (num_nodes, T).
        Pools multiple evenly spaced frames per window for ResNet50 features.
        """
        # 1) Audio features (WavLM) — chunked batch process
        n = batch.x.shape[0]
        audio_list = [batch.x[i].detach().cpu().numpy() for i in range(n)]
        audio_feats = torch.empty(n, self.audio_dim, device=device, dtype=torch.float32)
        for s in range(0, n, self.feat_chunk_size):
            e = min(n, s + self.feat_chunk_size)
            sub_list = audio_list[s:e]
            inputs = self.wav_fe(sub_list, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                out = self.wav_model(**inputs)
                hidden = out.last_hidden_state  # (B, T_feat, 768)
                # Windows are fixed-length, so uniform mean pooling over feature frames is valid
                pooled = hidden.mean(dim=1)
            audio_feats[s:e] = pooled.float()

        # 2) Image features (ResNet50) — pool K evenly spaced frames per node
        def _coerce_paths(obj):
            if isinstance(obj, (list, tuple)):
                return [p for p in obj if isinstance(p, str)]
            elif isinstance(obj, str):
                return [obj]
            return []

        img_paths_per_node = None
        if hasattr(batch, "img_paths"):
            ip = batch.img_paths
            if isinstance(ip, list):
                if len(ip) == n and all(isinstance(e, (list, tuple)) for e in ip):
                    img_paths_per_node = ip
                elif len(ip) == 1 and isinstance(ip[0], list) and len(ip[0]) == n:
                    img_paths_per_node = ip[0]

        # Flatten selected frames for batch processing
        flat_imgs: list[torch.Tensor] = []
        flat_nodes: list[int] = []
        if img_paths_per_node is not None:
            for i in range(n):
                paths = _coerce_paths(img_paths_per_node[i]) if img_paths_per_node[i] is not None else []
                if len(paths) == 0:
                    continue
                if len(paths) <= self.img_pool_num:
                    sel = paths
                else:
                    idxs = np.linspace(0, len(paths) - 1, num=self.img_pool_num)
                    sel = [paths[int(round(j))] for j in idxs]
                for p in sel:
                    if isinstance(p, str) and os.path.isfile(p):
                        try:
                            img = Image.open(p).convert("RGB")
                            img_t = self.vision_transform(img)
                            flat_imgs.append(img_t)
                            flat_nodes.append(i)
                        except Exception:
                            pass

        image_feats = torch.zeros(n, self.vision_dim, device=device, dtype=torch.float32)
        if len(flat_imgs) > 0:
            counts = torch.zeros(n, device=device, dtype=torch.float32)
            for s in range(0, len(flat_imgs), self.feat_chunk_size):
                e = min(len(flat_imgs), s + self.feat_chunk_size)
                sub_imgs = torch.stack(flat_imgs[s:e]).to(device, non_blocking=True)
                node_idx = torch.as_tensor(flat_nodes[s:e], device=device, dtype=torch.long)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                    feats = self.vision_model(sub_imgs).flatten(1)  # (b,2048)
                # Accumulate
                image_feats.index_add_(0, node_idx, feats.float())
                counts.index_add_(0, node_idx, torch.ones_like(node_idx, dtype=torch.float32))
            counts = counts.clamp_min_(1.0).unsqueeze(1)
            image_feats = image_feats / counts

        # 3) Concatenate features on device
        pos_enc = self._compute_positional_encoding(batch, device)
        x = torch.cat([audio_feats.float(), image_feats.float(), pos_enc.float()], dim=1)
        batch.x = x
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        if getattr(self, "_extractors_device", None) != device:
            self.wav_model.to(device)  # type: ignore[arg-type]
            self.vision_model.to(device)  # type: ignore[arg-type]
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
        mse = self.loss_module(predictions, y)
        pearson = getattr(self, f"{mode}_pearson")(predictions, y)
        ccc = self._ccc(predictions, y)
        ccc_loss = 1.0 - ccc.mean()
        loss = self.loss_alpha * mse + (1.0 - self.loss_alpha) * ccc_loss
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
    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3)
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=10)

    # Logger
    csv_logger = CSVLogger("logs", name="video_gnn_logs")

    # DataModule
    datamodule = VideoDataModule(
        video_dir="/home/azureuser/localfiles/datasets/multimodal/new_vids",
        cropped_img_dir="/home/azureuser/localfiles/datasets/multimodal/cropped_aligned_new_50_vids",
        annotation_root="/home/azureuser/localfiles/datasets/multimodal/VA_Estimation_Challenge",
        time_window_sec=1.0,
    num_workers=3
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

    # Model (auto-wired c_in: WavLM 768 + ResNet50 2048 + pos 4 = 2820)
    video_model = VideoGNN(
        model_name="GCN",
        c_hidden=64,
        c_out=2,
        num_classes=2,
        num_layers=3,
        layer_name="GCN",
        dp_rate=0.2,
        loss_alpha=0.5,
        img_pool_num=3,
        use_pos_enc=True,
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