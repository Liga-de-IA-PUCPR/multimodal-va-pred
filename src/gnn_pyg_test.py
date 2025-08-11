import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import torchmetrics
from torch_geometric.utils import remove_self_loops, add_self_loops, to_networkx
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch_geometric.nn as geom_nn
from PIL import Image

from graph_datamodule import VideoDataModule
import networkx as nx

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

class VideoSequentialGNN(L.LightningModule):
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

    def extract_features(self, batch, device):
        audio_feats = []
        image_feats = []

        # Align img paths per node: PyG may collate Python lists as [per_graph_obj]
        img_paths_per_node = None
        if hasattr(batch, "img_paths"):
            ip = batch.img_paths
            if isinstance(ip, list):
                # Case 1: already node-aligned list[list[str]]
                if len(ip) == batch.x.shape[0] and all(isinstance(e, list) for e in ip):
                    img_paths_per_node = ip
                # Case 2: batch_size==1 -> [list[list[str]]]
                elif len(ip) == 1 and isinstance(ip[0], list) and len(ip[0]) == batch.x.shape[0]:
                    img_paths_per_node = ip[0]

        for i in range(batch.x.shape[0]):
            audio_segment = batch.x[i].cpu().numpy()
            audio_feat = extract_wav2vec_features(
                audio_segment,
                16000,
                self.wav2vec_processor,
                self.wav2vec_model,
                device
            )
            audio_feats.append(audio_feat)

            if img_paths_per_node is not None:
                img_paths = img_paths_per_node[i]
                img_feat = extract_vggface_features(
                    img_paths,
                    self.vggface_transform,
                    self.vggface_model,
                    device
                )
                image_feats.append(img_feat)
            else:
                image_feats.append(np.zeros(512, dtype=np.float32))

        node_features = [np.concatenate([a, i]) for a, i in zip(audio_feats, image_feats)]
        x = torch.tensor(np.array(node_features), dtype=torch.float, device=device)
        batch.x = x
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        # Ensure extractor models are on the correct device (do once)
        if getattr(self, "_extractors_device", None) != device:
            self.wav2vec_model.to(device)  # type: ignore[arg-type]
            self.vggface_model.to(device)  # type: ignore[arg-type]
            self._extractors_device = device
        # Move PyG Data to device and run feature extraction
        batch = batch.to(device)
        batch = self.extract_features(batch, device)
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
    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass

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
        video_dir="/home/azureuser/localfiles/datasets/multimodal/new_vids",
        cropped_img_dir="/home/azureuser/localfiles/datasets/multimodal/cropped_aligned_new_50_vids",
        annotation_root="/home/azureuser/localfiles/datasets/multimodal/VA_Estimation_Challenge",
        time_window_sec=5.0,
        num_workers=1
    )
    datamodule.setup()

    # --- Pre-fit sanity checks (printed to stdout) ---
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
    video_model = VideoSequentialGNN(
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
        max_epochs=10,
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