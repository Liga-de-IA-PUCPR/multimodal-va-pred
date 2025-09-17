import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
import torchmetrics
from torch_geometric.utils import remove_self_loops, add_self_loops
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch_geometric.nn as geom_nn
from PIL import Image
from graph_datamodule import VideoDataModule
from utils.extractors import (
    WavLMExtractor, HuBERTExtractor, Wav2Vec2Extractor, MFCCExtractor,
    ResNet50FeatureExtractor, I3DExtractor, LSTMFeatureExtractor, ViTFeatureExtractor
)

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
    def __init__(self, model_name, audio_extractor_obj, vision_extractor_obj, num_classes=3, loss_alpha: float = 0.5, use_pos_enc: bool = True, **model_kwargs):
        super().__init__()
        
        # Store extractors (don't log them as they change each run)
        self.audio_extractor = audio_extractor_obj
        self.vision_extractor = vision_extractor_obj
        
        # Log hyperparameters excluding the extractor objects
        hyperparams = {
            'model_name': model_name,
            'num_classes': num_classes,
            'loss_alpha': loss_alpha,
            'use_pos_enc': use_pos_enc,
            'audio_extractor_type': type(audio_extractor_obj).__name__,
            'vision_extractor_type': type(vision_extractor_obj).__name__,
            **model_kwargs
        }
        self.save_hyperparameters(hyperparams)
        
        self.num_classes = num_classes
        self.loss_alpha = float(loss_alpha)
        self.use_pos_enc = bool(use_pos_enc)
        self.loss_module = nn.MSELoss()
        
        # Metrics
        self.train_pearson = torchmetrics.PearsonCorrCoef(num_outputs=2)
        self.val_pearson = torchmetrics.PearsonCorrCoef(num_outputs=2)
        self.test_pearson = torchmetrics.PearsonCorrCoef(num_outputs=2)

        # --- Modular feature extractors ---
        self.audio_extractor = audio_extractor
        self.vision_extractor = vision_extractor
        self.audio_dim = audio_extractor.get_feature_dim()
        self.vision_dim = vision_extractor.get_feature_dim()
        
        # Positional encoding
        self.pos_enc_dim = 4
        self.feat_chunk_size = 8
        
        # Auto-wire model input dim
        expected_c_in = self.audio_dim + self.vision_dim + (self.pos_enc_dim if self.use_pos_enc else 0)
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

    def _get_img_paths_per_node(self, batch):
        """Extract image paths from batch."""
        if hasattr(batch, "img_paths"):
            ip = batch.img_paths
            n = batch.x.shape[0]
            if isinstance(ip, list):
                if len(ip) == n and all(isinstance(e, (list, tuple)) for e in ip):
                    return ip
                elif len(ip) == 1 and isinstance(ip[0], list) and len(ip[0]) == n:
                    return ip[0]
        return None

    def extract_features(self, batch, device):
        """Batchified feature extraction."""
        # Audio features 
        audio_feats = self.audio_extractor.extract_features(batch.x, device, self.feat_chunk_size)

        # Image features 
        img_paths_per_node = self._get_img_paths_per_node(batch)
        if img_paths_per_node is None:
            # No image paths available, return zeros
            image_feats = torch.zeros(batch.x.size(0), self.vision_dim, device=device, dtype=torch.float32)
        else:
            image_feats = self.vision_extractor.extract_features(
                img_paths_per_node, device, self.feat_chunk_size
            )

        # Concatenate features
        pos_enc = self._compute_positional_encoding(batch, device)
        x = torch.cat([audio_feats.float(), image_feats.float(), pos_enc.float()], dim=1)
        batch.x = x
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        if getattr(self, "_extractors_device", None) != device:
            self.audio_extractor.to_device(device)
            self.vision_extractor.to_device(device)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_train_epoch_end(self):
        """Log gradient norms at the end of each training epoch."""
        # Calculate total gradient norm
        total_norm = 0
        param_count = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        total_norm = total_norm ** (1. / 2) if param_count > 0 else 0
        
        # Log to MLflow via Lightning's logger
        if hasattr(self.logger, 'experiment'):
            # Use Lightning's logging system instead of direct MLflow calls
            self.log("grad_norm_total", total_norm, logger=True)
            
            # Optional: Log per-layer gradient norms (first few layers only to avoid too many metrics)
            layer_count = 0
            for name, param in self.named_parameters():
                if param.grad is not None and layer_count < 5:  # Limit to first 5 layers
                    grad_norm = param.grad.data.norm(2).item()
                    # Clean up parameter name for logging
                    clean_name = name.replace('.', '_').replace('model_', '').replace('convs_', 'conv_')
                    self.log(f"grad_{clean_name}", grad_norm, logger=True)
                    layer_count += 1

def plot_loss_and_acc_from_mlflow(experiment_name="video_gnn_experiment", tracking_uri="file:./mlruns", 
                                 loss_ylim=(0.0, 0.9), acc_ylim=(0.3, 1.0), save_loss=None, save_acc=None):
    """Plot metrics from MLflow instead of CSV files."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found. Skipping plotting.")
            return
            
        runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        
        if not runs:
            print("No runs found. Skipping plotting.")
            return
            
        run_id = runs[0].info.run_id
        
        metrics = {}
        for metric_key in ["train_loss", "val_loss", "train_pearson", "val_pearson", "train_ccc", "val_ccc"]:
            try:
                metric_history = client.get_metric_history(run_id, metric_key)
                metrics[metric_key] = [(m.step, m.value) for m in metric_history]
            except Exception:
                continue
        
        if not metrics:
            print("No metrics found. Skipping plotting.")
            return
            
        # Convert to DataFrame-like structure
        max_steps = max(len(values) for values in metrics.values()) if metrics else 0
        df_data = {"epoch": list(range(max_steps))}
        
        for metric_key, values in metrics.items():
            df_data[metric_key] = [v[1] if i < len(values) else None for i, v in enumerate(values)]
        
        df_metrics = pd.DataFrame(df_data).dropna(subset=["epoch"])
        
        
        if "train_loss" in df_metrics.columns and "val_loss" in df_metrics.columns:
            df_metrics[["train_loss", "val_loss"]].plot(grid=True, legend=True, xlabel="Epoch", ylabel="Loss")
            plt.ylim(loss_ylim)
            if save_loss is not None:
                plt.savefig(save_loss)
                plt.close()
        
        
        corr_cols = [c for c in ["train_pearson", "val_pearson"] if c in df_metrics.columns]
        if len(corr_cols) >= 1:
            df_metrics[corr_cols].plot(grid=True, legend=True, xlabel="Epoch", ylabel="Pearson")
            plt.ylim(acc_ylim)
            if save_acc is not None:
                plt.savefig(save_acc)
                plt.close()
        
        
        ccc_cols = [c for c in ["train_ccc", "val_ccc"] if c in df_metrics.columns]
        if len(ccc_cols) >= 1:
            df_metrics[ccc_cols].plot(grid=True, legend=True, xlabel="Epoch", ylabel="CCC")
            plt.ylim(acc_ylim)
            if save_acc is not None and save_acc != "corr_ccc.png":
                plt.savefig("ccc.png")
                plt.close()
                
    except ImportError:
        print("MLflow not installed. Install with: pip install mlflow")
    except Exception as e:
        print(f"Error plotting from MLflow: {e}")

if __name__ == "__main__":
    
    L.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Callbacks
    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3)
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=10)

    # Logger
    mlflow_logger = MLFlowLogger(
        experiment_name="video_gnn_experiment", 
        tracking_uri="file:./mlruns",  # Local tracking directory
        run_name=f"video_gnn_{L.seed_everything(42)}"  # Use seed for reproducibility
    )

    # DataModule
    datamodule = VideoDataModule(
        video_dir="/home/blau/datasets/affwild2/batch1-video",
        cropped_img_dir="/home/blau/datasets/affwild2/batch1",
        annotation_root="/home/blau/datasets/affwild2/6th ABAW Annotations/VA_Estimation_Challenge",
        time_window_sec=1.0,
        batch_size=1,  
        num_workers=0  
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

    # Model with modular extractors (auto-wired c_in: WavLM 768 + ResNet50 2048 + pos 4 = 2820)
    audio_extractor = WavLMExtractor()
    vision_extractor = ResNet50FeatureExtractor()
    video_model = VideoGNN(
        model_name="GCN",
        audio_extractor_obj=audio_extractor,
        vision_extractor_obj=vision_extractor,
        c_hidden=32,  
        c_out=2,
        num_classes=2,
        num_layers=2,  
        layer_name="GCN",
        dp_rate=0.3, 
        loss_alpha=0.5,
        use_pos_enc=True,
    )

    # Trainer
    video_trainer = L.Trainer(
        callbacks=[model_checkpoint, early_stopping_callback],
        logger=mlflow_logger,
        accelerator="auto",
        max_epochs=25,  
        precision="32",  
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        enable_progress_bar=True,
        devices=1,
        accumulate_grad_batches=2,  
    )

    # --- Training, Testing, and Plotting ---
    video_trainer.fit(video_model, datamodule=datamodule)
    
    # Plot metrics from MLflow
    plot_loss_and_acc_from_mlflow(
        experiment_name="video_gnn_experiment",
        tracking_uri="file:./mlruns",
        save_loss="loss.png", 
        save_acc="corr_ccc.png"
    )
    plt.show()

    video_test_result = video_trainer.test(video_model, dataloaders=datamodule.test_dataloader())
    print(video_test_result)
    
    # To view MLflow experiments in browser:
    # Run: mlflow ui --backend-store-uri file:./mlruns
    # Then open http://localhost:5000 in your browser