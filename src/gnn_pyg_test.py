import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import remove_self_loops, add_self_loops
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch_geometric.nn as geom_nn
from graph_datamodule import VideoDataModule
from multiprocessing import set_start_method

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
        edge_index, x = edge_index.to(self.device), x.to(self.device)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        predictions = self.model(x, edge_index)
        y = data.y.to(self.device)
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
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5),
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def plot_loss_and_acc(log_dir, loss_ylim=(0.0, 0.9), acc_ylim=(0.3, 1.0), save_loss=None, save_acc=None):
    metrics = pd.read_csv(f"{log_dir}/metrics.csv")
    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)
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

# Main execution block to prevent CUDA multiprocessing errors
if __name__ == "__main__":
    # Set the start method for multiprocessing
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    L.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Callbacks
    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3)
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=25)

    # Logger
    csv_logger = CSVLogger("logs", name="video_gnn_logs")

    # DataModule
    datamodule = VideoDataModule(
        video_dir="/home/user/liga-ia/datasets/affwild2/batch1",
        cropped_img_dir="/home/user/liga-ia/datasets/affwild2/batch1-cropped",
        annotation_root="/home/user/liga-ia/datasets/affwild2/Annotations/VA_Estimation_Challenge",
        time_window_sec=5.0,
        num_workers=4  # Increase this value for better performance if your system supports it
    )
    datamodule.setup()

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
        max_epochs=50,
        precision="16-mixed",
        gradient_clip_val=1.0,
    )

    # --- Training, Testing, and Plotting ---
    video_trainer.fit(video_model, datamodule=datamodule)
    
    log_dir = video_trainer.logger.log_dir
    plot_loss_and_acc(log_dir, save_loss="loss.png", save_acc="corr_ccc.png")
    plt.show()

    video_test_result = video_trainer.test(video_model, dataloaders=datamodule.test_dataloader())
    print(video_test_result)