import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


# Import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# import lightning
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import torchmetrics

# import Pytorch Geometric
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data

# Seed
L.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
}
class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=2,
        layer_name="GCN",
        dp_rate=0.1,
        **kwargs
    ):
        """
        Initializes a GNNModel object.

        Args:
            c_in (int): The number of input channels.
            c_hidden (int): The number of hidden channels.
            c_out (int): The number of output channels.
            num_layers (int, optional): The number of GNN layers. Defaults to 2.
            layer_name (str, optional): The name of the GNN layer. Defaults to "GCN".
            dp_rate (float, optional): The dropout rate. Defaults to 0.1.
            **kwargs: Additional keyword arguments to be passed to the GNN layer.

        Returns:
            None
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []

        in_channels, out_channels = c_in, c_out
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels, c_hidden, *kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.
            edge_index (torch.Tensor): Edge index tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Se a camada atual é uma camada de passagem de mensagem,
        # então ela precisa de duas informações para realizar sua operação (x, edge_index)

        # Se a camada atual não é uma camada de passagem de mensagem, por exemplo ReLU. dropout
        # então ela só precisa da entrada x para realizar sua operação.

        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                x = layer(x=x, edge_index=edge_index)
            else:
                x = layer(x)
        return x


geom_nn.MessagePassing

# %%
class MLPModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers, dp_rate=0.1):
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

    def forward(self, x):
        return self.layers(x)

# %%
class VideoSequentialGNN(L.LightningModule):
    def __init__(self, model_name, num_classes=3, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        
        self.loss_module = nn.CrossEntropyLoss()
        
        self.acc_train = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.acc_val = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.acc_test = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def sequential_video_predict(self, data, mode="test"):
        x, edge_index = data.x, data.edge_index
        num_frames = x.shape[0]
        
        # Start with original frame features
        current_x = x.clone()
        predictions = []
        losses = []
        
        # For videos, predict frames in temporal order (0, 1, 2, ...)
        for frame_idx in range(num_frames):
            # Run GNN with current features
            frame_outputs = self.model(current_x, edge_index)
            
            # Get prediction for current frame
            frame_pred = frame_outputs[frame_idx]
            predicted_label = torch.argmax(frame_pred)
            predictions.append(predicted_label)
            
            # Calculate loss for this frame
            if mode == "train":
                step_loss = self.loss_module(frame_pred.unsqueeze(0), data.y[frame_idx].unsqueeze(0))
                losses.append(step_loss)
            
            # Update current frame's features with its prediction
            # This gives the model "memory" of what it predicted before
            one_hot_pred = torch.zeros(self.num_classes, device=x.device)
            one_hot_pred[predicted_label] = 1.0
            
            # Add prediction information to subsequent frames
            for future_frame in range(frame_idx + 1, min(frame_idx + 5, num_frames)):  # Affect next 5 frames
                decay = 0.8 ** (future_frame - frame_idx)  # Decay influence over time
                if current_x.shape[1] >= self.num_classes:
                    current_x[future_frame, :self.num_classes] += decay * 0.1 * one_hot_pred
            
            if frame_idx % 50 == 0:  # Print every 50 frames
                print(f"Frame {frame_idx}: Predicted {predicted_label.item()}")
        
        predictions = torch.stack(predictions)
        
        if mode == "train" and losses:
            total_loss = torch.stack(losses).mean()
            acc = self.acc_train(predictions, data.y)
            return total_loss, acc, predictions
        else:
            acc = getattr(self, f"acc_{mode}")(predictions, data.y)
            return None, acc, predictions

    def _shared_step(self, data, mode="train"):
        if mode == "train":
            loss, acc, predictions = self.sequential_video_predict(data, mode)
            return loss, acc
        else:
            _, acc, predictions = self.sequential_video_predict(data, mode)
            return torch.tensor(0.0, device=self.device), acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, mode="train")
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log("train_acc", acc, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self._shared_step(batch, mode="test")
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001, weight_decay=1e-4
        )
        return optimizer


# callbacks
## modelCheckpoint
ModelCheckpoint = L.pytorch.callbacks.ModelCheckpoint(monitor="train_loss", mode="min", save_top_k=3)
## earlystopping
early_stopping_callback = L.pytorch.callbacks.EarlyStopping(monitor="train_loss", patience=25)


## csv Logger
csv_logger = CSVLogger("logs", name="cora_logs")

# Note: You need to load your actual dataset here
# For Cora dataset: from torch_geometric.datasets import Planetoid
# cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')[0]


def plot_loss_and_acc(
    log_dir, loss_ylim=(0.0, 0.9), acc_ylim=(0.7, 1.0), save_loss=None, save_acc=None
):

    metrics = pd.read_csv(f"{log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )

    plt.ylim(loss_ylim)
    if save_loss is not None:
        plt.savefig(save_loss)

    df_metrics[["train_acc","val_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )

    plt.ylim(acc_ylim)
    if save_acc is not None:
        plt.savefig(save_acc)


# Example usage of VideoSequentialGNN for video data
def create_mock_video_dataset():
    """Create a mock video dataset for testing VideoSequentialGNN"""
    # Mock video data: 3 videos with different lengths
    video_data_list = []
    
    for video_idx in range(3):
        num_frames = np.random.randint(100, 300)  # Random number of frames
        num_features = 2  # 2-dimensional features
        
        # Create mock video features normalized between -1 and 1
        x = torch.rand(num_frames, num_features) * 2 - 1  # Scale [0,1] to [-1,1]
        
        # Create mock per-frame labels (replace with your actual annotations)
        y = torch.randint(0, 2, (num_frames,))  # 2 classes: 0, 1
        
        # Create temporal edges (each frame connects to next few frames)
        edge_list = []
        temporal_window = 5
        for i in range(num_frames):
            for j in range(1, temporal_window + 1):
                if i + j < num_frames:
                    edge_list.extend([[i, i + j], [i + j, i]])  # bidirectional
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # All frames available for training (no internal masks needed)
        train_mask = torch.ones(num_frames, dtype=torch.bool)
        val_mask = torch.ones(num_frames, dtype=torch.bool)
        test_mask = torch.ones(num_frames, dtype=torch.bool)
        
        video_graph = geom_data.Data(
            x=x, edge_index=edge_index, y=y,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
        )
        
        video_data_list.append(video_graph)
    
    return video_data_list

# Create mock video dataset
video_dataset = create_mock_video_dataset()
video_dataloader = geom_data.DataLoader(video_dataset, batch_size=1, shuffle=True)

# Create VideoSequentialGNN model
video_model = VideoSequentialGNN(
    model_name="GCN", 
    c_in=2,          # 2-dimensional features
    c_hidden=64,     # Hidden dimension  
    c_out=2,         # Number of classes
    num_classes=2,   # Must match c_out
    num_layers=3,    # Layers for temporal modeling
    layer_name="GCN", 
    dp_rate=0.2
)

# Setup trainer for video model
video_trainer = L.Trainer(
    callbacks=[ModelCheckpoint, early_stopping_callback],
    logger=CSVLogger("logs", name="video_gnn_logs"),
    accelerator="auto",
    max_epochs=50,
    precision="bf16-mixed",
)

# Train the video model
video_trainer.fit(video_model, video_dataloader, video_dataloader)

# Test the video model
video_test_result = video_trainer.test(video_model, dataloaders=video_dataloader)

# %%
