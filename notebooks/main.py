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
        **kwargs,
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


class NodeLvelGNN(L.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

        self.acc_train = torchmetrics.Accuracy(task="multiclass", num_classes=7)
        self.acc_val = torchmetrics.Accuracy(task="multiclass", num_classes=7)
        self.acc_test = torchmetrics.Accuracy(task="multiclass", num_classes=7)

    def _shared_step(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown mode: {mode}"

        loss = self.loss_module(x[mask], data.y[mask])
        acc = getattr(self, f"acc_{mode}")(x[mask], data.y[mask])
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, mode="train")
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        self.log("train_acc", acc, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, mode="val")
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self._shared_step(batch, mode="test")
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3
        )
        return optimizer


## modelCheckpoint
ModelCheckpoint = L.pytorch.callbacks.ModelCheckpoint(
    monitor="val_loss", mode="max", save_top_k=3
)
## earlystopping
early_stopping_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss", patience=25
)

# TODO create the  dataset

## csv Logger
csv_logger = CSVLogger("logs", name="cora_logs")
node_dataloader = geom_data.DataLoader(cora_dataset, batch_size=1)
trainer = L.Trainer(
    callbacks=[ModelCheckpoint, early_stopping_callback],
    logger=csv_logger,
    accelerator="auto",
    max_epochs=200,
    precision="bf16-mixed",
)


model = NodeLvelGNN(
    model_name="GCN",
    c_in=1433,
    c_hidden=16,
    c_out=7,
    num_layers=2,
    layer_name="GCN",
    dp_rate=0.1,
)
# model.compile() # discomment if you are using unix OS
model


trainer.fit(model, node_dataloader, node_dataloader)
test_result = trainer.test(model, dataloaders=node_dataloader)