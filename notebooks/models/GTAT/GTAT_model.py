import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax, remove_self_loops, add_self_loops
from torch_geometric.typing import OptTensor, Adj, Size, PairTensor
from typing import Union, Tuple, Optional

from utils.config import VISUAL_BACKBONE_OUT_DIM, AUDIO_MEL_N_MELS

class GTATConv(MessagePassing):
    _alpha1: OptTensor
    _alpha2: OptTensor

    def __init__(self, in_channels: int, out_channels: int, heads: int,
                 topology_channels: int = 15,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0., add_self_loops: bool = True,
                 bias: bool = True, share_weights: bool = False, **kwargs):
        super(GTATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.topology_channels = topology_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
        
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)
        
        self.att = Parameter(torch.Tensor(1, heads, out_channels))
        self.att2 = Parameter(torch.Tensor(1, heads, self.topology_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha1 = None
        self._alpha2 = None

        self.bias2 = Parameter(torch.Tensor(self.topology_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att)
        glorot(self.att2)
        zeros(self.bias)
        zeros(self.bias2)

    def forward(self, x: Union[torch.Tensor, PairTensor], edge_index: Adj,
                topology: torch.Tensor,
                size: Size = None, return_attention_weights: bool = None):
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, torch.Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)  # (N, heads, features)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None
        topology = topology.unsqueeze(dim=1)
        topology = topology.repeat(1, self.heads, 1)
        x_l = torch.cat((x_l, topology), dim=-1)
        x_r = torch.cat((x_r, topology), dim=-1)

        if self.add_self_loops:
            if isinstance(edge_index, torch.Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out_all = self.propagate(edge_index, x=(x_l, x_r), size=size)
        out = out_all[:, :, :self.out_channels]
        out2 = out_all[:, :, self.out_channels:]
        alpha1 = self._alpha1
        self._alpha1 = None
        alpha2 = self._alpha2
        self._alpha2 = None

        if self.concat:
            out = out.reshape(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        out2 = out2.mean(dim=1)
        out2 += self.bias2

        if isinstance(return_attention_weights, bool):
            assert alpha1 is not None and alpha2 is not None
            if isinstance(edge_index, torch.Tensor):
                return out, (edge_index, (alpha1, alpha2))
            # For SparseTensor handling (removed for brevity)
        else:
            return out, out2

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, index: torch.Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        x = x_i + x_j
        alpha1 = (x[:, :, :self.out_channels] * self.att).sum(dim=-1)
        alpha2 = (x[:, :, self.out_channels:] * self.att2).sum(dim=-1)
        alpha1 = F.leaky_relu(alpha1, self.negative_slope)
        alpha2 = F.leaky_relu(alpha2, self.negative_slope)
        alpha1 = softmax(alpha1, index, ptr, size_i)
        alpha2 = softmax(alpha2, index, ptr, size_i)
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        alpha1 = F.dropout(alpha1, p=self.dropout, training=self.training)
        alpha2 = F.dropout(alpha2, p=self.dropout, training=self.training)
        return torch.cat((x_j[:, :, :self.out_channels] * alpha2.unsqueeze(-1), 
                         x_j[:, :, self.out_channels:] * alpha1.unsqueeze(-1)), dim=-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GCATopo(nn.Module):
    """
    Graph Topology Aware Transformer Model for Emotion Recognition
    Predicts valence and arousal based on graph structure from video frames
    """
    def __init__(self, 
                 input_dim=VISUAL_BACKBONE_OUT_DIM + AUDIO_MEL_N_MELS, 
                 hidden_dim=128,
                 topology_dim=15, 
                 num_gtat_layers=2, 
                 gtat_heads=4, 
                 dropout=0.2):
        super(GCATopo, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.topology_dim = topology_dim

        # Topology feature extractor - extracts topology features from node features
        self.topology_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, topology_dim)
        )

        # GTAT layers
        self.gtat_layers = nn.ModuleList()
        
        # First GTAT layer takes input_dim features
        self.gtat_layers.append(
            GTATConv(
                in_channels=input_dim, 
                out_channels=hidden_dim,
                heads=gtat_heads,
                topology_channels=topology_dim,
                dropout=dropout
            )
        )
        
        # Subsequent GTAT layers take hidden_dim*gtat_heads features
        for _ in range(num_gtat_layers - 1):
            self.gtat_layers.append(
                GTATConv(
                    in_channels=hidden_dim * gtat_heads, 
                    out_channels=hidden_dim,
                    heads=gtat_heads,
                    topology_channels=topology_dim,
                    dropout=dropout
                )
            )
        
        # Output layer for valence and arousal prediction
        self.valence_out = nn.Sequential(
            nn.Linear(hidden_dim * gtat_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.arousal_out = nn.Sequential(
            nn.Linear(hidden_dim * gtat_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Extract topology features
        topology = self.topology_extractor(x)
        
        # Apply GTAT layers
        for i, gtat_layer in enumerate(self.gtat_layers):
            x, topology = gtat_layer(x, edge_index, topology)
        
        # Global mean pooling
        # For a batched graph, you would use global_mean_pool from PyG
        # But for a single graph, we can just take the mean
        if hasattr(data, 'batch') and data.batch is not None:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, data.batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Predict valence and arousal
        valence = self.valence_out(x)
        arousal = self.arousal_out(x)
        
        return valence, arousal