import torch
from torch_geometric.data import Data

def build_video_graph(visual_features, audio_features):
    """
    Construct a graph for a single video.
    Args:
        visual_features: Tensor [num_frames, visual_dim]
        audio_features: Tensor [audio_dim]
    Returns:
        data: PyTorch Geometric Data object
    """
    num_frames = visual_features.size(0)
    node_features = torch.cat([visual_features, audio_features.unsqueeze(0).repeat(num_frames, 1)], dim=1)  # [num_frames, visual_dim + audio_dim]
    
    # Temporal edges: frame_t ↔ frame_{t+1}
    edge_index = torch.tensor(
        [[i, i+1] for i in range(num_frames-1)] +
        [[i+1, i] for i in range(num_frames-1)],  # Bidirectional
        dtype=torch.long
    ).t()

    # Cross-modal edges: frame_t ↔ audio
    audio_idx = num_frames  # Last node is audio
    cross_edges = torch.tensor(
        [[i, audio_idx] for i in range(num_frames)] +
        [[audio_idx, i] for i in range(num_frames)],  # Bidirectional
        dtype=torch.long
    ).t()

    # Combine edges
    edge_index = torch.cat([edge_index, cross_edges], dim=1)

    # Create Data object
    data = Data(x=node_features, edge_index=edge_index)
    return data

