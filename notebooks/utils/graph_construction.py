import torch
from torch_geometric.data import Data

def build_video_graph(visual_features: torch.Tensor, audio_features: torch.Tensor, 
                      valence: torch.Tensor, arousal: torch.Tensor) -> Data:
    """
    Construct a graph for a single video, representing frames as nodes and temporal connections as edges.
    
    Args:
        visual_features (torch.Tensor): Tensor of shape [num_frames, visual_dim]. Expected on CPU.
        audio_features (torch.Tensor): Tensor of shape [num_frames, audio_dim]. Expected on CPU.
        valence (torch.Tensor): Tensor of shape [num_frames, 1] representing valence values per frame.
        arousal (torch.Tensor): Tensor of shape [num_frames, 1] representing arousal values per frame.
                                       
    Returns:
        torch_geometric.data.Data: A PyG Data object representing the video graph.
                                   Node features are a concatenation of visual, audio, valence, and arousal features.
                                   All tensors in the Data object are on CPU.
    """
    num_frames_visual = visual_features.size(0)
    num_frames_audio = audio_features.size(0)
    num_frames_valence = valence.size(0)
    num_frames_arousal = arousal.size(0)
    
    if num_frames_visual != num_frames_audio or num_frames_visual != num_frames_valence or num_frames_visual != num_frames_arousal:
        raise ValueError(
            f"Mismatch in frame count: visual={num_frames_visual}, audio={num_frames_audio}, "
            f"valence={num_frames_valence}, arousal={num_frames_arousal}. They must be aligned before graph construction."
        )
    
    num_frames = num_frames_visual  # Consistent number of frames

    # Ensure features are on CPU for PyG Data object creation
    visual_features = visual_features.cpu()
    audio_features = audio_features.cpu()
    valence = valence.cpu()
    arousal = arousal.cpu()

    if num_frames == 0:
        print("Warning: Building graph with 0 frames. Node features and edge index will be empty.")
        # Create an empty Data object with correctly typed empty tensors for x and edge_index
        # Infer feature dimension for x from input tensors if possible, otherwise use a placeholder sum if dims are known
        combined_feature_dim = (visual_features.size(1) + audio_features.size(1) + 
                                valence.size(1) + arousal.size(1)) if visual_features.ndim > 1 else 0
        return Data(x=torch.empty((0, combined_feature_dim), dtype=torch.float), 
                    edge_index=torch.empty((2,0), dtype=torch.long))

    node_features = torch.cat([visual_features, audio_features, valence, arousal], dim=1)
    
    edge_index = torch.empty((2, 0), dtype=torch.long)  # Default to no edges
    if num_frames > 1:
        source_nodes = torch.arange(0, num_frames - 1, dtype=torch.long)
        target_nodes = torch.arange(1, num_frames, dtype=torch.long)
        
        edge_index_forward = torch.stack([source_nodes, target_nodes], dim=0)
        edge_index_backward = torch.stack([target_nodes, source_nodes], dim=0)  # Bidirectional
        edge_index = torch.cat([edge_index_forward, edge_index_backward], dim=1)

    data = Data(x=node_features, edge_index=edge_index)
    
    return data