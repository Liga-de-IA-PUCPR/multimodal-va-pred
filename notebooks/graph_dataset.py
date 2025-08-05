import torch
import torch_geometric.data as geom_data

class TemporalGraphBuilder:
    """
    Builds a temporal graph from video features and annotations.
    
    Each node in the graph represents a time segment of the video.
    Edges are created between consecutive segments and contextually distant segments.
    """
    def __init__(self, time_interval=1.0, context_k=None):
        """
        Args:
            time_interval (float): The duration of each time segment in seconds.
            context_k (list of int, optional): A list of temporal distances (k) 
                                               to create context edges between nodes 
                                               (t and t+k). Defaults to [1, 2, 3, 4].
        """
        self.time_interval = time_interval
        self.context_k = context_k if context_k is not None else [1, 2, 3, 4]
    
    def _create_temporal_edges(self, num_nodes):
        """
        Creates temporal edges for the graph.
        
        Includes:
        - Consecutive edges (t -> t+1)
        - Context edges (t -> t+k)
        All edges are created as bidirectional.
        """
        edges = []
        
        # Consecutive temporal edges
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        
        # Context edges for longer temporal dependencies
        for k in self.context_k:
            for i in range(num_nodes - k):
                edges.append([i, i + k])
                edges.append([i + k, i])
        
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
            
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def build_video_graph(self, video_id, features, annotations, fps):
        """
        Builds a single temporal graph for a video.
        
        Args:
            video_id (int or str): The identifier for the video.
            features (torch.Tensor): A tensor of features for each frame.
            annotations (torch.Tensor): A tensor of annotations (e.g., valence, arousal) for each frame.
            fps (float): The frames per second of the video.
            
        Returns:
            torch_geometric.data.Data: A graph data object for the video.
        """
        if features is None or len(features) == 0:
            return None

        frames_per_segment = int(fps * self.time_interval)
        if frames_per_segment == 0:
            return None # Cannot create segments if interval is smaller than frame time

        num_segments = len(features) // frames_per_segment
        
        segment_features = []
        segment_labels = []
        
        for i in range(num_segments):
            start_frame = i * frames_per_segment
            end_frame = start_frame + frames_per_segment
            
            # Aggregate features for the segment (e.g., by averaging)
            seg_feat = torch.mean(features[start_frame:end_frame], dim=0)
            segment_features.append(seg_feat)
            
            # Aggregate annotations for the segment
            seg_label = torch.mean(annotations[start_frame:end_frame], dim=0)
            segment_labels.append(seg_label)
            
        if not segment_features:
            return None

        # Create edges based on the number of segments
        edge_index = self._create_temporal_edges(num_segments)
        
        # Create the PyG Data object
        graph_data = geom_data.Data(
            x=torch.stack(segment_features),
            y=torch.stack(segment_labels),
            edge_index=edge_index,
            video_id=video_id,
            num_segments=num_segments
        )
        
        return graph_data