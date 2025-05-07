from torch.utils.data import Dataset
from utils.audio import extract_audio_features_with_mel
from utils.video import extract_visual_features
from utils.graph_construction import build_video_graph
import os


class Affwild2GraphDataset(Dataset):
    def __init__(self, video_ids, root_dir):
        self.video_ids = video_ids
        self.root_dir = root_dir

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_folder = os.path.join(self.root_dir, "cropped_aligned_new_50_vids", str(video_id))
        video_path = os.path.join(self.root_dir, "new_vids", f"{video_id}.mp4")

        # Extract features
        visual_features = extract_visual_features(video_folder)
        audio_features = extract_audio_features_with_mel(video_path)

        # Build graph
        graph_data = build_video_graph(visual_features, audio_features)
        return graph_data
