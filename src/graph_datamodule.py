import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
import lightning as L
from moviepy.editor import VideoFileClip
import librosa

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 video_dir, 
                 cropped_img_dir, 
                 annotation_dir, 
                 split="train", 
                 fps=30, 
                 time_window_sec=1.0,
                 audio_sr=16000,
                 audio_feature_fn=None):
        """
        Args:
            video_dir (str): Path to raw video files.
            cropped_img_dir (str): Path to cropped face images (optional).
            annotation_dir (str): Path to annotation set folder (Train_Set or Validation_Set).
            split (str): 'train', 'val', or 'test'.
            fps (int): Video frames per second (assumed constant).
            time_window_sec (float): How many seconds each node covers.
            audio_sr (int): Audio sample rate for extraction.
            audio_feature_fn (callable): Function to extract audio features. Signature:
                                         fn(audio_segment: np.ndarray, sr: int) -> np.ndarray
        """
        self.video_dir = video_dir
        self.cropped_img_dir = cropped_img_dir
        self.annotation_dir = annotation_dir
        self.split = split
        self.fps = fps
        self.time_window_sec = time_window_sec
        self.audio_sr = audio_sr
        self.audio_feature_fn = audio_feature_fn or self.default_audio_feature_fn

        # Load all annotation files for this split
        self.video_entries = []
        ann_files = glob.glob(os.path.join(annotation_dir, "*.txt"))
        
        for ann_path in ann_files:
            video_name = os.path.splitext(os.path.basename(ann_path))[0]
            ann_df = pd.read_csv(ann_path)
            # Remove -5 entries
            ann_df = ann_df[(ann_df["valence"] != -5) & (ann_df["arousal"] != -5)]
            if ann_df.empty:
                continue
            
            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            if not os.path.exists(video_path):
                continue  # Skip missing videos
            
            self.video_entries.append({
                "video_name": video_name,
                "video_path": video_path,
                "annotations": ann_df.reset_index(drop=True)
            })

    def default_audio_feature_fn(self, audio_segment, sr):
        """Extracts MFCC as default audio features."""
        mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)  # average over time window

    def __len__(self):
        return len(self.video_entries)

    def __getitem__(self, idx):
        entry = self.video_entries[idx]
        video_path = entry["video_path"]
        ann_df = entry["annotations"]

        clip = VideoFileClip(video_path)
        audio = clip.audio.to_soundarray(fps=self.audio_sr)
        # Stereo → mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        window_size_frames = int(self.time_window_sec * self.fps)
        num_frames = len(ann_df)

        node_features = []
        labels = []
        edge_list = []

        # Iterate through chunks of frames
        for start_idx in range(0, num_frames, window_size_frames):
            end_idx = min(start_idx + window_size_frames, num_frames)
            if end_idx - start_idx < window_size_frames:
                continue  # skip incomplete chunk

            # Audio slice corresponding to this chunk
            audio_start_sec = start_idx / self.fps
            audio_end_sec = end_idx / self.fps
            audio_start_sample = int(audio_start_sec * self.audio_sr)
            audio_end_sample = int(audio_end_sec * self.audio_sr)
            audio_segment = audio[audio_start_sample:audio_end_sample]

            # Extract audio features
            audio_feat = self.audio_feature_fn(audio_segment, self.audio_sr)

            # (Optional) you could also add image/video features here
            # For now: just audio
            node_features.append(audio_feat)

            # Mean label over this window
            valence = ann_df["valence"].iloc[start_idx:end_idx].mean()
            arousal = ann_df["arousal"].iloc[start_idx:end_idx].mean()
            labels.append([valence, arousal])

        # Build edges (temporal chain)
        num_nodes = len(node_features)
        for i in range(num_nodes - 1):
            edge_list.append([i, i+1])
            edge_list.append([i+1, i])

        x = torch.tensor(np.array(node_features), dtype=torch.float)
        y = torch.tensor(np.array(labels), dtype=torch.float)
        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data


class VideoDataModule(L.LightningDataModule):
    def __init__(self, 
                 video_dir,
                 cropped_img_dir,
                 annotation_root,
                 fps=30,
                 time_window_sec=1.0,
                 audio_sr=16000,
                 audio_feature_fn=None,
                 batch_size=1,
                 num_workers=4):
        super().__init__()
        self.video_dir = video_dir
        self.cropped_img_dir = cropped_img_dir
        self.annotation_root = annotation_root
        self.fps = fps
        self.time_window_sec = time_window_sec
        self.audio_sr = audio_sr
        self.audio_feature_fn = audio_feature_fn
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_ann_dir = os.path.join(self.annotation_root, "Train_Set")
        val_ann_dir = os.path.join(self.annotation_root, "Validation_Set")
        # For now, using Validation_Set as test set as well
        self.train_dataset = VideoDataset(
            self.video_dir, self.cropped_img_dir, train_ann_dir,
            split="train", fps=self.fps, time_window_sec=self.time_window_sec,
            audio_sr=self.audio_sr, audio_feature_fn=self.audio_feature_fn
        )
        self.val_dataset = VideoDataset(
            self.video_dir, self.cropped_img_dir, val_ann_dir,
            split="val", fps=self.fps, time_window_sec=self.time_window_sec,
            audio_sr=self.audio_sr, audio_feature_fn=self.audio_feature_fn
        )
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return GeoDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return GeoDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return GeoDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
