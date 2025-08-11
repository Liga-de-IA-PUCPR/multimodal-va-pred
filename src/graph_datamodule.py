import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

# Imports for advanced feature extractors
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from facenet_pytorch import InceptionResnetV1

from PIL import Image
from torchvision import transforms

import lightning as L
from torchaudio.io import StreamReader

logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    def __init__(
        self,
        video_dir,
        cropped_img_dir,
        annotation_dir,
        split="train",
        fps=30,
        time_window_sec=1.0,
        audio_sr=16000,
        audio_feature_fn=None,
        image_feature_fn=None,
    ):
        self.video_dir = video_dir
        self.cropped_img_dir = cropped_img_dir
        self.annotation_dir = annotation_dir
        self.split = split
        self.fps = fps
        self.time_window_sec = time_window_sec
        self.audio_sr = audio_sr

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Feature extraction will run on device: {self.device}")

        # === 1. Prepare Wav2Vec 2.0 Model for Audio ===
        wav2vec_model_name = "facebook/wav2vec2-base-960h"
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model_name).to(self.device)
        self.wav2vec_model.eval()

        # === 2. Prepare VGGFace (InceptionResnetV1) Model for Images ===
        self.vggface_model = InceptionResnetV1(pretrained='vggface2').to(self.device)
        self.vggface_model.eval()
        self.vggface_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ])

        # === 3. Restore "Hot-Swapping" Functionality ===
        # Use the provided function OR fall back to the new default methods.
        self.audio_feature_fn = audio_feature_fn or self.wav2vec_feature_fn
        self.image_feature_fn = image_feature_fn or self.vggface_feature_fn

        # Load annotation entries (logic unchanged)
        self.video_entries = []
        ann_files = glob.glob(os.path.join(annotation_dir, "*.txt"))
        logger.info(f"Loading {split} dataset from {annotation_dir}")
        logger.info(f"Found {len(ann_files)} annotation files")

        window_size_frames = int(self.time_window_sec * self.fps)

        for ann_path in ann_files:
            video_name = os.path.splitext(os.path.basename(ann_path))[0]
            ann_df = pd.read_csv(ann_path)
            ann_df = ann_df[(ann_df["valence"] != -5) & (ann_df["arousal"] != -5)]
            if ann_df.empty:
                continue
            if len(ann_df) < window_size_frames:
                logger.debug(f"Skipping {video_name} for split {split}: {len(ann_df)} frames < {window_size_frames} required")
                continue
            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            if not os.path.exists(video_path):
                continue
            self.video_entries.append({
                "video_name": video_name,
                "video_path": video_path,
                "annotations": ann_df.reset_index(drop=True),
            })

        logger.info(f"Loaded {len(self.video_entries)} videos for {split} split")
        self.feature_dim = None

    def wav2vec_feature_fn(self, audio_segment, sr):
        """Default audio feature extractor: Wav2Vec 2.0."""
        if audio_segment.size == 0:
            return np.zeros(768, dtype=np.float32)

        inputs = self.wav2vec_processor(audio_segment, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.wav2vec_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        
        return features.squeeze().cpu().numpy()

    def vggface_feature_fn(self, img_paths):
        """Default image feature extractor: VGGFace."""
        imgs = [self.vggface_transform(Image.open(p).convert("RGB")) for p in img_paths if os.path.exists(p)]
        if not imgs:
            return np.zeros(512, dtype=np.float32)
        
        imgs_tensor = torch.stack(imgs).to(self.device)
        
        with torch.no_grad():
            feats = self.vggface_model(imgs_tensor)
        
        return feats.mean(dim=0).cpu().numpy()

    def __len__(self):
        return len(self.video_entries)

    def __getitem__(self, idx):
        entry = self.video_entries[idx]
        video_path = entry["video_path"]
        ann_df = entry["annotations"]
        video_name = entry["video_name"]

        window_size_frames = int(self.time_window_sec * self.fps)
        num_frames = len(ann_df)

        try:
            reader = StreamReader(video_path)
            reader.add_audio_stream(frames_per_chunk=-1)
            chunks = [chunk for (chunk,) in reader.stream()]
            if len(chunks) > 0:
                audio_tc = torch.cat(chunks, dim=0).mean(dim=1, keepdim=True)
                audio_t = audio_tc.squeeze(1).unsqueeze(0)
                duration_sec = num_frames / float(self.fps)
                target_len = max(1, int(duration_sec * self.audio_sr))
                wave = F.interpolate(audio_t.unsqueeze(0), size=target_len, mode='linear', align_corners=False)
                audio = wave.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            else:
                raise RuntimeError("No audio chunks decoded")
        except Exception as e:
            logger.error(f"torchaudio failed to load audio for {video_name}: {e}. Using silence.")
            duration_sec = num_frames / float(self.fps)
            audio = np.zeros(max(1, int(duration_sec * self.audio_sr)), dtype=np.float32)

        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        node_features, labels, edge_list = [], [], []

        for start_idx in range(0, num_frames, window_size_frames):
            end_idx = min(start_idx + window_size_frames, num_frames)
            if end_idx - start_idx < window_size_frames:
                continue

            audio_start_sec = start_idx / self.fps
            audio_end_sec = end_idx / self.fps
            audio_segment = audio[int(audio_start_sec * self.audio_sr): int(audio_end_sec * self.audio_sr)]
            audio_feat = self.audio_feature_fn(audio_segment, self.audio_sr)

            img_paths = [
                os.path.join(self.cropped_img_dir, f"{video_name}_frame_{frame_i:05d}.jpg")
                for frame_i in range(start_idx, end_idx)
            ]
            img_feat = self.image_feature_fn(img_paths)

            combined_feat = np.concatenate([audio_feat, img_feat])
            node_features.append(combined_feat)

            valence = ann_df["valence"].iloc[start_idx:end_idx].mean()
            arousal = ann_df["arousal"].iloc[start_idx:end_idx].mean()
            labels.append([valence, arousal])

        num_nodes = len(node_features)
        for i in range(num_nodes - 1):
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])

        x = torch.tensor(np.array(node_features), dtype=torch.float)
        y = torch.tensor(np.array(labels), dtype=torch.float)
        edge_arr = np.array(edge_list, dtype=np.int64)
        edge_index = torch.from_numpy(edge_arr.T).long() if edge_arr.size > 0 else torch.empty((2, 0), dtype=torch.long)
        
        if self.feature_dim is None and x.nelement() > 0:
            self.feature_dim = x.shape[1]

        return Data(x=x, edge_index=edge_index, y=y)


class VideoDataModule(L.LightningDataModule):
    def __init__(
        self,
        video_dir,
        cropped_img_dir,
        annotation_root,
        fps=30,
        time_window_sec=1.0,
        audio_sr=16000,
        batch_size=1,
        num_workers=4,
        audio_feature_fn=None,
        image_feature_fn=None,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.cropped_img_dir = cropped_img_dir
        self.annotation_root = annotation_root
        self.fps = fps
        self.time_window_sec = time_window_sec
        self.audio_sr = audio_sr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.audio_feature_fn = audio_feature_fn
        self.image_feature_fn = image_feature_fn

    def setup(self, stage=None):
        train_ann_dir = os.path.join(self.annotation_root, "Train_Set")
        val_ann_dir = os.path.join(self.annotation_root, "Validation_Set")

        self.train_dataset = VideoDataset(
            self.video_dir, self.cropped_img_dir, train_ann_dir,
            split="train", fps=self.fps, time_window_sec=self.time_window_sec, audio_sr=self.audio_sr,
            audio_feature_fn=self.audio_feature_fn, image_feature_fn=self.image_feature_fn
        )
        self.val_dataset = VideoDataset(
            self.video_dir, self.cropped_img_dir, val_ann_dir,
            split="val", fps=self.fps, time_window_sec=self.time_window_sec, audio_sr=self.audio_sr,
            audio_feature_fn=self.audio_feature_fn, image_feature_fn=self.image_feature_fn
        )
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return GeoDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return GeoDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return GeoDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)