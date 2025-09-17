import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

# Note: Feature extraction is handled in the LightningModule; no extractor imports here.

import lightning as L
from torchaudio.io import StreamReader
import torchaudio.functional as AF

logger = logging.getLogger(__name__)

def _find_video_file(video_dir: str, video_name: str) -> str | None:
    exts = [".mp4", ".avi", ".mov", ".mkv", ".m4v", ".MP4", ".AVI"]
    for ext in exts:
        p = os.path.join(video_dir, f"{video_name}{ext}")
        if os.path.exists(p):
            return p
    # Fallback: any file matching name.*
    matches = glob.glob(os.path.join(video_dir, f"{video_name}.*"))
    return matches[0] if matches else None

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

        # Feature extraction is now handled in the LightningModule, not here.
        self.audio_feature_fn = audio_feature_fn
        self.image_feature_fn = image_feature_fn

        self.video_entries = []
        ann_files = glob.glob(os.path.join(annotation_dir, "*.txt"))
        logger.debug(f"Loading {split} dataset from {annotation_dir}")
        logger.debug(f"Found {len(ann_files)} annotation files")

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
            video_path = _find_video_file(video_dir, video_name)
            if video_path is None:
                continue
            self.video_entries.append({
                "video_name": video_name,
                "video_path": video_path,
                "annotations": ann_df.reset_index(drop=True),
            })

        logger.debug(f"Loaded {len(self.video_entries)} videos for {split} split")


    def __len__(self):
        return len(self.video_entries)

    def __getitem__(self, idx):
        entry = self.video_entries[idx]
        video_path = entry["video_path"]
        ann_df = entry["annotations"]
        video_name = entry["video_name"]

        window_size_frames = int(self.time_window_sec * self.fps)
        num_frames = len(ann_df)

        # --- Audio decode and cache ---
        # Use per-worker cache to avoid multiprocessing issues
        worker_id = torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0
        cache_key = f"{worker_id}_{video_name}"
        
        if not hasattr(self, "_audio_cache"):
            self._audio_cache = {}
        if cache_key in self._audio_cache:
            audio = self._audio_cache[cache_key]
        else:
            try:
                reader = StreamReader(video_path)
                # Decode full audio; we'll resample to target sample rate if needed
                reader.add_audio_stream(frames_per_chunk=-1)
                chunks = [chunk for (chunk,) in reader.stream() if chunk is not None]
                if len(chunks) > 0:
                    # Concatenate along time dimension and convert to mono
                    tensors = [torch.as_tensor(c) for c in chunks]
                    audio_tc = torch.cat(tensors, dim=0)  # type: ignore[arg-type]
                    if audio_tc.dim() == 2:
                        audio_t = audio_tc.mean(dim=1)
                    else:
                        audio_t = audio_tc
                    # Get source sample rate and resample if needed
                    try:
                        out_info = reader.get_out_stream_info(0)
                        orig_sr = getattr(out_info, "sample_rate", self.audio_sr)
                    except Exception:
                        orig_sr = self.audio_sr
                    if orig_sr != self.audio_sr:
                        audio_resampled = AF.resample(audio_t.unsqueeze(0), orig_sr, self.audio_sr).squeeze(0)
                        audio_np = audio_resampled.cpu().numpy().astype(np.float32)
                    else:
                        audio_np = audio_t.cpu().numpy().astype(np.float32)
                    audio = audio_np
                else:
                    raise RuntimeError("No audio chunks decoded")
            except Exception as e:
                logger.debug(f"torchaudio failed to load audio for {video_name}: {e}. Using silence.")
                duration_sec = num_frames / float(self.fps)
                audio = np.zeros(max(1, int(duration_sec * self.audio_sr)), dtype=np.float32)
            self._audio_cache[cache_key] = audio

        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # --- Compute fixed window length in samples and helper to slice with padding ---
        samples_per_window = int(round(self.time_window_sec * self.audio_sr))
        total_samples = int(audio.shape[0]) if isinstance(audio, np.ndarray) else len(audio)

        def slice_audio_fixed(start_frame_idx: int) -> np.ndarray:
            # Use integer-aligned sample index to avoid cumulative float rounding
            start_sample = int(round(start_frame_idx * self.audio_sr / self.fps))
            end_sample = start_sample + samples_per_window
            # Slice and pad if necessary to enforce fixed length
            if start_sample >= total_samples:
                seg = np.zeros((samples_per_window,), dtype=np.float32)
            else:
                seg = audio[start_sample:min(end_sample, total_samples)].astype(np.float32, copy=False)
                if seg.shape[0] < samples_per_window:
                    pad = np.zeros((samples_per_window - seg.shape[0],), dtype=np.float32)
                    seg = np.concatenate([seg, pad], axis=0)
                elif seg.shape[0] > samples_per_window:
                    seg = seg[:samples_per_window]
            return seg

        node_features, labels, edge_list = [], [], []
        img_paths_per_window = []
        # Preload and sort available frame images for this video (if folder exists)
        frames_dir = os.path.join(self.cropped_img_dir, video_name)
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))) if os.path.isdir(frames_dir) else []

        for start_idx in range(0, num_frames, window_size_frames):
            end_idx = min(start_idx + window_size_frames, num_frames)
            if end_idx - start_idx < window_size_frames:
                continue

            # Fixed-length audio segment per window
            audio_segment = slice_audio_fixed(start_idx)
            # Feature extraction is now handled in LightningModule; we pass raw audio segment
            node_features.append(audio_segment)

            # Collect per-window image paths from preloaded frame list
            if frame_files:
                img_paths = frame_files[start_idx:end_idx]
            else:
                img_paths = []
            img_paths_per_window.append(img_paths)

            valence = ann_df["valence"].iloc[start_idx:end_idx].mean()
            arousal = ann_df["arousal"].iloc[start_idx:end_idx].mean()
            labels.append([valence, arousal])

        num_nodes = len(node_features)
        for i in range(num_nodes - 1):
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])

        # Stack into consistent tensors now that all windows are equal length
        x = torch.as_tensor(np.stack(node_features, axis=0), dtype=torch.float32)
        y = torch.as_tensor(np.array(labels, dtype=np.float32))
        edge_arr = np.array(edge_list, dtype=np.int64)
        edge_index = torch.from_numpy(edge_arr.T).long() if edge_arr.size > 0 else torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y, img_paths=img_paths_per_window)


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
        # Prefer Validation_Set, otherwise try common fallbacks
        cand_val_dirs = [
            os.path.join(self.annotation_root, name)
            for name in ["Validation_Set", "Val_Set", "Valid_Set", "Val", "Validation"]
        ]
        val_ann_dir = next((d for d in cand_val_dirs if os.path.isdir(d) and glob.glob(os.path.join(d, "*.txt"))), None)
        if val_ann_dir is None:
            # Try using Test_Set as validation
            test_cand = os.path.join(self.annotation_root, "Test_Set")
            if os.path.isdir(test_cand) and glob.glob(os.path.join(test_cand, "*.txt")):
                val_ann_dir = test_cand
                logger.warning("Validation annotations not found; using Test_Set for validation.")
            else:
                # Fallback to train annotations to avoid empty val loader
                val_ann_dir = train_ann_dir
                logger.warning("Validation annotations not found; falling back to Train_Set for validation.")

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
        # Test dataset: prefer Test_Set if available; otherwise reuse val
        test_ann_dir = os.path.join(self.annotation_root, "Test_Set")
        if os.path.isdir(test_ann_dir) and glob.glob(os.path.join(test_ann_dir, "*.txt")):
            self.test_dataset = VideoDataset(
                self.video_dir, self.cropped_img_dir, test_ann_dir,
                split="test", fps=self.fps, time_window_sec=self.time_window_sec, audio_sr=self.audio_sr,
                audio_feature_fn=self.audio_feature_fn, image_feature_fn=self.image_feature_fn
            )
        else:
            self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return GeoDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
        )  

    def val_dataloader(self):
        return GeoDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
        )  

    def test_dataloader(self):
        return GeoDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None,
        )

    @staticmethod
    def _worker_init_fn(worker_id):
        """Initialize worker process to avoid CUDA context issues."""
        import torch
        # Set unique seed for each worker
        torch.manual_seed(42 + worker_id)
        if torch.cuda.is_available():
            # Disable CUDA in worker processes to avoid context issues
            torch.cuda.set_device(-1)
