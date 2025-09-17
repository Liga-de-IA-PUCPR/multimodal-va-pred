import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from .base_extractors import VisionFeatureExtractor


class ResNet50FeatureExtractor(VisionFeatureExtractor):
    """ResNet50-based vision feature extractor."""

    def __init__(self, img_pool_num: int = 8):
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        modules = list(backbone.children())[:-1]  # remove FC, keep avgpool
        self.model = nn.Sequential(*modules)
        self.model.eval()
        self.transforms = weights.transforms()
        self.feature_dim = 2048
        self.img_pool_num = img_pool_num

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def to_device(self, device: torch.device):
        self.model.to(device)

    def extract_features(self, img_paths_per_node: list[list[str]], device: torch.device,
                        feat_chunk_size: int = 8) -> torch.Tensor:
        """Extract features using ResNet50 with frame pooling."""
        n = len(img_paths_per_node)

        # Flatten selected frames for batch processing
        flat_imgs: list[torch.Tensor] = []
        flat_nodes: list[int] = []

        for i in range(n):
            paths = img_paths_per_node[i] if img_paths_per_node[i] is not None else []
            if len(paths) == 0:
                continue
            if len(paths) <= self.img_pool_num:
                sel = paths
            else:
                idxs = np.linspace(0, len(paths) - 1, num=self.img_pool_num)
                sel = [paths[int(round(j))] for j in idxs]

            for p in sel:
                if isinstance(p, str) and os.path.isfile(p):
                    try:
                        img = Image.open(p).convert("RGB")
                        img_t = self.transforms(img)
                        flat_imgs.append(img_t)
                        flat_nodes.append(i)
                    except Exception:
                        pass

        image_feats = torch.zeros(n, self.feature_dim, device=device, dtype=torch.float32)
        if len(flat_imgs) > 0:
            counts = torch.zeros(n, device=device, dtype=torch.float32)
            for s in range(0, len(flat_imgs), feat_chunk_size):
                e = min(len(flat_imgs), s + feat_chunk_size)
                sub_imgs = torch.stack(flat_imgs[s:e]).to(device, non_blocking=True)
                node_idx = torch.as_tensor(flat_nodes[s:e], device=device, dtype=torch.long)

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                    feats = self.model(sub_imgs).flatten(1)

                image_feats.index_add_(0, node_idx, feats.float())
                counts.index_add_(0, node_idx, torch.ones_like(node_idx, dtype=torch.float32))

            counts = counts.clamp_min_(1.0).unsqueeze(1)
            image_feats = image_feats / counts

        return image_feats


class I3DExtractor(VisionFeatureExtractor):
    """I3D-based vision feature extractor for video sequences."""

    def __init__(self, kinetics_pretrained: bool = True, expected_frames: int = 16):
        # Import I3D model - you'll need to implement or import the actual I3D
        # This assumes you have the I3D model available
        try:
            from models.DeepMind_I3D.pytorch_i3d_new import InceptionI3d
            self.model = InceptionI3d(num_classes=400, in_channels=3)
            if kinetics_pretrained:
                # Load pretrained weights if available
                pretrained_path = "models/DeepMind_I3D/pretrained/rgb_imagenet.pt"
                if os.path.exists(pretrained_path):
                    self.model.load_state_dict(torch.load(pretrained_path))
            self.model.eval()
            self.feature_dim = 1024  # Adjust based on actual I3D output
        except ImportError:
            # Fallback if I3D not available
            print("I3D model not found, using ResNet50 as fallback")
            self.model = ResNet50FeatureExtractor()
            self.feature_dim = self.model.get_feature_dim()

        self.expected_frames = expected_frames

        # I3D preprocessing
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def to_device(self, device: torch.device):
        self.model.to(device)

    def extract_features(self, img_paths_per_node: list[list[str]], device: torch.device,
                        feat_chunk_size: int = 4) -> torch.Tensor:
        """Extract features using I3D on fixed-length temporal sequences."""
        n = len(img_paths_per_node)
        image_feats = torch.zeros(n, self.feature_dim, device=device, dtype=torch.float32)

        for i in range(n):
            paths = img_paths_per_node[i] if img_paths_per_node[i] is not None else []
            if len(paths) == 0:
                continue

            # Sample or pad to expected_frames
            if len(paths) >= self.expected_frames:
                # Sample evenly distributed frames
                idxs = np.linspace(0, len(paths) - 1, num=self.expected_frames, dtype=int)
                sel = [paths[j] for j in idxs]
            else:
                # Pad with the last frame if not enough frames
                sel = paths + [paths[-1]] * (self.expected_frames - len(paths))

            # Load and transform frames
            frames = []
            for p in sel:
                if isinstance(p, str) and os.path.isfile(p):
                    try:
                        img = Image.open(p).convert("RGB")
                        img_t = self.transforms(img)
                        frames.append(img_t)
                    except Exception:
                        # If loading fails, use a zero tensor as placeholder
                        frames.append(torch.zeros(3, 224, 224))
                else:
                    frames.append(torch.zeros(3, 224, 224))

            if len(frames) != self.expected_frames:
                continue

            # Stack frames into video tensor: (T, C, H, W) -> (T, 3, 224, 224)
            video_tensor = torch.stack(frames).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                # I3D forward pass - adjust based on actual model interface
                # This is a placeholder - you'll need to implement the actual feature extraction
                try:
                    features = self.model.extract_features(video_tensor)  # This method needs to be implemented
                    # Pool temporally to get single feature vector
                    pooled_features = torch.mean(features, dim=1)  # (1, feature_dim)
                except AttributeError:
                    # Fallback to frame-wise processing if extract_features not available
                    frame_features = []
                    for frame in frames:
                        frame_feat = self.model(frame.unsqueeze(0)).flatten(1)
                        frame_features.append(frame_feat)
                    pooled_features = torch.mean(torch.stack(frame_features), dim=0).unsqueeze(0)

            image_feats[i] = pooled_features.squeeze(0).float()

        return image_feats


class LSTMFeatureExtractor(VisionFeatureExtractor):
    """LSTM-based temporal feature extractor using ResNet50 + LSTM."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, num_layers: int = 2, max_sequence_length: int = 10):
        # First extract frame features with ResNet50
        self.frame_extractor = ResNet50FeatureExtractor()

        # Then process temporally with LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.feature_dim = hidden_dim * 2  # bidirectional

        # Optional projection layer
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

        self.max_sequence_length = max_sequence_length

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def to_device(self, device: torch.device):
        self.frame_extractor.to_device(device)
        self.lstm.to(device)
        self.projection.to(device)

    def extract_features(self, img_paths_per_node: list[list[str]], device: torch.device,
                        feat_chunk_size: int = 8) -> torch.Tensor:
        """Extract features using ResNet50 + LSTM on frame sequences."""
        n = len(img_paths_per_node)

        # First get frame-level features for all frames
        all_frame_features = []
        node_frame_counts = []

        for i in range(n):
            paths = img_paths_per_node[i] if img_paths_per_node[i] is not None else []
            if len(paths) == 0:
                all_frame_features.append(torch.zeros(1, self.frame_extractor.get_feature_dim(), device=device))
                node_frame_counts.append(1)
                continue

            # Extract features for frames of this node, up to max_sequence_length
            num_frames = min(len(paths), self.max_sequence_length)
            node_paths = [[p] for p in paths[:num_frames]]  # Each frame as separate "node"
            frame_features = self.frame_extractor.extract_features(node_paths, device,
                                                                feat_chunk_size=feat_chunk_size)
            all_frame_features.append(frame_features)
            node_frame_counts.append(num_frames)

        # Process each node's frame sequence through LSTM
        image_feats = torch.zeros(n, self.feature_dim, device=device, dtype=torch.float32)

        for i in range(n):
            frame_seq = all_frame_features[i]  # (num_frames, feature_dim)

            if frame_seq.size(0) == 0:
                continue

            # LSTM expects (batch_size, seq_len, input_dim)
            frame_seq = frame_seq.unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                lstm_out, (h_n, c_n) = self.lstm(frame_seq)
                # Use final hidden state (concatenate forward and backward)
                final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # bidirectional
                projected = self.projection(final_hidden)

            image_feats[i] = projected.squeeze(0)

        return image_feats


class ViTFeatureExtractor(VisionFeatureExtractor):
    """Vision Transformer-based feature extractor."""

    def __init__(self, model_name: str = "google/vit-base-patch16-224", img_pool_num: int = 8):
        from transformers import ViTModel, ViTFeatureExtractor
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.eval()
        self.feature_dim = self.model.config.hidden_size  # 768 for base
        self.img_pool_num = img_pool_num

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def to_device(self, device: torch.device):
        self.model.to(device)

    def extract_features(self, img_paths_per_node: list[list[str]], device: torch.device,
                        feat_chunk_size: int = 8) -> torch.Tensor:
        """Extract features using ViT with frame pooling."""
        n = len(img_paths_per_node)

        # Flatten selected frames for batch processing
        flat_imgs: list[torch.Tensor] = []
        flat_nodes: list[int] = []

        for i in range(n):
            paths = img_paths_per_node[i] if img_paths_per_node[i] is not None else []
            if len(paths) == 0:
                continue
            if len(paths) <= self.img_pool_num:
                sel = paths
            else:
                idxs = np.linspace(0, len(paths) - 1, num=self.img_pool_num)
                sel = [paths[int(round(j))] for j in idxs]

            for p in sel:
                if isinstance(p, str) and os.path.isfile(p):
                    try:
                        img = Image.open(p).convert("RGB")
                        # ViT feature extractor handles preprocessing
                        inputs = self.feature_extractor(images=img, return_tensors="pt")
                        img_t = inputs['pixel_values'].squeeze(0)
                        flat_imgs.append(img_t)
                        flat_nodes.append(i)
                    except Exception:
                        pass

        image_feats = torch.zeros(n, self.feature_dim, device=device, dtype=torch.float32)
        if len(flat_imgs) > 0:
            counts = torch.zeros(n, device=device, dtype=torch.float32)
            for s in range(0, len(flat_imgs), feat_chunk_size):
                e = min(len(flat_imgs), s + feat_chunk_size)
                sub_imgs = torch.stack(flat_imgs[s:e]).to(device, non_blocking=True)
                node_idx = torch.as_tensor(flat_nodes[s:e], device=device, dtype=torch.long)

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                    outputs = self.model(sub_imgs)
                    # Use [CLS] token representation
                    feats = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

                image_feats.index_add_(0, node_idx, feats.float())
                counts.index_add_(0, node_idx, torch.ones_like(node_idx, dtype=torch.float32))

            counts = counts.clamp_min_(1.0).unsqueeze(1)
            image_feats = image_feats / counts

        return image_feats