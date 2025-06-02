import os
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from utils.backbones import VisualBackbone
from utils.config import NUM_FRAMES, VISUAL_BACKBONE_OUT_DIM

# Initialize visual backbone
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_visual_backbone = VisualBackbone(out_dim=VISUAL_BACKBONE_OUT_DIM).to(_DEVICE)
_visual_backbone.eval()

# Transform for input images
_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_visual_features(video_frames_folder: str, num_frames_to_extract: int = NUM_FRAMES) -> torch.Tensor:
    """
    Extract visual features from video frames stored in a folder, processing the entire video in batches.
    Args:
        video_frames_folder: Path to folder containing video frame images (e.g., .jpg, .png)
        num_frames_to_extract: Number of frames to extract per batch. Defaults to config.NUM_FRAMES.
    Returns:
        torch.Tensor: Visual features of shape [total_frames, VISUAL_BACKBONE_OUT_DIM].
                      Features are returned on CPU.
    """
    try:
        frame_files = sorted([f for f in os.listdir(video_frames_folder) if os.path.isfile(os.path.join(video_frames_folder, f))])
    except FileNotFoundError:
        print(f"Error: Video frames folder not found: {video_frames_folder}. Returning empty tensor.")
        return torch.zeros(0, VISUAL_BACKBONE_OUT_DIM, device="cpu")

    if not frame_files:
        print(f"Warning: No frame files found in {video_frames_folder}. Returning empty tensor.")
        return torch.zeros(0, VISUAL_BACKBONE_OUT_DIM, device="cpu")

    total_frames = len(frame_files)
    all_features = []

    # Process frames in batches
    for batch_start in range(0, total_frames, num_frames_to_extract):
        batch_end = min(batch_start + num_frames_to_extract, total_frames)
        batch_indices = list(range(batch_start, batch_end))
        
        # If this is the last batch and we need padding
        if len(batch_indices) < num_frames_to_extract:
            padding_needed = num_frames_to_extract - len(batch_indices)
            batch_indices.extend([batch_indices[-1]] * padding_needed)

        batch_features = []
        for frame_idx in batch_indices:
            frame_path = os.path.join(video_frames_folder, frame_files[frame_idx])
            try:
                img = Image.open(frame_path).convert("RGB")
                img_tensor = _transform(img).unsqueeze(0).to(_DEVICE)
                with torch.no_grad():
                    feature_vec = _visual_backbone(img_tensor)  # [1, VISUAL_BACKBONE_OUT_DIM]
                batch_features.append(feature_vec.cpu())
            except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
                print(f"Error loading or processing image {frame_path}: {e}. Using zeros for this frame.")
                zero_feature_vec = torch.zeros(1, VISUAL_BACKBONE_OUT_DIM, device="cpu")
                batch_features.append(zero_feature_vec)

        if batch_features:
            batch_tensor = torch.cat(batch_features, dim=0)  # [batch_size, VISUAL_BACKBONE_OUT_DIM]
            all_features.append(batch_tensor)

    if not all_features:
        print(f"Warning: No features were extracted for {video_frames_folder}. Returning empty tensor.")
        return torch.zeros(0, VISUAL_BACKBONE_OUT_DIM, device="cpu")

    # Concatenate all batches
    final_features = torch.cat(all_features, dim=0)  # [total_frames, VISUAL_BACKBONE_OUT_DIM]
    return final_features  # Already on CPU