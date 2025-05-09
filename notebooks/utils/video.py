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
    Extract visual features from video frames stored in a folder.
    Args:
        video_frames_folder: Path to folder containing video frame images (e.g., .jpg, .png)
        num_frames_to_extract: Number of frames to extract features for. Defaults to config.NUM_FRAMES.
    Returns:
        torch.Tensor: Visual features of shape [num_frames_to_extract, VISUAL_BACKBONE_OUT_DIM].
                      Features are returned on CPU.
    """
    default_zero_features = torch.zeros(num_frames_to_extract, VISUAL_BACKBONE_OUT_DIM, device="cpu")
    
    try:
        frame_files = sorted([f for f in os.listdir(video_frames_folder) if os.path.isfile(os.path.join(video_frames_folder, f))])
    except FileNotFoundError:
        print(f"Error: Video frames folder not found: {video_frames_folder}. Returning zeros.")
        return default_zero_features

    if not frame_files:
        print(f"Warning: No frame files found in {video_frames_folder}. Returning zeros.")
        return default_zero_features

    total_frames_available = len(frame_files)
    
    indices = []
    if total_frames_available > 0 :
        if total_frames_available <= num_frames_to_extract:
            base_indices = list(range(total_frames_available))
            if total_frames_available < num_frames_to_extract:
                padding_needed = num_frames_to_extract - total_frames_available
                padding_indices = [total_frames_available - 1] * padding_needed # repeat last frame index
                indices = base_indices + padding_indices
            else: # total_frames_available == num_frames_to_extract
                indices = base_indices
        else: # total_frames_available > num_frames_to_extract
            indices = torch.linspace(0, total_frames_available - 1, num_frames_to_extract).long().tolist()
    
    if not indices and num_frames_to_extract > 0:
        print(f"Warning: Could not determine frame indices for {video_frames_folder} (available: {total_frames_available}, needed: {num_frames_to_extract}). Returning zeros.")
        return default_zero_features

    extracted_features_list = []
    for i, frame_idx in enumerate(indices):
        frame_path = os.path.join(video_frames_folder, frame_files[frame_idx])
        try:
            img = Image.open(frame_path).convert("RGB")
            img_tensor = _transform(img).unsqueeze(0).to(_DEVICE)
            with torch.no_grad():
                feature_vec = _visual_backbone(img_tensor) # [1, VISUAL_BACKBONE_OUT_DIM]
            extracted_features_list.append(feature_vec.cpu()) 
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            print(f"Error loading or processing image {frame_path}: {e}. Using zeros for this frame.")
            # Create a zero feature vector of the correct dimension on CPU
            zero_feature_vec = torch.zeros(1, VISUAL_BACKBONE_OUT_DIM, device="cpu")
            extracted_features_list.append(zero_feature_vec)
            continue
    
    if not extracted_features_list : 
        print(f"Warning: No features were extracted for {video_frames_folder}. Returning zeros.")
        return default_zero_features
        
    # Stack features: list of [1, VISUAL_BACKBONE_OUT_DIM] tensors becomes [num_frames_to_extract, 1, VISUAL_BACKBONE_OUT_DIM]
    stacked_features = torch.cat(extracted_features_list, dim=0) # Using cat since each element is [1, dim]
    
    # final_features shape: [num_frames_to_extract, VISUAL_BACKBONE_OUT_DIM]
    return stacked_features # Already on CPU