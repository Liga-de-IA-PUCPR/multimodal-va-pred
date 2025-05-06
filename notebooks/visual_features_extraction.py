import os
import torch
from torchvision import transforms
from PIL import Image
from backbones import VisualBackbone  # Assuming you already created this

# Initialize visual backbone
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
visual_backbone = VisualBackbone(out_dim=256).to(device)
visual_backbone.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_visual_features(video_folder):
    features = []
    for frame_file in sorted(os.listdir(video_folder)):  # Ensure frames are processed in order
        frame_path = os.path.join(video_folder, frame_file)
        img = Image.open(frame_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            feature = visual_backbone(img)  # Extract feature
        features.append(feature.cpu())
    return torch.stack(features)  # Shape: [num_frames, 256]

# Example usage
video_folder = "data/raw/cropped_aligned_new_50_vids/461"
visual_features = extract_visual_features(video_folder)
print(visual_features.shape)  # [num_frames, 256]