# Multimodal Video DataModule

This directory contains a Lightning DataModule for multimodal video analysis using Graph Neural Networks (GNNs). The DataModule handles video and audio data processing, creates temporal graphs, and integrates with PyTorch Lightning for streamlined training.

## Features

- **Multimodal Processing**: Handles both video and audio data
- **Temporal Graph Construction**: Creates graphs where video frames are nodes connected by temporal edges
- **Feature Extraction Hooks**: Provides skeleton for I3D video features and HuBERT audio features
- **Flexible Configuration**: Supports different dataset formats and use cases
- **Lightning Integration**: Full PyTorch Lightning DataModule with train/val/test splits

## File Structure

```
src/
├── data_module.py          # Main DataModule implementation
├── example_usage.py        # Example of how to use the DataModule
└── config/
    └── data_config.py      # Configuration classes and presets
```

## Quick Start

### 1. Using the Mock DataModule (for testing)

```python
from src.data_module import MockMultimodalVideoDataModule
import lightning as L

# Create mock datamodule for testing
datamodule = MockMultimodalVideoDataModule(
    num_videos=10,
    batch_size=1,
    num_classes=2,
    video_feature_dim=2,    # Small for testing
    audio_feature_dim=2,    # Small for testing
)

# Setup and use with Lightning
trainer = L.Trainer(max_epochs=10)
model = YourGNNModel(...)
trainer.fit(model, datamodule)
```

### 2. Using with Real Data

```python
from src.data_module import MultimodalVideoDataModule

# Configure for your dataset
datamodule = MultimodalVideoDataModule(
    data_dir="/path/to/your/dataset",
    video_dir="videos",           # Subdirectory with video files
    audio_dir="audio",            # Subdirectory with audio files  
    annotation_dir="annotations", # Subdirectory with label files
    video_feature_dim=2048,       # I3D feature dimension
    audio_feature_dim=768,        # HuBERT feature dimension
    num_classes=3,                # Number of output classes
    temporal_window=5,            # Frames to connect in graph
    max_frames=300,               # Maximum frames per video
    batch_size=1,
    num_workers=4
)
```

## Data Format Requirements

### Directory Structure
```
your_dataset/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── audio/
│   ├── video_001.wav
│   ├── video_002.wav  
│   └── ...
└── annotations/
    ├── video_001.csv
    ├── video_002.csv
    └── ...
```

### Annotation Format
Annotation files should contain frame-level labels. The current implementation expects:
- CSV files with frame-level annotations
- One label per frame
- Labels should be integers (0, 1, 2, ... for multi-class)

**Example CSV format:**
```csv
frame,label
0,1
1,1
2,0
3,2
...
```

## Implementation Status

### ✅ Completed
- DataModule skeleton with Lightning integration
- Temporal graph construction
- Train/validation/test splitting
- Mock data generation for testing
- Configuration system

### 🚧 TODO (Implementation Required)
- **Video Feature Extraction**: Replace mock implementation with actual I3D feature extraction
- **Audio Feature Extraction**: Replace mock implementation with actual HuBERT feature extraction  
- **Annotation Loading**: Implement loading of different annotation formats
- **Data Validation**: Add checks for data integrity
- **Preprocessing**: Add video/audio preprocessing pipelines

## Key Components

### VideoGraphDataset
- Loads individual videos as temporal graphs
- Handles multimodal feature combination
- Creates temporal edge connections between frames

### MultimodalVideoDataModule  
- Lightning DataModule for full training pipeline
- Handles data splitting and loading
- Manages file discovery and pairing

### MockMultimodalVideoDataModule
- Testing version that generates synthetic data
- Useful for development and debugging
- Mimics the structure of real data

## Integration with Existing GNN Model

The DataModule is designed to work with the existing `VideoSequentialGNN` model:

```python
# The DataModule produces graphs with combined video+audio features
total_features = video_feature_dim + audio_feature_dim

model = VideoSequentialGNN(
    model_name="GCN",
    c_in=total_features,      # Combined feature dimension
    c_hidden=64,
    c_out=num_classes,
    num_classes=num_classes,
    num_layers=3,
    layer_name="GCN",
    dp_rate=0.2
)
```

## Configuration

Use predefined configurations from `config/data_config.py`:

```python
from src.config.data_config import EMOTION_CONFIG, ACTION_CONFIG, MOCK_CONFIG

# For emotion recognition
datamodule = MultimodalVideoDataModule(**EMOTION_CONFIG.__dict__)

# For testing/development  
datamodule = MockMultimodalVideoDataModule(**MOCK_CONFIG.__dict__)
```

## Next Steps for Implementation

1. **Video Feature Extraction**:
   ```python
   def _extract_video_features(self, video_path: str) -> torch.Tensor:
       # Load video with cv2 or torchvision
       # Preprocess frames (resize, normalize)
       # Extract I3D features
       # Return tensor of shape (num_frames, 2048)
   ```

2. **Audio Feature Extraction**:
   ```python
   def _extract_audio_features(self, audio_path: str) -> torch.Tensor:
       # Load audio with librosa/torchaudio
       # Extract HuBERT features
       # Align with video frame timing
       # Return tensor of shape (num_frames, 768)
   ```

3. **Annotation Loading**:
   ```python
   def _load_annotations(self, annotation_path: str, num_frames: int) -> torch.Tensor:
       # Load CSV/JSON annotations
       # Handle temporal alignment
       # Convert to frame-level integer labels
       # Return tensor of shape (num_frames,)
   ```

## Example Usage

See `example_usage.py` for a complete example of training with the DataModule:

```bash
cd src
python example_usage.py
```

This will:
1. Create a mock dataset
2. Initialize the GNN model  
3. Train for a few epochs
4. Show data inspection capabilities
