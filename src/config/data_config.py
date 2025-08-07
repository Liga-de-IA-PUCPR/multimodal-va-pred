"""
Configuration file for MultimodalVideoDataModule
Defines different configurations for various use cases
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for the DataModule"""
    
    # Data paths
    data_dir: str = "/path/to/your/video/dataset"
    video_dir: str = "videos"
    audio_dir: str = "audio"
    annotation_dir: str = "annotations"
    
    # Data splitting
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Data loading
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    
    # Video processing
    temporal_window: int = 5
    max_frames: Optional[int] = 300
    
    # Feature dimensions
    video_feature_dim: int = 2048  # I3D features
    audio_feature_dim: int = 768   # HuBERT features
    
    # Classification
    num_classes: int = 3  # e.g., low/medium/high valence


# Predefined configurations for different scenarios

# Configuration for emotion recognition (valence/arousal)
EMOTION_CONFIG = DataConfig(
    data_dir="/path/to/emotion/dataset",
    num_classes=3,  # Low/Medium/High
    video_feature_dim=2048,  # I3D RGB features
    audio_feature_dim=768,   # HuBERT features
    max_frames=300,
    temporal_window=5,
    batch_size=1,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)

# Configuration for action recognition
ACTION_CONFIG = DataConfig(
    data_dir="/path/to/action/dataset", 
    num_classes=10,  # 10 different actions
    video_feature_dim=2048,  # I3D RGB features
    audio_feature_dim=512,   # Smaller audio features
    max_frames=200,          # Shorter sequences
    temporal_window=3,       # Smaller temporal window
    batch_size=2,           # Larger batch size
    train_split=0.8,
    val_split=0.1,
    test_split=0.1
)

# Configuration for testing/development
MOCK_CONFIG = DataConfig(
    data_dir="/tmp/mock_data",
    num_classes=2,           # Binary classification
    video_feature_dim=2,     # Small features for testing
    audio_feature_dim=2,     # Small features for testing
    max_frames=100,          # Smaller videos
    temporal_window=3,
    batch_size=1,
    num_workers=0,           # Single-threaded for debugging
    train_split=0.6,
    val_split=0.2,
    test_split=0.2
)

# Configuration for large-scale experiments
LARGE_SCALE_CONFIG = DataConfig(
    data_dir="/path/to/large/dataset",
    num_classes=5,
    video_feature_dim=4096,  # Larger features
    audio_feature_dim=1024,  # Larger audio features
    max_frames=500,          # Longer sequences
    temporal_window=10,      # Larger temporal window
    batch_size=4,           # Larger batches
    num_workers=8,          # More workers
    pin_memory=True,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1
)
