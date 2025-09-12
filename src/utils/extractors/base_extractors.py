from abc import ABC, abstractmethod
import torch

class AudioFeatureExtractor(ABC):
    """Abstract base class for audio feature extractors."""

    @abstractmethod
    def extract_features(self, audio_windows: torch.Tensor, device: torch.device,
                        feat_chunk_size: int = 8) -> torch.Tensor:
        """Extract features from audio windows.

        Args:
            audio_windows: Tensor of shape (num_nodes, time_steps) containing audio data
            device: Target device for computation
            feat_chunk_size: Batch size for processing

        Returns:
            Tensor of shape (num_nodes, feature_dim)
        """
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        pass

    @abstractmethod
    def to_device(self, device: torch.device):
        """Move model to specified device."""
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Return expected sample rate for audio input."""
        pass


class VisionFeatureExtractor(ABC):
    """Abstract base class for vision feature extractors."""

    @abstractmethod
    def extract_features(self, img_paths_per_node: list[list[str]], device: torch.device,
                        feat_chunk_size: int = 8) -> torch.Tensor:
        """Extract features from image paths.

        Args:
            img_paths_per_node: List of lists containing image paths for each node
            device: Target device for computation
            feat_chunk_size: Batch size for processing

        Returns:
            Tensor of shape (num_nodes, feature_dim)
        """
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        pass

    @abstractmethod
    def to_device(self, device: torch.device):
        """Move model to specified device."""
        pass