"""
Feature extractors for multimodal video analysis.

This package provides modular audio and vision feature extractors
that can be easily swapped during development.
"""

from .base_extractors import AudioFeatureExtractor, VisionFeatureExtractor
from .audio_extractors import (
    WavLMExtractor,
    HuBERTExtractor,
    Wav2Vec2Extractor,
    MFCCExtractor
)
from .vision_extractors import (
    ResNet50FeatureExtractor,
    I3DExtractor,
    LSTMFeatureExtractor,
    ViTFeatureExtractor
)

__all__ = [
    # Base classes
    'AudioFeatureExtractor',
    'VisionFeatureExtractor',

    # Audio extractors
    'WavLMExtractor',
    'HuBERTExtractor',
    'Wav2Vec2Extractor',
    'MFCCExtractor',

    # Vision extractors
    'ResNet50FeatureExtractor',
    'I3DExtractor',
    'LSTMFeatureExtractor',
    'ViTFeatureExtractor'
]