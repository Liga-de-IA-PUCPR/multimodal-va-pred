import os
import torch
import torch.nn as nn
import numpy as np
from .base_extractors import AudioFeatureExtractor


class WavLMExtractor(AudioFeatureExtractor):
    """WavLM-based audio feature extractor."""

    def __init__(self, model_name: str = "microsoft/wavlm-base-plus"):
        from transformers import WavLMModel, AutoFeatureExtractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.eval()
        self.sample_rate = self.feature_extractor.sampling_rate
        self.feature_dim = self.model.config.hidden_size  # 768 for base-plus

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def to_device(self, device: torch.device):
        self.model.to(device)

    def extract_features(self, audio_windows: torch.Tensor, device: torch.device,
                        feat_chunk_size: int = 8) -> torch.Tensor:
        """Extract features using WavLM."""
        n = audio_windows.shape[0]
        audio_list = [audio_windows[i].detach().cpu().numpy() for i in range(n)]
        audio_feats = torch.empty(n, self.feature_dim, device=device, dtype=torch.float32)

        for s in range(0, n, feat_chunk_size):
            e = min(n, s + feat_chunk_size)
            sub_list = audio_list[s:e]
            inputs = self.feature_extractor(sub_list, sampling_rate=self.sample_rate,
                                          return_tensors="pt", padding=True)
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                out = self.model(**inputs)
                hidden = out.last_hidden_state  # (B, T_feat, hidden_size)
                # Mean pooling over time dimension
                pooled = hidden.mean(dim=1)

            audio_feats[s:e] = pooled.float()

        return audio_feats


class HuBERTExtractor(AudioFeatureExtractor):
    """HuBERT-based audio feature extractor."""

    def __init__(self, model_name: str = "facebook/hubert-base-ls960"):
        from transformers import HubertModel, AutoFeatureExtractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.model.eval()
        self.sample_rate = self.feature_extractor.sampling_rate
        self.feature_dim = self.model.config.hidden_size  # 768 for base

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def to_device(self, device: torch.device):
        self.model.to(device)

    def extract_features(self, audio_windows: torch.Tensor, device: torch.device,
                        feat_chunk_size: int = 8) -> torch.Tensor:
        """Extract features using HuBERT."""
        n = audio_windows.shape[0]
        audio_list = [audio_windows[i].detach().cpu().numpy() for i in range(n)]
        audio_feats = torch.empty(n, self.feature_dim, device=device, dtype=torch.float32)

        for s in range(0, n, feat_chunk_size):
            e = min(n, s + feat_chunk_size)
            sub_list = audio_list[s:e]
            inputs = self.feature_extractor(sub_list, sampling_rate=self.sample_rate,
                                          return_tensors="pt", padding=True)
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                out = self.model(**inputs)
                hidden = out.last_hidden_state  # (B, T_feat, hidden_size)
                # Mean pooling over time dimension
                pooled = hidden.mean(dim=1)

            audio_feats[s:e] = pooled.float()

        return audio_feats


class Wav2Vec2Extractor(AudioFeatureExtractor):
    """Wav2Vec2-based audio feature extractor."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        from transformers import Wav2Vec2Model, AutoFeatureExtractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()
        self.sample_rate = self.feature_extractor.sampling_rate
        self.feature_dim = self.model.config.hidden_size  # 768 for base

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def to_device(self, device: torch.device):
        self.model.to(device)

    def extract_features(self, audio_windows: torch.Tensor, device: torch.device,
                        feat_chunk_size: int = 8) -> torch.Tensor:
        """Extract features using Wav2Vec2."""
        n = audio_windows.shape[0]
        audio_list = [audio_windows[i].detach().cpu().numpy() for i in range(n)]
        audio_feats = torch.empty(n, self.feature_dim, device=device, dtype=torch.float32)

        for s in range(0, n, feat_chunk_size):
            e = min(n, s + feat_chunk_size)
            sub_list = audio_list[s:e]
            inputs = self.feature_extractor(sub_list, sampling_rate=self.sample_rate,
                                          return_tensors="pt", padding=True)
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                out = self.model(**inputs)
                hidden = out.last_hidden_state  # (B, T_feat, hidden_size)
                # Mean pooling over time dimension
                pooled = hidden.mean(dim=1)

            audio_feats[s:e] = pooled.float()

        return audio_feats


class MFCCExtractor(AudioFeatureExtractor):
    """Traditional MFCC-based audio feature extractor."""

    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 40, n_fft: int = 400, hop_length: int = 160):
        import torchaudio
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": n_fft, "hop_length": hop_length}
        )
        self.feature_dim = n_mfcc

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def to_device(self, device: torch.device):
        self.mfcc_transform.to(device)

    def extract_features(self, audio_windows: torch.Tensor, device: torch.device,
                        feat_chunk_size: int = 8) -> torch.Tensor:
        """Extract MFCC features."""
        n = audio_windows.shape[0]
        audio_feats = torch.empty(n, self.feature_dim, device=device, dtype=torch.float32)

        for s in range(0, n, feat_chunk_size):
            e = min(n, s + feat_chunk_size)
            batch_audio = audio_windows[s:e].to(device)

            # Ensure proper shape for MFCC
            if batch_audio.dim() == 2:
                batch_audio = batch_audio.unsqueeze(1)  # Add channel dimension if needed

            with torch.no_grad():
                mfcc_features = self.mfcc_transform(batch_audio)
                # Mean pooling over time dimension
                pooled = mfcc_features.mean(dim=-1)  # (batch, n_mfcc)

            audio_feats[s:e] = pooled

        return audio_feats