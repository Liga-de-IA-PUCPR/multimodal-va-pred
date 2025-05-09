import torch
import torch.nn as nn
import torchvision.models as tvmodels
import torchaudio
import torchaudio.pipelines as ta_pipelines # Explicit import for clarity

# ----------------------------
# Visual backbone (ResNet-based)
# ----------------------------
class VisualBackbone(nn.Module):
    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = tvmodels.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tvmodels.resnet18(weights=weights)
        
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(resnet.fc.in_features, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images)
        x = x.view(x.size(0), -1)
        return self.proj(x)

# ----------------------------
# Audio backbone (Wav2Vec2 base)
# ----------------------------
class AudioBackbone(nn.Module):
    def __init__(self, out_dim: int = 512, freeze_encoder: bool = False):
        super().__init__()
        try:
            # Try modern torchaudio.pipelines access
            self.bundle = ta_pipelines.WAV2VEC2_BASE
        except AttributeError:
            # Fallback for potentially older torchaudio versions or different naming
            try:
                self.bundle = ta_pipelines.Wav2Vec2Bundle.WAV2VEC2_BASE
            except AttributeError:
                # Further fallback if Wav2Vec2Bundle is not directly in pipelines
                try:
                    from torchaudio.models.wav2vec2.utils import import_fairseq_model
                    # This is more involved, usually for custom fairseq models.
                    # For standard torchaudio bundles, the above should work.
                    # If truly old torchaudio, user might need hub access.
                    # For now, raise an error if common paths fail.
                    raise ImportError("Could not load WAV2VEC2_BASE bundle. "
                                      "Please check torchaudio version and installation.")
                except ImportError:
                     raise ImportError("Failed to load Wav2Vec2 model. Ensure torchaudio is correctly installed.")


        self.encoder = self.bundle.get_model()
        self.expected_sample_rate = self.bundle.sample_rate

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        
        # Determine hidden dimension. For WAV2VEC2_BASE, it's 768.
        # Using a known value is safer than relying on private attributes like _params.
        wav2vec2_hidden_dim = 768 
        # Example of how one might try to get it if structure is known:
        # if hasattr(self.encoder, 'config') and hasattr(self.encoder.config, 'hidden_size'):
        #     wav2vec2_hidden_dim = self.encoder.config.hidden_size
        # elif hasattr(self.bundle, '_params') and "encoder_embed_dim" in self.bundle._params: # Less ideal
        #    wav2vec2_hidden_dim = self.bundle._params["encoder_embed_dim"]


        self.proj = nn.Linear(wav2vec2_hidden_dim, out_dim)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor = None, return_sequence: bool = False) -> torch.Tensor:
        """
        Args:
            waveforms (torch.Tensor): [B, T_samples]. Must be at self.expected_sample_rate.
            lengths (torch.Tensor, optional): [B], valid lengths for padding mask.
            return_sequence (bool): If True, returns [B, T_features, wav2vec2_hidden_dim] (before projection).
                                    If False, returns [B, out_dim] (mean-pooled and projected).
        """
        # Using extract_features is generally for getting representations.
        # The Wav2Vec2Model itself can also be called directly if fine-tuning.
        # extract_features returns all hidden states if num_layers is None.
        all_hidden_states, feature_lengths = self.encoder.extract_features(
            waveforms, lengths=lengths, num_layers=None
        )
        
        last_layer_hidden_states = all_hidden_states[-1]  # [B, T_features, wav2vec2_hidden_dim]

        if return_sequence:
            return last_layer_hidden_states 

        # Mean-pool over the time dimension (T_features).
        # Proper masking should be applied if `feature_lengths` are used for variable length sequences.
        if feature_lengths is not None:
            mask = torch.arange(last_layer_hidden_states.size(1), device=last_layer_hidden_states.device).expand(len(feature_lengths), -1) < feature_lengths.unsqueeze(1)
            masked_features = last_layer_hidden_states * mask.unsqueeze(-1)
            summed_features = masked_features.sum(dim=1)
            pooled_features = summed_features / feature_lengths.unsqueeze(1).clamp(min=1) # Avoid division by zero
        else:
            pooled_features = last_layer_hidden_states.mean(dim=1)
        
        return self.proj(pooled_features)