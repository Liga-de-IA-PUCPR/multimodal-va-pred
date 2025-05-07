import torch
import torch.nn as nn
import torchvision.models as tvmodels
import torchaudio

# ----------------------------
# Visual backbone (ResNet-based)
# ----------------------------
class VisualBackbone(nn.Module):
    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        """
        A simple ResNet-18 visual feature extractor.
        We strip off the final FC and project into out_dim.
        """
        super().__init__()
        # load a ResNet-18
        resnet = tvmodels.resnet18(weights='DEFAULT')
        # drop the classifier head
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(resnet.fc.in_features, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W]
        x = self.encoder(images)          # [B, C, 1, 1]
        x = x.view(x.size(0), -1)         # [B, C]
        return self.proj(x)               # [B, out_dim]


# ----------------------------
# Audio backbone (Wav2Vec2 base)
# ----------------------------
class AudioBackbone(nn.Module):
    def __init__(self, out_dim: int = 512):
        """
        A Wav2Vec2 feature extractor from torchaudio.
        We take the last hidden features and project to out_dim.
        """
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.encoder = bundle.get_model()
        # freeze the encoder if you only want fixed features:
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        hidden_dim = bundle._params["encoder_embed_dim"] 
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        waveforms: [B, T]  — raw audio (mono, sampled at bundle.sample_rate)
        lengths:   [B]     — (optional) lengths for padding mask
        """
        # wav2vec2 returns a tuple (features, …)
        features, _ = self.encoder(waveforms, lengths)  
        # features: List[Tensor] over layers; take last:
        last = features[-1]                  # [B, T', hidden_dim]
        # mean‐pool over time
        x = last.mean(dim=1)                 # [B, hidden_dim]
        return self.proj(x)                  # [B, out_dim]