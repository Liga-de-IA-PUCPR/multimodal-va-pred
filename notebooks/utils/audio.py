import torchaudio
import torch
import os
import torch.nn.functional as F
from dotenv import load_dotenv
from torchaudio.transforms import MelSpectrogram
from utils.backbones import AudioBackbone

load_dotenv("notebooks/config/.env")

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE"))

audio_backbone = AudioBackbone(out_dim=256).to("cuda" if torch.cuda.is_available() else "cpu")
audio_backbone.eval()
BACKBONE_DEVICE = next(audio_backbone.parameters()).device

_first_conv = audio_backbone.encoder.feature_extractor.conv_layers[0]
ksz = _first_conv.kernel_size
MIN_KSZ = ksz if isinstance(ksz, int) else ksz[0]

mel_transform = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    hop_length=160,
    n_mels=128
)

def extract_audio_features_with_mel(
    video_path: str,
    num_frames: int = int(os.getenv("NUM_FRAMES")),
    fps: int = 30
) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(video_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
        sample_rate = SAMPLE_RATE

    total = waveform.size(1)
    spf = int(sample_rate / fps)
    needed = num_frames * spf
    if total < needed:
        waveform = F.pad(waveform, (0, needed - total))

    chunks = waveform.unfold(1, spf, spf).squeeze(0)  # [num_frames, spf]
    feats = []
    for chunk in chunks:
        # Ensure chunk size is at least 3x the kernel size for stability
        min_chunk_size = MIN_KSZ * 3
        if chunk.size(-1) < min_chunk_size:
            chunk = F.pad(chunk, (0, min_chunk_size - chunk.size(-1)))
        
        with torch.no_grad():
            f = audio_backbone(chunk.unsqueeze(0).to(BACKBONE_DEVICE))
        feats.append(f.cpu())

    return torch.cat(feats, dim=0)  # [num_frames, out_dim]              # [num_frames, out_dim]
