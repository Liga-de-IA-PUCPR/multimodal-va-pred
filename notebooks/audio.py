import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram
from backbones import AudioBackbone  # Assuming you already created this

# Initialize audio backbone
audio_backbone = AudioBackbone(out_dim=256).to("cuda" if torch.cuda.is_available() else "cpu")
audio_backbone.eval()

# Initialize MelSpectrogram transform
mel_transform = MelSpectrogram(
    sample_rate=16000,  # Match the sample rate of your audio backbone
    n_fft=400,          # Number of FFT bins
    hop_length=160,     # Hop length between frames
    n_mels=128          # Number of mel filter banks
)

def extract_audio_features_with_mel(video_path, num_frames, fps=25):
    """
    Extracts and aligns audio features using a MelSpectrogram representation.
    Args:
        video_path: Path to the video file
        num_frames: Total number of video frames
        fps: Frames per second of the video
    Returns:
        Aligned audio features [num_frames, 256]
    """
    waveform, sample_rate = torchaudio.load(video_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)  # Resample to 16kHz

    # Apply mel spectrogram transformation
    mel_spectrogram = mel_transform(waveform)  # Shape: [1, n_mels, time_steps]

    # Average mel spectrogram across time to create a feature vector
    mel_features = mel_spectrogram.mean(dim=2).squeeze(0)  # Shape: [n_mels]

    # Ensure the mel features align with the number of frames
    mel_features = mel_features.unsqueeze(0).repeat(num_frames, 1)  # Repeat for each frame

    # Pass through audio backbone
    with torch.no_grad():
        audio_features = audio_backbone(mel_features)  # Shape: [num_frames, 256]

    return audio_features

# Example usage
video_path = "data/raw/new_vids/461.mp4"
num_frames = 100  # Example number of frames
aligned_audio_features = extract_audio_features_with_mel(video_path, num_frames)
print(aligned_audio_features.shape)  # [num_frames, 256]