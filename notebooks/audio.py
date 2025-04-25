import os
import torch
import torchaudio
from torchvision import models
from torch import nn
from moviepy.editor import VideoFileClip
from torchaudio.transforms import MelSpectrogram

class AudioProcessor:
    def __init__(self, sampling_rate=44100, n_mels=64, segment_duration=1.0):
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.segment_duration = segment_duration
        self.mel_transform = MelSpectrogram(sample_rate=sampling_rate, n_mels=n_mels)

    def extract_audio(self, video_path, output_audio_path):
        """Extract audio from video using moviepy"""
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path, fps=self.sampling_rate)

    def segment_audio(self, audio_path):
        """Segment audio into chunks of fixed duration"""
        waveform, sample_rate = torchaudio.load(audio_path)
        segment_length = int(self.segment_duration * sample_rate)
        segments = waveform.unfold(1, segment_length, segment_length)
        return segments

    def extract_features(self, audio_segments):
        """Convert audio segments to spectrogram features"""
        features = [self.mel_transform(segment) for segment in audio_segments]
        return torch.stack(features)

class AudioModel(nn.Module):
    def __init__(self, pretrained=True):
        super(AudioModel, self).__init__()
        self.base_model = models.resnet18(pretrained=pretrained)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 128)  # Reduce to embedding size

    def forward(self, x):
        return self.base_model(x)

def process_audio(video_path, audio_output_dir, model):
    # Step 1: Initialize processor and model
    audio_processor = AudioProcessor()
    os.makedirs(audio_output_dir, exist_ok=True)

    # Step 2: Extract audio from video
    audio_path = os.path.join(audio_output_dir, "audio.wav")
    audio_processor.extract_audio(video_path, audio_path)

    # Step 3: Segment audio
    segments = audio_processor.segment_audio(audio_path)

    # Step 4: Extract features from segments
    features = audio_processor.extract_features(segments)

    # Step 5: Run features through audio model
    audio_features = model(features)
    return audio_features