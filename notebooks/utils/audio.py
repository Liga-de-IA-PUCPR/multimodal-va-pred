import torchaudio
import torch
from pydub import AudioSegment
import io
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, Resample
from utils.config import SAMPLE_RATE, NUM_FRAMES, AUDIO_MEL_N_MELS, AUDIO_MEL_FMAX

class AudioFeatureExtractor:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = AUDIO_MEL_N_MELS,
        n_fft: int = 1024,
        hop_length: int = 512,
        power: float = 2.0,
        normalized: bool = True,
        num_frames_to_extract: int = NUM_FRAMES,
        f_max: float = AUDIO_MEL_FMAX
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames_to_extract = num_frames_to_extract
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
            normalized=normalized,
            f_max=f_max
        ).to(self._device)
        
    def _load_and_preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}. Returning empty tensor.")
            return torch.empty(0, device=self._device)
            
        waveform = waveform.to(self._device)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if sr != self.sample_rate:
            resampler_module = Resample(sr, self.sample_rate)
            resampler_module = resampler_module.to(self._device)
            
            try:
                waveform = resampler_module(waveform)
            except RuntimeError as e:
                print(f"RuntimeError during resampling for {audio_path}!")
                print(f"  Waveform was on: {waveform.device}, shape: {waveform.shape}")
                print(f"  Resampler module buffers: {[(name, buf.device) for name, buf in resampler_module.named_buffers()]}")
                print(f"  Resampler module parameters: {[(name, param.device) for name, param in resampler_module.named_parameters()]}")
                raise e
            
        return waveform
    
    def _compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.numel() == 0:
             return torch.empty((waveform.size(0), self.n_mels, 0), device=self._device)
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        return mel_spec
    
    def extract_features(self, audio_path: str) -> torch.Tensor:
        target_num_frames = self.num_frames_to_extract # Should be NUM_FRAMES from your config

        waveform = self._load_and_preprocess_audio(audio_path)

        if waveform.numel() == 0:
            print(f"Warning: Waveform for {audio_path} is empty. Returning zero tensor of shape [{target_num_frames}, {self.n_mels}].")
            return torch.zeros(target_num_frames, self.n_mels, device="cpu")

        mel_spec = self._compute_mel_spectrogram(waveform) # Expected shape: [1, n_mels, time_steps]
        
        # Ensure mel_spec is on the correct device for processing
        mel_spec = mel_spec.to(self._device)

        # Squeeze channel dim (if present) and transpose to [time_steps, n_mels]
        if mel_spec.dim() == 3 and mel_spec.size(0) == 1:
            # Standard case: [1, n_mels, time_steps] -> [n_mels, time_steps] -> [time_steps, n_mels]
            mel_spec_processed = mel_spec.squeeze(0).transpose(0, 1)
        elif mel_spec.dim() == 2:
            # If _compute_mel_spectrogram somehow returns [n_mels, time_steps]
            mel_spec_processed = mel_spec.transpose(0, 1)
        else:
            print(f"Warning: Unexpected mel_spec shape {mel_spec.shape} for {audio_path}. Returning zero tensor of shape [{target_num_frames}, {self.n_mels}].")
            return torch.zeros(target_num_frames, self.n_mels, device="cpu")

        current_time_steps = mel_spec_processed.size(0)

        if current_time_steps == 0:
            print(f"Warning: Mel spectrogram for {audio_path} has 0 time steps after processing. Returning zero tensor of shape [{target_num_frames}, {self.n_mels}].")
            return torch.zeros(target_num_frames, self.n_mels, device="cpu")

        if current_time_steps == target_num_frames:
            final_features = mel_spec_processed

        elif current_time_steps > target_num_frames:
            # If more frames than target, uniformly sample
            indices = torch.linspace(0, current_time_steps - 1, steps=target_num_frames, device=self._device).long()
            final_features = mel_spec_processed[indices, :]

        else: # current_time_steps < target_num_frames
            
            # If fewer frames than target, pad
            padding_needed = target_num_frames - current_time_steps

            if mel_spec_processed.numel() > 0 : # Check if there's anything to pad from
                last_frame = mel_spec_processed[-1:, :] # Get the last available frame
                padding = last_frame.repeat(padding_needed, 1) # Repeat last frame for padding

            else: # Should be caught by current_time_steps == 0, but as a safeguard

                padding = torch.zeros(padding_needed, self.n_mels, device=self._device)
            final_features = torch.cat([mel_spec_processed, padding], dim=0)
        
        return final_features.cpu() # Ensure final tensor is on CPU as per original logic

_audio_feature_extractor = AudioFeatureExtractor(
    sample_rate=SAMPLE_RATE, 
    num_frames_to_extract=NUM_FRAMES,
    n_mels=AUDIO_MEL_N_MELS,
    f_max=AUDIO_MEL_FMAX
)

def extract_audio_features_for_video(video_audio_path: str, size_of_features: int) -> torch.Tensor:
    _audio_feature_extractor.num_frames_to_extract = size_of_features
    return _audio_feature_extractor.extract_features(video_audio_path)