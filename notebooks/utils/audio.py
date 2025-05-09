import torchaudio
import torch
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
            f_max=f_max # Use configured f_max
        ).to(self._device)
        
    def _load_and_preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}. Returning empty tensor.")
            return torch.empty(0, device=self._device) # Return on target device
            
        waveform = waveform.to(self._device)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if sr != self.sample_rate:
            resampler_module = Resample(sr, self.sample_rate)
            resampler_module = resampler_module.to(self._device)
            
            # --- Debugging device placement ---
            kernel_device_str = "N/A"
            # The kernel is a buffer. We try to access it.
            # In torchaudio.transforms.Resample, it uses _torchaudio.kaldi.ResampleWaveform
            # or functional._apply_sinc_resample_kernel. The kernel is created and stored as buffer.
            # Let's check the device of a buffer if one is named 'kernel' or similar.
            resampler_buffers = dict(resampler_module.named_buffers())
            if 'kernel' in resampler_buffers: # Common name for sinc kernel buffer
                kernel_device_str = str(resampler_buffers['kernel'].device)
            elif resampler_buffers: # Check device of the first available buffer as a proxy
                first_buffer_name = next(iter(resampler_buffers))
                kernel_device_str = f"first_buffer ('{first_buffer_name}'): {resampler_buffers[first_buffer_name].device}"
            else:
                kernel_device_str = "No buffers found in resampler_module"

            print(f"Debug Resample for {audio_path}: "
                  f"waveform.device={waveform.device}, "
                  f"resampler_kernel_approx_device={kernel_device_str}, "
                  f"target_device={self._device}, "
                  f"original_sr={sr}, target_sr={self.sample_rate}")
            # --- End Debugging ---
            
            try:
                waveform = resampler_module(waveform)
            except RuntimeError as e:
                print(f"RuntimeError during resampling for {audio_path}!")
                print(f"  Waveform was on: {waveform.device}, shape: {waveform.shape}")
                print(f"  Resampler module buffers: {[(name, buf.device) for name, buf in resampler_module.named_buffers()]}")
                print(f"  Resampler module parameters: {[(name, param.device) for name, param in resampler_module.named_parameters()]}")
                raise e # Re-raise the error after printing debug info
            
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
        waveform = self._load_and_preprocess_audio(audio_path)

        if waveform.numel() == 0:
            print(f"Warning: Waveform for {audio_path} is empty. Returning zeros.")
            return torch.zeros(self.num_frames_to_extract, self.n_mels, device="cpu")

        mel_spec = self._compute_mel_spectrogram(waveform)
        time_steps = mel_spec.size(-1)
        
        if time_steps == 0:
            print(f"Warning: Mel spectrogram for {audio_path} has 0 time steps. Returning zeros.")
            return torch.zeros(self.num_frames_to_extract, self.n_mels, device="cpu")

        if time_steps < self.num_frames_to_extract:
            indices = torch.linspace(0, time_steps - 1, self.num_frames_to_extract, device=mel_spec.device).long()
            indices = torch.clamp(indices, 0, time_steps - 1)
        else:
            indices = torch.linspace(0, time_steps - 1, self.num_frames_to_extract, device=mel_spec.device).long()

        sampled_mel_spec = mel_spec.squeeze(0)[:, indices]
        sampled_mel_spec = sampled_mel_spec.transpose(0, 1)
        return sampled_mel_spec.cpu()

_audio_feature_extractor = AudioFeatureExtractor(
    sample_rate=SAMPLE_RATE, 
    num_frames_to_extract=NUM_FRAMES,
    n_mels=AUDIO_MEL_N_MELS,
    f_max=AUDIO_MEL_FMAX
)

def extract_audio_features_for_video(video_audio_path: str) -> torch.Tensor:
    return _audio_feature_extractor.extract_features(video_audio_path)