import os
from dotenv import load_dotenv

# Path to the .env file, assuming config.py is in 'notebooks/'
# and .env is in 'notebooks/config/.env'
_CONFIG_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(_CONFIG_MODULE_DIR, "config", ".env")

if os.path.exists(DOTENV_PATH):
    load_dotenv(DOTENV_PATH)
    # print(f"Loaded .env file from {DOTENV_PATH}") # For debugging
else:
    print(f"Warning: .env file not found at {DOTENV_PATH}. Using default values for relevant settings.")

# --- Model & Feature Extraction Parameters ---
NUM_FRAMES = 124  # 124 frames per video, corresponding to 5 seconds at 24 fps

# --- Audio Parameters ---
DEFAULT_SAMPLE_RATE = 16000
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", str(DEFAULT_SAMPLE_RATE)))


# For MelSpectrogram's f_max. If None, uses sample_rate / 2.
# Setting it can help avoid warnings about empty high-frequency mel bins.
_default_fmax = None
if SAMPLE_RATE > 24000:  # e.g., for 44.1kHz or 48kHz, cap at 12kHz-16kHz
    _default_fmax = 12000.0 # Can be tuned, e.g. 8000.0, 16000.0
elif SAMPLE_RATE > 10000: # e.g., for 16kHz, cap at Nyquist (8kHz)
    _default_fmax = float(SAMPLE_RATE // 2)
# For lower sample rates, None (which defaults to sample_rate / 2) is fine.
# You can also directly set a value like: AUDIO_MEL_FMAX = float(os.getenv("AUDIO_MEL_FMAX", 8000.0))
AUDIO_MEL_FMAX = _default_fmax
print(f"Config: SAMPLE_RATE={SAMPLE_RATE}, AUDIO_MEL_FMAX={AUDIO_MEL_FMAX}")


# --- Subdirectory Names for Data ---
VISUAL_FRAMES_SUBDIR_NAME = "cropped_aligned_new_50_vids"
AUDIO_FILES_SUBDIR_NAME = "/Users/lfbf/Library/CloudStorage/OneDrive-GrupoMarista/AffWild2/Video_files"

# --- Backbone output dimensions (example) ---
VISUAL_BACKBONE_OUT_DIM = 256

AUDIO_MEL_N_MELS = 128 # Number of mel bands for mel spectrogram features