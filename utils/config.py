# config.py
import torch
import os # Added os

# Get the directory of the current config.py file
_config_dir = os.path.dirname(os.path.abspath(__file__))

# --- Model Configuration ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DECODER_MODEL_NAME = "gpt2" # We'll use GPT-2 as the base decoder

# --- Dataset Configuration ---
DATASET_NAME = "itsanmolgupta/mimic-cxr-cleaned-old"
IMAGE_COLUMN = "image"
FINDINGS_COLUMN = "findings"
IMPRESSION_COLUMN = "impression"
IMAGE_DIR = "/path/to/your/images" # IMPORTANT: Set this if images aren't in the dataset object directly

# --- Training Hyperparameters ---
BATCH_SIZE = 8
LEARNING_RATE_CLIP = 1e-5
LEARNING_RATE_DECODER = 5e-5
EPOCHS_CLIP = 5
EPOCHS_DECODER = 10
MAX_TEXT_LENGTH = 128

# --- Image Processing ---
IMAGE_SIZE = 224 # Standard CLIP input size

# --- Paths (now relative to this config file's directory) ---
# Assumes a 'models' subdirectory within the 'utils' directory
MODELS_SUBDIR = "models"
CLIP_SAVE_PATH = os.path.join(_config_dir, MODELS_SUBDIR, "clip_finetuned")
DECODER_SAVE_PATH = os.path.join(_config_dir, MODELS_SUBDIR, "decoder_trained")
PROCESSOR_SAVE_PATH = os.path.join(_config_dir, MODELS_SUBDIR, "processor")
TOKENIZER_SAVE_PATH = os.path.join(_config_dir, MODELS_SUBDIR, "tokenizer")

# --- Device ---
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available(): # For Apple Silicon GPUs
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")
