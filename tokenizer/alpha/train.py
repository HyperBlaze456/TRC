import sys
import os
from dotenv import load_dotenv
from huggingface_hub import login
# Add the TRC directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from functools import partial
from dataclasses import dataclass

# Import our modules
from tokenizer.alpha.model import AudioTokenizer
from tokenizer.previous_modules.loss import (
    compute_generator_loss,
    compute_discriminator_loss,
    create_phoneme_vocabulary,
    extract_encoder_lengths
)
from tokenizer.alpha.components.discriminators import (
    MultiScaleDiscriminator,
    MultiPeriodDiscriminator,
    STFTDiscriminator
)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Set token globally for HuggingFace
if hf_token:
    print(f"Setting HF token (first 8 chars): {hf_token[:8]}...")
    try:
        login(token=hf_token)
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("HF token set successfully")
    except Exception as e:
        print(f"Warning: Failed to login with HF token: {e}")
else:
    print("Warning: No HF_TOKEN found in environment")

@dataclass
class TrainingState:
    """Training state containing models and optimizers."""
    generator: AudioTokenizer
    msd: MultiScaleDiscriminator  # Multi-Scale Discriminator
    mpd: MultiPeriodDiscriminator  # Multi-Period Discriminator
    stftd: STFTDiscriminator  # STFT Discriminator
    gen_optimizer: nnx.Optimizer
    msd_optimizer: nnx.Optimizer
    mpd_optimizer: nnx.Optimizer
    stftd_optimizer: nnx.Optimizer
