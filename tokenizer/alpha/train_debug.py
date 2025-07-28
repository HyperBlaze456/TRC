"""
Debug version of training script with detailed memory profiling.
"""

import os
import sys

# Set JAX traceback filtering off for detailed error traces
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# Add the TRC directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
from huggingface_hub import login

# Setup environment
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

# Import memory profiler first
from tokenizer.alpha.memory_profiler import (
    enable_memory_profiling,
    profile_memory_usage,
    wrap_jit_with_profiling,
    analyze_pytree_memory,
    check_model_memory,
    profile_function_memory
)

# Enable memory profiling
enable_memory_profiling()

# Now import everything else
from dataclasses import dataclass, field
from functools import partial
import time

import orbax.checkpoint as ocp
from flax import nnx
import jax
import jax.numpy as jnp
import optax

from tokenizer.alpha.components.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    STFTDiscriminator,
)
from tokenizer.alpha.losses.discriminator_loss import (
    compute_discriminator_loss_hinge,
    compute_discriminator_loss_lsgan,
)
from tokenizer.alpha.losses.generator_loss import (
    compute_generator_loss_hinge,
    compute_generator_loss_lsgan,
)
from tokenizer.alpha.losses.phoneme_loss import phoneme_ctc_loss
from tokenizer.alpha.model import SpeechTokenizer
from tokenizer.utils.data.phoneme_utils import PHONEME_VOCAB_SIZE
from tokenizer.utils.data.simple_loader import AudioConfig, create_emilia_ds


@dataclass
class DebugConfig:
    """Minimal config for debugging."""
    # Model architecture
    hidden_size: int = 512
    encoder_depth: int = 4
    encoder_heads: int = 8
    phoneme_codebook_size: int = PHONEME_VOCAB_SIZE
    bsq_spherical_dim: int = 256
    decoder_output_48khz: bool = False
    
    # Batch settings - start very small
    batch_size: int = 1  # Start with batch size 1
    
    # Learning rates
    generator_lr: float = 1e-4
    discriminator_lr: float = 2e-4
    
    # Loss settings
    loss_type: str = "lsgan"
    loss_weights: dict = field(default_factory=lambda: {
        "l1": 1.0,
        "l2": 1.0,
        "mel": 15.0,
        "stft_sc": 2.0,
        "stft_lm": 1.0,
        "vq_commit": 0.1,
        "bsq_commit": 1.0,
        "adversarial": 1.0,
        "feature_match": 10.0,
        "ctc": 10.0,
    })


def test_model_creation():
    """Test creating models one by one."""
    print("\n" + "="*80)
    print("TESTING MODEL CREATION")
    print("="*80)
    
    config = DebugConfig()
    rngs = nnx.Rngs(42)
    
    # Test creating generator
    print("\n1. Creating SpeechTokenizer...")
    profile_memory_usage()
    
    generator = SpeechTokenizer(
        hidden_size=config.hidden_size,
        encoder_depth=config.encoder_depth,
        encoder_heads=config.encoder_heads,
        phoneme_codebook_size=config.phoneme_codebook_size,
        bsq_spherical_dim=config.bsq_spherical_dim,
        decoder_output_48khz=config.decoder_output_48khz,
        rngs=rngs,
    )
    
    print("SpeechTokenizer created successfully")
    profile_memory_usage()
    check_model_memory(generator, dummy_input_shape=(1, 480, 1))  # 20ms at 24kHz
    
    # Test creating discriminators
    print("\n2. Creating MultiScaleDiscriminator...")
    profile_memory_usage()
    msd = MultiScaleDiscriminator(rngs=rngs)
    print("MultiScaleDiscriminator created successfully")
    profile_memory_usage()
    
    print("\n3. Creating MultiPeriodDiscriminator...")
    profile_memory_usage()
    mpd = MultiPeriodDiscriminator(rngs=rngs)
    print("MultiPeriodDiscriminator created successfully")
    profile_memory_usage()
    
    print("\n4. Creating STFTDiscriminator...")
    profile_memory_usage()
    stftd = STFTDiscriminator(rngs=rngs)
    print("STFTDiscriminator created successfully")
    profile_memory_usage()
    
    return generator, msd, mpd, stftd


def test_jit_compilation():
    """Test JIT compilation of training functions."""
    print("\n" + "="*80)
    print("TESTING JIT COMPILATION")
    print("="*80)
    
    config = DebugConfig()
    rngs = nnx.Rngs(42)
    
    # Create models
    generator, msd, mpd, stftd = test_model_creation()
    
    # Create optimizers
    print("\n5. Creating optimizers...")
    gen_optimizer = nnx.Optimizer(
        generator,
        optax.adamw(learning_rate=config.generator_lr, b1=0.5, b2=0.9, weight_decay=1e-4)
    )
    msd_optimizer = nnx.Optimizer(
        msd,
        optax.adamw(learning_rate=config.discriminator_lr, b1=0.5, b2=0.9, weight_decay=1e-4)
    )
    mpd_optimizer = nnx.Optimizer(
        mpd,
        optax.adamw(learning_rate=config.discriminator_lr, b1=0.5, b2=0.9, weight_decay=1e-4)
    )
    stftd_optimizer = nnx.Optimizer(
        stftd,
        optax.adamw(learning_rate=config.discriminator_lr, b1=0.5, b2=0.9, weight_decay=1e-4)
    )
    
    print("Optimizers created successfully")
    profile_memory_usage()
    
    # Create dummy batch
    print("\n6. Creating dummy batch...")
    batch_size = 1
    seq_len = 24000  # 1 second at 24kHz
    
    audio = jnp.ones((batch_size, seq_len, 1))
    encoder_causal_mask = jnp.ones((batch_size, 1, 1, seq_len // 480))  # Assuming 480x compression
    padding_mask = jnp.ones((batch_size, seq_len))
    encoder_mask = jnp.ones((batch_size, seq_len // 480))
    phonemes = jnp.zeros((batch_size, 100), dtype=jnp.int32)
    phoneme_mask = jnp.ones((batch_size, 100))
    
    print(f"Dummy batch created with shapes:")
    print(f"  audio: {audio.shape}")
    print(f"  encoder_causal_mask: {encoder_causal_mask.shape}")
    print(f"  padding_mask: {padding_mask.shape}")
    print(f"  encoder_mask: {encoder_mask.shape}")
    print(f"  phonemes: {phonemes.shape}")
    print(f"  phoneme_mask: {phoneme_mask.shape}")
    profile_memory_usage()
    
    # Test discriminator step compilation
    print("\n7. Compiling discriminator training step...")
    profile_memory_usage()
    
    # Import the original training functions
    from tokenizer.alpha.train import train_discriminator_step as original_disc_step
    
    # Wrap with profiling
    train_discriminator_step = wrap_jit_with_profiling(
        original_disc_step,
        static_argnums=(10,)  # loss_type
    )
    
    try:
        # First call will trigger compilation
        disc_metrics, _ = train_discriminator_step(
            generator, msd, mpd, stftd,
            msd_optimizer, mpd_optimizer, stftd_optimizer,
            audio, encoder_causal_mask, padding_mask,
            config.loss_type
        )
        print("Discriminator step compiled successfully!")
        print(f"Metrics: {disc_metrics}")
    except Exception as e:
        print(f"ERROR during discriminator compilation: {e}")
        import traceback
        traceback.print_exc()
    
    profile_memory_usage()
    
    # Test generator step compilation
    print("\n8. Compiling generator training step...")
    profile_memory_usage()
    
    # Import the original training function
    from tokenizer.alpha.train import train_generator_step as original_gen_step
    
    # Wrap with profiling
    train_generator_step = wrap_jit_with_profiling(
        original_gen_step,
        static_argnums=(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    )
    
    try:
        # First call will trigger compilation
        gen_metrics, _ = train_generator_step(
            generator, msd, mpd, stftd,
            gen_optimizer,
            audio, encoder_causal_mask, padding_mask, encoder_mask,
            phonemes, phoneme_mask,
            config.loss_type,
            config.loss_weights["l1"],
            config.loss_weights["l2"],
            config.loss_weights["mel"],
            config.loss_weights["stft_sc"],
            config.loss_weights["stft_lm"],
            config.loss_weights["vq_commit"],
            config.loss_weights["bsq_commit"],
            config.loss_weights["adversarial"],
            config.loss_weights["feature_match"],
            config.loss_weights["ctc"]
        )
        print("Generator step compiled successfully!")
        print(f"Metrics: {gen_metrics}")
    except Exception as e:
        print(f"ERROR during generator compilation: {e}")
        import traceback
        traceback.print_exc()
    
    profile_memory_usage()


def test_individual_losses():
    """Test compiling individual loss functions."""
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL LOSS FUNCTIONS")
    print("="*80)
    
    # Create dummy data
    batch_size = 1
    seq_len = 24000
    hidden_size = 512
    
    pred_audio = jnp.ones((batch_size, seq_len, 1))
    target_audio = jnp.ones((batch_size, seq_len, 1))
    encoder_output = jnp.ones((batch_size, seq_len // 480, hidden_size))
    vq_quantized = jnp.ones((batch_size, seq_len // 480, hidden_size))
    bsq_quantized = jnp.ones((batch_size, seq_len // 480, hidden_size))
    vq_residual = jnp.ones((batch_size, seq_len // 480, hidden_size))
    disc_outputs = [jnp.ones((batch_size, 1)) for _ in range(3)]
    disc_features_real = [jnp.ones((batch_size, 10, 64)) for _ in range(3)]
    disc_features_fake = [jnp.ones((batch_size, 10, 64)) for _ in range(3)]
    padding_mask = jnp.ones((batch_size, seq_len))
    encoder_mask = jnp.ones((batch_size, seq_len // 480))
    
    print("\n1. Testing generator loss function...")
    profile_memory_usage()
    
    try:
        loss, metrics = profile_function_memory(
            compute_generator_loss_lsgan,
            pred_audio=pred_audio,
            target_audio=target_audio,
            encoder_output=encoder_output,
            vq_quantized=vq_quantized,
            bsq_quantized=bsq_quantized,
            vq_residual=vq_residual,
            disc_outputs=disc_outputs,
            disc_features_real=disc_features_real,
            disc_features_fake=disc_features_fake,
            padding_mask=padding_mask,
            encoder_mask=encoder_mask,
            w_l1=1.0,
            w_l2=1.0,
            w_mel=15.0,
            w_stft_sc=2.0,
            w_stft_lm=1.0,
            w_vq_commit=0.1,
            w_bsq_commit=1.0,
            w_adversarial=1.0,
            w_feature_match=10.0,
            stft_fft_sizes=(2048, 1024, 512, 256, 128),
            stft_hop_sizes=(512, 256, 128, 64, 32),
            stft_win_sizes=(2048, 1024, 512, 256, 128)
        )
        print(f"Generator loss computed successfully: {loss}")
    except Exception as e:
        print(f"ERROR in generator loss: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("JAX Devices:", jax.devices())
    print("JAX Default Backend:", jax.default_backend())
    
    # Run tests
    try:
        # Test individual components first
        test_individual_losses()
        
        # Then test full JIT compilation
        test_jit_compilation()
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        profile_memory_usage()