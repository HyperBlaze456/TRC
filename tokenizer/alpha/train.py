"""
Training script for SpeechTokenizer with adversarial training.

This script trains the generator (SpeechTokenizer) and three discriminators
(MultiScale, MultiPeriod, STFT) using step-based training suitable for
large streaming datasets like Emilia (4TB).
"""

import os
import sys

# Set JAX traceback filtering off for detailed error traces
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

from dotenv import load_dotenv
from huggingface_hub import login

# Add the TRC directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataclasses import dataclass, field
from functools import partial
import time

import orbax.checkpoint as ocp
from flax import nnx
import jax
import optax
import wandb

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

# Import our modules
from tokenizer.alpha.model import SpeechTokenizer
from tokenizer.utils.data.phoneme_utils import PHONEME_VOCAB_SIZE
from tokenizer.utils.data.simple_loader import AudioConfig, create_emilia_ds

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


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model architecture
    hidden_size: int = 512
    encoder_depth: int = 4
    encoder_heads: int = 8
    phoneme_codebook_size: int = PHONEME_VOCAB_SIZE
    bsq_spherical_dim: int = 256
    decoder_output_48khz: bool = False

    # Training steps
    warmup_steps: int = 10_000

    # Checkpointing and logging
    checkpoint_every: int = 5_000
    log_every: int = 100
    profile_first_n_steps: int = 5
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Batch settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 4

    # Learning rates
    generator_lr: float = 1e-4
    discriminator_lr: float = 2e-4

    # Loss settings
    loss_type: str = "lsgan"  # or "hinge"
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

    # W&B settings
    use_wandb: bool = True
    wandb_project: str = "speech-tokenizer"
    wandb_name: str = None


# ============================================================================
# Training State
# ============================================================================

@dataclass
class TrainingState:
    """Training state containing models and optimizers."""
    generator: SpeechTokenizer
    msd: MultiScaleDiscriminator
    mpd: MultiPeriodDiscriminator
    stftd: STFTDiscriminator
    gen_optimizer: nnx.Optimizer
    msd_optimizer: nnx.Optimizer
    mpd_optimizer: nnx.Optimizer
    stftd_optimizer: nnx.Optimizer
    step: int = 0


# ============================================================================
# JAX Profiler Setup
# ============================================================================

def setup_profiler(log_dir: str):
    """Setup JAX profiler for performance monitoring."""
    profile_dir = os.path.join(log_dir, "profiles")
    os.makedirs(profile_dir, exist_ok=True)

    # Log device memory usage
    print("\n=== Initial Device Memory Info ===")
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            print(f"Device {device}: {stats}")
    print("=" * 50 + "\n")

    return profile_dir


def log_memory_usage(label: str):
    """Log current GPU memory usage."""
    for i, device in enumerate(jax.devices()):
        stats = device.memory_stats()
        if stats:
            used_gb = stats['bytes_in_use'] / (1024**3)
            limit_gb = stats.get('bytes_limit', 0) / (1024**3)
            print(f"[{label}] GPU {i}: {used_gb:.2f}GB / {limit_gb:.2f}GB used")

def profile_step(step: int, profile_dir: str, func, *args, **kwargs):
    """Profile a training step if within profiling range."""
    if step < 5:  # Profile first 5 steps to catch JIT compilations
        trace_path = os.path.join(profile_dir, f"step_{step}")
        
        # Log memory before
        log_memory_usage(f"Before {func.__name__} step {step}")
        
        with jax.profiler.trace(trace_path):
            result = func(*args, **kwargs)
            
        # Log memory after
        log_memory_usage(f"After {func.__name__} step {step}")
        
        print(f"Profiled step {step} -> {trace_path}")
    else:
        result = func(*args, **kwargs)
    return result


# ============================================================================
# Model Creation
# ============================================================================

def create_models_and_optimizers(config: TrainingConfig, rngs: nnx.Rngs) -> TrainingState:
    """Create models and optimizers."""

    # Create generator (SpeechTokenizer)
    generator = SpeechTokenizer(
        hidden_size=config.hidden_size,
        encoder_depth=config.encoder_depth,
        encoder_heads=config.encoder_heads,
        phoneme_codebook_size=config.phoneme_codebook_size,
        bsq_spherical_dim=config.bsq_spherical_dim,
        decoder_output_48khz=config.decoder_output_48khz,
        rngs=rngs,
    )

    # Create discriminators
    msd = MultiScaleDiscriminator(rngs=rngs)
    mpd = MultiPeriodDiscriminator(rngs=rngs)
    stftd = STFTDiscriminator(rngs=rngs)

    # Create optimizers
    gen_optimizer = nnx.Optimizer(
        generator,
        optax.adamw(
            learning_rate=config.generator_lr,
            b1=0.5,
            b2=0.9,
            weight_decay=1e-4
        )
    )

    msd_optimizer = nnx.Optimizer(
        msd,
        optax.adamw(
            learning_rate=config.discriminator_lr,
            b1=0.5,
            b2=0.9,
            weight_decay=1e-4
        )
    )

    mpd_optimizer = nnx.Optimizer(
        mpd,
        optax.adamw(
            learning_rate=config.discriminator_lr,
            b1=0.5,
            b2=0.9,
            weight_decay=1e-4
        )
    )

    stftd_optimizer = nnx.Optimizer(
        stftd,
        optax.adamw(
            learning_rate=config.discriminator_lr,
            b1=0.5,
            b2=0.9,
            weight_decay=1e-4
        )
    )

    return TrainingState(
        generator=generator,
        msd=msd,
        mpd=mpd,
        stftd=stftd,
        gen_optimizer=gen_optimizer,
        msd_optimizer=msd_optimizer,
        mpd_optimizer=mpd_optimizer,
        stftd_optimizer=stftd_optimizer,
    )


# ============================================================================
# Data Loading
# ============================================================================

def create_data_iterator(config: TrainingConfig):
    """Create streaming iterator for massive dataset."""
    dataset = create_emilia_ds(AudioConfig(
        streaming=True,  # Critical for 4TB dataset
        batch_size=config.batch_size,
        sample_rate=24000,
    ))

    # Create infinite iterator
    return iter(dataset)


def prepare_batch(batch: dict) -> dict:
    """Prepare batch by converting masks to proper shapes."""
    # Convert 4D masks to 2D for our loss functions
    if "padding_mask" in batch and batch["padding_mask"].ndim == 4:
        batch["padding_mask_2d"] = batch["padding_mask"][:, 0, 0, :]  # [B, T]

    return batch


# ============================================================================
# Training Steps (JIT-compiled)
# ============================================================================

@partial(nnx.jit, static_argnums=(10,))
def train_discriminator_step(
        # Individual modules
        generator: SpeechTokenizer,
        msd: MultiScaleDiscriminator,
        mpd: MultiPeriodDiscriminator,
        stftd: STFTDiscriminator,
        msd_optimizer: nnx.Optimizer,
        mpd_optimizer: nnx.Optimizer,
        stftd_optimizer: nnx.Optimizer,
        # Batch arrays
        audio: jax.Array,
        encoder_causal_mask: jax.Array,
        padding_mask: jax.Array,
        # Mentioned before; This might be consumed by discriminator modules later for more accuracy
        # Static config
        loss_type: str = "lsgan"
) -> tuple[dict, tuple[SpeechTokenizer, MultiScaleDiscriminator, MultiPeriodDiscriminator, STFTDiscriminator]]:
    """Train discriminators for one step."""

    # Select loss function based on type
    if loss_type == "lsgan":
        disc_loss_fn = compute_discriminator_loss_lsgan
    else:
        disc_loss_fn = compute_discriminator_loss_hinge

    # Get real audio
    real_audio = audio

    # Reconstruction from fixed generator
    fake_audio, _, _, _, _, _, _, _ = generator(
        real_audio,
        encoder_causal_mask
    )
    fake_audio = jax.lax.stop_gradient(fake_audio)

    def msd_loss_fn(msd):
        msd_real, _ = msd(real_audio, training=True)
        msd_fake, _ = msd(fake_audio, training=True)
        loss, metrics = disc_loss_fn(msd_real, msd_fake)
        return loss, metrics

    grad_fn = nnx.value_and_grad(msd_loss_fn, has_aux=True)
    (msd_loss, msd_metrics), msd_grads = grad_fn(msd)
    msd_optimizer.update(msd_grads)

    def mpd_loss_fn(mpd):
        mpd_real, _ = mpd(real_audio)
        mpd_fake, _ = mpd(fake_audio)
        loss, metrics = disc_loss_fn(mpd_real, mpd_fake)
        return loss, metrics

    grad_fn = nnx.value_and_grad(mpd_loss_fn, has_aux=True)
    (mpd_loss, mpd_metrics), mpd_grads = grad_fn(mpd)
    mpd_optimizer.update(mpd_grads)

    def stftd_loss_fn(stftd):
        stftd_real, _ = stftd(real_audio, training=True)
        stftd_fake, _ = stftd(fake_audio, training=True)
        loss, metrics = disc_loss_fn(stftd_real, stftd_fake)
        return loss, metrics

    grad_fn = nnx.value_and_grad(stftd_loss_fn, has_aux=True)
    (stftd_loss, stftd_metrics), stftd_grads = grad_fn(stftd)
    stftd_optimizer.update(stftd_grads)

    # Combine metrics
    metrics = {
        "disc/msd": msd_metrics,
        "disc/mpd": mpd_metrics,
        "disc/stftd": stftd_metrics,
        "disc/total": msd_loss + mpd_loss + stftd_loss,
    }

    return metrics, (generator, msd, mpd, stftd)


# Arguments 0-10: modules and arrays (not static)
# Arguments 11-24: static parameters
@partial(nnx.jit, static_argnums=(11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21))
def train_generator_step(
        # Individual modules
        generator: SpeechTokenizer,
        msd: MultiScaleDiscriminator,
        mpd: MultiPeriodDiscriminator,
        stftd: STFTDiscriminator,
        gen_optimizer: nnx.Optimizer,
        # Batch arrays
        audio: jax.Array,
        encoder_causal_mask: jax.Array,
        padding_mask: jax.Array,
        encoder_mask: jax.Array,
        phonemes: jax.Array,
        phoneme_mask: jax.Array,
        # Loss weights (static)
        loss_type: str,
        w_l1: float,
        w_l2: float,
        w_mel: float,
        w_stft_sc: float,
        w_stft_lm: float,
        w_vq_commit: float,
        w_bsq_commit: float,
        w_adversarial: float,
        w_feature_match: float,
        w_ctc: float,
) -> tuple[dict, SpeechTokenizer]:
    """Train generator for one step."""

    # Select loss function based on config
    if loss_type == "lsgan":
        gen_loss_fn = compute_generator_loss_lsgan
    else:
        gen_loss_fn = compute_generator_loss_hinge

    def generator_loss(generator):
        # Forward pass - now returns all intermediate values needed for losses
        reconstructed, phoneme_indices, acoustic_codes, encoder_output, phoneme_logits, vq_quantized, bsq_quantized, vq_residual = generator(
            audio,
            encoder_causal_mask
        )

        # Get discriminator outputs (no gradients for discriminators)
        msd_fake, msd_feat_fake = msd(reconstructed, training=False)
        msd_real, msd_feat_real = msd(audio, training=False)

        mpd_fake, mpd_feat_fake = mpd(reconstructed)
        mpd_real, mpd_feat_real = mpd(audio)

        stftd_fake, stftd_feat_fake = stftd(reconstructed, training=False)
        stftd_real, stftd_feat_real = stftd(audio, training=False)

        # Combine discriminator outputs
        disc_outputs = msd_fake + mpd_fake + stftd_fake
        disc_features_real = msd_feat_real + mpd_feat_real + stftd_feat_real
        disc_features_fake = msd_feat_fake + mpd_feat_fake + stftd_feat_fake

        # Compute generator losses
        gen_loss, gen_metrics = gen_loss_fn(
            pred_audio=reconstructed,
            target_audio=audio,
            encoder_output=encoder_output,
            vq_quantized=vq_quantized,
            bsq_quantized=bsq_quantized,
            vq_residual=vq_residual,
            disc_outputs=disc_outputs,
            disc_features_real=disc_features_real,
            disc_features_fake=disc_features_fake,
            padding_mask=padding_mask,
            encoder_mask=encoder_mask,
            w_l1=w_l1,
            w_l2=w_l2,
            w_mel=w_mel,
            w_stft_sc=w_stft_sc,
            w_stft_lm=w_stft_lm,
            w_vq_commit=w_vq_commit,
            w_bsq_commit=w_bsq_commit,
            w_adversarial=w_adversarial,
            w_feature_match=w_feature_match,
        )

        # Compute CTC loss
        ctc_loss, ctc_metrics = phoneme_ctc_loss(
            phoneme_logits=phoneme_logits,
            encoder_mask=encoder_mask,
            phoneme_indices=phonemes,
            phoneme_mask=phoneme_mask
        )

        # Total loss
        total_loss = gen_loss + w_ctc * ctc_loss

        # Update metrics
        gen_metrics.update({
            "gen/ctc_loss": ctc_loss,
            **{f"gen/ctc_{k}": v for k, v in ctc_metrics.items()},
        })
        gen_metrics["gen/total"] = total_loss

        return total_loss, gen_metrics

    # Compute gradients and update
    grad_fn = nnx.value_and_grad(generator_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(generator)
    gen_optimizer.update(grads)

    return metrics, generator


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(state: TrainingState, step: int, checkpoint_dir: str):

    checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Create checkpoint manager
    ckptr = ocp.PyTreeCheckpointer()

    # Save state
    ckptr.save(
        checkpoint_path,
        {
            "generator": state.generator,
            "msd": state.msd,
            "mpd": state.mpd,
            "stftd": state.stftd,
            "gen_optimizer": state.gen_optimizer,
            "msd_optimizer": state.msd_optimizer,
            "mpd_optimizer": state.mpd_optimizer,
            "stftd_optimizer": state.stftd_optimizer,
            "step": step,
        }
    )

    print(f"Saved checkpoint at step {step}")


# ============================================================================
# Logging
# ============================================================================

def setup_logging(config: TrainingConfig):
    """Setup logging (W&B or console)."""
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name or f"run_{int(time.time())}",
            config=config.__dict__
        )

    os.makedirs(config.log_dir, exist_ok=True)


def log_metrics(step: int, metrics: dict, config: TrainingConfig):
    """Log metrics to console and W&B."""
    # Console logging
    if step % config.log_every == 0:
        print(f"\nStep {step}")
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    print(f"  {key}/{subkey}: {float(subvalue):.4f}")
            else:
                print(f"  {key}: {float(value):.4f}")

    # W&B logging
    if config.use_wandb:
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_metrics[f"{key}/{subkey}"] = float(subvalue)
            else:
                flat_metrics[key] = float(value)
        wandb.log(flat_metrics, step=step)


# ============================================================================
# Main Training Loop
# ============================================================================

def train(config: TrainingConfig):
    """Main training function."""
    print("Starting training...")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")

    # Setup
    rngs = nnx.Rngs(42)
    state = create_models_and_optimizers(config, rngs)
    data_iter = create_data_iterator(config)
    profile_dir = setup_profiler(config.log_dir)
    setup_logging(config)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Training loop
    start_time = time.time()
    step = 0
    while True:  # Run forever(later on fix this to epoch based)
        batch = next(data_iter)
        batch = prepare_batch(batch)

        # get all jax arrays
        audio = batch["audio"]
        encoder_causal_mask = batch["encoder_causal_mask"]
        padding_mask = batch.get("padding_mask_2d", batch["padding_mask"][:, 0, 0, :])
        encoder_mask = batch["encoder_mask"]
        phonemes = batch["phonemes"]
        phoneme_mask = batch["phoneme_mask"]

        # Train discriminators
        if step < config.profile_first_n_steps:
            disc_metrics, (_, new_msd, new_mpd, new_stftd) = profile_step(
                step, profile_dir,
                train_discriminator_step,
                state.generator, state.msd, state.mpd, state.stftd,
                state.msd_optimizer, state.mpd_optimizer, state.stftd_optimizer,
                audio, encoder_causal_mask, padding_mask,
                config.loss_type
            )
        else:
            disc_metrics, (_, new_msd, new_mpd, new_stftd) = train_discriminator_step(
                state.generator, state.msd, state.mpd, state.stftd,
                state.msd_optimizer, state.mpd_optimizer, state.stftd_optimizer,
                audio, encoder_causal_mask, padding_mask,
                config.loss_type
            )

        # Update discriminator
        state.msd = new_msd
        state.mpd = new_mpd
        state.stftd = new_stftd

        # Train generator
        if step < config.profile_first_n_steps:
            gen_metrics, new_generator = profile_step(
                step, profile_dir,
                train_generator_step,
                state.generator, state.msd, state.mpd, state.stftd,
                state.gen_optimizer,
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
        else:
            gen_metrics, new_generator = train_generator_step(
                state.generator, state.msd, state.mpd, state.stftd,
                state.gen_optimizer,
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

        # Update generator model
        state.generator = new_generator

        # Update step
        state.step = step
        step += 1

        # Logging
        if step % config.log_every == 0:
            all_metrics = {**disc_metrics, **gen_metrics}
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            all_metrics["train/steps_per_sec"] = steps_per_sec
            log_metrics(step, all_metrics, config)

        # Checkpointing
        if step > 0 and step % config.checkpoint_every == 0:
            save_checkpoint(state, step, config.checkpoint_dir)

        # Profiling complete message
        if step == config.profile_first_n_steps:
            print(f"\nProfiling complete. Check {profile_dir} for results")
            print("JIT compilation should be complete now")
            print(f"Training speed: {steps_per_sec:.2f} steps/sec\n")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    config = TrainingConfig()
    train(config)
