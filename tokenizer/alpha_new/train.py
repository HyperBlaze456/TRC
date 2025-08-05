import os
import sys
from pathlib import Path
from dataclasses import dataclass
from functools import partial

# Suppress XLA compilation warnings
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_softmax_fusion=true'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path for relative imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp

from flax import nnx
import optax
import orbax.checkpoint as ocp

from tokenizer.alpha_new.discriminators import MPD, MSD, MSTFTD
from tokenizer.utils.metrics.wandb import init_wandb, log_generator_metrics, log_discriminator_metrics, finish_wandb
from tokenizer.alpha_new.loss import (
    l1_loss, mel_spectrogram_loss, multi_scale_stft_loss,
    vq_loss, bsq_loss,
    adversarial_g_loss_hinge, adversarial_g_loss_lsgan,
    compute_discriminator_loss_hinge, compute_discriminator_loss_lsgan,
    phoneme_ctc_loss,
    feature_matching_loss
)
from tokenizer.alpha_new.model import SpeechTokenizer
from tokenizer.utils.data.phoneme_utils import PHONEME_VOCAB_SIZE
from tokenizer.utils.data.loader import AudioConfig, create_emilia_ds


@dataclass(frozen=True)
class TrainingConfig:
    # Model hyperparameters
    hidden_size: int = 512
    encoder_mlp_dim: int = 2048
    encoder_depth: int = 4
    encoder_heads: int = 8
    phoneme_codebook_size: int = PHONEME_VOCAB_SIZE
    bsq_spherical_dim: int = 256
    temperature: float = 1.0
    decoder_output_48khz: bool = False
    
    # Training hyperparameters
    batch_size: int = 16  # Total batch size across all devices
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    warmup_steps: int = 2000
    total_steps: int = 200000
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    
    # Loss weights
    reconstruction_weight: float = 1.0
    mel_weight: float = 45.0
    stft_weight: float = 45.0
    adversarial_weight: float = 1.0
    feature_matching_weight: float = 2.0
    vq_commitment_weight: float = 0.25  # Encoder → VQ codebook
    vq_codebook_weight: float = 1.0      # VQ codebook → Encoder  
    bsq_commitment_weight: float = 0.25  # Residual → BSQ
    bsq_codebook_weight: float = 1.0     # BSQ → Residual
    phoneme_encoder_weight: float = 2.5   # Phoneme loss → Encoder
    phoneme_codebook_weight: float = 10.0 # Phoneme loss → VQ codebook
    
    # Discriminator settings
    disc_start_step: int = 10000
    disc_update_freq: int = 1
    
    # Audio configuration
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints/speech_tokenizer"
    checkpoint_every: int = 50 # very small, test
    
    # RNGs seed
    seed: int = 42


def create_models_and_optimizers(config, rngs):
    """Create all models and optimizers.
    
    Returns a dictionary containing all models and optimizers.
    """
    
    # Initialize models
    generator = SpeechTokenizer(
        hidden_size=config.hidden_size,
        encoder_mlp_dim=config.encoder_mlp_dim,
        encoder_depth=config.encoder_depth,
        encoder_heads=config.encoder_heads,
        phoneme_codebook_size=config.phoneme_codebook_size,
        bsq_spherical_dim=config.bsq_spherical_dim,
        temperature=config.temperature,
        decoder_output_48khz=config.decoder_output_48khz,
        rngs=rngs
    )
    
    mpd = MPD(rngs=rngs)
    msd = MSD(rngs=rngs)
    mstftd = MSTFTD(rngs=rngs)
    
    # Learning rate schedules
    gen_lr = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate_g,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps - config.warmup_steps,
        end_value=config.learning_rate_g * 0.01
    )
    
    disc_lr = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate_d,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps - config.warmup_steps,
        end_value=config.learning_rate_d * 0.01
    )
    
    # Optimizers
    gen_tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(gen_lr, weight_decay=config.weight_decay)
    )
    
    disc_tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(disc_lr, weight_decay=config.weight_decay)
    )
    
    gen_optimizer = nnx.Optimizer(generator, gen_tx)
    mpd_optimizer = nnx.Optimizer(mpd, disc_tx)
    msd_optimizer = nnx.Optimizer(msd, disc_tx)
    mstftd_optimizer = nnx.Optimizer(mstftd, disc_tx)
    
    return {
        'generator': generator,
        'mpd': mpd,
        'msd': msd,
        'mstftd': mstftd,
        'gen_optimizer': gen_optimizer,
        'mpd_optimizer': mpd_optimizer,
        'msd_optimizer': msd_optimizer,
        'mstftd_optimizer': mstftd_optimizer
    }

def create_data_iterator(config: TrainingConfig):
    dataset = create_emilia_ds(
        AudioConfig(
            streaming=True,
            batch_size=config.batch_size,
            sample_rate=config.sample_rate,
        )
    )

    return iter(dataset)

@partial(nnx.jit, static_argnums=10)
def train_discriminator_step(
        generator: SpeechTokenizer,
        msd: MSD,
        mstftd: MSTFTD,
        mpd: MPD,
        msd_optimizer: nnx.Optimizer,
        mpd_optimizer: nnx.Optimizer,
        stftd_optimizer: nnx.Optimizer,

        audio: jax.Array,
        padding_mask: jax.Array,
        encoder_causal_mask: jax.Array,
        loss_type: str = "lsgan"
):
    if loss_type == "lsgan":
        disc_loss_fn = compute_discriminator_loss_lsgan
    else:
        disc_loss_fn = compute_discriminator_loss_hinge

    generator.eval()
    fake_audio, _, _, _, _, _, _, _ = generator(audio, encoder_causal_mask)

    def msd_loss_fn(msd):
        real_features = msd(audio)  # List of feature maps from each scale
        fake_features = msd(fake_audio)
        
        # Extract only the final output (last item) from each feature map
        real_outputs = [feat_map[-1] for feat_map in real_features]
        fake_outputs = [feat_map[-1] for feat_map in fake_features]
        
        loss = disc_loss_fn(real_outputs, fake_outputs)
        return loss
    
    def mpd_loss_fn(mpd):
        real_features = mpd(audio)
        fake_features = mpd(fake_audio)
        
        # Extract only the final output (last item) from each feature map
        real_outputs = [feat_map[-1] for feat_map in real_features]
        fake_outputs = [feat_map[-1] for feat_map in fake_features]
        
        loss = disc_loss_fn(real_outputs, fake_outputs)
        return loss
    
    def stftd_loss_fn(mstftd):
        real_features = mstftd(audio)
        fake_features = mstftd(fake_audio)
        
        # Extract only the final output (last item) from each feature map
        real_outputs = [feat_map[-1] for feat_map in real_features]
        fake_outputs = [feat_map[-1] for feat_map in fake_features]
        
        loss = disc_loss_fn(real_outputs, fake_outputs)
        return loss

    msd.train()
    mpd.train()
    mstftd.train()
    # Compute gradients and losses for each discriminator
    msd_loss_val, msd_grads = nnx.value_and_grad(msd_loss_fn)(msd)
    mpd_loss_val, mpd_grads = nnx.value_and_grad(mpd_loss_fn)(mpd)
    stftd_loss_val, stftd_grads = nnx.value_and_grad(stftd_loss_fn)(mstftd)
    
    # Update discriminators
    msd_optimizer.update(msd_grads)
    mpd_optimizer.update(mpd_grads)
    stftd_optimizer.update(stftd_grads)
    
    # Return loss values for logging
    return {
        'msd_loss': msd_loss_val,
        'mpd_loss': mpd_loss_val,
        'mstftd_loss': stftd_loss_val,
        'total_disc_loss': msd_loss_val + mpd_loss_val + stftd_loss_val
    }


@partial(nnx.jit, static_argnums=(11, 12, 13))
def train_generator_step(
        generator: SpeechTokenizer,
        gen_optimizer: nnx.Optimizer,
        msd: MSD,
        mstftd: MSTFTD,
        mpd: MPD,
        
        audio: jax.Array,
        padding_mask: jax.Array,
        encoder_causal_mask: jax.Array,
        encoder_mask: jax.Array,
        phoneme_indices: jax.Array = None,
        phoneme_mask: jax.Array = None,
        
        use_discriminators: bool = True,
        loss_type: str = "lsgan",
        config: TrainingConfig = None,
):
    """Train generator with all losses.
    
    Args:
        generator: SpeechTokenizer model
        gen_optimizer: Generator optimizer
        msd, mstftd, mpd: Discriminators (used if use_discriminators=True)
        audio: Raw audio [B, T, 1]
        padding_mask: Audio padding mask [B, T] where True = valid
        encoder_causal_mask: Encoder attention mask [B, T', T']
        encoder_mask: Encoder validity mask [B, T'] where True = valid
        phoneme_indices: Target phoneme indices [B, N] (optional)
        phoneme_mask: Phoneme padding mask [B, N] where 1.0 = padded (optional)
        use_discriminators: Whether to use adversarial losses
        loss_type: "lsgan" or "hinge" for adversarial loss
        config: Training configuration with loss weights
    """
    
    if loss_type == "lsgan":
        adv_g_loss_fn = adversarial_g_loss_lsgan
    else:
        adv_g_loss_fn = adversarial_g_loss_hinge
    
    def generator_loss_fn(generator):
        # Forward pass
        (
            reconstructed,
            phoneme_indices_pred,
            acoustic_codes,
            encoder_output,
            phoneme_logits,
            vq_quantized,
            bsq_quantized,
            vq_residual,
        ) = generator(audio, encoder_causal_mask)
        
        total_loss = 0.0
        loss_dict = {}
        
        # 1. Reconstruction losses
        # L1 loss
        l1_loss_val = l1_loss(reconstructed, audio, padding_mask)
        total_loss += config.reconstruction_weight * l1_loss_val
        loss_dict["l1_loss"] = l1_loss_val
        
        # Mel-spectrogram loss (expects [B, T] not [B, T, 1])
        audio_2d = jnp.squeeze(audio, axis=-1)
        reconstructed_2d = jnp.squeeze(reconstructed, axis=-1)
        mel_loss_val = mel_spectrogram_loss(reconstructed_2d, audio_2d, padding_mask)
        total_loss += config.mel_weight * mel_loss_val
        loss_dict["mel_loss"] = mel_loss_val
        
        # Multi-scale STFT loss
        stft_loss_val = multi_scale_stft_loss(reconstructed, audio, padding_mask)
        total_loss += config.stft_weight * stft_loss_val
        loss_dict["stft_loss"] = stft_loss_val
        
        # 2. Quantizer commitment losses (bidirectional)
        # VQ losses - two directions with different weights
        # a) Commitment loss: pulls encoder output toward VQ codebook
        vq_commitment_loss_val = vq_loss(
            jax.lax.stop_gradient(vq_quantized),  # Fixed target
            encoder_output,                        # Updates encoder
            encoder_mask, 
            vq_weight=config.vq_commitment_weight  # 0.25
        )
        total_loss += vq_commitment_loss_val
        loss_dict["vq_commitment_loss"] = vq_commitment_loss_val
        
        # b) Codebook loss: pulls VQ codebook toward encoder output
        vq_codebook_loss_val = vq_loss(
            vq_quantized,                              # Updates VQ codebook
            jax.lax.stop_gradient(encoder_output),     # Fixed target
            encoder_mask,
            vq_weight=config.vq_codebook_weight        # 1.0
        )
        total_loss += vq_codebook_loss_val
        loss_dict["vq_codebook_loss"] = vq_codebook_loss_val
        
        # BSQ losses - two directions with different weights
        # a) Commitment loss: pulls residual toward BSQ output
        bsq_commitment_loss_val = bsq_loss(
            jax.lax.stop_gradient(bsq_quantized),  # Fixed target
            vq_residual,                            # Updates through encoder/VQ
            encoder_mask,
            bsq_weight=config.bsq_commitment_weight  # 0.25
        )
        total_loss += bsq_commitment_loss_val
        loss_dict["bsq_commitment_loss"] = bsq_commitment_loss_val
        
        # b) Codebook loss: pulls BSQ toward residual
        bsq_codebook_loss_val = bsq_loss(
            bsq_quantized,                          # Updates BSQ parameters
            jax.lax.stop_gradient(vq_residual),     # Fixed target
            encoder_mask,
            bsq_weight=config.bsq_codebook_weight   # 1.0
        )
        total_loss += bsq_codebook_loss_val
        loss_dict["bsq_codebook_loss"] = bsq_codebook_loss_val
        
        # 3. Phoneme CTC loss (if phoneme labels provided)
        if phoneme_indices is not None and phoneme_mask is not None:
            # a) Phoneme loss for encoder update
            encoder_output_for_phoneme = encoder_output
            codebook_detached = jax.lax.stop_gradient(generator.quantizer.phoneme_vq.codebook)
            diff_squared = jnp.square(encoder_output_for_phoneme[:, :, None, :] - codebook_detached[None, None, :, :])
            distances_encoder = jnp.sum(diff_squared, axis=-1)
            phoneme_logits_encoder = jax.nn.log_softmax(-distances_encoder, axis=-1)
            
            phoneme_encoder_loss = phoneme_ctc_loss(
                phoneme_logits_encoder,
                encoder_mask,
                phoneme_indices,
                phoneme_mask,
                blank_id=0,
                reduction="mean"
            )
            total_loss += config.phoneme_encoder_weight * phoneme_encoder_loss
            loss_dict["phoneme_encoder_loss"] = phoneme_encoder_loss
            
            # b) Phoneme loss for VQ codebook update
            encoder_stopped = jax.lax.stop_gradient(encoder_output)
            distances_codebook = generator.quantizer.phoneme_vq.get_distances(encoder_stopped)
            phoneme_logits_codebook = jax.nn.log_softmax(-distances_codebook, axis=-1)
            
            phoneme_codebook_loss = phoneme_ctc_loss(
                phoneme_logits_codebook,
                encoder_mask,
                phoneme_indices,
                phoneme_mask,
                blank_id=0,
                reduction="mean"
            )
            total_loss += config.phoneme_codebook_weight * phoneme_codebook_loss
            loss_dict["phoneme_codebook_loss"] = phoneme_codebook_loss
        
        # 4. Adversarial and feature matching losses (if discriminators enabled)
        if use_discriminators:
            # Get discriminator outputs for fake audio
            msd_fake_features = msd(reconstructed)
            mpd_fake_features = mpd(reconstructed)
            stftd_fake_features = mstftd(reconstructed)
            
            # Extract final outputs for adversarial loss
            msd_fake_outputs = [feat_map[-1] for feat_map in msd_fake_features]
            mpd_fake_outputs = [feat_map[-1] for feat_map in mpd_fake_features]
            stftd_fake_outputs = [feat_map[-1] for feat_map in stftd_fake_features]
            
            # Combine all discriminator outputs
            all_fake_outputs = msd_fake_outputs + mpd_fake_outputs + stftd_fake_outputs
            
            # Adversarial loss
            adv_loss_val = adv_g_loss_fn(all_fake_outputs)
            total_loss += config.adversarial_weight * adv_loss_val
            loss_dict["adv_loss"] = adv_loss_val
            
            # Feature matching loss (using all intermediate features)
            msd.eval()
            mpd.eval()
            mstftd.eval()
            msd_real_features = msd(audio)
            mpd_real_features = mpd(audio)
            stftd_real_features = mstftd(audio)
            # concatenate all feature maps, anyway it is element wise...
            real_features = msd_real_features + mpd_real_features + stftd_real_features
            fake_features = msd_fake_features + mpd_fake_features + stftd_fake_features
            
            fm_loss_val = feature_matching_loss(real_features, fake_features)
            total_loss += config.feature_matching_weight * fm_loss_val
            loss_dict["fm_loss"] = fm_loss_val
        
        return total_loss, loss_dict
    
    # Compute gradients and update
    generator.train()
    (total_loss, loss_dict), grads = nnx.value_and_grad(
        generator_loss_fn, has_aux=True
    )(generator)
    
    gen_optimizer.update(grads)
    
    # Return loss values for logging
    return loss_dict


# Create pmap versions of training steps for data parallel training
train_discriminator_step_pmap = nnx.pmap(
    train_discriminator_step,
    in_axes=(None, None, None, None, None, None, None, 0, 0, 0, None),  # Shard data, replicate models
    out_axes=0,  # Shard outputs across devices
    axis_name="devices",
    static_broadcasted_argnums=(10,)  # loss_type is static
)

train_generator_step_pmap = nnx.pmap(
    train_generator_step,
    in_axes=(None, None, None, None, None, 0, 0, 0, 0, 0, 0, None, None, None),  # Shard data, replicate models
    out_axes=0,  # Shard outputs across devices
    axis_name="devices",
    static_broadcasted_argnums=(11, 12, 13)  # use_discriminators, loss_type, config are static
)


def shard_batch(batch, num_devices):
    """Shard batch across devices.
    
    Args:
        batch: Dictionary with keys like 'audio', 'padding_mask', etc.
        num_devices: Number of devices to shard across
    
    Returns:
        Sharded batch with first dimension split across devices
    """
    def shard_array(arr):
        # Reshape to (num_devices, batch_per_device, ...)
        batch_size = arr.shape[0]
        if batch_size % num_devices != 0:
            raise ValueError(f"Batch size {batch_size} not divisible by {num_devices} devices")
        
        batch_per_device = batch_size // num_devices
        new_shape = (num_devices, batch_per_device) + arr.shape[1:]
        return arr.reshape(new_shape)
    
    sharded_batch = {}
    for key, value in batch.items():
        if isinstance(value, jax.Array):
            sharded_batch[key] = shard_array(value)
        else:
            sharded_batch[key] = value
    
    return sharded_batch


def save_checkpoint(
    checkpoint_path: str,
    step: int,
    generator: SpeechTokenizer,
    gen_optimizer: nnx.Optimizer,
    msd: MSD,
    mpd: MPD,
    mstftd: MSTFTD,
    msd_optimizer: nnx.Optimizer,
    mpd_optimizer: nnx.Optimizer,
    mstftd_optimizer: nnx.Optimizer,
):
    """Save checkpoint using orbax."""
    # Create checkpoint directory
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Create checkpoint manager
    ckptr = ocp.StandardCheckpointer()
    
    # Save models and optimizers
    checkpoint_data = {
        'step': step,
        'generator': nnx.state(generator),
        'gen_optimizer': nnx.state(gen_optimizer),
        'msd': nnx.state(msd),
        'mpd': nnx.state(mpd),
        'mstftd': nnx.state(mstftd),
        'msd_optimizer': nnx.state(msd_optimizer),
        'mpd_optimizer': nnx.state(mpd_optimizer),
        'mstftd_optimizer': nnx.state(mstftd_optimizer),
    }
    
    ckptr.save(checkpoint_path, checkpoint_data)
    print(f"Checkpoint saved at step {step}")


def load_checkpoint(
    checkpoint_path: str,
    generator: SpeechTokenizer,
    gen_optimizer: nnx.Optimizer,
    msd: MSD,
    mpd: MPD,
    mstftd: MSTFTD,
    msd_optimizer: nnx.Optimizer,
    mpd_optimizer: nnx.Optimizer,
    mstftd_optimizer: nnx.Optimizer,
):
    """Load checkpoint using orbax."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0
    
    # Create checkpoint manager
    ckptr = ocp.StandardCheckpointer()
    
    # Load checkpoint data
    checkpoint_data = ckptr.restore(checkpoint_path)
    
    # Restore model and optimizer states
    nnx.update(generator, checkpoint_data['generator'])
    nnx.update(gen_optimizer, checkpoint_data['gen_optimizer'])
    nnx.update(msd, checkpoint_data['msd'])
    nnx.update(mpd, checkpoint_data['mpd'])
    nnx.update(mstftd, checkpoint_data['mstftd'])
    nnx.update(msd_optimizer, checkpoint_data['msd_optimizer'])
    nnx.update(mpd_optimizer, checkpoint_data['mpd_optimizer'])
    nnx.update(mstftd_optimizer, checkpoint_data['mstftd_optimizer'])
    
    step = checkpoint_data['step']
    print(f"Restored checkpoint from step {step}")
    return step


def train(config: TrainingConfig):
    """Main training loop with data parallel training."""
    
    # Setup
    num_devices = jax.device_count()
    print(f"Training on {num_devices} devices")
    
    # Initialize wandb
    wandb_run = init_wandb(config, project="speech-tokenizer")
    
    # Initialize RNGs
    key = jax.random.PRNGKey(config.seed)
    key, model_key = jax.random.split(key)
    rngs = nnx.Rngs(model_key)
    
    # Create models and optimizers
    models = create_models_and_optimizers(config, rngs)
    generator = models['generator']
    gen_optimizer = models['gen_optimizer']
    msd = models['msd']
    mpd = models['mpd']
    mstftd = models['mstftd']
    msd_optimizer = models['msd_optimizer']
    mpd_optimizer = models['mpd_optimizer']
    mstftd_optimizer = models['mstftd_optimizer']
    
    # Create data iterator
    data_iter = create_data_iterator(config)
    
    # Training loop
    for step in range(config.total_steps):
        # Get batch
        batch = next(data_iter)
        
        # Shard batch across devices
        sharded_batch = shard_batch(batch, num_devices)
        
        # Extract arrays from batch
        audio = sharded_batch['audio']  # [num_devices, batch_per_device, T, 1]
        padding_mask = sharded_batch['padding_mask']  # [num_devices, batch_per_device, T]
        encoder_causal_mask = sharded_batch.get('encoder_causal_mask')  # [num_devices, batch_per_device, T', T']
        encoder_mask = sharded_batch.get('encoder_mask')  # [num_devices, batch_per_device, T']
        phoneme_indices = sharded_batch.get('phoneme_indices')  # Optional
        phoneme_mask = sharded_batch.get('phoneme_mask')  # Optional
        
        # Discriminator training step (if enabled)
        if step >= config.disc_start_step and step % config.disc_update_freq == 0:
            disc_losses = train_discriminator_step_pmap(
                generator, msd, mstftd, mpd,
                msd_optimizer, mpd_optimizer, mstftd_optimizer,
                audio, padding_mask, encoder_causal_mask,
                "lsgan"  # Pass as positional argument
            )
            # Average losses across devices
            disc_losses = jax.tree_map(lambda x: jnp.mean(x).item(), disc_losses)
            log_discriminator_metrics(disc_losses, step)
        
        # Generator training step
        use_discriminators = step >= config.disc_start_step
        gen_losses = train_generator_step_pmap(
            generator, gen_optimizer,
            msd, mstftd, mpd,
            audio, padding_mask, encoder_causal_mask, encoder_mask,
            phoneme_indices, phoneme_mask,
            use_discriminators,  # Pass as positional argument
            "lsgan",  # Pass as positional argument
            config  # Pass as positional argument
        )
        # Average losses across devices
        gen_losses = jax.tree_map(lambda x: jnp.mean(x).item(), gen_losses)
        log_generator_metrics(gen_losses, step)
        
        # Simple logging
        if step % 10 == 0:
            print(f"Step {step}/{config.total_steps}")
        
        # Checkpointing
        if step % config.checkpoint_every == 0 and step > 0:
            checkpoint_path = f"{config.checkpoint_dir}/step_{step}"
            print(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(
                checkpoint_path,
                step,
                generator,
                gen_optimizer,
                msd,
                mpd,
                mstftd,
                msd_optimizer,
                mpd_optimizer,
                mstftd_optimizer
            )
            
    print("Training completed!")
    finish_wandb()


if __name__ == "__main__":
    config = TrainingConfig()
    train(config)

