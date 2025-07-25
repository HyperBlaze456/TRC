"""
Generator losses for audio tokenizer training (DAC-style recipe).

This module implements all losses needed for training the generator/encoder-decoder
in a GAN-based audio codec, following the DAC (Descript Audio Codec) approach.
"""

import jax
import jax.numpy as jnp
import optax
from functools import partial


# ============================================================================
# Reconstruction Losses (Time Domain)
# ============================================================================

def l1_loss(
    predictions: jax.Array,
    targets: jax.Array,
    mask: jax.Array = None
) -> jax.Array:
    """L1 loss with optional masking.
    
    Args:
        predictions: Predicted audio [B, T, C]
        targets: Target audio [B, T, C]
        mask: Optional padding mask [B, T] where True = valid
        
    Returns:
        Scalar L1 loss
    """
    loss = jnp.abs(predictions - targets)
    
    if mask is not None:
        # Expand mask to match audio shape
        if mask.ndim == 2 and loss.ndim == 3:
            mask = mask[:, :, None]  # [B, T, 1]
        
        # Apply mask and compute mean over valid positions
        loss = loss * mask
        return jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
        return jnp.mean(loss)


def l2_loss(
    predictions: jax.Array,
    targets: jax.Array,
    mask: jax.Array = None
) -> jax.Array:
    """L2 (MSE) loss with optional masking.
    
    Args:
        predictions: Predicted audio [B, T, C]
        targets: Target audio [B, T, C]
        mask: Optional padding mask [B, T] where True = valid
        
    Returns:
        Scalar L2 loss
    """
    loss = jnp.square(predictions - targets)
    
    if mask is not None:
        # Expand mask to match audio shape
        if mask.ndim == 2 and loss.ndim == 3:
            mask = mask[:, :, None]  # [B, T, 1]
        
        # Apply mask and compute mean over valid positions
        loss = loss * mask
        return jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
        return jnp.mean(loss)


# ============================================================================
# Mel-Spectrogram Loss
# ============================================================================

def mel_spectrogram_loss(
    predictions: jax.Array,
    targets: jax.Array,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
    mask: jax.Array = None
) -> jax.Array:
    """Mel-spectrogram L1 loss for perceptual quality.
    
    Args:
        predictions: Predicted audio [B, T, C]
        targets: Target audio [B, T, C]
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT
        n_mels: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency (defaults to sample_rate/2)
        mask: Optional padding mask [B, T]
        
    Returns:
        Scalar mel-spectrogram loss
    """
    if fmax is None:
        fmax = sample_rate / 2.0
    
    # For now, return a placeholder
    # In practice, you'd use a proper mel-spectrogram implementation
    # This is a simplified version using FFT
    
    # Remove channel dimension for processing
    if predictions.ndim == 3:
        predictions = predictions[:, :, 0]  # [B, T]
        targets = targets[:, :, 0]  # [B, T]
    
    # Simple magnitude spectrum as placeholder for mel-spectrogram
    # In real implementation, use proper mel filterbank
    pred_fft = jnp.fft.rfft(predictions, n=n_fft, axis=-1)
    target_fft = jnp.fft.rfft(targets, n=n_fft, axis=-1)
    
    pred_mag = jnp.abs(pred_fft)
    target_mag = jnp.abs(target_fft)
    
    # Log magnitude
    eps = 1e-5
    pred_log_mag = jnp.log(pred_mag + eps)
    target_log_mag = jnp.log(target_mag + eps)
    
    loss = jnp.abs(pred_log_mag - target_log_mag)
    
    if mask is not None:
        # Create frequency mask based on time mask
        # This is simplified - proper implementation would handle STFT framing
        freq_mask = mask[:, :pred_log_mag.shape[1]]
        if freq_mask.ndim == 2 and loss.ndim == 3:
            freq_mask = freq_mask[:, :, None]
        
        loss = loss * freq_mask
        return jnp.sum(loss) / jnp.maximum(jnp.sum(freq_mask), 1.0)
    else:
        return jnp.mean(loss)


# ============================================================================
# Multi-Resolution STFT Losses
# ============================================================================

def stft_loss(
    predictions: jax.Array,
    targets: jax.Array,
    n_fft: int,
    hop_length: int = None,
    mask: jax.Array = None
) -> tuple[jax.Array, jax.Array]:
    """Single-resolution STFT loss (spectral convergence + log magnitude).
    
    Args:
        predictions: Predicted audio [B, T, C]
        targets: Target audio [B, T, C]
        n_fft: FFT size
        hop_length: Hop length (defaults to n_fft // 4)
        mask: Optional padding mask [B, T]
        
    Returns:
        spectral_convergence: Relative spectral error
        log_magnitude: Absolute log magnitude error
    """
    if hop_length is None:
        hop_length = n_fft // 4
    
    # Remove channel dimension
    if predictions.ndim == 3:
        predictions = predictions[:, :, 0]
        targets = targets[:, :, 0]
    
    # Compute STFT (simplified - real implementation needs proper windowing)
    pred_stft = jnp.fft.rfft(predictions, n=n_fft, axis=-1)
    target_stft = jnp.fft.rfft(targets, n=n_fft, axis=-1)
    
    pred_mag = jnp.abs(pred_stft)
    target_mag = jnp.abs(target_stft)
    
    # Spectral convergence loss
    sc_loss = jnp.sqrt(jnp.sum(jnp.square(pred_mag - target_mag), axis=-1)) / (
        jnp.sqrt(jnp.sum(jnp.square(target_mag), axis=-1)) + 1e-6
    )
    
    # Log magnitude loss
    eps = 1e-5
    lm_loss = jnp.abs(jnp.log(pred_mag + eps) - jnp.log(target_mag + eps))
    
    if mask is not None:
        # Apply mask
        batch_size = predictions.shape[0]
        freq_frames = pred_mag.shape[1]
        
        # Simple mask interpolation for frequency domain
        if mask.shape[1] != freq_frames:
            # Downsample mask to match STFT frames
            indices = jnp.linspace(0, mask.shape[1] - 1, freq_frames).astype(jnp.int32)
            mask_freq = mask[:, indices]
        else:
            mask_freq = mask[:, :freq_frames]
        
        sc_loss = sc_loss * mask_freq
        sc_loss = jnp.sum(sc_loss) / jnp.maximum(jnp.sum(mask_freq), 1.0)
        
        lm_loss = lm_loss * mask_freq[:, :, None]
        lm_loss = jnp.sum(lm_loss) / jnp.maximum(jnp.sum(mask_freq[:, :, None]), 1.0)
    else:
        sc_loss = jnp.mean(sc_loss)
        lm_loss = jnp.mean(lm_loss)
    
    return sc_loss, lm_loss


def multi_resolution_stft_loss(
    predictions: jax.Array,
    targets: jax.Array,
    n_ffts: list[int] = [512, 1024, 2048],
    hop_lengths: list[int] = None,
    mask: jax.Array = None
) -> tuple[jax.Array, jax.Array]:
    """Multi-resolution STFT loss.
    
    Args:
        predictions: Predicted audio [B, T, C]
        targets: Target audio [B, T, C]
        n_ffts: List of FFT sizes
        hop_lengths: List of hop lengths (defaults to n_fft // 4)
        mask: Optional padding mask [B, T]
        
    Returns:
        total_sc: Total spectral convergence loss
        total_lm: Total log magnitude loss
    """
    if hop_lengths is None:
        hop_lengths = [n_fft // 4 for n_fft in n_ffts]
    
    total_sc = 0.0
    total_lm = 0.0
    
    for n_fft, hop_length in zip(n_ffts, hop_lengths):
        sc, lm = stft_loss(predictions, targets, n_fft, hop_length, mask)
        total_sc += sc
        total_lm += lm
    
    # Average over resolutions
    total_sc = total_sc / len(n_ffts)
    total_lm = total_lm / len(n_ffts)
    
    return total_sc, total_lm


# ============================================================================
# Quantization Commitment Losses
# ============================================================================

def vq_commitment_loss(
    encoder_output: jax.Array,
    quantized: jax.Array,
    mask: jax.Array = None,
    beta: float = 0.1
) -> jax.Array:
    """VQ commitment loss for phoneme quantizer.
    
    Args:
        encoder_output: Output before quantization [B, T, D]
        quantized: Quantized output [B, T, D]
        mask: Optional encoder mask [B, T]
        beta: Commitment weight (low for flexibility)
        
    Returns:
        Scalar commitment loss
    """
    # Stop gradient on quantized for commitment
    loss = jnp.square(encoder_output - jax.lax.stop_gradient(quantized))
    
    if mask is not None:
        if mask.ndim == 2 and loss.ndim == 3:
            mask = mask[:, :, None]  # [B, T, 1]
        loss = loss * mask
        loss = jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
        loss = jnp.mean(loss)
    
    return beta * loss


def bsq_commitment_loss(
    residual: jax.Array,
    quantized: jax.Array,
    mask: jax.Array = None,
    gamma: float = 1.0
) -> jax.Array:
    """BSQ commitment loss for residual quantizer.
    
    Args:
        residual: Residual before quantization [B, T, D]
        quantized: Quantized residual [B, T, D]
        mask: Optional encoder mask [B, T]
        gamma: Commitment weight
        
    Returns:
        Scalar commitment loss
    """
    loss = jnp.square(residual - jax.lax.stop_gradient(quantized))
    
    if mask is not None:
        if mask.ndim == 2 and loss.ndim == 3:
            mask = mask[:, :, None]  # [B, T, 1]
        loss = loss * mask
        loss = jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
        loss = jnp.mean(loss)
    
    return gamma * loss


# ============================================================================
# Adversarial Losses
# ============================================================================

def adversarial_g_loss(
    disc_outputs: list[jax.Array],
    loss_type: str = "lsgan"
) -> jax.Array:
    """Generator adversarial loss.
    
    Args:
        disc_outputs: List of discriminator outputs for generated samples
        loss_type: "lsgan" or "hinge"
        
    Returns:
        Scalar adversarial loss
    """
    total_loss = 0.0
    
    for output in disc_outputs:
        if loss_type == "lsgan":
            # Generator wants discriminator to output 1
            loss = jnp.mean(jnp.square(output - 1))
        elif loss_type == "hinge":
            # Generator wants positive discriminator outputs
            loss = -jnp.mean(output)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        total_loss += loss
    
    return total_loss / len(disc_outputs)


def feature_matching_loss(
    real_features: list[list[jax.Array]],
    fake_features: list[list[jax.Array]]
) -> jax.Array:
    """Feature matching loss for training stability.
    
    Args:
        real_features: Features from discriminators processing real audio
        fake_features: Features from discriminators processing fake audio
        
    Returns:
        Scalar feature matching loss
    """
    total_loss = 0.0
    num_features = 0
    
    for real_feats, fake_feats in zip(real_features, fake_features):
        for real_f, fake_f in zip(real_feats, fake_feats):
            # L1 loss between features
            loss = jnp.mean(jnp.abs(fake_f - jax.lax.stop_gradient(real_f)))
            total_loss += loss
            num_features += 1
    
    # Average over all features
    return total_loss / max(num_features, 1)


# ============================================================================
# Combined Generator Loss
# ============================================================================

def compute_generator_loss(
    pred_audio: jax.Array,
    target_audio: jax.Array,
    encoder_output: jax.Array,
    vq_quantized: jax.Array,
    bsq_quantized: jax.Array,
    vq_residual: jax.Array,
    disc_outputs: list[jax.Array],
    disc_features_real: list[list[jax.Array]],
    disc_features_fake: list[list[jax.Array]],
    padding_mask: jax.Array,
    encoder_mask: jax.Array,
    # Loss configuration
    loss_weights: dict = None,
    adversarial_loss_type: str = "lsgan",
    n_ffts: list[int] = [512, 1024, 2048]
) -> tuple[jax.Array, dict]:
    """Compute all generator losses with DAC-style weighting.
    
    Args:
        pred_audio: Reconstructed audio [B, T, C]
        target_audio: Original audio [B, T, C]
        encoder_output: Encoder output before quantization [B, T', D]
        vq_quantized: VQ output [B, T', D]
        bsq_quantized: BSQ output [B, T', D]
        vq_residual: Residual before BSQ [B, T', D]
        disc_outputs: Discriminator outputs for generated audio
        disc_features_real: Discriminator features for real audio
        disc_features_fake: Discriminator features for fake audio
        padding_mask: Audio-level mask [B, T] where True = valid
        encoder_mask: Encoder-level mask [B, T'] where True = valid
        loss_weights: Dictionary of loss weights
        adversarial_loss_type: "lsgan" or "hinge"
        n_ffts: FFT sizes for multi-resolution STFT
        
    Returns:
        total_loss: Combined scalar loss
        metrics: Dictionary of individual losses
    """
    if loss_weights is None:
        loss_weights = {
            # Reconstruction
            'l1': 1.0,
            'l2': 1.0,
            'mel': 15.0,  # High weight for perceptual quality
            'stft_sc': 2.0,  # Spectral convergence
            'stft_lm': 1.0,  # Log magnitude
            
            # Quantization
            'vq_commit': 0.1,  # Low for flexibility
            'bsq_commit': 1.0,
            
            # Adversarial
            'adversarial': 1.0,
            'feature_match': 10.0,
        }
    
    metrics = {}
    
    # Reconstruction losses
    metrics['l1'] = l1_loss(pred_audio, target_audio, padding_mask)
    metrics['l2'] = l2_loss(pred_audio, target_audio, padding_mask)
    metrics['mel'] = mel_spectrogram_loss(pred_audio, target_audio, mask=padding_mask)
    
    # Multi-resolution STFT losses
    sc_loss, lm_loss = multi_resolution_stft_loss(
        pred_audio, target_audio, n_ffts=n_ffts, mask=padding_mask
    )
    metrics['stft_sc'] = sc_loss
    metrics['stft_lm'] = lm_loss
    
    # Quantization losses
    metrics['vq_commit'] = vq_commitment_loss(
        encoder_output, vq_quantized, encoder_mask, beta=1.0
    )
    metrics['bsq_commit'] = bsq_commitment_loss(
        vq_residual, bsq_quantized, encoder_mask, gamma=1.0
    )
    
    # Adversarial losses
    metrics['adversarial'] = adversarial_g_loss(disc_outputs, adversarial_loss_type)
    metrics['feature_match'] = feature_matching_loss(disc_features_real, disc_features_fake)
    
    # Combine all losses
    total_loss = sum(
        loss_weights.get(name, 0.0) * value 
        for name, value in metrics.items()
    )
    
    metrics['total'] = total_loss
    
    return total_loss, metrics


# ============================================================================
# JIT-compiled versions
# ============================================================================

# Create versions with static loss configuration
compute_generator_loss_lsgan = partial(
    compute_generator_loss, 
    adversarial_loss_type="lsgan"
)

compute_generator_loss_hinge = partial(
    compute_generator_loss, 
    adversarial_loss_type="hinge"
)