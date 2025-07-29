"""
Generator losses for audio tokenizer training (DAC-style recipe).

This module implements all losses needed for training the generator/encoder-decoder
in a GAN-based audio codec, following the DAC (Descript Audio Codec) approach.
All functions are JIT-compatible with no conditional logic.
"""

from functools import partial

import jax
import jax.numpy as jnp

from tokenizer.utils.mel import MelSpectrogramJAX

# ============================================================================
# Reconstruction Losses (Time Domain)
# ============================================================================


def l1_loss(predictions: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    """L1 loss with masking.

    Args:
        predictions: Predicted audio [B, T, C]
        targets: Target audio [B, T, C]
        mask: Padding mask [B, T] where True = valid

    Returns:
        Scalar L1 loss
    """
    # Expand mask to match audio shape
    mask = jnp.expand_dims(mask, axis=-1)  # [B, T, 1]

    loss = jnp.abs(predictions - targets)
    loss = loss * mask
    return jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)


def l2_loss(predictions: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    """L2 (MSE) loss with masking.

    Args:
        predictions: Predicted audio [B, T, C]
        targets: Target audio [B, T, C]
        mask: Padding mask [B, T] where True = valid

    Returns:
        Scalar L2 loss
    """
    # Expand mask to match audio shape
    mask = jnp.expand_dims(mask, axis=-1)  # [B, T, 1]

    loss = jnp.square(predictions - targets)
    loss = loss * mask
    return jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)


# ============================================================================
# Mel-Spectrogram Loss
# ============================================================================

# Pre-initialize mel spectrogram for 24kHz
mel_transform_24k = MelSpectrogramJAX(
    sample_rate=24000, n_fft=1024, hop_length=256, n_mels=128, fmin=0.0, fmax=12000.0
)


def mel_spectrogram_loss(
    predictions: jax.Array, targets: jax.Array, mask: jax.Array
) -> jax.Array:
    """Mel-spectrogram L1 loss for perceptual quality.

    Args:
        predictions: Predicted audio [B, T]
        targets: Target audio [B, T]
        mask: Padding mask [B, T] where True = valid

    Returns:
        Scalar mel-spectrogram loss
    """
    # Compute mel spectrograms
    pred_mel = jax.vmap(mel_transform_24k)(predictions)  # [B, n_mels, n_frames]
    target_mel = jax.vmap(mel_transform_24k)(targets)  # [B, n_mels, n_frames]

    # Convert to log scale
    eps = 1e-5
    pred_log_mel = jnp.log(pred_mel + eps)
    target_log_mel = jnp.log(target_mel + eps)

    # L1 loss
    loss = jnp.abs(pred_log_mel - target_log_mel)

    # Create mel-domain mask from time-domain mask
    # Downsample mask to match mel frames
    hop_length = 256
    n_frames = loss.shape[-1]
    indices = jnp.arange(n_frames) * hop_length
    indices = jnp.minimum(indices, mask.shape[1] - 1)
    mask_mel = mask[:, indices]  # [B, n_frames]
    mask_mel = jnp.expand_dims(mask_mel, axis=1)  # [B, 1, n_frames]

    loss = loss * mask_mel
    return jnp.sum(loss) / jnp.maximum(jnp.sum(mask_mel), 1.0)


# ============================================================================
# Multi-Resolution STFT Losses
# ============================================================================


def stft_loss(
    predictions: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    n_fft: int,
    hop_length: int,
) -> tuple[jax.Array, jax.Array]:
    """Single-resolution STFT loss (spectral convergence + log magnitude).

    Args:
        predictions: Predicted audio [B, T]
        targets: Target audio [B, T]
        mask: Padding mask [B, T] where True = valid
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        spectral_convergence: Relative spectral error
        log_magnitude: Absolute log magnitude error
    """
    # Compute STFT
    pred_stft = jax.vmap(lambda x: jnp.fft.rfft(x, n=n_fft))(predictions)
    target_stft = jax.vmap(lambda x: jnp.fft.rfft(x, n=n_fft))(targets)

    pred_mag = jnp.abs(pred_stft)
    target_mag = jnp.abs(target_stft)

    # Spectral convergence loss
    diff = pred_mag - target_mag
    sc_num = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1))
    sc_den = jnp.sqrt(jnp.sum(jnp.square(target_mag), axis=-1)) + 1e-6
    sc_loss = sc_num / sc_den

    # Log magnitude loss
    eps = 1e-5
    lm_loss = jnp.abs(jnp.log(pred_mag + eps) - jnp.log(target_mag + eps))

    # Apply mask - for simplified FFT, we need to handle single-frame output
    # STFT on full signal produces shape [B, n_fft//2 + 1]
    # We'll use the first element of mask as a simple scalar mask
    mask_scalar = mask[:, 0]  # [B]

    sc_loss = sc_loss * mask_scalar
    sc_loss = jnp.sum(sc_loss) / jnp.maximum(jnp.sum(mask_scalar), 1.0)

    # For log magnitude, we need to handle [B, n_freqs] shape
    mask_freq = jnp.expand_dims(mask_scalar, axis=-1)  # [B, 1]
    lm_loss = lm_loss * mask_freq
    lm_loss = jnp.sum(lm_loss) / jnp.maximum(jnp.sum(mask_freq), 1.0)

    return sc_loss, lm_loss


# Static partial functions for different FFT sizes
stft_loss_512 = partial(stft_loss, n_fft=512, hop_length=128)
stft_loss_1024 = partial(stft_loss, n_fft=1024, hop_length=256)
stft_loss_2048 = partial(stft_loss, n_fft=2048, hop_length=512)


def multi_resolution_stft_loss(
    predictions: jax.Array, targets: jax.Array, mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Multi-resolution STFT loss with fixed resolutions.

    Args:
        predictions: Predicted audio [B, T]
        targets: Target audio [B, T]
        mask: Padding mask [B, T] where True = valid

    Returns:
        total_sc: Total spectral convergence loss
        total_lm: Total log magnitude loss
    """
    # Compute losses at each resolution
    sc1, lm1 = stft_loss_512(predictions, targets, mask)
    sc2, lm2 = stft_loss_1024(predictions, targets, mask)
    sc3, lm3 = stft_loss_2048(predictions, targets, mask)

    # Average over resolutions
    total_sc = (sc1 + sc2 + sc3) / 3.0
    total_lm = (lm1 + lm2 + lm3) / 3.0

    return total_sc, total_lm


# ============================================================================
# Quantization Commitment Losses
# ============================================================================


def vq_commitment_loss(
    encoder_output: jax.Array, quantized: jax.Array, mask: jax.Array, beta: float = 0.1
) -> jax.Array:
    """VQ commitment loss for phoneme quantizer.

    Args:
        encoder_output: Output before quantization [B, T, D]
        quantized: Quantized output [B, T, D]
        mask: Encoder mask [B, T] where True = valid
        beta: Commitment weight

    Returns:
        Scalar commitment loss
    """
    # Expand mask to match encoder shape
    mask = jnp.expand_dims(mask, axis=-1)  # [B, T, 1]

    loss = jnp.square(encoder_output - jax.lax.stop_gradient(quantized))
    loss = loss * mask
    return beta * jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)


def bsq_commitment_loss(
    residual: jax.Array, quantized: jax.Array, mask: jax.Array, gamma: float = 1.0
) -> jax.Array:
    """BSQ commitment loss for residual quantizer.

    Args:
        residual: Residual before quantization [B, T, D]
        quantized: Quantized residual [B, T, D]
        mask: Encoder mask [B, T] where True = valid
        gamma: Commitment weight

    Returns:
        Scalar commitment loss
    """
    # Expand mask to match encoder shape
    mask = jnp.expand_dims(mask, axis=-1)  # [B, T, 1]

    loss = jnp.square(residual - jax.lax.stop_gradient(quantized))
    loss = loss * mask
    return gamma * jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)


# ============================================================================
# Adversarial Losses
# ============================================================================


def adversarial_g_loss_lsgan(disc_outputs: list[jax.Array]) -> jax.Array:
    """LSGAN generator adversarial loss.

    Args:
        disc_outputs: List of discriminator outputs for generated samples

    Returns:
        Scalar adversarial loss
    """
    total_loss = 0.0
    for output in disc_outputs:
        loss = jnp.mean(jnp.square(output - 1))
        total_loss = total_loss + loss
    return total_loss / len(disc_outputs)


def adversarial_g_loss_hinge(disc_outputs: list[jax.Array]) -> jax.Array:
    """Hinge generator adversarial loss.

    Args:
        disc_outputs: List of discriminator outputs for generated samples

    Returns:
        Scalar adversarial loss
    """
    total_loss = 0.0
    for output in disc_outputs:
        loss = -jnp.mean(output)
        total_loss = total_loss + loss
    return total_loss / len(disc_outputs)


def feature_matching_loss(
    real_features: list[list[jax.Array]], fake_features: list[list[jax.Array]]
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

    for real_feats, fake_feats in zip(real_features, fake_features, strict=False):
        for real_f, fake_f in zip(real_feats, fake_feats, strict=False):
            loss = jnp.mean(jnp.abs(fake_f - jax.lax.stop_gradient(real_f)))
            total_loss = total_loss + loss
            num_features = num_features + 1

    return total_loss / jnp.maximum(num_features, 1)


# ============================================================================
# Combined Generator Loss
# ============================================================================


def compute_generator_loss_base(
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
    adversarial_loss_fn,
    # Static weights
    w_l1: float = 1.0,
    w_l2: float = 1.0,
    w_mel: float = 15.0,
    w_stft_sc: float = 2.0,
    w_stft_lm: float = 1.0,
    w_vq_commit: float = 0.1,
    w_bsq_commit: float = 1.0,
    w_adversarial: float = 1.0,
    w_feature_match: float = 10.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
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
        adversarial_loss_fn: Function for adversarial loss
        w_*: Loss weights

    Returns:
        total_loss: Combined scalar loss
        metrics: Dictionary of individual losses
    """
    # Reconstruction losses
    l1 = l1_loss(pred_audio, target_audio, padding_mask)
    l2 = l2_loss(pred_audio, target_audio, padding_mask)

    # Remove channel dimension for spectral losses
    pred_audio_2d = pred_audio[:, :, 0]
    target_audio_2d = target_audio[:, :, 0]

    mel = mel_spectrogram_loss(pred_audio_2d, target_audio_2d, padding_mask)

    # Multi-resolution STFT losses
    stft_sc, stft_lm = multi_resolution_stft_loss(
        pred_audio_2d, target_audio_2d, padding_mask
    )

    # Quantization losses
    vq_commit = vq_commitment_loss(encoder_output, vq_quantized, encoder_mask, beta=1.0)
    bsq_commit = bsq_commitment_loss(
        vq_residual, bsq_quantized, encoder_mask, gamma=1.0
    )

    # Adversarial losses
    adversarial = adversarial_loss_fn(disc_outputs)
    feature_match = feature_matching_loss(disc_features_real, disc_features_fake)

    # Combine all losses
    total_loss = (
        w_l1 * l1
        + w_l2 * l2
        + w_mel * mel
        + w_stft_sc * stft_sc
        + w_stft_lm * stft_lm
        + w_vq_commit * vq_commit
        + w_bsq_commit * bsq_commit
        + w_adversarial * adversarial
        + w_feature_match * feature_match
    )

    metrics = {
        "l1": l1,
        "l2": l2,
        "mel": mel,
        "stft_sc": stft_sc,
        "stft_lm": stft_lm,
        "vq_commit": vq_commit,
        "bsq_commit": bsq_commit,
        "adversarial": adversarial,
        "feature_match": feature_match,
        "total": total_loss,
    }

    return total_loss, metrics


# Create versions with static adversarial loss functions
compute_generator_loss_lsgan = partial(
    compute_generator_loss_base, adversarial_loss_fn=adversarial_g_loss_lsgan
)

compute_generator_loss_hinge = partial(
    compute_generator_loss_base, adversarial_loss_fn=adversarial_g_loss_hinge
)
