"""
CTC-based phoneme matching loss for audio tokenizer.

This module implements the phoneme prediction loss using Connectionist Temporal Classification (CTC).
The loss aligns encoder output frames with phoneme sequences without requiring explicit alignment.
"""

import jax
import jax.numpy as jnp
import optax


def compute_ctc_loss(
    logits: jax.Array,
    logit_paddings: jax.Array,
    labels: jax.Array,
    label_paddings: jax.Array,
    blank_id: int = 0,
    log_epsilon: float = -100000.0
) -> jax.Array:
    """Compute CTC loss for phoneme prediction.
    
    Args:
        logits: Model predictions with shape [B, T, K] where:
            B = batch size
            T = encoder output time steps (e.g., 50Hz frames)
            K = phoneme vocabulary size (218 including blank)
        logit_paddings: Padding mask for logits [B, T] where 1.0 = padded
        labels: Target phoneme indices [B, N] where N = max phoneme sequence length
        label_paddings: Padding mask for labels [B, N] where 1.0 = padded
        blank_id: Index of blank token (default: 0)
        log_epsilon: Small value for numerical stability
        
    Returns:
        ctc_losses: Per-sequence CTC loss [B]
    """
    # Compute CTC loss using optax
    ctc_losses = optax.ctc_loss(
        logits=logits,
        logit_paddings=logit_paddings,
        labels=labels,
        label_paddings=label_paddings,
        blank_id=blank_id,
        log_epsilon=log_epsilon
    )

    return ctc_losses


def phoneme_ctc_loss(
    phoneme_logits: jax.Array,
    encoder_mask: jax.Array,
    phoneme_indices: jax.Array,
    phoneme_mask: jax.Array,
    blank_id: int = 0,
    reduction: str = "mean"
) -> tuple[jax.Array, dict]:
    """Compute phoneme CTC loss with proper masking.
    
    This function handles the conversion from encoder masks to CTC-compatible
    padding format and computes the loss.
    
    Args:
        phoneme_logits: Phoneme predictions from model [B, T, K] where:
            - B = batch size
            - T = encoder output time steps
            - K = phoneme vocabulary size
        encoder_mask: Encoder validity mask [B, T] where True = valid
        phoneme_indices: Target phoneme indices [B, N]
        phoneme_mask: Phoneme padding mask [B, N] where 1.0 = padded (CTC format)
        blank_id: Index of blank token (default: 0)
        reduction: How to reduce batch losses ('mean', 'sum', or 'none')
        
    Returns:
        loss: Scalar loss (if reduction != 'none') or per-sequence losses [B]
        metrics: Dictionary with additional metrics for logging
    """
    # Convert encoder mask to CTC padding format
    # encoder_mask shape: [B, T] with True = valid
    # Need: [B, T] with 1.0 = padded
    logit_paddings = 1.0 - encoder_mask.astype(jnp.float32)  # Invert: 1.0 = padded

    # Compute CTC loss
    ctc_losses = compute_ctc_loss(
        logits=phoneme_logits,
        logit_paddings=logit_paddings,
        labels=phoneme_indices,
        label_paddings=phoneme_mask,  # Must be in CTC format, 1.0 for padded and 0.0 for not. Should be from the batch.
        blank_id=blank_id # 0
    )

    # Apply reduction
    if reduction == "mean":
        loss = jnp.mean(ctc_losses)
    elif reduction == "sum":
        loss = jnp.sum(ctc_losses)
    else:  # 'none'
        loss = ctc_losses

    # Should be commented out after test, very niche.
    # Count valid frames and phonemes for monitoring
    valid_frames = jnp.sum(encoder_mask)
    valid_phonemes = jnp.sum(1.0 - phoneme_mask)  # Invert mask to count valid

    # Compute per-sequence accuracy (greedy decoding)
    predictions = jnp.argmax(phoneme_logits, axis=-1)  # [B, T]

    metrics = {
        "ctc_loss": loss,
        "valid_frames": valid_frames,
        "valid_phonemes": valid_phonemes,
        "blank_predictions": jnp.sum(predictions == blank_id) / valid_frames,
    }

    return loss, metrics

"""
Note: The loss function has non-array arguments that changes behavior(reduction, blank_id).
This will result error while JIT, so must use functools.partial / static_arg... / topmost closure.
"""
