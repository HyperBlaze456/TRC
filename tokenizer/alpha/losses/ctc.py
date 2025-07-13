import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import functools


def log_sum_exp(a: jnp.ndarray, axis: int = -1, keepdims: bool = False) -> jnp.ndarray:
    """Numerically stable log-sum-exp operation."""
    a_max = jnp.max(a, axis=axis, keepdims=True)
    result = a_max + jnp.log(jnp.sum(jnp.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        result = jnp.squeeze(result, axis=axis)
    return result


@functools.partial(jax.jit, static_argnames=['blank_id'])
def ctc_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    logit_lengths: jnp.ndarray,
    target_lengths: jnp.ndarray,
    blank_id: int = 0,
    reduction: str = 'mean'
) -> jnp.ndarray:
    """CTC loss implementation in JAX.
    
    Computes the Connectionist Temporal Classification loss for sequence-to-sequence
    problems where the alignment between input and output is unknown.
    
    Args:
        logits: Log probabilities of shape [B, T, V] where B is batch size,
            T is max sequence length, V is vocabulary size (including blank).
        targets: Target sequences of shape [B, S] where S is max target length.
            Should NOT include blank tokens.
        logit_lengths: Actual lengths of logit sequences [B].
        target_lengths: Actual lengths of target sequences [B].
        blank_id: ID of the blank token (usually 0).
        reduction: 'mean', 'sum', or 'none'.
    
    Returns:
        CTC loss value(s).
    """
    batch_size, max_time, vocab_size = logits.shape
    _, max_target_len = targets.shape
    
    # Convert to log probabilities if not already
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    def compute_ctc_loss_single(log_probs_single, targets_single, T, S):
        """Compute CTC loss for a single sequence pair."""
        # Add blanks between targets and at the beginning/end
        # Extended target sequence: blank, t1, blank, t2, blank, ..., tS, blank
        extended_size = 2 * S + 1
        
        # Initialize forward variable alpha
        # alpha[t, s] = log probability of all paths ending at time t in state s
        alpha = jnp.full((T, extended_size), -jnp.inf)
        
        # Initialize: can start with blank or first target symbol
        alpha = alpha.at[0, 0].set(log_probs_single[0, blank_id])
        if S > 0:
            alpha = alpha.at[0, 1].set(log_probs_single[0, targets_single[0]])
        
        # Forward pass
        for t in range(1, T):
            # Update blank states (even indices)
            for s in range(0, extended_size, 2):
                # Can come from same state or previous non-blank state
                if s == 0:
                    alpha = alpha.at[t, s].set(
                        log_probs_single[t, blank_id] + alpha[t-1, s]
                    )
                else:
                    alpha = alpha.at[t, s].set(
                        log_probs_single[t, blank_id] + 
                        log_sum_exp(jnp.array([alpha[t-1, s], alpha[t-1, s-1]]))
                    )
            
            # Update non-blank states (odd indices)
            for s in range(1, extended_size, 2):
                target_idx = (s - 1) // 2
                target_label = targets_single[target_idx]
                
                # Can come from previous blank, same state, or previous different non-blank
                candidates = [alpha[t-1, s-1]]  # From previous blank
                
                # From same state (if previous target is different)
                if s > 2 and targets_single[target_idx] != targets_single[target_idx-1]:
                    candidates.append(alpha[t-1, s])
                    # From previous non-blank (skip blank)
                    candidates.append(alpha[t-1, s-2])
                elif s <= 2 or targets_single[target_idx] == targets_single[target_idx-1]:
                    candidates.append(alpha[t-1, s])
                
                alpha = alpha.at[t, s].set(
                    log_probs_single[t, target_label] + 
                    log_sum_exp(jnp.array(candidates))
                )
        
        # Total probability is sum of ending in last blank or last target
        if S > 0:
            log_prob = log_sum_exp(jnp.array([alpha[T-1, -1], alpha[T-1, -2]]))
        else:
            log_prob = alpha[T-1, 0]
        
        return -log_prob
    
    # Vectorize over batch
    losses = jax.vmap(compute_ctc_loss_single)(
        log_probs, targets, logit_lengths, target_lengths
    )
    
    # Apply reduction
    if reduction == 'mean':
        return jnp.mean(losses)
    elif reduction == 'sum':
        return jnp.sum(losses)
    else:
        return losses


def ctc_loss_with_phoneme_prior(
    logits: jnp.ndarray,
    phoneme_logits: jnp.ndarray,
    targets: jnp.ndarray,
    logit_lengths: jnp.ndarray,
    target_lengths: jnp.ndarray,
    blank_id: int = 0,
    phoneme_weight: float = 0.1,
    reduction: str = 'mean'
) -> Tuple[jnp.ndarray, dict]:
    """CTC loss with additional phoneme classification regularization.
    
    This variant adds a phoneme classification loss to encourage the model
    to learn meaningful phoneme representations in the first codebook.
    
    Args:
        logits: Log probabilities from decoder [B, T, V].
        phoneme_logits: Direct phoneme predictions from encoder [B, T', P]
            where P is phoneme vocabulary size.
        targets: Target sequences [B, S].
        logit_lengths: Lengths of logit sequences [B].
        target_lengths: Lengths of target sequences [B].
        blank_id: ID of blank token.
        phoneme_weight: Weight for phoneme classification loss.
        reduction: 'mean', 'sum', or 'none'.
    
    Returns:
        total_loss: Combined CTC and phoneme loss.
        loss_dict: Dictionary with individual loss components.
    """
    # Standard CTC loss
    ctc_loss_value = ctc_loss(
        logits, targets, logit_lengths, target_lengths, 
        blank_id=blank_id, reduction=reduction
    )
    
    # Phoneme classification loss (cross-entropy)
    # This encourages the encoder to produce phoneme-like representations
    batch_size = phoneme_logits.shape[0]
    
    # Create phoneme targets by downsampling the target sequence
    # This is a simplified approach - in practice you might want proper alignment
    downsample_factor = logits.shape[1] // phoneme_logits.shape[1]
    phoneme_targets = targets[:, ::downsample_factor]
    
    # Ensure phoneme_targets matches phoneme_logits time dimension
    max_phoneme_len = phoneme_logits.shape[1]
    if phoneme_targets.shape[1] > max_phoneme_len:
        phoneme_targets = phoneme_targets[:, :max_phoneme_len]
    elif phoneme_targets.shape[1] < max_phoneme_len:
        pad_len = max_phoneme_len - phoneme_targets.shape[1]
        phoneme_targets = jnp.pad(
            phoneme_targets, 
            ((0, 0), (0, pad_len)), 
            constant_values=blank_id
        )
    
    # Compute cross-entropy loss for phoneme classification
    phoneme_ce = jax.nn.sparse_softmax_cross_entropy_with_logits(
        labels=phoneme_targets,
        logits=phoneme_logits
    )
    
    # Mask out padded positions
    phoneme_lengths = (target_lengths / downsample_factor).astype(jnp.int32)
    mask = jnp.arange(max_phoneme_len)[None, :] < phoneme_lengths[:, None]
    phoneme_ce = phoneme_ce * mask
    
    # Reduce phoneme loss
    if reduction == 'mean':
        phoneme_loss = jnp.sum(phoneme_ce) / jnp.sum(mask)
    elif reduction == 'sum':
        phoneme_loss = jnp.sum(phoneme_ce)
    else:
        phoneme_loss = jnp.sum(phoneme_ce, axis=1) / jnp.sum(mask, axis=1)
    
    # Combine losses
    total_loss = ctc_loss_value + phoneme_weight * phoneme_loss
    
    loss_dict = {
        'ctc_loss': ctc_loss_value,
        'phoneme_loss': phoneme_loss,
        'total_ctc_loss': total_loss
    }
    
    return total_loss, loss_dict