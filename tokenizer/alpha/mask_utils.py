import jax
import jax.numpy as jnp
from typing import Optional, Tuple


def create_padding_mask(
        lengths: jax.Array,
        max_length: int,
        causal: bool = False,
        dtype: jnp.dtype = jnp.bool_
) -> jax.Array:
    """Create attention mask for padded sequences with left-padding.

    Args:
        lengths: Array of actual sequence lengths [B]
        max_length: Maximum sequence length in the batch
        causal: Whether to apply causal masking on top of padding mask
        dtype: Data type for the mask

    Returns:
        mask: Boolean mask [B, 1, 1, T] or [B, 1, T, T] where True = valid position
              Shape is ready for attention operations
    """
    batch_size = lengths.shape[0]

    # Create position indices
    positions = jnp.arange(max_length)[None, :]  # [1, T]

    # Create padding mask (True = valid, False = padded)
    # Since we use left-padding, valid positions are at the end
    padding_mask = positions >= (max_length - lengths[:, None])  # [B, T]

    if causal:
        # Create causal mask
        causal_mask = jnp.tril(jnp.ones((max_length, max_length), dtype=dtype))  # [T, T]
        # Combine with padding mask
        # Expand padding_mask for broadcasting: [B, 1, T]
        padding_mask_expanded = padding_mask[:, None, :]
        # Final mask: [B, T, T]
        mask = padding_mask_expanded & causal_mask[None, :, :]
        # Reshape for attention: [B, 1, T, T]
        mask = mask[:, None, :, :]
    else:
        # Just padding mask for non-causal attention
        # Reshape to [B, 1, 1, T] for attention operations
        mask = padding_mask[:, None, None, :]

    return mask.astype(dtype)


def downsample_mask(
        mask: Optional[jax.Array],
        downsample_factor: int
) -> Optional[jax.Array]:
    """Downsample attention mask to match encoder output time dimension.

    Args:
        mask: Input mask [B, 1, 1, T] or [B, 1, T, T]
        downsample_factor: How much the encoder downsamples (e.g., 480 for 24kHz->50Hz)

    Returns:
        Downsampled mask with appropriate shape
    """
    if mask is None:
        return None

    # Get the time dimension (last dimension)
    T = mask.shape[-1]
    T_down = T // downsample_factor

    if mask.ndim == 4 and mask.shape[-2] == mask.shape[-1]:  # Causal mask [B, 1, T, T]
        # For causal masks, we need to handle both dimensions
        # Take every downsample_factor-th position
        indices = jnp.arange(T_down) * downsample_factor
        mask_down = mask[:, :, indices, :][:, :, :, indices]
    else:  # Non-causal mask [B, 1, 1, T]
        # For simple padding masks, just take every downsample_factor-th position
        indices = jnp.arange(T_down) * downsample_factor
        mask_down = mask[:, :, :, indices]

    return mask_down


def create_lengths_from_mask(mask: jax.Array) -> jax.Array:
    """Extract sequence lengths from a mask.

    Args:
        mask: Boolean mask [B, 1, 1, T] or [B, 1, T, T] or [B, T]

    Returns:
        lengths: Array of sequence lengths [B]
    """
    # Handle different mask shapes
    if mask.ndim == 4:
        # Take the last time dimension
        mask_1d = mask[:, 0, 0, :]  # [B, T]
    elif mask.ndim == 3:
        mask_1d = mask[:, 0, :]  # [B, T]
    else:
        mask_1d = mask  # [B, T]

    # Count valid positions
    lengths = jnp.sum(mask_1d, axis=-1)

    return lengths


def pad_sequences_left(
        sequences: jax.Array,
        max_length: Optional[int] = None,
        pad_value: float = 0.0
) -> Tuple[jax.Array, jax.Array]:
    """Pad sequences with left-padding to create a batch.

    Args:
        sequences: List or array of sequences with potentially different lengths
        max_length: Target length (Recommended to use None as it infers longest sequence)
        pad_value: Value to use for padding

    Returns:
        padded: Padded sequences [B, max_length, ...]
        lengths: Original lengths of sequences [B]
    """
    if isinstance(sequences, list):
        # Convert list of arrays to padded array
        lengths = jnp.array([len(seq) for seq in sequences])
        if max_length is None:
            max_length = int(lengths.max())

        # Pad each sequence
        padded_list = []
        for seq in sequences:
            seq_len = len(seq)
            if seq_len < max_length:
                pad_width = [(max_length - seq_len, 0)] + [(0, 0)] * (seq.ndim - 1)
                padded_seq = jnp.pad(seq, pad_width, constant_values=pad_value)
            else:
                padded_seq = seq[:max_length]
            padded_list.append(padded_seq)

        padded = jnp.stack(padded_list)
    else:
        # Assume it's already an array, just return it with lengths
        lengths = jnp.array([sequences.shape[1]] * sequences.shape[0])
        padded = sequences

    return padded, lengths


def combine_masks(
        mask1: jax.Array,
        mask2: jax.Array,
        operation: str = "and"
) -> jax.Array:
    """Combine two masks using specified bitwise operation.
    Will support more, probably.
    Args:
        mask1: First mask
        mask2: Second mask (must have compatible shape)
        operation: "and" or "or"

    Returns:
        Combined mask
    """
    if operation == "and":
        return mask1 & mask2
    elif operation == "or":
        return mask1 | mask2
    else:
        raise ValueError(f"Unknown operation: {operation}")