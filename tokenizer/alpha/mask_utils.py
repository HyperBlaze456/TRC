import jax
import jax.numpy as jnp


def create_padding_mask(
    lengths: jax.Array,
    max_length: int,
    dtype: jnp.dtype = jnp.bool_,
) -> jax.Array:
    """Create attention mask for padded sequences with left-padding.

    Args:
        lengths: Array of actual sequence lengths [B]
        max_length: Maximum sequence length in the batch
        causal: Whether to apply causal masking on top of padding mask
        dtype: Data type for the mask

    Returns:
        mask: Boolean mask [B, T] where True = valid position
    """

    # Create position indices
    positions = jnp.arange(max_length)[None, :]  # [1, T]

    # Create padding mask (True = valid, False = padded)
    # Since we use left-padding, valid positions are at the end
    padding_mask = positions >= (max_length - lengths[:, None])  # [B, T]

    return padding_mask.astype(dtype)


def create_encoder_masks(
    lengths: jax.Array,
    max_length: int,
    downsample_factor: int,
    dtype: jnp.dtype = jnp.bool_,
) -> tuple[jax.Array, jax.Array]:
    """Create encoder masks at downsampled resolution for attention

    This function creates both non-causal and causal masks directly at the
    encoder resolution, avoiding the memory overhead of full-resolution causal masks.

    Args:
        lengths: Array of actual sequence lengths [B]
        max_length: Maximum sequence length in the batch (at audio resolution)
        downsample_factor: How much the encoder downsamples (e.g., 480 for 24kHz->50Hz)
        dtype: Data type for the mask

    Returns:
        encoder_mask: Non-causal padding mask at encoder resolution [B, T']
        encoder_causal_mask: Causal mask at encoder resolution [B, T', T']
    """
    batch_size = lengths.shape[0]
    # Use ceiling division to match causal convolution output length
    encoder_max_length = -(-max_length // downsample_factor)  # Ceiling division

    # Create encoder lengths
    # A frame is valid if it contains ANY valid audio sample
    encoder_lengths = jnp.ceil(lengths / downsample_factor).astype(jnp.int32)

    # Create position indices at encoder resolution
    positions = jnp.arange(encoder_max_length)[None, :]  # [1, T']

    # Create padding mask at encoder resolution
    padding_mask = positions < encoder_lengths[:, None]  # [B, T']

    # Non-causal mask for encoder (2D)
    encoder_mask = padding_mask  # [B, T']

    # Causal mask at encoder resolution (much smaller than audio resolution)
    causal_mask = jnp.tril(
        jnp.ones((encoder_max_length, encoder_max_length), dtype=dtype)
    )  # [T', T']

    # Combine with padding mask
    padding_mask_expanded = padding_mask[:, None, :]  # [B, 1, T']
    encoder_causal_mask = padding_mask_expanded & causal_mask[None, :, :]  # [B, T', T']

    return encoder_mask.astype(dtype), encoder_causal_mask.astype(dtype)

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
    sequences: list[jax.Array], max_length: int = None, pad_value: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """Pad sequences with left-padding to create a batch.

    Args:
        sequences: List or array of sequences with potentially different lengths
        max_length: Target length (Recommended to use None as it infers longest sequence)
        pad_value: Value to use for padding

    Returns:
        padded: Padded sequences [B, max_length, ...]
        lengths: Original lengths of sequences [B]
    """
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

    return padded, lengths


def combine_masks(
    mask1: jax.Array, mask2: jax.Array, operation: str = "and"
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
    if operation == "or":
        return mask1 | mask2
    raise ValueError(f"Unknown operation: {operation}")
