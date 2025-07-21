"""Loss functions for training the audio tokenizer with phoneme VQ + BSQ quantization.

The VQ layer in this architecture serves as a phoneme classifier using CTC loss,
while the BSQ layer handles acoustic residual information for reconstruction.

Usage Flow:
1. Text → G2P → Phoneme sequences (IPA symbols)
2. Phoneme sequences → Indices using phoneme vocabulary
3. Audio → Encoder → 50Hz features
4. Features → VQ layer → Phoneme predictions (CTC loss with targets)
5. Residual → BSQ layer → Acoustic features (reconstruction loss)

The phoneme vocabulary should map IPA symbols to indices, with:
- Index 0: CTC blank token '_'
- Index 1: Unknown token 'UNK'
- Index 2: Padding token 'PAD'
- Index 3+: IPA phoneme symbols
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Callable
from functools import partial
import optax


def extract_encoder_lengths(
        mask: jax.Array,
        downsample_factor: int
) -> jax.Array:
    """Extract encoder sequence lengths from audio-level mask.

    Args:
        mask: Audio-level mask [B, 1, 1, T] where True = valid
        downsample_factor: How much encoder downsamples (e.g., 480)

    Returns:
        Encoder lengths [B]
    """
    if mask is None:
        raise ValueError("Mask is required to compute encoder lengths")

    # Get audio lengths
    if mask.ndim == 4:
        mask_1d = mask[:, 0, 0, :]
    else:
        mask_1d = mask

    audio_lengths = jnp.sum(mask_1d, axis=-1)

    # Convert to encoder lengths
    encoder_lengths = audio_lengths // downsample_factor

    return encoder_lengths.astype(jnp.int32)


# ============================================================================
# Masked Loss Utilities
# ============================================================================

def apply_mask_to_loss(
        loss: jax.Array,
        mask: jax.Array,
        reduction: str = "mean"
) -> jax.Array:
    """Apply mask to loss values, handling padding properly.

    Args:
        loss: Loss values with shape matching mask or broadcastable
        mask: Boolean mask where True = valid position
        reduction: "none", "mean", or "sum"

    Returns:
        Masked loss value
    """
    # Handle different mask formats
    # If mask is [B, 1, 1, T] and loss is [B, T, D], we need to reshape
    if mask.ndim == 4 and loss.ndim == 3:
        # Extract time dimension from mask
        mask = mask[:, 0, 0, :]  # [B, T]
        # Add dimension for features
        mask = mask[:, :, None]  # [B, T, 1]
    elif mask.ndim == 4 and loss.ndim == 4:
        # Keep mask as is for 4D losses
        pass
    elif loss.ndim > mask.ndim:
        # General case: broadcast mask to match loss dimensions
        for _ in range(loss.ndim - mask.ndim):
            mask = mask[..., None]

    # Apply mask
    masked_loss = loss * mask.astype(loss.dtype)

    if reduction == "none":
        return masked_loss
    elif reduction == "sum":
        return jnp.sum(masked_loss)
    elif reduction == "mean":
        # Compute mean only over valid positions
        valid_elements = jnp.sum(mask.astype(jnp.float32))
        return jnp.sum(masked_loss) / (valid_elements + 1e-8)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# ============================================================================
# Reconstruction Losses
# ============================================================================

def l1_loss(
        predictions: jax.Array,
        targets: jax.Array,
        mask: Optional[jax.Array] = None,
        reduction: str = "mean"
) -> jax.Array:
    """L1 loss with optional masking.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional mask for valid positions
        reduction: Type of reduction

    Returns:
        L1 loss value
    """
    loss = jnp.abs(predictions - targets)

    if mask is not None:
        loss = apply_mask_to_loss(loss, mask, reduction)
    elif reduction == "mean":
        loss = jnp.mean(loss)
    elif reduction == "sum":
        loss = jnp.sum(loss)

    return loss


def l2_loss(
        predictions: jax.Array,
        targets: jax.Array,
        mask: Optional[jax.Array] = None,
        reduction: str = "mean"
) -> jax.Array:
    """L2 loss with optional masking.

    Args:
        predictions: Predicted values
        targets: Target values
        mask: Optional mask for valid positions
        reduction: Type of reduction

    Returns:
        L2 loss value
    """
    loss = jnp.square(predictions - targets)

    if mask is not None:
        loss = apply_mask_to_loss(loss, mask, reduction)
    elif reduction == "mean":
        loss = jnp.mean(loss)
    elif reduction == "sum":
        loss = jnp.sum(loss)

    return loss


# ============================================================================
# Spectral Losses
# ============================================================================

def stft(
        audio: jax.Array,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None
) -> jax.Array:
    """Compute STFT of audio signal.

    Args:
        audio: Audio signal [B, T] or [B, T, 1]
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length (defaults to n_fft)

    Returns:
        Complex STFT [B, F, T']
    """
    if audio.ndim == 3:
        audio = audio.squeeze(-1)

    if win_length is None:
        win_length = n_fft

    # Use JAX's built-in STFT for efficiency
    _, _, Zxx = jax.scipy.signal.stft(
        audio, 
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
        axis=-1
    )
    
    return Zxx


def spectral_convergence_loss(
        pred_audio: jax.Array,
        target_audio: jax.Array,
        n_ffts: List[int] = [512, 1024, 2048],
        hop_lengths: Optional[List[int]] = None,
        mask: Optional[jax.Array] = None
) -> jax.Array:
    """Multi-resolution spectral convergence loss.

    Args:
        pred_audio: Predicted audio [B, T, 1]
        target_audio: Target audio [B, T, 1]
        n_ffts: List of FFT sizes
        hop_lengths: List of hop lengths (defaults to n_fft // 4)
        mask: Optional mask for valid positions

    Returns:
        Spectral convergence loss
    """
    if hop_lengths is None:
        hop_lengths = [n // 4 for n in n_ffts]

    total_loss = 0.0

    for n_fft, hop_length in zip(n_ffts, hop_lengths):
        # Compute magnitude spectrograms
        pred_spec = jnp.abs(stft(pred_audio, n_fft, hop_length))
        target_spec = jnp.abs(stft(target_audio, n_fft, hop_length))

        # Spectral convergence
        loss = jnp.linalg.norm(target_spec - pred_spec, axis=(1, 2))
        loss = loss / (jnp.linalg.norm(target_spec, axis=(1, 2)) + 1e-8)

        if mask is not None:
            # Create time-domain mask for this resolution
            downsample_factor = hop_length
            T_spec = pred_spec.shape[-1]

            # Simple downsampling of mask
            if mask.ndim == 4:  # [B, 1, 1, T]
                mask_1d = mask[:, 0, 0, :]
            else:
                mask_1d = mask

            # Take every hop_length-th position
            indices = jnp.arange(T_spec) * hop_length
            indices = jnp.minimum(indices, mask_1d.shape[-1] - 1)
            spec_mask = mask_1d[:, indices]

            # Apply mask
            valid_frames = jnp.sum(spec_mask, axis=1)
            loss = jnp.sum(loss * spec_mask.any(axis=1)) / (valid_frames.sum() + 1e-8)
        else:
            loss = jnp.mean(loss)

        total_loss += loss

    return total_loss / len(n_ffts)


def log_magnitude_loss(
        pred_audio: jax.Array,
        target_audio: jax.Array,
        n_ffts: List[int] = [512, 1024, 2048],
        hop_lengths: Optional[List[int]] = None,
        mask: Optional[jax.Array] = None
) -> jax.Array:
    """Multi-resolution log magnitude loss.

    Args:
        pred_audio: Predicted audio [B, T, 1]
        target_audio: Target audio [B, T, 1]
        n_ffts: List of FFT sizes
        hop_lengths: List of hop lengths
        mask: Optional mask for valid positions

    Returns:
        Log magnitude loss
    """
    if hop_lengths is None:
        hop_lengths = [n // 4 for n in n_ffts]

    total_loss = 0.0

    for n_fft, hop_length in zip(n_ffts, hop_lengths):
        # Compute magnitude spectrograms
        pred_spec = jnp.abs(stft(pred_audio, n_fft, hop_length)) + 1e-8
        target_spec = jnp.abs(stft(target_audio, n_fft, hop_length)) + 1e-8

        # Log magnitude loss
        loss = jnp.abs(jnp.log(target_spec) - jnp.log(pred_spec))

        if mask is not None:
            # Downsample mask for spectral domain
            T_spec = pred_spec.shape[-1]
            if mask.ndim == 4:
                mask_1d = mask[:, 0, 0, :]
            else:
                mask_1d = mask

            indices = jnp.arange(T_spec) * hop_length
            indices = jnp.minimum(indices, mask_1d.shape[-1] - 1)
            spec_mask = mask_1d[:, indices]

            # Expand mask for frequency dimension
            spec_mask = spec_mask[:, None, :]  # [B, 1, T']

            loss = apply_mask_to_loss(loss, spec_mask, reduction="mean")
        else:
            loss = jnp.mean(loss)

        total_loss += loss

    return total_loss / len(n_ffts)


# ============================================================================
# Quantizer-specific Losses
# ============================================================================

def phoneme_commitment_loss(
        z: jax.Array,
        z_q: jax.Array,
        mask: Optional[jax.Array] = None,
        beta: float = 0.1
) -> jax.Array:
    """Commitment loss for VQ layer (phoneme).

    Since the VQ layer focuses on phoneme classification rather than
    reconstruction, we use a smaller beta to allow more flexibility.

    Args:
        z: Encoder output before quantization [B, T, D]
        z_q: Quantized output from VQ [B, T, D]
        mask: Optional mask [B, 1, 1, T]
        beta: Commitment loss weight (default small for classification task)

    Returns:
        Commitment loss
    """
    commitment = l2_loss(z, jax.lax.stop_gradient(z_q), mask=mask)
    return beta * commitment


def bsq_commitment_loss(
        residual: jax.Array,
        residual_q: jax.Array,
        mask: Optional[jax.Array] = None,
        gamma: float = 1.0
) -> jax.Array:
    """Commitment loss for BSQ layer (acoustic features).

    The BSQ layer is responsible for reconstruction, so we use
    a stronger commitment to ensure good acoustic modeling.

    Args:
        residual: Residual before BSQ [B, T, D]
        residual_q: Quantized residual from BSQ [B, T, D]
        mask: Optional mask [B, 1, 1, T]
        gamma: Commitment loss weight

    Returns:
        BSQ commitment loss
    """
    commitment = l2_loss(residual, jax.lax.stop_gradient(residual_q), mask=mask)
    return gamma * commitment


def ctc_loss(
        logits: jax.Array,
        phoneme_targets: jax.Array,
        logit_lengths: jax.Array,
        target_lengths: jax.Array,
        blank_id: int = 0
) -> jax.Array:
    """CTC loss for phoneme prediction from VQ layer.

    Args:
        logits: Log probabilities from VQ [B, T, vocab_size]
        phoneme_targets: Target phoneme indices [B, max_target_len]
        logit_lengths: Valid lengths for logits [B]
        target_lengths: Valid lengths for targets [B]
        blank_id: Index for CTC blank token (0)

    Returns:
        CTC loss value
    """
    # Import optax for CTC loss
    import optax

    # Compute CTC loss using optax
    loss = optax.ctc_loss(
        logits=logits,
        logit_paddings=_lengths_to_paddings(logit_lengths, logits.shape[1]),
        labels=phoneme_targets,
        label_paddings=_lengths_to_paddings(target_lengths, phoneme_targets.shape[1]),
        blank_id=blank_id
    )

    # Average over batch
    return jnp.mean(loss)


def _lengths_to_paddings(lengths: jax.Array, max_length: int) -> jax.Array:
    """Convert lengths to padding mask for CTC loss.

    Args:
        lengths: Array of valid lengths [B]
        max_length: Maximum sequence length

    Returns:
        Padding mask where 1.0 = padded, 0.0 = valid [B, max_length]
    """
    positions = jnp.arange(max_length)[None, :]  # [1, T]
    valid_mask = positions < lengths[:, None]  # [B, T]
    return 1.0 - valid_mask.astype(jnp.float32)  # Invert for padding


def phoneme_ctc_loss(
        encoder_output: jax.Array,
        phoneme_codebook: jax.Array,
        phoneme_targets: jax.Array,
        encoder_lengths: jax.Array,
        target_lengths: jax.Array,
        temperature: float = 1.0,
        blank_id: int = 0
) -> Tuple[jax.Array, jax.Array]:
    """Compute CTC loss for phoneme prediction using VQ similarity scores.

    This computes the similarity between encoder outputs and VQ codebook entries,
    then applies CTC loss for phoneme sequence prediction.

    Args:
        encoder_output: Encoder outputs [B, T, D]
        phoneme_codebook: VQ codebook embeddings [vocab_size, D]
        phoneme_targets: Target phoneme sequences (IPA indices) [B, max_target_len]
        encoder_lengths: Valid lengths for encoder outputs [B]
        target_lengths: Valid lengths for phoneme targets [B]
        temperature: Temperature for softmax over codebook similarities
        blank_id: Index for CTC blank token (usually 0)

    Returns:
        CTC loss value and predicted phoneme indices
    """
    batch_size, seq_len, hidden_dim = encoder_output.shape
    vocab_size = phoneme_codebook.shape[0]

    # Compute distances to all codebook entries
    # encoder_output: [B, T, 1, D]
    # phoneme_codebook: [1, 1, vocab_size, D]
    encoder_expanded = encoder_output[:, :, None, :]
    codebook_expanded = phoneme_codebook[None, None, :, :]

    # L2 distances
    distances = jnp.sum(jnp.square(encoder_expanded - codebook_expanded), axis=-1)  # [B, T, vocab_size]

    # Convert distances to log probabilities (closer = higher probability)
    logits = -distances / temperature  # [B, T, vocab_size]
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Get predicted indices (for inference)
    predicted_indices = jnp.argmin(distances, axis=-1)  # [B, T]

    # Compute CTC loss
    loss = ctc_loss(
        logits=log_probs,
        phoneme_targets=phoneme_targets,
        logit_lengths=encoder_lengths,
        target_lengths=target_lengths,
        blank_id=blank_id
    )

    return loss, predicted_indices


def prepare_phoneme_targets(
        texts: List[str],
        g2p_function: Callable[[str], List[str]],
        phoneme_vocab: Dict[str, int],
        max_length: Optional[int] = None,
        pad_id: int = 0
) -> Tuple[jax.Array, jax.Array]:
    """Convert text to phoneme sequences for CTC loss.

    Args:
        texts: List of text strings
        g2p_function: Grapheme-to-phoneme conversion function
        phoneme_vocab: Dictionary mapping phoneme symbols to indices
        max_length: Maximum sequence length (auto-computed if None)
        pad_id: Padding index (usually same as blank_id)

    Returns:
        Padded phoneme sequences [B, max_length] and lengths [B]
    """
    phoneme_sequences = []
    lengths = []

    for text in texts:
        # Convert text to phonemes
        phonemes = g2p_function(text)

        # Convert phonemes to indices
        indices = [phoneme_vocab.get(p, phoneme_vocab.get('UNK', 0)) for p in phonemes]

        phoneme_sequences.append(indices)
        lengths.append(len(indices))

    # Pad sequences
    if max_length is None:
        max_length = max(lengths)

    padded_sequences = []
    for seq in phoneme_sequences:
        if len(seq) < max_length:
            # Right-pad for CTC
            seq = seq + [pad_id] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        padded_sequences.append(seq)

    return jnp.array(padded_sequences), jnp.array(lengths)


def phoneme_error_rate(
        predicted_indices: jax.Array,
        target_indices: jax.Array,
        predicted_lengths: jax.Array,
        target_lengths: jax.Array,
        blank_id: int = 0
) -> jax.Array:
    """Compute phoneme error rate for monitoring.

    Args:
        predicted_indices: Predicted phoneme indices [B, T]
        target_indices: Target phoneme indices [B, max_target_len]
        predicted_lengths: Valid lengths for predictions [B]
        target_lengths: Valid lengths for targets [B]
        blank_id: CTC blank token ID

    Returns:
        Average phoneme error rate
    """
    batch_size = predicted_indices.shape[0]
    total_errors = 0.0
    total_length = 0.0

    for i in range(batch_size):
        # Extract valid sequences
        pred_len = predicted_lengths[i]
        target_len = target_lengths[i]

        pred_seq = predicted_indices[i, :pred_len]
        target_seq = target_indices[i, :target_len]

        # Remove CTC blanks and consecutive duplicates from predictions
        pred_seq_clean = remove_ctc_blanks(pred_seq, blank_id)

        # Compute edit distance
        errors = edit_distance(pred_seq_clean, target_seq)

        total_errors += errors
        total_length += target_len

    return total_errors / (total_length + 1e-8)


def remove_ctc_blanks(sequence: jax.Array, blank_id: int) -> jax.Array:
    """Remove CTC blanks and merge consecutive duplicates.

    Args:
        sequence: Sequence with potential blanks and duplicates
        blank_id: ID of blank token

    Returns:
        Cleaned sequence
    """
    # Remove blanks
    non_blank_mask = sequence != blank_id
    sequence_no_blanks = sequence[non_blank_mask]

    # Remove consecutive duplicates
    if len(sequence_no_blanks) == 0:
        return sequence_no_blanks

    # Check where values change
    changes = jnp.concatenate([
        jnp.array([True]),
        sequence_no_blanks[1:] != sequence_no_blanks[:-1]
    ])

    return sequence_no_blanks[changes]


def edit_distance(seq1: jax.Array, seq2: jax.Array) -> float:
    """Compute edit distance between two sequences.

    Simple implementation for monitoring purposes.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Edit distance
    """
    len1, len2 = len(seq1), len(seq2)

    # Create distance matrix
    dp = jnp.zeros((len1 + 1, len2 + 1))

    # Initialize base cases
    dp = dp.at[:, 0].set(jnp.arange(len1 + 1))
    dp = dp.at[0, :].set(jnp.arange(len2 + 1))

    # Fill matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp = dp.at[i, j].set(dp[i - 1, j - 1])
            else:
                dp = dp.at[i, j].set(1 + jnp.minimum(
                    dp[i - 1, j],  # Deletion
                    jnp.minimum(dp[i, j - 1], dp[i - 1, j - 1])  # Insertion, Substitution
                ))

    return dp[len1, len2]


# ============================================================================
# Phoneme Vocabulary Utilities
# ============================================================================

def create_phoneme_vocabulary(
        ipa_symbols: Optional[List[str]] = None,
        blank_token: str = "_",
        unk_token: str = "UNK",
        pad_token: str = "PAD"
) -> Dict[str, int]:
    """Create phoneme vocabulary mapping symbols to indices.

    Args:
        ipa_symbols: List of IPA phoneme symbols (uses common set if None)
        blank_token: CTC blank token
        unk_token: Unknown phoneme token
        pad_token: Padding token

    Returns:
        Dictionary mapping phoneme symbols to indices
    """
    if ipa_symbols is None:
        # Common IPA symbols for English (can be extended for multilingual)
        ipa_symbols = [
            # Vowels
            'i', 'ɪ', 'e', 'ɛ', 'æ', 'a', 'ɑ', 'ɔ', 'o', 'ʊ', 'u',
            'ə', 'ɚ', 'ɝ', 'ʌ',
            # Diphthongs
            'eɪ', 'aɪ', 'ɔɪ', 'aʊ', 'oʊ',
            # Consonants
            'p', 'b', 't', 'd', 'k', 'g', 'ʔ',
            'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h',
            'm', 'n', 'ŋ',
            'l', 'ɹ', 'j', 'w',
            'tʃ', 'dʒ',
            # Stress and syllable markers
            'ˈ', 'ˌ', '.',
            # Additional symbols for other languages can be added
        ]

    # Build vocabulary with special tokens first
    vocab = {
        blank_token: 0,  # CTC blank must be at index 0
        unk_token: 1,
        pad_token: 2,
    }

    # Add IPA symbols
    for i, symbol in enumerate(ipa_symbols):
        vocab[symbol] = i + 3

    return vocab


def adversarial_g_loss(
        disc_outputs: List[jax.Array],
        loss_type: str = "hinge"
) -> jax.Array:
    """Generator adversarial loss.

    Args:
        disc_outputs: List of discriminator outputs for generated audio
        loss_type: "hinge" or "lsgan"

    Returns:
        Generator adversarial loss
    """
    total_loss = 0.0

    for output in disc_outputs:
        if loss_type == "hinge":
            loss = -jnp.mean(output)
        elif loss_type == "lsgan":
            loss = jnp.mean(jnp.square(output - 1))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        total_loss += loss

    return total_loss / len(disc_outputs)


def adversarial_d_loss(
        real_outputs: List[jax.Array],
        fake_outputs: List[jax.Array],
        loss_type: str = "hinge"
) -> jax.Array:
    """Discriminator adversarial loss.

    Args:
        real_outputs: List of discriminator outputs for real audio
        fake_outputs: List of discriminator outputs for generated audio
        loss_type: "hinge" or "lsgan"

    Returns:
        Discriminator adversarial loss
    """
    total_loss = 0.0

    for real_out, fake_out in zip(real_outputs, fake_outputs):
        if loss_type == "hinge":
            real_loss = jnp.mean(jax.nn.relu(1 - real_out))
            fake_loss = jnp.mean(jax.nn.relu(1 + fake_out))
        elif loss_type == "lsgan":
            real_loss = jnp.mean(jnp.square(real_out - 1))
            fake_loss = jnp.mean(jnp.square(fake_out))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        total_loss += real_loss + fake_loss

    return total_loss / len(real_outputs)


def feature_matching_loss(
        real_features: List[List[jax.Array]],
        fake_features: List[List[jax.Array]],
        mask: Optional[jax.Array] = None
) -> jax.Array:
    """Feature matching loss for better training stability.

    Args:
        real_features: List of feature lists from discriminators (real)
        fake_features: List of feature lists from discriminators (fake)
        mask: Optional mask for valid positions

    Returns:
        Feature matching loss
    """
    total_loss = 0.0

    for real_feats, fake_feats in zip(real_features, fake_features):
        for real_f, fake_f in zip(real_feats, fake_feats):
            loss = l1_loss(
                fake_f,
                jax.lax.stop_gradient(real_f),
                mask=mask if mask is not None and real_f.ndim == 3 else None
            )
            total_loss += loss

    # Normalize by total number of features
    num_features = sum(len(f) for f in real_features)
    return total_loss / num_features


# ============================================================================
# Combined Loss Functions
# ============================================================================

def compute_generator_loss(
        pred_audio: jax.Array,
        target_audio: jax.Array,
        encoder_output: jax.Array,
        phoneme_quantized: jax.Array,
        residual: jax.Array,
        residual_quantized: jax.Array,
        phoneme_indices: jax.Array,
        phoneme_targets: jax.Array,
        phoneme_codebook: jax.Array,
        encoder_lengths: jax.Array,
        target_lengths: jax.Array,
        disc_outputs_fake: List[jax.Array],
        disc_features_real: List[List[jax.Array]],
        disc_features_fake: List[List[jax.Array]],
        mask: Optional[jax.Array] = None,
        encoder_mask: Optional[jax.Array] = None,
        config: Optional[Dict] = None
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute all generator losses.

    Args:
        pred_audio: Reconstructed audio [B, T, 1]
        target_audio: Original audio [B, T, 1]
        encoder_output: Encoder output before quantization [B, T', D]
        phoneme_quantized: VQ output [B, T', D]
        residual: Residual before BSQ [B, T', D]
        residual_quantized: BSQ output [B, T', D]
        phoneme_indices: Phoneme indices [B, T'] (for monitoring)
        phoneme_targets: Target phoneme sequences [B, max_target_len]
        phoneme_codebook: VQ codebook embeddings [vocab_size, D]
        encoder_lengths: Valid lengths for encoder outputs [B]
        target_lengths: Valid lengths for phoneme targets [B]
        disc_outputs_fake: Discriminator outputs for fake audio
        disc_features_real: Discriminator features for real audio
        disc_features_fake: Discriminator features for fake audio
        mask: Optional mask for audio-level losses [B, 1, 1, T]
        encoder_mask: Optional mask for encoder-level losses [B, 1, 1, T']
        config: Loss configuration weights

    Returns:
        Total loss and dictionary of individual losses
    """
    if config is None:
        config = {
            'l1_weight': 1.0,
            'l2_weight': 1.0,
            'spectral_weight': 2.0,
            'log_mag_weight': 1.0,
            'ctc_weight': 10.0,  # High weight for phoneme transcription task
            'phoneme_commit_weight': 0.1,  # Lower commitment for VQ flexibility
            'bsq_commit_weight': 1.0,
            'adversarial_weight': 1.0,
            'feature_match_weight': 10.0,
            'ctc_temperature': 1.0,  # Temperature for CTC softmax
        }

    losses = {}

    # Reconstruction losses
    losses['l1'] = l1_loss(pred_audio, target_audio, mask=mask)
    losses['l2'] = l2_loss(pred_audio, target_audio, mask=mask)
    losses['spectral_convergence'] = spectral_convergence_loss(
        pred_audio, target_audio, mask=mask
    )
    losses['log_magnitude'] = log_magnitude_loss(
        pred_audio, target_audio, mask=mask
    )

    # Use provided encoder_mask or compute it from audio mask if not provided
    if encoder_mask is None and mask is not None:
        # Fallback: compute encoder mask from audio mask
        # This should ideally be avoided by always providing encoder_mask
        downsample_factor = target_audio.shape[1] // encoder_output.shape[1]
        if mask.ndim == 4:
            mask_1d = mask[:, 0, 0, :]
        else:
            mask_1d = mask

        # Simple downsampling
        T_enc = encoder_output.shape[1]
        indices = jnp.arange(T_enc) * downsample_factor
        indices = jnp.minimum(indices, mask_1d.shape[-1] - 1)
        encoder_mask = mask_1d[:, indices]
        encoder_mask = encoder_mask[:, None, None, :]  # [B, 1, 1, T']

    # CTC loss for phoneme prediction
    ctc_loss_value, _ = phoneme_ctc_loss(
        encoder_output=encoder_output,
        phoneme_codebook=phoneme_codebook,
        phoneme_targets=phoneme_targets,
        encoder_lengths=encoder_lengths,
        target_lengths=target_lengths,
        temperature=config.get('ctc_temperature', 1.0),
        blank_id=0
    )
    losses['ctc'] = ctc_loss_value

    # Quantizer commitment losses
    losses['phoneme_commitment'] = phoneme_commitment_loss(
        encoder_output, phoneme_quantized, mask=encoder_mask, beta=0.1
    )
    losses['bsq_commitment'] = bsq_commitment_loss(
        residual, residual_quantized, mask=encoder_mask
    )

    # Adversarial losses
    losses['adversarial_g'] = adversarial_g_loss(disc_outputs_fake)
    losses['feature_matching'] = feature_matching_loss(
        disc_features_real, disc_features_fake
    )

    # Combine all losses
    total_loss = (
            config['l1_weight'] * losses['l1'] +
            config['l2_weight'] * losses['l2'] +
            config['spectral_weight'] * losses['spectral_convergence'] +
            config['log_mag_weight'] * losses['log_magnitude'] +
            config['ctc_weight'] * losses['ctc'] +
            config['phoneme_commit_weight'] * losses['phoneme_commitment'] +
            config['bsq_commit_weight'] * losses['bsq_commitment'] +
            config['adversarial_weight'] * losses['adversarial_g'] +
            config['feature_match_weight'] * losses['feature_matching']
    )

    losses['total'] = total_loss
    return total_loss, losses


def compute_discriminator_loss(
        disc_outputs_real: List[jax.Array],
        disc_outputs_fake: List[jax.Array],
        loss_type: str = "hinge"
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute discriminator loss.

    Args:
        disc_outputs_real: Discriminator outputs for real audio
        disc_outputs_fake: Discriminator outputs for fake audio
        loss_type: Type of adversarial loss

    Returns:
        Total loss and loss dictionary
    """
    d_loss = adversarial_d_loss(disc_outputs_real, disc_outputs_fake, loss_type)

    losses = {
        'adversarial_d': d_loss,
        'total': d_loss
    }

    return d_loss, losses



def compute_phoneme_metrics(
        encoder_output: jax.Array,
        phoneme_codebook: jax.Array,
        phoneme_targets: jax.Array,
        encoder_lengths: jax.Array,
        target_lengths: jax.Array,
        temperature: float = 1.0,
        blank_id: int = 0
) -> Dict[str, jax.Array]:
    """Compute phoneme-related metrics for monitoring.

    Args:
        encoder_output: Encoder outputs [B, T, D]
        phoneme_codebook: VQ codebook embeddings [vocab_size, D]
        phoneme_targets: Target phoneme sequences [B, max_target_len]
        encoder_lengths: Valid lengths for encoder outputs [B]
        target_lengths: Valid lengths for phoneme targets [B]
        temperature: Temperature for softmax
        blank_id: CTC blank token ID

    Returns:
        Dictionary with 'ctc_loss' and 'per' (phoneme error rate)
    """
    # Get CTC loss and predictions
    ctc_loss_value, predicted_indices = phoneme_ctc_loss(
        encoder_output=encoder_output,
        phoneme_codebook=phoneme_codebook,
        phoneme_targets=phoneme_targets,
        encoder_lengths=encoder_lengths,
        target_lengths=target_lengths,
        temperature=temperature,
        blank_id=blank_id
    )

    # Compute phoneme error rate
    per = phoneme_error_rate(
        predicted_indices=predicted_indices,
        target_indices=phoneme_targets,
        predicted_lengths=encoder_lengths,
        target_lengths=target_lengths,
        blank_id=blank_id
    )

    return {
        'ctc_loss': ctc_loss_value,
        'per': per,
        'predicted_phonemes': predicted_indices
    }


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of how to use these loss functions with G2P."""
    # Example with phonemizer or g2p_en
    try:
        from phonemizer import phonemize

        def g2p_function(text):
            # Use phonemizer for IPA conversion
            phonemes = phonemize(
                text,
                language='en-us',
                backend='espeak',
                strip=True,
                preserve_punctuation=False,
            )
            # Split into individual phonemes
            return list(phonemes.replace(' ', ''))
    except ImportError:
        # Fallback to g2p_en
        from g2p_en import G2p
        g2p = G2p()

        def g2p_function(text):
            return g2p(text)

    # Create phoneme vocabulary
    phoneme_vocab = create_phoneme_vocabulary()

    # Example texts
    texts = ["Hello world", "How are you"]

    # Convert to phoneme targets
    phoneme_targets, target_lengths = prepare_phoneme_targets(
        texts=texts,
        g2p_function=g2p_function,
        phoneme_vocab=phoneme_vocab
    )

    print(f"Phoneme vocabulary size: {len(phoneme_vocab)}")
    print(f"Phoneme targets shape: {phoneme_targets.shape}")
    print(f"Target lengths: {target_lengths}")

    return phoneme_vocab, phoneme_targets, target_lengths