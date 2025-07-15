import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional
import functools


def _compute_stft(
    signal: jnp.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int] = None,
    window: str = 'hann'
) -> jnp.ndarray:
    """Compute STFT of a signal.
    
    Args:
        signal: Input signal [B, T] or [B, T, 1]
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length (defaults to n_fft)
        window: Window type
    
    Returns:
        Complex STFT [B, F, T_frames]
    """
    if signal.ndim == 3:
        signal = signal.squeeze(-1)
    
    if win_length is None:
        win_length = n_fft
    
    # Create window
    if window == 'hann':
        window_fn = jnp.hanning(win_length)
    elif window == 'hamming':
        window_fn = jnp.hamming(win_length)
    else:
        raise ValueError(f"Unknown window type: {window}")
    
    # Pad window to n_fft if necessary
    if win_length < n_fft:
        pad = (n_fft - win_length) // 2
        window_fn = jnp.pad(window_fn, (pad, pad))
    
    batch_size = signal.shape[0]
    signal_length = signal.shape[1]
    
    # Calculate number of frames
    n_frames = 1 + (signal_length - n_fft) // hop_length
    
    # Create frame indices for all batches
    frame_starts = jnp.arange(n_frames) * hop_length
    frame_indices = frame_starts[:, None] + jnp.arange(n_fft)
    
    # Extract frames for all batches
    frames = signal[:, frame_indices] * window_fn  # [B, n_frames, n_fft]
    
    # Compute FFT
    stft = jnp.fft.rfft(frames, n=n_fft, axis=-1)  # [B, n_frames, F]
    
    return stft.transpose(0, 2, 1)  # [B, F, T_frames]


def multi_scale_spectrogram_loss(
    real: jnp.ndarray,
    generated: jnp.ndarray,
    fft_sizes: List[int] = [2048, 1024, 512, 256, 128],
    hop_lengths: Optional[List[int]] = None,
    win_lengths: Optional[List[int]] = None,
    loss_type: str = 'l1',
    mag_weight: float = 1.0,
    log_weight: float = 1.0
) -> Tuple[jnp.ndarray, dict]:
    """Multi-scale spectrogram loss as used in DAC.
    
    Computes L1/L2 loss on both linear and log-magnitude spectrograms
    at multiple scales to capture different time-frequency resolutions.
    
    Args:
        real: Real audio [B, T] or [B, T, 1]
        generated: Generated audio [B, T] or [B, T, 1]
        fft_sizes: List of FFT sizes for multi-scale analysis
        hop_lengths: List of hop lengths (defaults to fft_size // 4)
        win_lengths: List of window lengths (defaults to fft_size)
        loss_type: 'l1' or 'l2'
        mag_weight: Weight for linear magnitude loss
        log_weight: Weight for log magnitude loss
    
    Returns:
        total_loss: Scalar loss value
        loss_dict: Dictionary with individual loss components
    """
    if hop_lengths is None:
        hop_lengths = [n // 4 for n in fft_sizes]
    if win_lengths is None:
        win_lengths = fft_sizes
    
    assert len(fft_sizes) == len(hop_lengths) == len(win_lengths)
    
    loss_fn = jnp.abs if loss_type == 'l1' else lambda x: x ** 2
    
    mag_losses = []
    log_mag_losses = []
    
    for n_fft, hop_length, win_length in zip(fft_sizes, hop_lengths, win_lengths):
        # Compute STFTs
        stft_real = _compute_stft(real, n_fft, hop_length, win_length)
        stft_gen = _compute_stft(generated, n_fft, hop_length, win_length)
        
        # Magnitude spectrograms
        mag_real = jnp.abs(stft_real)
        mag_gen = jnp.abs(stft_gen)
        
        # Linear magnitude loss
        mag_loss = jnp.mean(loss_fn(mag_real - mag_gen))
        mag_losses.append(mag_loss)
        
        # Log magnitude loss (with stability epsilon)
        log_mag_real = jnp.log(mag_real + 1e-7)
        log_mag_gen = jnp.log(mag_gen + 1e-7)
        log_mag_loss = jnp.mean(loss_fn(log_mag_real - log_mag_gen))
        log_mag_losses.append(log_mag_loss)
    
    # Average across scales
    avg_mag_loss = jnp.mean(jnp.array(mag_losses))
    avg_log_mag_loss = jnp.mean(jnp.array(log_mag_losses))
    
    # Total loss
    total_loss = mag_weight * avg_mag_loss + log_weight * avg_log_mag_loss
    
    loss_dict = {
        'mag_loss': avg_mag_loss,
        'log_mag_loss': avg_log_mag_loss,
        'total_spectrogram_loss': total_loss
    }
    
    return total_loss, loss_dict


def time_domain_loss(
    real: jnp.ndarray,
    generated: jnp.ndarray,
    loss_type: str = 'l1'
) -> jnp.ndarray:
    """Simple time-domain reconstruction loss.
    
    Args:
        real: Real audio [B, T] or [B, T, 1]
        generated: Generated audio [B, T] or [B, T, 1]
        loss_type: 'l1' or 'l2'
    
    Returns:
        Scalar loss value
    """
    if loss_type == 'l1':
        return jnp.mean(jnp.abs(real - generated))
    elif loss_type == 'l2':
        return jnp.mean((real - generated) ** 2)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def perceptual_stft_loss(
    real: jnp.ndarray,
    generated: jnp.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    perceptual_weighting: bool = True
) -> jnp.ndarray:
    """Perceptual STFT loss with optional frequency weighting.
    
    Args:
        real: Real audio [B, T] or [B, T, 1]
        generated: Generated audio [B, T] or [B, T, 1]
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        perceptual_weighting: Apply A-weighting curve approximation
    
    Returns:
        Scalar loss value
    """
    # Compute STFTs
    stft_real = _compute_stft(real, n_fft, hop_length, win_length)
    stft_gen = _compute_stft(generated, n_fft, hop_length, win_length)
    
    # Magnitude and phase
    mag_real = jnp.abs(stft_real)
    mag_gen = jnp.abs(stft_gen)
    phase_real = jnp.angle(stft_real)
    phase_gen = jnp.angle(stft_gen)
    
    # Magnitude loss
    mag_loss = jnp.mean(jnp.abs(mag_real - mag_gen))
    
    # Phase loss (wrapped)
    phase_diff = phase_real - phase_gen
    phase_diff = jnp.angle(jnp.exp(1j * phase_diff))  # Wrap to [-pi, pi]
    phase_loss = jnp.mean(jnp.abs(phase_diff))
    
    # Optional perceptual weighting (simplified A-weighting)
    if perceptual_weighting:
        freqs = jnp.linspace(0, 22050, n_fft // 2 + 1)  # Assuming 44.1kHz
        # Simple approximation of A-weighting curve
        weight = (freqs / 1000) ** 2 / ((freqs / 1000) ** 2 + 1)
        weight = weight[None, :, None]  # [1, F, 1]
        mag_loss = jnp.mean(weight * jnp.abs(mag_real - mag_gen))
    
    return mag_loss + 0.1 * phase_loss


def mel_spectrogram_loss(
    real: jnp.ndarray,
    generated: jnp.ndarray,
    sample_rate: int = 24000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> jnp.ndarray:
    """Mel-spectrogram loss for perceptually weighted reconstruction.
    
    Args:
        real: Real audio [B, T] or [B, T, 1]
        generated: Generated audio [B, T] or [B, T, 1]
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop length
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency (defaults to sample_rate / 2)
    
    Returns:
        Scalar loss value
    """
    from tokenizer.utils.mel import MelSpectrogramJAX
    
    # Create mel spectrogram computer
    mel_computer = MelSpectrogramJAX(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax or sample_rate / 2,
        power=1.0  # Use magnitude
    )
    
    # Ensure inputs are 2D
    if real.ndim == 3:
        real = real.squeeze(-1)
    if generated.ndim == 3:
        generated = generated.squeeze(-1)
    
    # Compute mel spectrograms for each sample in batch
    mel_real = jax.vmap(mel_computer)(real)
    mel_gen = jax.vmap(mel_computer)(generated)
    
    # L1 loss on mel spectrograms
    return jnp.mean(jnp.abs(mel_real - mel_gen))