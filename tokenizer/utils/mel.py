import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Literal
import functools


class MelSpectrogramJAX:
    """JAX-based Mel Spectrogram computation with JIT and parallel processing support."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int | None = None,
        window: Literal["hann", "hamming", "blackman", "bartlett", "kaiser"] = "hann",
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: float | None = None,
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: Literal["slaney"] | None = None,
        htk: bool = False,
    ):
        """
        Initialize Mel Spectrogram parameters.

        Args:
            sample_rate: Sample rate of audio signal
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            win_length: Window length (defaults to n_fft)
            window: Window function type
            n_mels: Number of Mel bands
            fmin: Lowest frequency (Hz)
            fmax: Highest frequency (Hz) (defaults to sample_rate/2)
            power: Exponent for magnitude (1.0 for amplitude, 2.0 for power)
            center: Whether to center the signal
            pad_mode: Padding mode for edge treatment
            norm: Mel filter normalization ('slaney' or None)
            htk: Use HTK formula for mel scale
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2
        self.power = power
        self.center = center
        self.pad_mode = pad_mode
        self.norm = norm
        self.htk = htk

        # Pre-compute window and mel filterbank
        self.window_func = self._get_window()
        self.mel_basis = self._create_mel_filterbank()

    def _get_window(self) -> jax.Array:
        """Generate window function."""
        if self.window == "hann":
            return jnp.hanning(self.win_length)
        elif self.window == "hamming":
            return jnp.hamming(self.win_length)
        elif self.window == "blackman":
            return jnp.blackman(self.win_length)
        elif self.window == "bartlett":
            return jnp.bartlett(self.win_length)
        elif self.window == "kaiser":
            return jnp.kaiser(self.win_length, 8.0)
        else:
            raise ValueError(f"Unknown window type: {self.window}")

    @functools.partial(jit, static_argnums=(0,))
    def hz_to_mel(self, frequencies: jax.Array) -> jax.Array:
        """Convert Hz to Mel scale."""
        if self.htk:
            return 2595.0 * jnp.log10(1.0 + frequencies / 700.0)
        else:
            # Slaney's formula
            f_min = 0.0
            f_sp = 200.0 / 3
            min_log_hz = 1000.0
            min_log_mel = (min_log_hz - f_min) / f_sp
            logstep = jnp.log(6.4) / 27.0

            mels = jnp.where(
                frequencies < min_log_hz,
                (frequencies - f_min) / f_sp,
                min_log_mel + jnp.log(frequencies / min_log_hz) / logstep,
            )
            return mels

    @functools.partial(jit, static_argnums=(0,))
    def mel_to_hz(self, mels: jax.Array) -> jax.Array:
        """Convert Mel scale to Hz."""
        if self.htk:
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        else:
            # Slaney's formula
            f_min = 0.0
            f_sp = 200.0 / 3
            min_log_hz = 1000.0
            min_log_mel = (min_log_hz - f_min) / f_sp
            logstep = jnp.log(6.4) / 27.0

            freqs = jnp.where(
                mels < min_log_mel,
                f_min + f_sp * mels,
                min_log_hz * jnp.exp(logstep * (mels - min_log_mel)),
            )
            return freqs

    def _create_mel_filterbank(self) -> jax.Array:
        """Create Mel filterbank matrix."""
        # Frequency bins
        fft_freqs = jnp.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)

        # Mel points
        mel_min = self.hz_to_mel(jnp.array(self.fmin))
        mel_max = self.hz_to_mel(jnp.array(self.fmax))
        mel_points = jnp.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)

        # Create filterbank
        mel_basis = jnp.zeros((self.n_mels, self.n_fft // 2 + 1))

        for i in range(self.n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]

            # Rising edge
            rising = (fft_freqs - left) / (center - left)
            rising = jnp.clip(rising, 0, 1)

            # Falling edge
            falling = (right - fft_freqs) / (right - center)
            falling = jnp.clip(falling, 0, 1)

            # Combine
            mel_basis = mel_basis.at[i].set(jnp.minimum(rising, falling))

        # Normalize
        if self.norm == "slaney":
            enorm = 2.0 / (hz_points[2 : self.n_mels + 2] - hz_points[: self.n_mels])
            mel_basis = mel_basis * enorm[:, None]

        return mel_basis

    @functools.partial(jit, static_argnums=(0,))
    def _pad_signal(self, signal: jax.Array) -> jax.Array:
        """Pad signal for centered STFT."""
        if self.center:
            pad_length = self.n_fft // 2
            if self.pad_mode == "reflect":
                return jnp.pad(signal, pad_length, mode="reflect")
            elif self.pad_mode == "constant":
                return jnp.pad(signal, pad_length, mode="constant")
            else:
                return jnp.pad(signal, pad_length, mode=self.pad_mode)
        return signal

    @functools.partial(jit, static_argnums=(0,))
    def _stft_single_frame(self, frame: jax.Array) -> jax.Array:
        """Compute FFT for a single frame."""
        # Apply window
        windowed = frame[: self.win_length] * self.window_func

        # Pad to n_fft if necessary
        if self.win_length < self.n_fft:
            windowed = jnp.pad(windowed, (0, self.n_fft - self.win_length))

        # Compute FFT
        fft_result = jnp.fft.rfft(windowed)
        return fft_result

    @functools.partial(jit, static_argnums=(0,))
    def stft(self, signal: jax.Array) -> jax.Array:
        """
        Compute Short-Time Fourier Transform.

        Args:
            signal: Input audio signal

        Returns:
            Complex STFT matrix of shape (n_freqs, n_frames)
        """
        # Pad signal if centering
        signal = self._pad_signal(signal)

        # Calculate number of frames
        n_frames = 1 + (len(signal) - self.n_fft) // self.hop_length

        # Create frame indices
        frame_starts = jnp.arange(n_frames) * self.hop_length

        # Extract frames using advanced indexing
        frame_indices = frame_starts[:, None] + jnp.arange(self.n_fft)
        frames = signal[frame_indices]

        # Apply STFT to all frames using vmap
        stft_frames = vmap(self._stft_single_frame)(frames)

        return stft_frames.T

    @functools.partial(jit, static_argnums=(0,))
    def __call__(self, signal: jax.Array) -> jax.Array:
        """
        Compute Mel spectrogram from audio signal.

        Args:
            signal: Input audio signal (1D array)

        Returns:
            Mel spectrogram of shape (n_mels, n_frames)
        """
        # Compute STFT
        stft_matrix = self.stft(signal)

        # Convert to magnitude/power spectrogram
        magnitude = jnp.abs(stft_matrix)
        if self.power != 1.0:
            magnitude = magnitude**self.power

        # Apply mel filterbank
        mel_spec = jnp.dot(self.mel_basis, magnitude)

        return mel_spec

    @functools.partial(jit, static_argnums=(0,))
    def compute_log_mel(
        self, signal: jax.Array, ref: float = 1.0, amin: float = 1e-10
    ) -> jax.Array:
        """
        Compute log-scaled Mel spectrogram.

        Args:
            signal: Input audio signal
            ref: Reference value for dB calculation
            amin: Minimum value for numerical stability

        Returns:
            Log-scaled Mel spectrogram in dB
        """
        mel_spec = self(signal)

        # Convert to dB
        log_spec = 10.0 * jnp.log10(jnp.maximum(amin, mel_spec))
        log_spec -= 10.0 * jnp.log10(jnp.maximum(amin, ref))

        return log_spec
