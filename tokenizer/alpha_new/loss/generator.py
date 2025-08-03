from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.signal import stft
from tokenizer.utils.mel import MelSpectrogramJAX

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
        predictions: Features from predicted(generator output) [B, T, C]
        targets: True feature extracted [B, T, C]
        mask: Mask [B, T] where True = valid

    Returns:
        Scalar L2 loss
    """
    # Expand mask to match audio shape
    mask = jnp.expand_dims(mask, axis=-1)  # [B, T, 1]

    loss = jnp.square(predictions - targets)
    loss = loss * mask
    return jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)

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


def single_scale_stft_loss(predictions, targets, mask, n_fft, hop_length,
                           mag_weight=1.0, log_mag_weight=1.0, power=2.0, eps=1e-5):
    """
    Calculate STFT loss of given audio pair using a single scale.
    L1 loss of raw magnitude of STFT and L1 of log magnitude of STFT is computed
    Weighted as configured and summed.

    Args:
        predictions: Predicted audio from generator [B, T, 1]
        targets: Ground truth audio [B, T, 1]
        mask: Padding mask [B, T]
        n_fft: FFT size (window_length)
        hop_length: Hop length
        mag_weight: raw magnitude L1 loss weight
        log_mag_weight: log magnitude L1 loss weight
        power: log magnitude powered
        eps: 1e-5 no div by zero

    Returns:
        Scalar loss of all batches summed.
    """

    mask_expanded = jnp.expand_dims(mask, axis=-1)
    predictions_masked = predictions * mask_expanded
    targets_masked = targets * mask_expanded

    # remove last dim for stft
    predictions_squeezed = jnp.squeeze(predictions_masked, axis=-1)
    targets_squeezed = jnp.squeeze(targets_masked, axis=-1)

    def _stft_fn(audio):
        # nperseg is window_length, noverlap is window_length - hop_length
        # padded=True, boundary='zeros' ensures padding to work.
        _, _, Zxx = stft(audio,
                         nperseg=n_fft,
                         noverlap=n_fft - hop_length,
                         nfft=n_fft,
                         window='hann',
                         padded=True,
                         boundary='zeros')
        return Zxx

    vmapped_stft = jax.vmap(_stft_fn) # verify if vmap is needed?

    pred_stft = vmapped_stft(predictions_squeezed)  #[B, n_freqs, n_frames]
    target_stft = vmapped_stft(targets_squeezed)

    # calc magnitude
    pred_mag = jnp.abs(pred_stft)
    target_mag = jnp.abs(target_stft)

    total_loss = 0.0

    num_valid_samples = jnp.maximum(jnp.sum(mask), 1.0)
    # 1. raw magnitude L1 loss
    if mag_weight > 0:
        mag_loss = jnp.abs(pred_mag - target_mag)
        total_loss += mag_weight * (jnp.sum(mag_loss) / num_valid_samples)

    # 2. log magnitude L1 loss
    if log_mag_weight > 0:
        pred_log_mag = jnp.log(jnp.power(pred_mag, power) + eps)
        target_log_mag = jnp.log(jnp.power(target_mag, power) + eps)

        log_mag_loss = jnp.abs(pred_log_mag - target_log_mag)
        total_loss += log_mag_weight * (jnp.sum(log_mag_loss) / num_valid_samples)

    return total_loss


@partial(jax.jit, static_argnames=("n_ffts", "hop_length_ratio", "mag_weight", "log_mag_weight", "power"))
def multi_scale_stft_loss(predictions, targets, mask,
                          n_ffts=(2048, 512),
                          hop_length_ratio=0.25,
                          mag_weight=1.0,
                          log_mag_weight=1.0,
                          power=2.0):
    total_loss = 0.0

    # This for loop would get 'unrolled' when JIT compiled as n_ffts are static.
    for n_fft in n_ffts:
        hop_length = int(n_fft * hop_length_ratio)
        loss = single_scale_stft_loss(predictions, targets, mask, n_fft, hop_length,
                                      mag_weight, log_mag_weight, power)
        total_loss += loss

    return total_loss

@partial(jax.jit, static_argnames=("vq_weight", ))
def vq_commitment_loss(
        encoder_output: jax.Array, quantized: jax.Array, mask: jax.Array, vq_weight: float = 0.25
):
    # mask must be encoder mask, not regular padding mask
    loss = l2_loss(quantized, encoder_output, mask) # TODO: check where stop_gradient is needed.
    return vq_weight * loss

@partial(jax.jit, static_argnames=("bsq_weight", ))
def bsq_commitment_loss(
        residual: jax.Array, quantized: jax.Array, mask: jax.Array, bsq_weight: float = 1.0
):
    # mask must be encoder mask, not regular padding mask.
    loss = l2_loss(quantized, residual, mask) # TODO: check where stop_gradient is needed.
    return bsq_weight * loss

def adversarial_g_loss_lsgan(disc_outputs: list[jax.Array]) -> jax.Array:
    total_loss = 0.0
    for output in disc_outputs:
        loss = jnp.mean(jnp.square(output - 1))
        total_loss = total_loss + loss
    return total_loss / len(disc_outputs)

def adversarial_g_loss_hinge(disc_outputs: list[jax.Array]) -> jax.Array:
    total_loss = 0.0
    for output in disc_outputs:
        loss = -jnp.mean(output)
        total_loss = total_loss + loss
    return total_loss / len(disc_outputs)

def feature_matching_loss(
    real_features: list[list[jax.Array]], fake_features: list[list[jax.Array]]
) -> jax.Array:
    total_loss = 0.0
    num_features = 0

    for real_feats, fake_feats in zip(real_features, fake_features, strict=False):
        for real_f, fake_f in zip(real_feats, fake_feats, strict=False):
            loss = jnp.mean(jnp.abs(fake_f - real_f)) # TODO: check where stop_gradient is needed.
            total_loss = total_loss + loss
            num_features = num_features + 1

    return total_loss / jnp.maximum(num_features, 1)

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    subkey, key = jax.random.split(key)

    audio_true = jax.random.normal(key, (4, 505440))
    audio_fake = jax.random.normal(subkey, (4, 505440))
    mask = jnp.ones((4, 505440))

    # test mel spectrogram loss
    mel_loss = mel_spectrogram_loss(audio_fake, audio_true, mask)
    print(f"Mel spectrogram loss: {mel_loss}")
    
    # Test multi-scale STFT loss
    audio_true_3d = jnp.expand_dims(audio_true, axis=-1)  # [B, T, 1]
    audio_fake_3d = jnp.expand_dims(audio_fake, axis=-1)  # [B, T, 1]
    
    stft_loss = multi_scale_stft_loss(audio_fake_3d, audio_true_3d, mask)
    print(f"Multi-scale STFT loss: {stft_loss}")
    
    # Test individual components
    single_loss_2048 = single_scale_stft_loss(audio_fake_3d, audio_true_3d, mask, 2048, 512)
    single_loss_512 = single_scale_stft_loss(audio_fake_3d, audio_true_3d, mask, 512, 128)
    
    print(f"Single scale STFT loss (n_fft=2048): {single_loss_2048}")
    print(f"Single scale STFT loss (n_fft=512): {single_loss_512}")
    print(f"Sum of both scales: {single_loss_2048 + single_loss_512}")
    
    # Test with partial mask
    mask_partial = jnp.ones((4, 505440))
    mask_partial = mask_partial.at[:, :400000].set(0)
    print(mask_partial[0])
    stft_loss_masked = multi_scale_stft_loss(audio_fake_3d, audio_true_3d, mask_partial)
    print(f"Multi-scale STFT loss with partial mask: {stft_loss_masked}")

