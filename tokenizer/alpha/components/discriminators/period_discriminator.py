import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple


class PeriodDiscriminator(nnx.Module):
    """Single period discriminator sub-network."""
    
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, rngs: nnx.Rngs = None):
        self.period = period
        
        self.convs = [
            nnx.Conv(1, 32, kernel_size=(kernel_size, 1), strides=(stride, 1), padding=((2, 2), (0, 0)), rngs=rngs),
            nnx.Conv(32, 128, kernel_size=(kernel_size, 1), strides=(stride, 1), padding=((2, 2), (0, 0)), rngs=rngs),
            nnx.Conv(128, 512, kernel_size=(kernel_size, 1), strides=(stride, 1), padding=((2, 2), (0, 0)), rngs=rngs),
            nnx.Conv(512, 1024, kernel_size=(kernel_size, 1), strides=(stride, 1), padding=((2, 2), (0, 0)), rngs=rngs),
            nnx.Conv(1024, 1024, kernel_size=(kernel_size, 1), strides=(1, 1), padding=((2, 2), (0, 0)), rngs=rngs),
        ]
        
        self.conv_post = nnx.Conv(1024, 1, kernel_size=(3, 1), strides=(1, 1), padding=((1, 1), (0, 0)), rngs=rngs)
        
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Args:
            x: Input tensor of shape [B, T]
            
        Returns:
            output: Discriminator output of shape [B, T', 1]
            feature_maps: List of intermediate feature maps for feature matching loss
        """
        batch_size, time_steps = x.shape
        
        # Pad to make time divisible by period
        if time_steps % self.period != 0:
            pad_size = self.period - (time_steps % self.period)
            x = jnp.pad(x, ((0, 0), (0, pad_size)), mode='reflect')
            time_steps = x.shape[1]
        
        # Reshape to 2D representation: [B, T/P, P]
        x = x.reshape(batch_size, time_steps // self.period, self.period)
        
        # Add channel dimension: [B, T/P, P, 1]
        x = x[..., jnp.newaxis]
        
        # Apply convolutions
        feature_maps = []
        for conv in self.convs:
            x = conv(x)
            x = nnx.leaky_relu(x, negative_slope=0.1)
            feature_maps.append(x)
        
        # Final convolution
        x = self.conv_post(x)
        feature_maps.append(x)
        
        # Flatten output: [B, T', 1]
        x = x.reshape(batch_size, -1, 1)
        
        return x, feature_maps


class MultiPeriodDiscriminator(nnx.Module):
    """Multi-Period Discriminator (MPD) from HiFi-GAN."""
    
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11], kernel_size: int = 5, stride: int = 3, rngs: nnx.Rngs = None):
        """
        Args:
            periods: List of periods for sub-discriminators
            kernel_size: Kernel size for convolutions
            stride: Stride for convolutions
            rngs: Random number generators for initialization
        """
        self.discriminators = [
            PeriodDiscriminator(period, kernel_size, stride, rngs)
            for period in periods
        ]
    
    def __call__(self, x: jnp.ndarray) -> Tuple[List[jnp.ndarray], List[List[jnp.ndarray]]]:
        """
        Args:
            x: Input waveform of shape [B, T] or [B, T, 1]
            
        Returns:
            outputs: List of discriminator outputs from each sub-discriminator
            feature_maps: List of feature maps from each sub-discriminator
        """
        # Handle [B, T, 1] input
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        
        outputs = []
        all_feature_maps = []
        
        for discriminator in self.discriminators:
            output, features = discriminator(x)
            outputs.append(output)
            all_feature_maps.append(features)
        
        return outputs, all_feature_maps
