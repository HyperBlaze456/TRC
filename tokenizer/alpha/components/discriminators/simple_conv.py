import jax.numpy as jnp
from flax import nnx
from typing import List


class ConvDiscriminator(nnx.Module):
    def __init__(
        self,
        channels: List[int] = [32, 64, 128, 256, 512],
        kernel_size: int = 4,
        strides: int = 2,
        padding: str = "SAME",
        rngs: nnx.Rngs = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        self.conv_blocks = []
        
        # Build convolutional blocks
        in_channels = 1  # Assuming mono audio input
        for i, out_channels in enumerate(channels):
            # Conv layer
            conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                rngs=rngs,
            )
            
            # Batch norm (except for first layer)
            bn = nnx.BatchNorm(
                num_features=out_channels,
                rngs=rngs,
            ) if i > 0 else None
            
            self.conv_blocks.append({
                'conv': conv,
                'bn': bn,
            })
            
            in_channels = out_channels
        
        # Final conv to single channel
        self.final_conv = nnx.Conv(
            in_features=channels[-1],
            out_features=1,
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            rngs=rngs,
        )
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # x shape: [B, T, 1] (batch, time, channels)
        
        for i, block in enumerate(self.conv_blocks):
            x = block['conv'](x)
            
            # Apply batch norm if not first layer
            if block['bn'] is not None:
                x = block['bn'](x, use_running_average=not training)
            
            # LeakyReLU activation
            x = nnx.leaky_relu(x, negative_slope=0.2)
        
        # Final conv
        x = self.final_conv(x)
        
        # Global average pooling to get single value per sample
        x = jnp.mean(x, axis=1, keepdims=True)  # [B, 1, 1]
        x = jnp.squeeze(x, axis=-1)  # [B, 1]
        
        return x