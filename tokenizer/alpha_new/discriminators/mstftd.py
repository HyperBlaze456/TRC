from flax import nnx
import jax
import jax.numpy as jnp

class STFTDiscriminator(nnx.Module):
    def __init__(self, fft_size: int, hop_length: int, win_length: int, rngs: nnx.Rngs):
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length

        self.convs = [
            nnx.Conv(
                2, 32,
                kernel_size=(3, 9),
                strides=(1, 2),
                padding="SAME",
                rngs=rngs
            ),
            nnx.Conv(
                32, 64,
                kernel_size=(3, 9),
                strides=(1, 2),
                padding="SAME",
                rngs=rngs
            ),
            nnx.Conv(
                64, 128,
                kernel_size=(3, 9),
                strides=(1, 2),
                padding="SAME",
                rngs=rngs
            ),
            nnx.Conv(
                128, 256,
                kernel_size=(3, 9),
                strides=(1, 2),
                padding="SAME",
                rngs=rngs
            ),
            nnx.Conv(
                256, 512,
                kernel_size=(3, 9),
                strides=(1, 2),
                padding="SAME",
                rngs=rngs
            ),
        ]

        self.conv_post = nnx.Conv(
            512, 1,
            kernel_size=(3, 9),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs
        )

    def __call__(self, x: jax.Array):
        """
        Args:
            x: Input tensor of shape [B, T, 1]
        Returns:
            feature_map: Contains output features from each convolutional layer
        """
        
        def get_memory_usage():
            """Get current GPU memory usage"""
            devices = jax.devices()
            if devices and devices[0].platform == "gpu":
                stats = devices[0].memory_stats()
                if stats:
                    return stats.get('bytes_in_use', 0) / (1024**3)  # GB
            return 0.0
        
        def get_tensor_size_mb(tensor):
            """Calculate tensor size in MB"""
            return tensor.nbytes / (1024**2)
        
        def get_conv_param_size_mb(conv):
            """Calculate conv layer parameter size in MB"""
            # Conv has kernel and bias
            kernel_size = conv.kernel.value.nbytes / (1024**2)
            bias_size = conv.bias.value.nbytes / (1024**2) if hasattr(conv, 'bias') and conv.bias is not None else 0
            return kernel_size + bias_size

        x = x.squeeze(-1)  # [B, T]
        input_size = get_tensor_size_mb(x)
        jax.debug.print("Input audio shape: {shape}, size: {size:.2f} MB", shape=x.shape, size=input_size)

        # Returns (freqs, times, Zxx) where Zxx is complex STFT
        _, _, Zxx = jax.scipy.signal.stft(
            x,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.fft_size,
            boundary=None,
            padded=False
        )

        stft_size = Zxx.nbytes / (1024**2)  # Complex64 = 8 bytes per element
        jax.debug.print("STFT output shape: {shape}, size: {size:.2f} MB (complex64)", shape=Zxx.shape, size=stft_size)

        # Zxx shape: [B, F, T_frames]
        # Stack magnitude and phase as channels
        mag = jnp.abs(Zxx)
        phase = jnp.angle(Zxx)
        
        mag_size = get_tensor_size_mb(mag)
        phase_size = get_tensor_size_mb(phase)
        jax.debug.print("Magnitude size: {size:.2f} MB, Phase size: {size2:.2f} MB", size=mag_size, size2=phase_size)

        # Transpose to [B, T_frames, F, 2] for 2D convolution
        x = jnp.stack([mag, phase], axis=-1)  # [B, F, T_frames, 2]
        x = jnp.transpose(x, (0, 2, 1, 3))  # [B, T_frames, F, 2]
        
        stacked_size = get_tensor_size_mb(x)
        jax.debug.print("Stacked mag+phase shape: {shape}, size: {size:.2f} MB", shape=x.shape, size=stacked_size)

        # Get initial memory
        initial_mem = get_memory_usage()
        jax.debug.print("\n=== Starting convolutions ===")
        jax.debug.print("Initial GPU memory: {mem:.3f} GB", mem=initial_mem)

        feature_map = []
        total_feature_size = 0.0
        total_param_size = 0.0
        
        for i, conv in enumerate(self.convs):
            # Get conv param size
            param_size = get_conv_param_size_mb(conv)
            total_param_size += param_size
            
            x = conv(x)
            x = nnx.leaky_relu(x, negative_slope=0.1)
            feature_map.append(x)
            
            # Calculate theoretical tensor size
            tensor_size = get_tensor_size_mb(x)
            total_feature_size += tensor_size
            
            # Force synchronization and get memory
            jax.block_until_ready(x)
            current_mem = get_memory_usage()
            jax.debug.print("Conv {i}: shape={shape}, tensor={tensor:.2f}MB, params={params:.2f}MB, GPU={mem:.3f}GB (Δ={delta:.3f}GB)", 
                           i=i,
                           shape=x.shape,
                           tensor=tensor_size,
                           params=param_size,
                           mem=current_mem,
                           delta=current_mem - initial_mem)

        # Conv post
        param_size = get_conv_param_size_mb(self.conv_post)
        total_param_size += param_size
        
        x = self.conv_post(x)
        feature_map.append(x)
        
        tensor_size = get_tensor_size_mb(x)
        total_feature_size += tensor_size
        
        # Final memory stats
        jax.block_until_ready(x)
        final_mem = get_memory_usage()
        jax.debug.print("Conv_post: shape={shape}, tensor={tensor:.2f}MB, params={params:.2f}MB, GPU={mem:.3f}GB (Δ={delta:.3f}GB)", 
                       shape=x.shape,
                       tensor=tensor_size,
                       params=param_size,
                       mem=final_mem,
                       delta=final_mem - initial_mem)
        
        jax.debug.print("\n=== Summary ===")
        jax.debug.print("Total feature map size (theoretical): {size:.2f} MB", size=total_feature_size)
        jax.debug.print("Total parameter size: {size:.2f} MB", size=total_param_size)
        jax.debug.print("Total theoretical memory: {size:.2f} MB", size=total_feature_size + total_param_size)
        jax.debug.print("Actual GPU memory increase: {delta:.3f} GB", delta=final_mem - initial_mem)

        return feature_map


class MSTFTD(nnx.Module):
    def __init__(
            self,
            fft_sizes: list[int] = [2048, 1024, 512],
            hop_lengths: list[int] = [512, 256, 128],
            win_lengths: list[int] = [2048, 1024, 512],
            rngs: nnx.Rngs = None,
    ):
        self.discriminators = []
        for fft_size, hop_length, win_length in zip(fft_sizes, hop_lengths, win_lengths):
            disc = STFTDiscriminator(
                fft_size=fft_size,
                hop_length=hop_length,
                win_length=win_length,
                rngs=rngs
            )
            self.discriminators.append(disc)

    def __call__(self, x: jax.Array):
        """
        Args:
            x: Input tensor of shape [B, T, 1]
        Returns:
            feature_maps: List of feature maps from each STFT resolution
        """
        feature_maps = []
        for discriminator in self.discriminators:
            feature_map = discriminator(x)
            feature_maps.append(feature_map)

        return feature_maps

# Can't JIT print

if __name__ == '__main__':
    # Get initial memory state
    devices = jax.devices()
    print(f"Running on devices: {devices}")
    
    if devices and devices[0].platform == "gpu":
        initial_stats = devices[0].memory_stats()
        if initial_stats:
            print(f"Initial GPU memory: {initial_stats.get('bytes_in_use', 0) / (1024**3):.3f} GB")
    
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(0)

    model = MSTFTD(rngs=rngs)
    mock_audio = jax.random.normal(key, (32, 168_000, 1))

    # Run without JIT to see memory usage properly
    print("\nRunning MSTFTD forward pass (no JIT)...")
    featmap = model(mock_audio)
    
    # Ensure all computations are complete
    for scale_maps in featmap:
        for feat in scale_maps:
            jax.block_until_ready(feat)
    
    # Get final memory state
    if devices and devices[0].platform == "gpu":
        final_stats = devices[0].memory_stats()
        if final_stats:
            print(f"\nFinal GPU memory: {final_stats.get('bytes_in_use', 0) / (1024**3):.3f} GB")
            print(f"Total memory allocated: {final_stats.get('bytes_in_use', 0) / (1024**3):.3f} GB")
            
            if initial_stats:
                delta = (final_stats.get('bytes_in_use', 0) - initial_stats.get('bytes_in_use', 0)) / (1024**3)
                print(f"Memory increase: {delta:.3f} GB")