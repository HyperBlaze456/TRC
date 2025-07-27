from flax import nnx
import jax

from tokenizer.utils.activation import Snake


class CausalConvTranspose(nnx.Module):
    def __init__(
        self,
        in_features:  int,
        out_features: int,
        kernel_size:  int,
        rngs       :  nnx.Rngs,
        stride:       int = 1,
        dilation:     int = 1,
        *,
        use_bias: bool = True,
        **kwargs,               # initializers, dtype …
    ):
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.deconv = nnx.ConvTranspose(
            in_features = in_features,
            out_features = out_features,
            kernel_size = kernel_size,
            strides = stride,
            padding = "VALID",
            kernel_dilation= dilation,
            use_bias = use_bias,
            rngs = rngs,
            **kwargs,
        )

        self._crop = max(0, dilation * (kernel_size - 1) + 1 - stride)

    def __call__(self, x):
        y = self.deconv(x)
        if self._crop:
            y = y[:, :-self._crop, :]
        return y

class RawDecoder(nnx.Module):
    """Mirrored Encoder, converting latent frames back to waveform.
    Either 480x or 960x. Things could change any time.
    """

    def __init__(
        self,
        hidden_size: int,
        rngs: nnx.Rngs,
        output_48khz: bool = False,
    ):
        """
        Args:
            hidden_size: Input embedding dimension
            rngs: Random number generators
            output_48khz: Whether to output 48kHz (True) or 24kHz (False)
        """
        self.output_48khz = output_48khz

        # Mirror of encoder stages (in reverse order)
        # Stage 4: 50Hz -> 400Hz (stride 8)
        self.deconv4 = CausalConvTranspose(
            in_features=hidden_size,
            out_features=64,
            kernel_size=12,
            stride=8,
            rngs=rngs,
        )
        self.snake4 = Snake(64)

        # Stage 3: 400Hz -> 2kHz (stride 5)
        self.deconv3 = CausalConvTranspose(
            in_features=64,
            out_features=16,
            kernel_size=9,
            stride=5,
            rngs=rngs,
        )
        self.snake3 = Snake(16)

        # Stage 2: 2kHz -> 6kHz (stride 3)
        self.deconv2 = CausalConvTranspose(
            in_features=16,
            out_features=8,
            kernel_size=5,
            stride=3,
            rngs=rngs,
        )
        self.snake2 = Snake(8)

        # Stage 1: 6kHz -> 24kHz (stride 4)
        self.deconv1 = CausalConvTranspose(
            in_features=8,
            out_features=1,
            kernel_size=12,
            stride=4,
            rngs=rngs,
        )

        # Optional 24kHz -> 48kHz
        if self.output_48khz:
            self.deconv_to48 = CausalConvTranspose(
                in_features=1,
                out_features=1,
                kernel_size=4,
                stride=2,
                rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Compressed embeddings [B, T, hidden_size] at 50Hz
            
        Returns:
            Reconstructed waveform [B, T*480, 1] at 24kHz or [B, T*960, 1] at 48kHz
        """
        # Progressive upsampling (reverse of encoder)
        x = self.deconv4(x)
        x = self.snake4(x)

        x = self.deconv3(x)
        x = self.snake3(x)

        x = self.deconv2(x)
        x = self.snake2(x)

        x = self.deconv1(x)
        # No activation after final layer to allow full waveform range

        # Optional 24kHz to 48kHz upsampling if needed.
        if self.output_48khz:
            x = self.deconv_to48(x)

        return x
