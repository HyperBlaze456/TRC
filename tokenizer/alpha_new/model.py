from flax import nnx
import jax

from tokenizer.alpha_new.components.decoder import RawDecoder
from tokenizer.alpha_new.components.encoder import RawEncoder
from tokenizer.alpha_new.components.quantizer import PhonemeBSQQuantizer


class SpeechTokenizer(nnx.Module):
    """Main audio tokenizer model with phoneme VQ + acoustic BSQ.

    Processes raw audio waveforms through:
    1. Encoder: Downsamples audio from 24/48kHz to 50Hz representations
    2. Quantizer: Phoneme VQ + residual BSQ for efficient tokenization
    3. Decoder: Reconstructs high-quality audio from quantized codes
    """

    def __init__(
        self,
        hidden_size: int = 512,
        encoder_mlp_dim: int = 2048,
        encoder_depth: int = 4,
        encoder_heads: int = 8,
        phoneme_codebook_size: int = 100,
        bsq_spherical_dim: int = 256,
        temperature: float = 1.0,
        decoder_output_48khz: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.downsample_factor = 960 if decoder_output_48khz else 480
        self.encoder = RawEncoder(
            hidden_size=hidden_size,
            mlp_dim=encoder_mlp_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            rngs=rngs,
            is_48khz=decoder_output_48khz,
        )

        self.quantizer = PhonemeBSQQuantizer(
            input_dim=hidden_size,
            phoneme_codebook_size=phoneme_codebook_size,
            spherical_dim=bsq_spherical_dim,
            rngs=rngs,
            temperature=temperature,
        )
        self.decoder = RawDecoder(
            hidden_size=hidden_size,
            rngs=rngs,
            output_48khz=decoder_output_48khz,
        )

    def __call__(
        self, x: jax.Array, mask: jax.Array = None
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        """Forward pass through the tokenizer.

        Args:
            x: Raw audio waveform [B, T, 1] (channels last)
            mask: Pre-computed encoder mask at encoder resolution [B, T', T']

        Returns:
            reconstructed: Reconstructed audio [B, T, 1]
            phoneme_indices: Phoneme codebook indices [B, T']
            acoustic_codes: Binary acoustic codes [B, T', spherical_dim]
            encoder_output: Encoder output before quantization [B, T', D]
            phoneme_logits: Phoneme log probabilities [B, T', phoneme_codebook_size]
            vq_quantized: VQ output for commitment loss [B, T', D]
            bsq_quantized: BSQ output for commitment loss [B, T', D]
            vq_residual: Residual before BSQ for commitment loss [B, T', D]
        """
        encoder_output = self.encoder(x, mask=mask)

        (
            quantized,
            phoneme_indicies,
            acoustic_codes,
            phoneme_logits,
            vq_quantized,
            bsq_quantized,
            vq_residual,
        ) = self.quantizer(x)

        reconstructed = self.decoder(quantized)

        return (
            reconstructed,
            phoneme_indicies,
            acoustic_codes,
            encoder_output,
            phoneme_logits,
            vq_quantized,
            bsq_quantized,
            vq_residual,
        )

    def encode(self, x, mask: jax.Array = None) -> tuple[jax.Array, jax.Array]:
        """
        TODO
        Encoding only function.

        Args:
            x: Raw audio waveform [B, T, 1]. Channel dim included
            mask: Pre-computed encoder mask at encoder resolution [B, T', T'] or [B, 1, T', T']

        Returns:
            phoneme_indicies: First layer(Phoneme VQ) indicies
            acoustic_codes: Second layer(residual BSQ) indicies
        """
