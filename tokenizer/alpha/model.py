from flax import nnx
import jax

from tokenizer.alpha.components.encoder import RawEncoder
from tokenizer.alpha.components.quantizer import PhonemeBSQQuantizer
from tokenizer.alpha.components.decoder import RawDecoder


class AudioTokenizer(nnx.Module):
    """Main audio tokenizer model with phoneme VQ + acoustic BSQ.
    
    Processes raw audio waveforms through:
    1. Encoder: Downsamples audio from 24/48kHz to 50Hz representations
    2. Quantizer: Phoneme VQ + residual BSQ for efficient tokenization
    3. Decoder: Reconstructs high-quality audio from quantized codes
    """
    
    def __init__(
        self,
        hidden_size: int = 512,
        encoder_depth: int = 4,
        encoder_heads: int = 8,
        phoneme_codebook_size: int = 100,
        bsq_spherical_dim: int = 256,
        decoder_output_48khz: bool = False,
        rngs: nnx.Rngs = None,
    ):
        """
        Args:
            hidden_size: Hidden dimension size for encoder/decoder
            encoder_depth: Number of transformer blocks in encoder
            encoder_heads: Number of attention heads in encoder
            phoneme_codebook_size: Size of VQ codebook for phoneme classification
            bsq_spherical_dim: Spherical dimension for BSQ acoustic features
            decoder_output_48khz: Whether decoder outputs 48kHz (True) or 24kHz (False)
            rngs: Random number generators
        """
        if rngs is None:
            raise ValueError("rngs parameter is required")
            
        self.hidden_size = hidden_size
        
        # Encoder: Raw audio to compressed representation
        self.encoder = RawEncoder(
            hidden_size=hidden_size,
            depth=encoder_depth,
            num_heads=encoder_heads,
            rngs=rngs,
            mlp_ratio=4.0,
            is_48khz=decoder_output_48khz,  # Match input/output sample rates
        )
        
        # Quantizer: Phoneme VQ + Acoustic BSQ
        self.quantizer = PhonemeBSQQuantizer(
            input_dim=hidden_size,
            phoneme_codebook_size=phoneme_codebook_size,
            spherical_dim=bsq_spherical_dim,
            rngs=rngs,
            temperature=1.0,
        )
        
        # Decoder: Compressed representation back to audio
        self.decoder = RawDecoder(
            hidden_size=hidden_size,
            rngs=rngs,
            output_48khz=decoder_output_48khz,
        )
    
    def __call__(self, x: jax.Array):
        """Forward pass through the tokenizer.
        
        Args:
            x: Raw audio waveform [B, T, 1] (channels last)
            
        Returns:
            reconstructed: Reconstructed audio [B, T, 1]
            phoneme_indices: Phoneme codebook indices [B, T']
            acoustic_codes: Binary acoustic codes [B, T', spherical_dim]
            encoder_output: Encoder output before quantization [B, T', D]
        """
        # Encode audio to latent representation
        encoder_output = self.encoder(x)
        
        # Quantize with phoneme VQ + acoustic BSQ
        quantized, phoneme_indices, acoustic_codes = self.quantizer(encoder_output)
        
        # Decode back to audio
        reconstructed = self.decoder(quantized)
        
        return reconstructed, phoneme_indices, acoustic_codes, encoder_output
    
    def encode(self, x: jax.Array):
        """Encode audio to discrete tokens.
        
        Args:
            x: Raw audio waveform [B, T, 1]
            
        Returns:
            phoneme_indices: Phoneme codebook indices [B, T']
            acoustic_codes: Binary acoustic codes [B, T', spherical_dim]
        """
        # Encode to latent
        encoder_output = self.encoder(x)
        
        # Get discrete codes
        phoneme_indices, acoustic_codes = self.quantizer.encode(encoder_output)
        
        return phoneme_indices, acoustic_codes
    
    def decode(self, phoneme_indices: jax.Array, acoustic_codes: jax.Array):
        """Decode discrete tokens back to audio.
        
        Args:
            phoneme_indices: Phoneme codebook indices [B, T']
            acoustic_codes: Binary acoustic codes [B, T', spherical_dim]
            
        Returns:
            Reconstructed audio [B, T, 1]
        """
        # Decode tokens to latent
        latent = self.quantizer.decode(phoneme_indices, acoustic_codes)
        
        # Decode to audio
        audio = self.decoder(latent)
        
        return audio
