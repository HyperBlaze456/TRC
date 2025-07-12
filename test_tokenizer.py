import jax
import jax.numpy as jnp
from flax import nnx

from tokenizer.alpha.model import AudioTokenizer

def test_audio_tokenizer():
    """Test the AudioTokenizer model with sample input."""
    
    # Initialize RNG
    rngs = nnx.Rngs(params=42)
    
    # Create model
    model = AudioTokenizer(
        hidden_size=128,  # Smaller for testing
        encoder_depth=2,
        encoder_heads=4,
        phoneme_codebook_size=50,  # Small codebook for testing
        bsq_spherical_dim=64,  # Smaller BSQ dim for testing
        decoder_output_48khz=False,  # 24kHz output
        rngs=rngs,
    )
    
    # Create dummy input: [batch=2, time=24000, channels=1] (1 second at 24kHz)
    batch_size = 2
    sample_rate = 24000
    duration = 1.0  # seconds
    x = jax.random.normal(rngs.params(), shape=(batch_size, int(sample_rate * duration), 1))
    
    print(f"Input shape: {x.shape}")
    
    # Test forward pass
    reconstructed, phoneme_indices, acoustic_codes, encoder_output = model(x)
    
    print(f"\nForward pass results:")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Phoneme indices shape: {phoneme_indices.shape}")
    print(f"Acoustic codes shape: {acoustic_codes.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    
    # Test encode/decode
    phoneme_indices_enc, acoustic_codes_enc = model.encode(x)
    reconstructed_dec = model.decode(phoneme_indices_enc, acoustic_codes_enc)
    
    print(f"\nEncode/decode results:")
    print(f"Phoneme indices (encode) shape: {phoneme_indices_enc.shape}")
    print(f"Acoustic codes (encode) shape: {acoustic_codes_enc.shape}")
    print(f"Reconstructed (decode) shape: {reconstructed_dec.shape}")
    
    # Verify compression ratio
    compression_ratio = x.shape[1] / encoder_output.shape[1]
    print(f"\nCompression ratio: {compression_ratio:.1f}x")
    print(f"Expected: ~480x (24kHz -> 50Hz)")
    
    # Check that shapes match
    assert reconstructed.shape == x.shape, f"Output shape mismatch: {reconstructed.shape} != {x.shape}"
    #assert jnp.allclose(reconstructed, reconstructed_dec, atol=1e-5), "Forward pass and encode/decode mismatch"
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_audio_tokenizer()