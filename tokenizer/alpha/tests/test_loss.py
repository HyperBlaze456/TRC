"""Test file for compute_generator_loss and compute_discriminator_loss functions.

This file tests the loss functions with synthetic data to check for bugs and ensure
all shapes and computations work correctly.
"""


import jax
import jax.numpy as jnp

# Import the loss functions
from tokenizer.previous_things.loss import (
    compute_discriminator_loss,
    compute_generator_loss,
    create_phoneme_vocabulary,
)


def create_test_data(
    batch_size: int = 4,
    audio_length: int = 24000,  # 1 second at 24kHz
    encoder_length: int = 50,   # 50Hz features
    hidden_dim: int = 256,
    vocab_size: int = 100,
    num_discriminators: int = 3,
    num_disc_layers: int = 4
) -> dict:
    """Create synthetic test data with appropriate shapes.
    
    Args:
        batch_size: Batch size
        audio_length: Audio sequence length
        encoder_length: Encoder output sequence length (50Hz)
        hidden_dim: Hidden dimension size
        vocab_size: Phoneme vocabulary size
        num_discriminators: Number of discriminators
        num_disc_layers: Number of layers per discriminator
    
    Returns:
        Dictionary containing all test data
    """
    # Initialize random key
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 20)

    # Audio data
    pred_audio = jax.random.normal(keys[0], (batch_size, audio_length, 1))
    target_audio = jax.random.normal(keys[1], (batch_size, audio_length, 1))

    # Encoder outputs
    encoder_output = jax.random.normal(keys[2], (batch_size, encoder_length, hidden_dim))
    phoneme_quantized = jax.random.normal(keys[3], (batch_size, encoder_length, hidden_dim))

    # Residual for BSQ
    residual = jax.random.normal(keys[4], (batch_size, encoder_length, hidden_dim))
    residual_quantized = jax.random.normal(keys[5], (batch_size, encoder_length, hidden_dim))

    # Phoneme indices (discrete values)
    phoneme_indices = jax.random.randint(keys[6], (batch_size, encoder_length), 0, vocab_size)

    # Phoneme targets and lengths
    max_target_len = 30
    phoneme_targets = jax.random.randint(keys[7], (batch_size, max_target_len), 0, vocab_size)
    target_lengths = jax.random.randint(keys[8], (batch_size,), 10, max_target_len)
    encoder_lengths = jnp.full((batch_size,), encoder_length)  # Full length for simplicity

    # Phoneme codebook
    phoneme_codebook = jax.random.normal(keys[9], (vocab_size, hidden_dim))

    # Discriminator outputs (multiple discriminators with different scales)
    disc_outputs_fake = []
    disc_outputs_real = []
    for i in range(num_discriminators):
        # Each discriminator may have different output shape
        disc_length = encoder_length // (2 ** i)  # Progressively smaller
        disc_outputs_fake.append(
            jax.random.normal(keys[10 + i], (batch_size, disc_length))
        )
        disc_outputs_real.append(
            jax.random.normal(keys[13 + i], (batch_size, disc_length))
        )

    # Discriminator features (intermediate layer outputs)
    disc_features_fake = []
    disc_features_real = []
    for i in range(num_discriminators):
        fake_features = []
        real_features = []
        for j in range(num_disc_layers):
            # Feature maps get smaller deeper in the network
            feat_length = encoder_length // (2 ** (i + j // 2))
            feat_dim = hidden_dim // (2 ** j)

            fake_features.append(
                jax.random.normal(keys[16], (batch_size, feat_length, feat_dim))
            )
            real_features.append(
                jax.random.normal(keys[17], (batch_size, feat_length, feat_dim))
            )

        disc_features_fake.append(fake_features)
        disc_features_real.append(real_features)

    # Mask (optional) - using left padding
    mask = jnp.ones((batch_size, 1, 1, audio_length), dtype=jnp.bool_)
    # Add some padding to test masking
    mask = mask.at[0, :, :, -audio_length//4:].set(False)
    mask = mask.at[1, :, :, -audio_length//3:].set(False)

    return {
        "pred_audio": pred_audio,
        "target_audio": target_audio,
        "encoder_output": encoder_output,
        "phoneme_quantized": phoneme_quantized,
        "residual": residual,
        "residual_quantized": residual_quantized,
        "phoneme_indices": phoneme_indices,
        "phoneme_targets": phoneme_targets,
        "phoneme_codebook": phoneme_codebook,
        "encoder_lengths": encoder_lengths,
        "target_lengths": target_lengths,
        "disc_outputs_fake": disc_outputs_fake,
        "disc_outputs_real": disc_outputs_real,
        "disc_features_real": disc_features_real,
        "disc_features_fake": disc_features_fake,
        "mask": mask
    }


def test_generator_loss():
    """Test compute_generator_loss function."""
    print("Testing compute_generator_loss...")

    # Create test data
    data = create_test_data()

    # Test configuration
    config = {
        "l1_weight": 1.0,
        "l2_weight": 1.0,
        "spectral_weight": 2.0,
        "log_mag_weight": 1.0,
        "ctc_weight": 10.0,
        "phoneme_commit_weight": 0.1,
        "bsq_commit_weight": 1.0,
        "adversarial_weight": 1.0,
        "feature_match_weight": 10.0,
        "ctc_temperature": 1.0,
    }

    try:
        # Test with mask
        total_loss, losses = compute_generator_loss(
            pred_audio=data["pred_audio"],
            target_audio=data["target_audio"],
            encoder_output=data["encoder_output"],
            phoneme_quantized=data["phoneme_quantized"],
            residual=data["residual"],
            residual_quantized=data["residual_quantized"],
            phoneme_indices=data["phoneme_indices"],
            phoneme_targets=data["phoneme_targets"],
            phoneme_codebook=data["phoneme_codebook"],
            encoder_lengths=data["encoder_lengths"],
            target_lengths=data["target_lengths"],
            disc_outputs_fake=data["disc_outputs_fake"],
            disc_features_real=data["disc_features_real"],
            disc_features_fake=data["disc_features_fake"],
            mask=data["mask"],
            config=config
        )

        print("✓ Generator loss with mask computed successfully")
        print(f"  Total loss: {total_loss:.4f}")
        print("  Individual losses:")
        for name, value in losses.items():
            if name != "total":
                print(f"    {name}: {value:.4f}")

        # Test without mask
        total_loss_no_mask, losses_no_mask = compute_generator_loss(
            pred_audio=data["pred_audio"],
            target_audio=data["target_audio"],
            encoder_output=data["encoder_output"],
            phoneme_quantized=data["phoneme_quantized"],
            residual=data["residual"],
            residual_quantized=data["residual_quantized"],
            phoneme_indices=data["phoneme_indices"],
            phoneme_targets=data["phoneme_targets"],
            phoneme_codebook=data["phoneme_codebook"],
            encoder_lengths=data["encoder_lengths"],
            target_lengths=data["target_lengths"],
            disc_outputs_fake=data["disc_outputs_fake"],
            disc_features_real=data["disc_features_real"],
            disc_features_fake=data["disc_features_fake"],
            mask=None,
            config=config
        )

        print("\n✓ Generator loss without mask computed successfully")
        print(f"  Total loss: {total_loss_no_mask:.4f}")

        # Check that all losses are finite
        for name, value in losses.items():
            if not jnp.isfinite(value):
                print(f"✗ WARNING: {name} loss is not finite: {value}")

    except Exception as e:
        print(f"✗ Error in compute_generator_loss: {e!s}")
        raise


def test_discriminator_loss():
    """Test compute_discriminator_loss function."""
    print("\n\nTesting compute_discriminator_loss...")

    # Create test data
    data = create_test_data()

    # Test with hinge loss
    try:
        d_loss_hinge, losses_hinge = compute_discriminator_loss(
            disc_outputs_real=data["disc_outputs_real"],
            disc_outputs_fake=data["disc_outputs_fake"],
            loss_type="hinge"
        )

        print("✓ Discriminator hinge loss computed successfully")
        print(f"  Total loss: {d_loss_hinge:.4f}")
        print("  Individual losses:")
        for name, value in losses_hinge.items():
            if name != "total":
                print(f"    {name}: {value:.4f}")

        # Test with lsgan loss
        d_loss_lsgan, losses_lsgan = compute_discriminator_loss(
            disc_outputs_real=data["disc_outputs_real"],
            disc_outputs_fake=data["disc_outputs_fake"],
            loss_type="lsgan"
        )

        print("\n✓ Discriminator LSGAN loss computed successfully")
        print(f"  Total loss: {d_loss_lsgan:.4f}")

        # Check that losses are finite
        if not jnp.isfinite(d_loss_hinge):
            print(f"✗ WARNING: Hinge loss is not finite: {d_loss_hinge}")
        if not jnp.isfinite(d_loss_lsgan):
            print(f"✗ WARNING: LSGAN loss is not finite: {d_loss_lsgan}")

    except Exception as e:
        print(f"✗ Error in compute_discriminator_loss: {e!s}")
        raise


def test_shape_compatibility():
    """Test with various shape configurations."""
    print("\n\nTesting shape compatibility...")

    # Test different batch sizes
    for batch_size in [1, 2, 8]:
        print(f"\n  Testing batch size {batch_size}...")
        data = create_test_data(batch_size=batch_size)

        try:
            total_loss, _ = compute_generator_loss(
                pred_audio=data["pred_audio"],
                target_audio=data["target_audio"],
                encoder_output=data["encoder_output"],
                phoneme_quantized=data["phoneme_quantized"],
                residual=data["residual"],
                residual_quantized=data["residual_quantized"],
                phoneme_indices=data["phoneme_indices"],
                phoneme_targets=data["phoneme_targets"],
                phoneme_codebook=data["phoneme_codebook"],
                encoder_lengths=data["encoder_lengths"],
                target_lengths=data["target_lengths"],
                disc_outputs_fake=data["disc_outputs_fake"],
                disc_features_real=data["disc_features_real"],
                disc_features_fake=data["disc_features_fake"],
                mask=data["mask"]
            )
            print(f"    ✓ Batch size {batch_size} works correctly")
        except Exception as e:
            print(f"    ✗ Batch size {batch_size} failed: {e!s}")

    # Test different sequence lengths
    for audio_length, encoder_length in [(12000, 25), (48000, 100)]:
        print(f"\n  Testing audio_length={audio_length}, encoder_length={encoder_length}...")
        data = create_test_data(audio_length=audio_length, encoder_length=encoder_length)

        try:
            total_loss, _ = compute_generator_loss(
                pred_audio=data["pred_audio"],
                target_audio=data["target_audio"],
                encoder_output=data["encoder_output"],
                phoneme_quantized=data["phoneme_quantized"],
                residual=data["residual"],
                residual_quantized=data["residual_quantized"],
                phoneme_indices=data["phoneme_indices"],
                phoneme_targets=data["phoneme_targets"],
                phoneme_codebook=data["phoneme_codebook"],
                encoder_lengths=data["encoder_lengths"],
                target_lengths=data["target_lengths"],
                disc_outputs_fake=data["disc_outputs_fake"],
                disc_features_real=data["disc_features_real"],
                disc_features_fake=data["disc_features_fake"],
                mask=None  # No mask for simplicity
            )
            print("    ✓ Sequence lengths work correctly")
        except Exception as e:
            print(f"    ✗ Sequence lengths failed: {e!s}")


def test_edge_cases():
    """Test edge cases and potential issues."""
    print("\n\nTesting edge cases...")

    # Test with very small values (potential log issues)
    data = create_test_data()
    data["pred_audio"] = data["pred_audio"] * 1e-8
    data["target_audio"] = data["target_audio"] * 1e-8

    try:
        total_loss, losses = compute_generator_loss(
            pred_audio=data["pred_audio"],
            target_audio=data["target_audio"],
            encoder_output=data["encoder_output"],
            phoneme_quantized=data["phoneme_quantized"],
            residual=data["residual"],
            residual_quantized=data["residual_quantized"],
            phoneme_indices=data["phoneme_indices"],
            phoneme_targets=data["phoneme_targets"],
            phoneme_codebook=data["phoneme_codebook"],
            encoder_lengths=data["encoder_lengths"],
            target_lengths=data["target_lengths"],
            disc_outputs_fake=data["disc_outputs_fake"],
            disc_features_real=data["disc_features_real"],
            disc_features_fake=data["disc_features_fake"],
            mask=None
        )
        print("✓ Small value test passed")
        if not jnp.isfinite(total_loss):
            print(f"  ⚠ Warning: Total loss is not finite with small values: {total_loss}")
    except Exception as e:
        print(f"✗ Small value test failed: {e!s}")

    # Test with mismatched shapes (should fail gracefully)
    print("\n  Testing mismatched shapes (expected to fail)...")
    data = create_test_data()
    data["pred_audio"] = data["pred_audio"][:, :-100, :]  # Wrong shape

    try:
        total_loss, _ = compute_generator_loss(
            pred_audio=data["pred_audio"],
            target_audio=data["target_audio"],
            encoder_output=data["encoder_output"],
            phoneme_quantized=data["phoneme_quantized"],
            residual=data["residual"],
            residual_quantized=data["residual_quantized"],
            phoneme_indices=data["phoneme_indices"],
            phoneme_targets=data["phoneme_targets"],
            phoneme_codebook=data["phoneme_codebook"],
            encoder_lengths=data["encoder_lengths"],
            target_lengths=data["target_lengths"],
            disc_outputs_fake=data["disc_outputs_fake"],
            disc_features_real=data["disc_features_real"],
            disc_features_fake=data["disc_features_fake"],
            mask=None
        )
        print("  ⚠ Mismatched shapes did not raise error (unexpected)")
    except Exception as e:
        print(f"  ✓ Mismatched shapes correctly raised error: {type(e).__name__}")


def test_phoneme_vocabulary():
    """Test phoneme vocabulary creation."""
    print("\n\nTesting phoneme vocabulary creation...")

    vocab = create_phoneme_vocabulary()
    print(f"✓ Created phoneme vocabulary with {len(vocab)} entries")
    print(f"  First 5 entries: {list(vocab.items())[:5]}")
    print(f"  Contains blank token '_': {('_' in vocab)}")
    print(f"  Blank token index: {vocab.get('_', 'NOT FOUND')}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing loss functions for audio tokenizer")
    print("=" * 60)

    # Set up JAX
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print()

    # Run tests
    test_generator_loss()
    test_discriminator_loss()
    test_shape_compatibility()
    test_edge_cases()
    test_phoneme_vocabulary()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
