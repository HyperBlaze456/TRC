import jax.numpy as jnp
from mask_utils import create_padding_mask, downsample_mask, pad_sequences_left


def print_mask_demo():
    print("=" * 80)
    print("MASK UTILITIES DEMONSTRATION")
    print("=" * 80)

    # Create sample audio sequences of different lengths
    audio_lengths = [8, 6, 10, 4]  # Smaller lengths for easier visualization
    max_length = 10

    print(f"\nOriginal audio lengths: {audio_lengths}")
    print(f"Max length for padding: {max_length}")

    # Create dummy audio sequences
    audio_sequences = []
    for i, length in enumerate(audio_lengths):
        audio = jnp.ones(length) * (i + 1)  # Simple values for clarity
        audio_sequences.append(audio)

    # Pad sequences with left padding
    padded_audio, lengths = pad_sequences_left(audio_sequences, max_length=max_length)

    print("\n" + "=" * 40)
    print("LEFT-PADDED SEQUENCES")
    print("=" * 40)
    for i, seq in enumerate(padded_audio):
        print(f"Audio {i+1} (len={audio_lengths[i]}): {seq}")

    # Create non-causal padding mask
    non_causal_mask = create_padding_mask(lengths, max_length, causal=False)

    print("\n" + "=" * 40)
    print("NON-CAUSAL PADDING MASK")
    print("=" * 40)
    print("Shape:", non_causal_mask.shape, "(B, 1, 1, T)")
    print("\nMask values (True=Valid, False=Padded):")
    for i in range(len(audio_lengths)):
        mask_1d = non_causal_mask[i, 0, 0, :]
        print(f"Audio {i+1}: {mask_1d}")
        print(f"         {''.join(['  V' if m else '  P' for m in mask_1d])}")

    # Create causal padding mask
    causal_mask = create_padding_mask(lengths, max_length, causal=True)

    print("\n" + "=" * 40)
    print("CAUSAL PADDING MASK")
    print("=" * 40)
    print("Shape:", causal_mask.shape, "(B, 1, T, T)")

    # Show causal mask for one sample
    sample_idx = 1  # Audio 2
    print(f"\nCausal mask for Audio {sample_idx+1} (length={audio_lengths[sample_idx]}):")
    causal_single = causal_mask[sample_idx, 0, :, :]

    # Print with row/column headers
    print("\n    ", end="")
    for j in range(max_length):
        print(f" {j:2}", end="")
    print("\n    " + "-" * (max_length * 3))

    for i in range(max_length):
        print(f"{i:2} |", end="")
        for j in range(max_length):
            if causal_single[i, j]:
                print("  1", end="")
            else:
                print("  0", end="")
        print()

    # Demonstrate downsampling
    print("\n" + "=" * 40)
    print("DOWNSAMPLING DEMONSTRATION")
    print("=" * 40)

    downsample_factor = 2
    print(f"Downsample factor: {downsample_factor}")

    # Downsample non-causal mask
    downsampled_non_causal = downsample_mask(non_causal_mask, downsample_factor)
    print(f"\nOriginal mask shape: {non_causal_mask.shape}")
    print(f"Downsampled mask shape: {downsampled_non_causal.shape}")

    print("\nDownsampled mask values:")
    for i in range(len(audio_lengths)):
        mask_1d = downsampled_non_causal[i, 0, 0, :]
        print(f"Audio {i+1}: {mask_1d}")

    # Show the relationship between positions
    print("\n" + "=" * 40)
    print("POSITION MAPPING (Left-padding)")
    print("=" * 40)

    for i in range(len(audio_lengths)):
        print(f"\nAudio {i+1} (original length: {audio_lengths[i]}):")
        padding_positions = max_length - audio_lengths[i]
        print(f"  Padding positions: 0 to {padding_positions-1}")
        print(f"  Valid positions: {padding_positions} to {max_length-1}")

        # Show the actual data with positions
        seq = padded_audio[i]
        mask = non_causal_mask[i, 0, 0, :]
        print("  Position: ", end="")
        for j in range(max_length):
            print(f"{j:4}", end="")
        print("\n  Value:    ", end="")
        for j in range(max_length):
            print(f"{seq[j]:4.0f}", end="")
        print("\n  Valid:    ", end="")
        for j in range(max_length):
            print(f"{'T':4}" if mask[j] else f"{'F':4}", end="")
        print()

if __name__ == "__main__":
    print_mask_demo()
