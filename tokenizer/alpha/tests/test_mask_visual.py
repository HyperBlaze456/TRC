import jax
import jax.numpy as jnp
from mask_utils import create_padding_mask, downsample_mask, pad_sequences_left
import matplotlib.pyplot as plt
import numpy as np


def visualize_masks():
    # Create sample audio sequences of different lengths
    # Simulating batch of 4 audio sequences with different lengths
    audio_lengths = [100, 80, 120, 60]  # Original audio lengths
    max_length = 120  # Maximum length in batch

    # Create dummy audio data (just for visualization)
    audio_sequences = []
    for i, length in enumerate(audio_lengths):
        # Create a simple sine wave with different frequencies for each audio
        t = np.linspace(0, length/24000 * 2 * np.pi * (i+1), length)
        audio = np.sin(t * 100 * (i+1))
        audio_sequences.append(jnp.array(audio))

    # Pad sequences with left padding
    padded_audio, lengths = pad_sequences_left(audio_sequences, max_length=max_length)

    # Create non-causal padding mask
    non_causal_mask = create_padding_mask(lengths, max_length, causal=False)

    # Create causal padding mask
    causal_mask = create_padding_mask(lengths, max_length, causal=True)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Mask Utilities Visualization", fontsize=16)

    # Plot 1: Padded audio sequences
    ax = axes[0, 0]
    for i in range(len(padded_audio)):
        ax.plot(padded_audio[i], label=f"Audio {i+1} (len={audio_lengths[i]})", alpha=0.7)
    ax.set_title("Left-Padded Audio Sequences")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Non-causal padding mask visualization
    ax = axes[0, 1]
    # Reshape mask for visualization [B, 1, 1, T] -> [B, T]
    mask_2d = non_causal_mask[:, 0, 0, :]
    im = ax.imshow(mask_2d, aspect="auto", cmap="RdBu", interpolation="nearest")
    ax.set_title("Non-Causal Padding Mask\n(True=Valid, False=Padded)")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Batch samples")
    ax.set_yticks(range(len(audio_lengths)))
    ax.set_yticklabels([f"Audio {i+1}" for i in range(len(audio_lengths))])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Valid (1) / Padded (0)")

    # Plot 3: Individual causal mask for one sample
    ax = axes[1, 0]
    # Show causal mask for the first sample
    causal_single = causal_mask[0, 0, :, :]  # [T, T]
    im = ax.imshow(causal_single, aspect="equal", cmap="RdBu", interpolation="nearest")
    ax.set_title(f"Causal Mask for Audio 1 (len={audio_lengths[0]})")
    ax.set_xlabel("Key positions")
    ax.set_ylabel("Query positions")

    # Plot 4: Downsampled masks
    ax = axes[1, 1]
    downsample_factor = 10  # Simulate downsampling by factor of 10

    # Downsample non-causal mask
    downsampled_non_causal = downsample_mask(non_causal_mask, downsample_factor)
    downsampled_2d = downsampled_non_causal[:, 0, 0, :]

    im = ax.imshow(downsampled_2d, aspect="auto", cmap="RdBu", interpolation="nearest")
    ax.set_title(f"Downsampled Non-Causal Mask\n(factor={downsample_factor})")
    ax.set_xlabel("Downsampled time steps")
    ax.set_ylabel("Batch samples")
    ax.set_yticks(range(len(audio_lengths)))
    ax.set_yticklabels([f"Audio {i+1}" for i in range(len(audio_lengths))])

    # Plot 5: Mask statistics
    ax = axes[2, 0]
    ax.axis("off")
    stats_text = "Mask Statistics:\n\n"
    for i in range(len(audio_lengths)):
        valid_positions = jnp.sum(mask_2d[i])
        stats_text += f"Audio {i+1}:\n"
        stats_text += f"  Original length: {audio_lengths[i]}\n"
        stats_text += f"  Padded length: {max_length}\n"
        stats_text += f"  Valid positions: {valid_positions}\n"
        stats_text += f"  Padding positions: {max_length - valid_positions}\n\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
            verticalalignment="top", fontsize=10, family="monospace")

    # Plot 6: Causal vs Non-causal comparison
    ax = axes[2, 1]
    # Show difference between causal and non-causal for one sample
    sample_idx = 2  # Use Audio 3 which has full length
    non_causal_sample = jnp.ones((max_length, max_length)) * mask_2d[sample_idx, None, :]
    causal_sample = causal_mask[sample_idx, 0, :, :]

    # Stack them side by side
    comparison = jnp.hstack([non_causal_sample, jnp.ones((max_length, 5))*0.5, causal_sample])
    im = ax.imshow(comparison, aspect="auto", cmap="RdBu", interpolation="nearest")
    ax.set_title(f"Non-Causal (left) vs Causal (right)\nAudio {sample_idx+1}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")

    # Add vertical line to separate
    ax.axvline(x=max_length, color="black", linewidth=2)
    ax.axvline(x=max_length+5, color="black", linewidth=2)

    plt.tight_layout()
    plt.savefig("mask_visualization_1.png", dpi=150, bbox_inches="tight")
    print("Saved mask_visualization_1.png")

    # Additional visualization for understanding padding behavior
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle("Detailed Padding Behavior", fontsize=16)

    # Show how padding affects actual data
    for i in range(min(4, len(audio_lengths))):
        ax = axes2[i//2, i%2]

        # Plot the padded sequence
        ax.plot(padded_audio[i], "b-", alpha=0.7, label="Padded audio")

        # Highlight valid vs padded regions
        valid_mask = mask_2d[i]
        padded_region = ~valid_mask

        # Color regions
        ax.fill_between(range(max_length),
                       padded_audio[i].min()-0.1,
                       padded_audio[i].max()+0.1,
                       where=padded_region,
                       alpha=0.3,
                       color="red",
                       label="Padded region")

        ax.fill_between(range(max_length),
                       padded_audio[i].min()-0.1,
                       padded_audio[i].max()+0.1,
                       where=valid_mask,
                       alpha=0.3,
                       color="green",
                       label="Valid region")

        ax.set_title(f"Audio {i+1}: Original length={audio_lengths[i]}")
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add text annotations
        padding_start = max_length - audio_lengths[i]
        if padding_start > 0:
            ax.axvline(x=padding_start, color="red", linestyle="--", alpha=0.7)
            ax.text(padding_start/2, 0, "Padding", ha="center", va="center",
                   bbox=dict(boxstyle="round", facecolor="red", alpha=0.3))

    plt.tight_layout()
    plt.savefig("mask_visualization_2.png", dpi=150, bbox_inches="tight")
    print("Saved mask_visualization_2.png")

if __name__ == "__main__":
    # Set up JAX
    print("JAX version:", jax.__version__)
    print("Devices:", jax.devices())

    # Run visualization
    visualize_masks()
