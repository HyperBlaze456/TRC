import jax
import jax.numpy as jnp
from tokenizer.utils.data.mask_utils import pad_sequences_left, create_padding_mask

# Test rounding functionality
print("Testing padding with rounding to nearest half-second (12000 samples):")
print("-" * 60)

# Create test sequences with various lengths
test_sequences = [
    jnp.ones((24000, 1)),
    jnp.ones((24001, 1)),
    jnp.ones((30000, 1)),
    jnp.ones((35999, 1)),
    jnp.ones((35888, 1)),
    jnp.ones((12000, 1)),
    jnp.ones((1, 1)),
]

# Test without rounding
print("\nWithout rounding:")
padded_no_round, lengths = pad_sequences_left(test_sequences)
print(f"Max length: {padded_no_round.shape[1]}")
print(f"Original lengths: {lengths}")

# Test with rounding to 12000
print("\nWith rounding to 12000:")
padded_rounded, lengths = pad_sequences_left(test_sequences, round_to_multiple=12000)
print(f"Rounded shape: {padded_rounded.shape}")
print(f"Original lengths: {lengths}")

# Verify masks work correctly with rounded lengths
print("\nVerifying padding masks:")
padding_mask = create_padding_mask(lengths, padded_rounded.shape[1])
for i, length in enumerate(lengths):
    valid_positions = jnp.sum(padding_mask[i])
    print(f"Sequence {i}: length={length}, valid positions in mask={valid_positions}")
    assert valid_positions == length, f"Mask mismatch for sequence {i}"

print("\nAll tests passed! The rounding functionality works correctly.")

# Show JIT compilation benefit
print("\n" + "-" * 60)
print("JIT compilation benefit demonstration:")

# Example of unique shapes without rounding
unique_shapes_no_round = set()
for i in range(24000, 48001, 1):  # All possible lengths from 1s to 2s
    unique_shapes_no_round.add(i)
print(f"Without rounding: {len(unique_shapes_no_round)} unique shapes")

# Example of unique shapes with rounding
unique_shapes_rounded = set()
for i in range(24000, 48001, 1):
    rounded = ((i + 11999) // 12000) * 12000
    unique_shapes_rounded.add(rounded)
print(f"With rounding: {len(unique_shapes_rounded)} unique shapes")
print(f"Reduction factor: {len(unique_shapes_no_round) / len(unique_shapes_rounded):.1f}x")