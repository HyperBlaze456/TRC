from flax import nnx
import jax
from tokenizer.alpha.components.discriminators.stft_discriminator import STFTDiscriminator

key = jax.random.PRNGKey(42)
model = STFTDiscriminator(rngs=nnx.Rngs(key))
# Dummy 1‑second, 16 kHz waveform batch
x = jax.random.normal(key, (4, 16000))  # [B, T]

logits = model(x, training=False)  # [R, B]
print("logits shape:", logits.shape)
# Sanity check — expect (num_res, B)
assert logits.shape == (len(model.fft_sizes), x.shape[0])

# JIT compile round‑trip
jit_fn = jax.jit(model)
_ = jit_fn(x, training=False)
print("JIT compile success ✓")
