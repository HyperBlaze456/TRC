from tokenizer.alpha_new.components.encoder import RawEncoder
from tokenizer.alpha_new.components.decoder import RawDecoder

import jax
from flax import nnx
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
rngs = nnx.Rngs(0)
test_audio = jax.random.uniform(key, shape=(32, 48000, 1))

encoder = RawEncoder(hidden_size=512, depth=4, num_heads=8, rngs=rngs, mlp_dim=2048)
decoder = RawDecoder(hidden_size=512, rngs=rngs)
encoder_latent = encoder(test_audio)
reconstructed_audio = decoder(encoder_latent)

print(encoder_latent.shape)
print(reconstructed_audio.shape)
