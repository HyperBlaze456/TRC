import jax
from flax import nnx
import optax

from tokenizer.alpha.model import AudioTokenizer
from tokenizer.alpha.components.discriminators import MultiPeriodDiscriminator, STFTDiscriminator, MultiScaleDiscriminator

def