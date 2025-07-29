from flax import nnx


class SimpleGenerator(nnx.Module):
    def __init__(self, hidden_dim: int, output_dim: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
