from flax import nnx

class ScaleDiscriminator(nnx.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 48000):

