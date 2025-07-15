from .simple_conv import ConvDiscriminator
from .stft_discriminator import STFTDiscriminator, STFTResolutionDiscriminator
from .period_discriminator import MultiPeriodDiscriminator, PeriodDiscriminator
from .scale_discriminator import MultiScaleDiscriminator, ScaleDiscriminator

__all__ = ["ConvDiscriminator", "STFTDiscriminator", "STFTResolutionDiscriminator", "MultiPeriodDiscriminator",
           "PeriodDiscriminator", "MultiScaleDiscriminator", "ScaleDiscriminator"]