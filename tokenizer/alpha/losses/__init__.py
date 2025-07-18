from .adversarial import (
    hinge_loss_generator,
    hinge_loss_discriminator,
    least_squares_loss_generator,
    least_squares_loss_discriminator,
    feature_matching_loss
)
from .reconstruction import (
    multi_scale_spectrogram_loss,
    time_domain_loss,
    perceptual_stft_loss
)
from .ctc import ctc_loss
# deprecate this
__all__ = [
    'hinge_loss_generator',
    'hinge_loss_discriminator',
    'least_squares_loss_generator',
    'least_squares_loss_discriminator',
    'feature_matching_loss',
    'multi_scale_spectrogram_loss',
    'time_domain_loss',
    'perceptual_stft_loss',
    'ctc_loss'
]