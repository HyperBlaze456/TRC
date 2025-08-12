# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .quantize import ResidualVQ
from .vocos import VocosBackbone


def WNConv1d(*args, **kwargs):
    """Weight normalized 1D convolution layer."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """Weight normalized 1D transposed convolution layer.""" 
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    """Snake activation function."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """1D Snake activation layer."""
    
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def init_weights(m):
    """Initialize weights for Conv1d and Linear layers."""
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions and Snake activation."""
    
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    """Encoder block with residual units and downsampling convolution."""
    
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and residual units."""
    
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class CodecEncoder(nn.Module):
    """
    Audio codec encoder with progressive downsampling through EncoderBlocks.
    
    Args:
        d_model (int): Initial number of channels. Defaults to 64.
        up_ratios (list): Downsampling ratios for each encoder block. Defaults to [4, 5, 5, 6].
        out_channels (int): Output channels. Defaults to 256.
        use_tanh (bool): Whether to apply Tanh activation at output. Defaults to False.
        cfg: Optional configuration object to override defaults.
    """
    
    def __init__(
        self,
        d_model: int = 64,
        up_ratios: list = [4, 5, 5, 6],
        out_channels: int = 256,
        use_tanh: bool = False,
        cfg=None,
    ):
        super().__init__()

        # Override with config values if provided
        d_model = cfg.d_model if cfg is not None else d_model
        up_ratios = cfg.up_ratios if cfg is not None else up_ratios
        out_channels = cfg.out_channels if cfg is not None else out_channels
        use_tanh = cfg.use_tanh if cfg is not None else use_tanh

        # Create first convolution layer
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in up_ratios:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create final convolution layer
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        # Add Tanh activation if requested
        if use_tanh:
            self.block += [nn.Tanh()]

        # Wrap blocks into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

        self.reset_parameters()

    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input audio tensor of shape (B, 1, T)
            
        Returns:
            torch.Tensor: Encoded features of shape (B, out_channels, T')
        """
        return self.block(x)

    def reset_parameters(self):
        """Reset all parameters using the init_weights function."""
        self.apply(init_weights)


class CodecDecoder(nn.Module):
    """
    Audio codec decoder with quantization and upsampling reconstruction.
    
    Args:
        in_channels (int): Input feature channels. Defaults to 256.
        upsample_initial_channel (int): Initial channel count for upsampling. Defaults to 1536.
        up_ratios (list): Upsampling ratios for decoder blocks. Defaults to [5, 5, 4, 2].
        num_quantizers (int): Number of quantizers in RVQ. Defaults to 8.
        codebook_size (int): Size of each codebook. Defaults to 1024.
        codebook_dim (int): Dimension of codebook vectors. Defaults to 256.
        quantizer_type (str): Type of quantizer ("vq", "fvq", "lfq"). Defaults to "vq".
        quantizer_dropout (float): Dropout rate for quantizers during training. Defaults to 0.5.
        commitment (float): Commitment loss weight. Defaults to 0.25.
        codebook_loss_weight (float): Codebook loss weight. Defaults to 1.0.
        use_l2_normlize (bool): Whether to use L2 normalization. Defaults to False.
        codebook_type (str): Type of codebook distance metric. Defaults to "euclidean".
        kmeans_init (bool): Whether to use k-means initialization. Defaults to False.
        kmeans_iters (int): Number of k-means iterations. Defaults to 10.
        decay (float): Exponential moving average decay. Defaults to 0.8.
        eps (float): Small epsilon for numerical stability. Defaults to 1e-5.
        threshold_ema_dead_code (int): Threshold for EMA dead code detection. Defaults to 2.
        weight_init (bool): Whether to initialize weights. Defaults to False.
        use_vocos (bool): Whether to use Vocos backbone. Defaults to False.
        vocos_dim (int): Vocos model dimension. Defaults to 384.
        vocos_intermediate_dim (int): Vocos intermediate dimension. Defaults to 1152.
        vocos_num_layers (int): Number of Vocos layers. Defaults to 8.
        n_fft (int): FFT size for Vocos. Defaults to 800.
        hop_size (int): Hop size for Vocos. Defaults to 200.
        padding (str): Padding type for Vocos. Defaults to "same".
        cfg: Optional configuration object to override defaults.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        upsample_initial_channel: int = 1536,
        up_ratios: list = [5, 5, 4, 2],
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        quantizer_type: str = "vq",
        quantizer_dropout: float = 0.5,
        commitment: float = 0.25,
        codebook_loss_weight: float = 1.0,
        use_l2_normlize: bool = False,
        codebook_type: str = "euclidean",
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.8,
        eps: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        weight_init: bool = False,
        use_vocos: bool = False,
        vocos_dim: int = 384,
        vocos_intermediate_dim: int = 1152,
        vocos_num_layers: int = 8,
        n_fft: int = 800,
        hop_size: int = 200,
        padding: str = "same",
        cfg=None,
    ):
        super().__init__()

        # Override parameters with config values if provided
        if cfg is not None:
            in_channels = getattr(cfg, 'in_channels', in_channels)
            upsample_initial_channel = getattr(cfg, 'upsample_initial_channel', upsample_initial_channel)
            up_ratios = getattr(cfg, 'up_ratios', up_ratios)
            num_quantizers = getattr(cfg, 'num_quantizers', num_quantizers)
            codebook_size = getattr(cfg, 'codebook_size', codebook_size)
            codebook_dim = getattr(cfg, 'codebook_dim', codebook_dim)
            quantizer_type = getattr(cfg, 'quantizer_type', quantizer_type)
            quantizer_dropout = getattr(cfg, 'quantizer_dropout', quantizer_dropout)
            commitment = getattr(cfg, 'commitment', commitment)
            codebook_loss_weight = getattr(cfg, 'codebook_loss_weight', codebook_loss_weight)
            use_l2_normlize = getattr(cfg, 'use_l2_normlize', use_l2_normlize)
            codebook_type = getattr(cfg, 'codebook_type', codebook_type)
            kmeans_init = getattr(cfg, 'kmeans_init', kmeans_init)
            kmeans_iters = getattr(cfg, 'kmeans_iters', kmeans_iters)
            decay = getattr(cfg, 'decay', decay)
            eps = getattr(cfg, 'eps', eps)
            threshold_ema_dead_code = getattr(cfg, 'threshold_ema_dead_code', threshold_ema_dead_code)
            weight_init = getattr(cfg, 'weight_init', weight_init)
            use_vocos = getattr(cfg, 'use_vocos', use_vocos)
            vocos_dim = getattr(cfg, 'vocos_dim', vocos_dim)
            vocos_intermediate_dim = getattr(cfg, 'vocos_intermediate_dim', vocos_intermediate_dim)
            vocos_num_layers = getattr(cfg, 'vocos_num_layers', vocos_num_layers)
            n_fft = getattr(cfg, 'n_fft', n_fft)
            hop_size = getattr(cfg, 'hop_size', hop_size)
            padding = getattr(cfg, 'padding', padding)

        # Initialize quantizer based on type
        if quantizer_type == "fvq":
            self.quantizer = ResidualVQ(
                input_dim=in_channels,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_type=quantizer_type,
                quantizer_dropout=quantizer_dropout,
                commitment=commitment,
                codebook_loss_weight=codebook_loss_weight,
                use_l2_normlize=use_l2_normlize,
            )
        else:
            raise ValueError(f"Only 'fvq' quantizer type is supported, got {quantizer_type}")

        # Build decoder model
        if not use_vocos:
            # Traditional convolutional decoder
            channels = upsample_initial_channel
            layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]

            # Add upsampling + decoder blocks
            for i, stride in enumerate(up_ratios):
                input_dim = channels // 2**i
                output_dim = channels // 2 ** (i + 1)
                layers += [DecoderBlock(input_dim, output_dim, stride)]

            # Add final conv layer
            layers += [
                Snake1d(output_dim),
                WNConv1d(output_dim, 1, kernel_size=7, padding=3),
                nn.Tanh(),
            ]

            self.model = nn.Sequential(*layers)
        else:
            # Vocos-based decoder  
            self.model = VocosBackbone(
                input_channels=in_channels,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
                adanorm_num_embeddings=None,
            )

        self.reset_parameters()

    def forward(self, x=None, vq=False, eval_vq=False, n_quantizers=None):
        """
        Forward pass through decoder.
        
        Args:
            x (torch.Tensor): Input tensor
            vq (bool): If True, x is encoder output to be quantized. If False, x is quantized output to be decoded.
            eval_vq (bool): Whether to set quantizer to eval mode.
            n_quantizers (int, optional): Number of quantizers to use.
            
        Returns:
            If vq=True: Tuple of (quantized_out, all_indices, all_commit_losses, all_codebook_losses, all_quantized)
            If vq=False: Decoded audio output
        """
        if vq is True:
            if eval_vq:
                self.quantizer.eval()
            (
                quantized_out,
                all_indices,
                all_commit_losses,
                all_codebook_losses,
                all_quantized,
            ) = self.quantizer(x, n_quantizers=n_quantizers)
            return (
                quantized_out,
                all_indices,
                all_commit_losses,
                all_codebook_losses,
                all_quantized,
            )

        return self.model(x)

    def quantize(self, x, n_quantizers=None):
        """
        Quantize input features.
        
        Args:
            x (torch.Tensor): Input features to quantize
            n_quantizers (int, optional): Number of quantizers to use
            
        Returns:
            Tuple of (quantized_out, vq_indices)
        """
        self.quantizer.eval()
        quantized_out, vq, _, _, _ = self.quantizer(x, n_quantizers=n_quantizers)
        return quantized_out, vq

    def vq2emb(self, vq, n_quantizers=None):
        """
        Convert VQ indices to embeddings.
        
        Args:
            vq (torch.Tensor): VQ indices
            n_quantizers (int, optional): Number of quantizers used
            
        Returns:
            torch.Tensor: Embedding representations
        """
        return self.quantizer.vq2emb(vq, n_quantizers=n_quantizers)

    def decode(self, x):
        """
        Decode quantized features to audio.
        
        Args:
            x (torch.Tensor): Quantized features
            
        Returns:
            torch.Tensor: Decoded audio
        """
        return self.model(x)

    def latent2dist(self, x, n_quantizers=None):
        """
        Convert latents to distance distributions.
        
        Args:
            x (torch.Tensor): Latent features
            n_quantizers (int, optional): Number of quantizers to use
            
        Returns:
            Distance distributions
        """
        return self.quantizer.latent2dist(x, n_quantizers=n_quantizers)

    def reset_parameters(self):
        """Reset all parameters using the init_weights function."""
        self.apply(init_weights)