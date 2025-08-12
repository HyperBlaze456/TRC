"""
Configuration classes for MaskGCT components
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SemanticCodecConfig:
    """Configuration for semantic tokenizer (RepCodec)"""
    codebook_size: int = 8192
    hidden_size: int = 1024
    codebook_dim: int = 8
    vocos_dim: int = 384
    vocos_intermediate_dim: int = 2048
    vocos_num_layers: int = 12
    num_quantizers: int = 1
    downsample_scale: int = 1


@dataclass
class AcousticEncoderConfig:
    """Configuration for acoustic encoder"""
    d_model: int = 96
    up_ratios: List[int] = None
    out_channels: int = 256
    use_tanh: bool = False
    
    def __post_init__(self):
        if self.up_ratios is None:
            self.up_ratios = [3, 4, 5, 8]


@dataclass
class AcousticDecoderConfig:
    """Configuration for acoustic decoder"""
    in_channels: int = 256
    upsample_initial_channel: int = 1024
    up_ratios: List[int] = None
    vq_num_q_c: int = 12  # num_quantizers
    vq_num_q_p: int = 1
    vq_num_q_r: int = 3
    vq_dim: int = 256
    codebook_dim: int = 8
    codebook_size: int = 1024
    quantizer_type: str = "fvq"
    quantizer_dropout: float = 0.0
    commitment: float = 0.15
    codebook_loss_weight: float = 1.0
    use_l2_normlize: bool = True
    use_vocos: bool = True
    vocos_dim: int = 512
    vocos_intermediate_dim: int = 2048
    vocos_num_layers: int = 30
    
    def __post_init__(self):
        if self.up_ratios is None:
            self.up_ratios = [5, 5, 4, 2]


@dataclass
class S2AModelConfig:
    """Configuration for Semantic-to-Acoustic model"""
    num_quantizer: int = 12
    hidden_size: int = 1024
    num_layers: int = 16
    num_heads: int = 16
    codebook_size: int = 1024
    cfg_scale: float = 0.15
    mask_layer_schedule: str = "linear"
    cond_codebook_size: int = 8192
    cond_dim: int = 1024
    predict_layer_1: bool = False


@dataclass
class MaskGCTConfig:
    """Complete configuration for MaskGCT system"""
    semantic_codec: SemanticCodecConfig = None
    acoustic_encoder: AcousticEncoderConfig = None
    acoustic_decoder: AcousticDecoderConfig = None
    s2a_model: S2AModelConfig = None
    
    def __post_init__(self):
        if self.semantic_codec is None:
            self.semantic_codec = SemanticCodecConfig()
        if self.acoustic_encoder is None:
            self.acoustic_encoder = AcousticEncoderConfig()
        if self.acoustic_decoder is None:
            self.acoustic_decoder = AcousticDecoderConfig()
        if self.s2a_model is None:
            self.s2a_model = S2AModelConfig()
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        semantic_cfg = SemanticCodecConfig(**config_dict.get("semantic_codec", {}))
        encoder_cfg = AcousticEncoderConfig(**config_dict.get("acoustic_encoder", {}))
        decoder_cfg = AcousticDecoderConfig(**config_dict.get("acoustic_decoder", {}))
        s2a_cfg = S2AModelConfig(**config_dict.get("s2a_model", {}))
        
        return cls(
            semantic_codec=semantic_cfg,
            acoustic_encoder=encoder_cfg,
            acoustic_decoder=decoder_cfg,
            s2a_model=s2a_cfg
        )
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "semantic_codec": self.semantic_codec.__dict__,
            "acoustic_encoder": self.acoustic_encoder.__dict__,
            "acoustic_decoder": self.acoustic_decoder.__dict__,
            "s2a_model": self.s2a_model.__dict__
        }


# Default configurations
def get_default_config():
    """Get default MaskGCT configuration"""
    return MaskGCTConfig()


def get_small_config():
    """Get small model configuration for testing"""
    return MaskGCTConfig(
        semantic_codec=SemanticCodecConfig(
            codebook_size=4096,
            vocos_num_layers=6
        ),
        acoustic_encoder=AcousticEncoderConfig(
            d_model=64,
            up_ratios=[2, 4, 4, 8]
        ),
        acoustic_decoder=AcousticDecoderConfig(
            vq_num_q_c=8,
            vocos_num_layers=12
        ),
        s2a_model=S2AModelConfig(
            num_quantizer=8,
            hidden_size=512,
            num_layers=8,
            num_heads=8
        )
    )


def get_large_config():
    """Get large model configuration"""
    return MaskGCTConfig(
        semantic_codec=SemanticCodecConfig(
            codebook_size=16384,
            hidden_size=1536,
            vocos_num_layers=18
        ),
        acoustic_encoder=AcousticEncoderConfig(
            d_model=128,
            out_channels=512
        ),
        acoustic_decoder=AcousticDecoderConfig(
            vq_num_q_c=16,
            vocos_num_layers=40,
            vocos_dim=768
        ),
        s2a_model=S2AModelConfig(
            num_quantizer=16,
            hidden_size=1536,
            num_layers=24,
            num_heads=24,
            cond_codebook_size=16384
        )
    )