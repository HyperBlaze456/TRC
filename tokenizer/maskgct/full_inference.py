#!/usr/bin/env python3
"""
Unified inference example for complete MaskGCT pipeline
Demonstrates: Audio → Semantic Tokens → S2A → Acoustic Tokens → Reconstructed Audio
"""

import torch
import librosa
import soundfile as sf
import argparse
from pathlib import Path

# Import all components
from .semantic_utils import build_semantic_model, build_semantic_codec, SemanticTokenizer
from .acoustic.codec import CodecEncoder, CodecDecoder
from .s2a.maskgct_s2a import MaskGCT_S2A
from .config import MaskGCTConfig, get_default_config


class MaskGCTInference:
    """Complete MaskGCT inference pipeline"""
    
    def __init__(self, config: MaskGCTConfig, device="cuda"):
        self.config = config
        self.device = device
        
        print("Loading models...")
        
        # Build semantic tokenizer
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(device)
        self.semantic_codec = build_semantic_codec(config.semantic_codec, device)
        self.semantic_tokenizer = SemanticTokenizer(
            self.semantic_model,
            self.semantic_codec,
            self.semantic_mean,
            self.semantic_std,
            device
        )
        
        # Build acoustic codec
        self.acoustic_encoder = CodecEncoder(
            d_model=config.acoustic_encoder.d_model,
            up_ratios=config.acoustic_encoder.up_ratios,
            out_channels=config.acoustic_encoder.out_channels,
            use_tanh=config.acoustic_encoder.use_tanh
        ).to(device)
        
        self.acoustic_decoder = CodecDecoder(
            in_channels=config.acoustic_decoder.in_channels,
            upsample_initial_channel=config.acoustic_decoder.upsample_initial_channel,
            up_ratios=config.acoustic_decoder.up_ratios,
            vq_num_q_c=config.acoustic_decoder.vq_num_q_c,
            vq_dim=config.acoustic_decoder.vq_dim,
            codebook_dim=config.acoustic_decoder.codebook_dim,
            codebook_size=config.acoustic_decoder.codebook_size,
            quantizer_type=config.acoustic_decoder.quantizer_type,
            use_vocos=config.acoustic_decoder.use_vocos,
            vocos_dim=config.acoustic_decoder.vocos_dim,
            vocos_intermediate_dim=config.acoustic_decoder.vocos_intermediate_dim,
            vocos_num_layers=config.acoustic_decoder.vocos_num_layers
        ).to(device)
        
        # Build S2A model
        self.s2a_model = MaskGCT_S2A(
            num_quantizer=config.s2a_model.num_quantizer,
            hidden_size=config.s2a_model.hidden_size,
            num_layers=config.s2a_model.num_layers,
            num_heads=config.s2a_model.num_heads,
            codebook_size=config.s2a_model.codebook_size,
            cfg_scale=config.s2a_model.cfg_scale,
            mask_layer_schedule=config.s2a_model.mask_layer_schedule,
            cond_codebook_size=config.s2a_model.cond_codebook_size,
            cond_dim=config.s2a_model.cond_dim,
            predict_layer_1=config.s2a_model.predict_layer_1
        ).to(device)
        
        self.acoustic_encoder.eval()
        self.acoustic_decoder.eval()
        self.s2a_model.eval()
        
        print("All models loaded successfully!")
    
    @torch.no_grad()
    def audio_to_semantic_tokens(self, audio, sample_rate=16000):
        """Convert audio to semantic tokens"""
        print("Extracting semantic tokens...")
        semantic_tokens = self.semantic_tokenizer.tokenize(audio, sampling_rate=sample_rate)
        return semantic_tokens
    
    @torch.no_grad()
    def audio_to_acoustic_tokens(self, audio, sample_rate=24000):
        """Convert audio to acoustic tokens for prompt/reference"""
        print("Extracting acoustic tokens...")
        
        # Ensure audio is tensor with correct shape
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)  # Add batch dim
        
        audio = audio.to(self.device)
        
        # Encode and quantize
        encoded = self.acoustic_encoder(audio)
        _, indices, _, _, _ = self.acoustic_decoder(encoded, vq=True)
        
        return indices  # Shape: (num_quantizers, batch, time)
    
    @torch.no_grad()
    def semantic_to_acoustic(self, semantic_tokens, prompt_acoustic_tokens=None,
                           n_timesteps=10, temperature=1.0, cfg_scale=1.0):
        """Convert semantic tokens to acoustic tokens using S2A model"""
        print("Generating acoustic tokens from semantic tokens...")
        
        # Prepare inputs
        if semantic_tokens.dim() == 2:
            semantic_tokens = semantic_tokens.unsqueeze(0)  # Add layer dim
        
        # Generate acoustic tokens
        if prompt_acoustic_tokens is not None:
            # With prompt (continuation)
            acoustic_tokens = self.s2a_model.reverse_diffusion(
                semantic_tokens,
                prompt_acoustic_tokens,
                n_timesteps=n_timesteps,
                temperature=temperature,
                cfg_scale=cfg_scale
            )
        else:
            # Without prompt (from scratch)
            batch_size = semantic_tokens.shape[1]
            seq_len = semantic_tokens.shape[2]
            
            # Initialize with mask tokens
            acoustic_tokens = torch.full(
                (self.config.s2a_model.num_quantizer, batch_size, seq_len),
                self.config.s2a_model.codebook_size,  # mask token id
                device=self.device
            )
            
            acoustic_tokens = self.s2a_model.reverse_diffusion(
                semantic_tokens,
                acoustic_tokens,
                n_timesteps=n_timesteps,
                temperature=temperature,
                cfg_scale=cfg_scale
            )
        
        return acoustic_tokens
    
    @torch.no_grad()
    def acoustic_tokens_to_audio(self, acoustic_tokens):
        """Convert acoustic tokens back to audio"""
        print("Decoding acoustic tokens to audio...")
        
        # Convert tokens to embeddings
        audio = self.acoustic_decoder.vq2emb(acoustic_tokens)
        
        # Decode to waveform
        audio = self.acoustic_decoder(audio)
        
        return audio.squeeze().cpu().numpy()
    
    def full_pipeline(self, input_audio_path, output_audio_path,
                     prompt_audio_path=None, sample_rate=24000):
        """Run complete pipeline"""
        
        # Load input audio
        print(f"Loading input audio from {input_audio_path}...")
        audio, sr = librosa.load(input_audio_path, sr=16000)  # Semantic uses 16kHz
        
        # Extract semantic tokens
        semantic_tokens = self.audio_to_semantic_tokens(audio, sample_rate=16000)
        print(f"Semantic tokens shape: {semantic_tokens.shape}")
        
        # Load prompt if provided
        prompt_acoustic = None
        if prompt_audio_path:
            print(f"Loading prompt audio from {prompt_audio_path}...")
            prompt_audio, _ = librosa.load(prompt_audio_path, sr=sample_rate)
            prompt_acoustic = self.audio_to_acoustic_tokens(prompt_audio, sample_rate)
            print(f"Prompt acoustic tokens shape: {prompt_acoustic.shape}")
        
        # Generate acoustic tokens
        acoustic_tokens = self.semantic_to_acoustic(
            semantic_tokens,
            prompt_acoustic_tokens=prompt_acoustic,
            n_timesteps=10,
            temperature=1.0,
            cfg_scale=1.0
        )
        print(f"Generated acoustic tokens shape: {acoustic_tokens.shape}")
        
        # Decode to audio
        reconstructed_audio = self.acoustic_tokens_to_audio(acoustic_tokens)
        
        # Save output
        print(f"Saving output audio to {output_audio_path}...")
        sf.write(output_audio_path, reconstructed_audio, sample_rate)
        
        return reconstructed_audio


def main():
    parser = argparse.ArgumentParser(description="MaskGCT Full Inference Pipeline")
    parser.add_argument("input_audio", type=str, help="Path to input audio file")
    parser.add_argument("output_audio", type=str, help="Path to output audio file")
    parser.add_argument("--prompt_audio", type=str, default=None,
                       help="Path to prompt audio for continuation")
    parser.add_argument("--config", type=str, default="default",
                       choices=["default", "small", "large"],
                       help="Model configuration to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--sample_rate", type=int, default=24000,
                       help="Sample rate for audio processing")
    
    args = parser.parse_args()
    
    # Get configuration
    if args.config == "small":
        from config import get_small_config
        config = get_small_config()
    elif args.config == "large":
        from config import get_large_config
        config = get_large_config()
    else:
        config = get_default_config()
    
    # Create inference pipeline
    pipeline = MaskGCTInference(config, device=args.device)
    
    # Run inference
    pipeline.full_pipeline(
        args.input_audio,
        args.output_audio,
        prompt_audio_path=args.prompt_audio,
        sample_rate=args.sample_rate
    )
    
    print("Inference complete!")


if __name__ == "__main__":
    import numpy as np
    main()