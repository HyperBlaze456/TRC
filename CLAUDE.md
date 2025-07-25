# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research-focused JAX/Flax repository for developing new audio generative models, particularly focusing on audio tokenization and GANs. While model building is straightforward, the real challenges lie in proper testing and especially in training these models adequately. The codebase uses JAX 0.6.2 with CUDA support and Flax 0.10.7 (NNX API).

## Environment Setup

```bash
# Always work from this directory
cd /mnt/d/리서치/JAXLearn/TRC

# Activate the uv-managed Python environment
source /usr/programming/uv/machine_learning/bin/activate
```

## Development Guidelines

- **Working Directory**: Always expect to operate in `/mnt/d/리서치/JAXLearn/TRC`
  - We are currently working on alpha version of speech tokenizer.
- **Framework**: Claude is expected to not know the latest docs of Flax NNX or any other helper libraries like optax, orbax. Claude is absolutely recommended to search up the official documentations

## Project Structure
- **`/tokenizer/`**: Main audio tokenization model implementation
  - `alpha/`: Primary model version with EnCodec-like architecture.
    - `components/`: Core model components (encoder, decoder, quantizer, discriminator)
    - `losses/`: Contains loss functions that are required to train the model. It must be JAX/NNX JIT friendly, minimizing the usage of configuration that changes behavior conditionally and notating them with static_argnums/static_argnames if needed.
      - `discriminator_loss.py`: Losses for STFT, MPD, MSD specifically.
      - `generator_loss.py`: 
    - `mask_utils.py`: Left-padding and validation mask generation utilities
    - `model.py`: Model assembly
  - `utils/`: Shared utilities and building blocks
    - `data/`: Huggingface data loading modules here. Currently, each baches yield left-padded audio, padding valid mask for compressed and non-compressed representations and compressed case causal mask.
    - `attention.py`: Multi-head attention with RoPE support
    - `activation.py`: Custom activations (Snake activation)
    - `dit.py`: Diffusion Transformer (DiT) implementation
    - `embeddings.py`: Positional embeddings (RoPE)
    - `flow.py`: Flow-based modules
    - `mel.py`: Mel-spectrogram utilities
    - `norm.py`: Normalization layers (AdaRMSZero)
    - `metrics/`: Logging utilities (W&B, TensorBoard, JAX profiling)

- **`/HyperTTS/`**: Text-to-speech related experiments and possibly end-output resulted. Not accessed yet.

## Key Architecture Details

### Audio Tokenizer Alpha (`/tokenizer/alpha/`)
Audio tokenizer inspired by EnCodec and SoundStream:
- **Architecture**: Pure 1D convolution-based encoder/decoder for dimension reduction. `nnx.Conv` was just a simple int all times, ensuring it is indeed 1D conv.
- **Input/Output**: Accepts 24kHz or 48kHz raw waveforms having the shape of [B, N, 1](channels last convention), compresses to 50Hz with shape of [B, N, D] (potentially 25Hz, but not yet deployed)
- **RawEncoder**: Progressive downsampling through convolutions followed by transformer blocks with RoPE
- **Quantization**: Core component using RVQ (Residual Vector Quantization) with 2 different codebooks
  - First codebook: For each frame produced by encoder, this VQ would produce phonemes or <blank> token.
  - Implements both Binary Spherical Quantizer and Vector Quantizer options
- **RawDecoder**: Reconstructs high-quality audio from quantized representations

This model works like typical Descript Audio Codec, downsampling raw audio to latent frames and then involving quantizer, then returning it back using decoder. The Encoder and Decoder are fully convolutional, both of them mirrored.
For the Quantizer, the model usees two layer RVQ method. First VQ has relatively small size, because it serves the role of CTC-like phoneme inference for those frames. The residual would be handled by BSQ. VQ is focused on minimizing phoneme matching loss solely, while BSQ is focused on capturing all the residuals and minimizing divergence of original.

### DiT Model (`/tokenizer/utils/dit.py`)
- Implements Diffusion Transformer for denoising tasks
- Supports conditional generation with speaker embeddings
- Uses AdaRMSZero normalization and RoPE positional embeddings

### Core Design Patterns
- Heavy use of Flax NNX's module system (PyTorch-like interface)
- Snake activation functions for audio processing
- Rotary positional embeddings (RoPE) throughout transformer blocks
- Causal approaches(convolution, if adequate attention too) for temporal consistency and future streaming support

## Development Commands

```bash
# Ensure environment is activated
source /usr/programming/uv/machine_learning/bin/activate

# Run Jupyter notebooks for development
jupyter lab

# Training scripts (to be implemented)
python -m tokenizer.alpha.train
```

## Technical Notes

- Models use JAX's functional programming paradigm with Flax NNX for stateful components
- Attention implementations support both masked and causal attention patterns
- The codebase is GPU-optimized with CUDA 12 support
- Mixed use of convolutions for downsampling and transformers for sequence modeling

## Critical Development Challenges

1. **Training Infrastructure**: Implementing robust, distributed training loops in `tokenizer/alpha/train.py`
2. **Model Convergence**: Achieving stable training for GANs and ensuring audio quality metrics
3. **Testing Framework**: Building comprehensive test suites for model correctness and performance
4. **Production Deployment**: Optimizing inference speed and memory usage for real-world applications
5. **Loss Engineering**: Designing effective external loss functions for various tasks (reconstruction, adversarial, perceptual)