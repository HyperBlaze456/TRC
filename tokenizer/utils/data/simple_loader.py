import os
import sys

# Add the TRC directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Audio
from dataclasses import dataclass
import warnings

from tokenizer.alpha.mask_utils import pad_sequences_left, create_padding_mask, create_encoder_masks

warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Set token globally for HuggingFace
if hf_token:
    print(f"Setting HF token (first 8 chars): {hf_token[:8]}...")
    try:
        login(token=hf_token)
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("HF token set successfully")
    except Exception as e:
        print(f"Warning: Failed to login with HF token: {e}")

@dataclass
class AudioConfig:
    """Configuration for simple audio loading."""
    dataset_name: str = "amphion/Emilia-Dataset"
    dataset_config: str = "default"
    split: str = "train"
    sample_rate: int = 24000
    batch_size: int = 16
    unified: float = 30.0 # later on, this will serve efficient packing, but this is too complicated ngl
    streaming: bool = True

def extract_text_language(batch):
    """Extract language and text from json column."""
    return {
        'language': batch['json']['language'],
        'text': batch['json']['text']
    }

def process_audio_batch(batch):
    """Process audio batch with left-padding and create masks.

    Args:
        batch: Dataset batch containing 'mp3' audio data and text/language fields

    Returns:
        dict: Processed batch with padded audio and masks
    """
    # Extract audio arrays from the batch
    # batch['mp3'] is a list of dicts with 'array' key
    print(len(batch['mp3']))
    print(batch['mp3'][0])
    audio_arrays = [item['array'] for item in batch['mp3']]

    # Add channel dimension if not present (assume mono audio)
    audio_arrays = [
        arr[:, None] if arr.ndim == 1 else arr
        for arr in audio_arrays
    ]

    # Pad sequences with left-padding
    padded_audio, lengths = pad_sequences_left(audio_arrays)
    max_length = padded_audio.shape[1]

    # Create padding mask (non-causal)
    padding_mask = create_padding_mask(
        lengths=lengths,
        max_length=max_length,
        causal=False
    )

    # Create encoder masks (both non-causal and causal)
    # Assuming 480x downsampling (24kHz -> 50Hz)
    downsample_factor = 480
    encoder_mask, encoder_causal_mask = create_encoder_masks(
        lengths=lengths,
        max_length=max_length,
        downsample_factor=downsample_factor
    )

    # Return processed batch
    return {
        'audio': padded_audio,  # [B, T, C]
        'lengths': lengths,  # [B]
        'padding_mask': padding_mask,  # [B, 1, 1, T]
        'encoder_mask': encoder_mask,  # [B, 1, 1, T']
        'encoder_causal_mask': encoder_causal_mask,  # [B, 1, T', T']
        'language': batch['language'],
        'text': batch['text']
    }

def create_emilia_ds(config: AudioConfig):
    dataset = load_dataset(
        config.dataset_name,
        split=config.split,
        streaming=config.streaming,
    )
    dataset = dataset.cast_column(
            "mp3",
            Audio(sampling_rate=config.sample_rate)
        )
    # Apply preprocessing to extract language and text
    dataset = dataset.map(extract_text_language)
    # Batch the dataset
    dataset = dataset.batch(config.batch_size)
    # Apply audio processing with padding and masks
    dataset = dataset.map(process_audio_batch)
    dataset = dataset.with_format("jax")
    return dataset

config = AudioConfig(
    dataset_name="amphion/Emilia-Dataset",
    dataset_config="default",
    split="train",
    sample_rate=24000,
    batch_size=32,
    streaming=True,
)