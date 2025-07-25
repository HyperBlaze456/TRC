import numpy as np
import jax.numpy as jnp
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from typing import List, Dict, Union, Optional, Tuple
import warnings

# Suppress phonemizer warnings
warnings.filterwarnings('ignore', category=UserWarning, module='phonemizer')

# Comprehensive IPA phoneme inventory - blank token must be first for CTC
IPA_PHONEMES = [
    # CTC blank token - MUST BE AT INDEX 0
    '<blank>',
    
    # Vowels - Close
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
    # Vowels - Near-close
    'ɪ', 'ʏ', 'ʊ',
    # Vowels - Close-mid
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
    # Vowels - Mid
    'ə',
    # Vowels - Open-mid
    'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
    # Vowels - Near-open
    'æ', 'ɐ',
    # Vowels - Open
    'a', 'ɶ', 'ä', 'ɑ', 'ɒ',
    
    # Consonants - Plosives
    'p', 'b', 't', 'd', 'ʈ', 'ɖ', 'c', 'ɟ', 'k', 'ɡ', 'q', 'ɢ', 'ʔ',
    # Consonants - Nasals
    'm', 'ɱ', 'n', 'ɳ', 'ɲ', 'ŋ', 'ɴ',
    # Consonants - Trills
    'ʙ', 'r', 'ʀ',
    # Consonants - Taps/Flaps
    'ⱱ', 'ɾ', 'ɽ',
    # Consonants - Fricatives
    'ɸ', 'β', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'ʂ', 'ʐ', 'ç', 'ʝ',
    'x', 'ɣ', 'χ', 'ʁ', 'ħ', 'ʕ', 'h', 'ɦ', 'ɕ',
    # Consonants - Lateral fricatives
    'ɬ', 'ɮ',
    # Consonants - Approximants
    'ʋ', 'ɹ', 'ɻ', 'j', 'ɰ', 'w',
    # Consonants - Lateral approximants
    'l', 'ɭ', 'ʎ', 'ʟ',
    # Consonants - Affricates (common ones)
    'ts', 'dz', 'tʃ', 'dʒ', 'tɕ', 'dʑ', 'tʂ', 'dʐ',
    
    # Suprasegmentals
    'ˈ', 'ˌ', 'ː', 'ˑ', '|', '‖',
    
    # Tone numbers (for languages like Chinese)
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    
    # Diacritics and modifiers
    '̥', '̬', '̤', '̰', '̼', '̴', '̪', '̺', '̻', '̹', '̜', '̟', '̠', '̈', '̽',
    '̩', '̯', '˞', '̮', '̙', '̘', '̞', '̝', '̎', '̋', '́', '̄', '̀', '̏', '̌', '̂',
    '᷄', '᷅', '᷈', '̃', '̊', 'ⁿ', 'ˡ', 'ˤ', 'ˠ', 'ʲ', 'ʷ', 'ᵝ',
    
    # Other symbols
    'ɚ', 'ɝ', 'ɫ',
    
    # Separators and special
    ' ', '.', '-', '_',
]

# Create mapping
PHONEME_TO_IDX = {ph: idx for idx, ph in enumerate(IPA_PHONEMES)}
IDX_TO_PHONEME = {idx: ph for ph, idx in PHONEME_TO_IDX.items()}

# CTC blank token index - MUST be 0
BLANK_IDX = 0
assert PHONEME_TO_IDX['<blank>'] == BLANK_IDX, "Blank token must be at index 0 for CTC"

# Language mapping
LANG_TO_ESPEAK = {
    'zh': 'cmn',
    'en-us': 'en-us',
    'en': 'en-us',
    'ja': 'ja',
    'fr': 'fr',
    'de': 'de',
    'ge': 'de',  # Common typo
    'ko': 'ko',
}

def get_espeak_lang(language: str) -> str:
    """Convert language code to espeak-ng format."""
    if language in LANG_TO_ESPEAK:
        return LANG_TO_ESPEAK[language]
    
    print(f"Warning: Unknown language '{language}', defaulting to English")
    return 'en-us'

def text_to_phonemes(text: str, language: str = 'en-us') -> str:
    """Convert text to IPA phonemes using phonemizer."""
    try:
        espeak_lang = get_espeak_lang(language)
        
        phonemes = phonemize(
            text,
            language=espeak_lang,
            backend='espeak',
            strip=True,
            preserve_punctuation=False,
            with_stress=True,
            language_switch='remove-flags'
        )
        
        return phonemes
    except Exception as e:
        print(f"Error phonemizing text: {e}")
        return ""

def phonemes_to_indices(phonemes: str) -> List[int]:
    """Convert phoneme string to list of indices (no special tokens for CTC)."""
    indices = []
    
    # Handle multi-character phonemes
    i = 0
    while i < len(phonemes):
        # Check for multi-character phonemes (up to 3 chars)
        found = False
        for length in [3, 2, 1]:
            if i + length <= len(phonemes):
                substr = phonemes[i:i+length]
                if substr in PHONEME_TO_IDX:
                    indices.append(PHONEME_TO_IDX[substr])
                    i += length
                    found = True
                    break
        
        if not found:
            # Skip unknown characters - no UNK token for CTC
            print(f"Warning: Unknown phoneme character '{phonemes[i]}' - skipping")
            i += 1
    
    return indices

def indices_to_phonemes(indices: List[int]) -> str:
    """Convert list of indices back to phoneme string."""
    phonemes = []
    for idx in indices:
        if idx in IDX_TO_PHONEME:
            phoneme = IDX_TO_PHONEME[idx]
            # Skip blank token in reconstruction
            if phoneme != '<blank>':
                phonemes.append(phoneme)
    return ''.join(phonemes)

def create_phoneme_array(indices: List[int], max_length: Optional[int] = None) -> jnp.ndarray:
    """Create JAX array from phoneme indices with left-padding to match audio."""
    if max_length is None:
        max_length = len(indices)
    
    # Pad with BLANK_IDX (0) for CTC using LEFT padding
    padded = np.full(max_length, BLANK_IDX, dtype=np.int32)
    actual_length = min(len(indices), max_length)
    # Place valid indices at the END (left-padding)
    padded[max_length - actual_length:] = indices[:actual_length]
    
    return jnp.array(padded)

def create_phoneme_mask(indices_list: List[List[int]], max_length: int) -> jnp.ndarray:
    """Create CTC-compatible mask for left-padded phoneme sequences.
    
    Returns mask where 1.0 = padded position, 0.0 = valid position (CTC convention).
    """
    batch_size = len(indices_list)
    # Start with all 1s (all padded)
    mask = np.ones((batch_size, max_length), dtype=np.float32)
    
    for i, indices in enumerate(indices_list):
        actual_length = min(len(indices), max_length)
        # Mark valid positions at the END as 0.0 (left-padding)
        mask[i, max_length - actual_length:] = 0.0
    
    return jnp.array(mask)

def batch_text_to_phoneme_arrays(
    texts: List[str], 
    languages: List[str]
) -> Tuple[jnp.ndarray, jnp.ndarray, List[int]]:
    """Convert batch of texts to phoneme arrays with masks.
    
    Args:
        texts: List of text strings
        languages: List of language codes
        
    Returns:
        phoneme_indices: JAX array of shape [B, L] with phoneme indices
        phoneme_mask: JAX array of shape [B, L] with CTC padding (1.0 = padded, 0.0 = valid)
        lengths: List of actual sequence lengths
    """
    all_indices = []
    lengths = []
    
    for text, lang in zip(texts, languages):
        phonemes = text_to_phonemes(text, lang)
        indices = phonemes_to_indices(phonemes)
        all_indices.append(indices)
        lengths.append(len(indices))
    
    # Find max length
    max_length = max(lengths) if lengths else 1
    
    # Create padded arrays
    phoneme_arrays = []
    for indices in all_indices:
        arr = create_phoneme_array(indices, max_length)
        phoneme_arrays.append(arr)
    
    # Stack into batch
    phoneme_indices = jnp.stack(phoneme_arrays) if phoneme_arrays else jnp.zeros((0, max_length), dtype=jnp.int32)
    
    # Create mask
    phoneme_mask = create_phoneme_mask(all_indices, max_length)
    
    return phoneme_indices, phoneme_mask, lengths

# Vocabulary size for model configuration
PHONEME_VOCAB_SIZE = len(IPA_PHONEMES)