import numpy as np
import jax.numpy as jnp
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from typing import List, Dict, Union, Optional, Tuple
import warnings
import unicodedata
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Suppress phonemizer warnings that we're handling ourselves
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")

# Comprehensive IPA phoneme inventory - blank token must be first for CTC
IPA_PHONEMES = [
    # CTC blank token - MUST BE AT INDEX 0
    "<blank>",
    # Vowels - Close
    "i",
    "y",
    "ɨ",
    "ʉ",
    "ɯ",
    "u",
    # Vowels - Near-close
    "ɪ",
    "ʏ",
    "ʊ",
    # Vowels - Close-mid
    "e",
    "ø",
    "ɘ",
    "ɵ",
    "ɤ",
    "o",
    # Vowels - Mid
    "ə",
    # Vowels - Open-mid
    "ɛ",
    "œ",
    "ɜ",
    "ɞ",
    "ʌ",
    "ɔ",
    # Vowels - Near-open
    "æ",
    "ɐ",
    # Vowels - Open
    "a",
    "ɶ",
    "ä",
    "ɑ",
    "ɒ",
    # Consonants - Plosives
    "p",
    "b",
    "t",
    "d",
    "ʈ",
    "ɖ",
    "c",
    "ɟ",
    "k",
    "ɡ",
    "q",
    "ɢ",
    "ʔ",
    # Consonants - Nasals
    "m",
    "ɱ",
    "n",
    "ɳ",
    "ɲ",
    "ŋ",
    "ɴ",
    # Consonants - Trills
    "ʙ",
    "r",
    "ʀ",
    # Consonants - Taps/Flaps
    "ⱱ",
    "ɾ",
    "ɽ",
    # Consonants - Fricatives
    "ɸ",
    "β",
    "f",
    "v",
    "θ",
    "ð",
    "s",
    "z",
    "ʃ",
    "ʒ",
    "ʂ",
    "ʐ",
    "ç",
    "ʝ",
    "x",
    "ɣ",
    "χ",
    "ʁ",
    "ħ",
    "ʕ",
    "h",
    "ɦ",
    "ɕ",
    # Consonants - Lateral fricatives
    "ɬ",
    "ɮ",
    # Consonants - Approximants
    "ʋ",
    "ɹ",
    "ɻ",
    "j",
    "ɰ",
    "w",
    # Consonants - Lateral approximants
    "l",
    "ɭ",
    "ʎ",
    "ʟ",
    # Consonants - Affricates (common ones)
    "ts",
    "dz",
    "tʃ",
    "dʒ",
    "tɕ",
    "dʑ",
    "tʂ",
    "dʐ",
    "pf",  # German affricate
    "ks",  # German x-sound cluster
    # Suprasegmentals
    "ˈ",
    "ˌ",
    "ː",
    "ˑ",
    "|",
    "‖",
    # Tone numbers (for languages like Chinese)
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    # Diacritics and modifiers
    "̥",
    "̬",
    "̤",
    "̰",
    "̼",
    "̴",
    "̪",
    "̺",
    "̻",
    "̹",
    "̜",
    "̟",
    "̠",
    "̈",
    "̽",
    "̩",
    "̯",
    "˞",
    "̮",
    "̙",
    "̘",
    "̞",
    "̝",
    "̎",
    "̋",
    "́",
    "̄",
    "̀",
    "̏",
    "̌",
    "̂",
    "᷄",
    "᷅",
    "᷈",
    "̃",
    "̊",
    "ⁿ",
    "ˡ",
    "ˤ",
    "ˠ",
    "ʲ",
    "ʷ",
    "ᵝ",
    # Diphthongs (especially for German)
    "aɪ",  # German "ei", "ai"
    "aʊ",  # German "au"
    "ɔʏ",  # German "eu", "äu"
    "ɔɪ",  # Alternative for German "eu"
    "oʊ",  # English "go"
    "eɪ",  # English "day"
    "ɪə",  # English "ear"
    "eə",  # English "air"
    "ʊə",  # English "tour"
    # Other symbols and r-colored vowels
    "ɚ",
    "ɝ",
    "ɫ",
    "ɐ̯",  # German non-syllabic schwa (r-ending)
    # Separators and special
    " ",
    ".",
    "-",
    "_",
]

# Create mapping
PHONEME_TO_IDX = {ph: idx for idx, ph in enumerate(IPA_PHONEMES)}
IDX_TO_PHONEME = {idx: ph for ph, idx in PHONEME_TO_IDX.items()}

# CTC blank token index - MUST be 0
BLANK_IDX = 0
assert PHONEME_TO_IDX["<blank>"] == BLANK_IDX, "Blank token must be at index 0 for CTC"

# Language mapping
LANG_TO_ESPEAK = {
    "zh": "cmn",
    "en-us": "en-us",
    "en": "en-us",
    "ja": "ja",
    "fr": "fr",
    "de": "de",
    "ge": "de",  # Common typo
    "ko": "ko",
}


def get_espeak_lang(language: str) -> str:
    """Convert language code to espeak-ng format."""
    if language in LANG_TO_ESPEAK:
        return LANG_TO_ESPEAK[language]

    print(f"Warning: Unknown language '{language}', defaulting to English")
    return "en-us"


def text_to_phonemes(text: str, language: str = "en-us") -> str:
    """Convert text to IPA phonemes using phonemizer.
    
    Args:
        text: Text to convert to phonemes
        language: Language code
    """
    try:
        espeak_lang = get_espeak_lang(language)
        
        phonemes = phonemize(
            text,
            language=espeak_lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            with_stress=True,
            language_switch="remove-flags",
        )
        
        return phonemes
    except Exception as e:
        logger.error(f"Error phonemizing text in {language}: {e}")
        logger.error(f"  Text sample: '{text[:100]}...'")
        return ""


def phonemes_to_indices(phonemes: str) -> List[int]:
    """Convert phoneme string to list of indices.
    
    Args:
        phonemes: Phoneme string to convert
    """
    indices = []

    # Handle multi-character phonemes
    i = 0
    while i < len(phonemes):
        # Check for multi-character phonemes (up to 3 chars)
        found = False
        for length in [3, 2, 1]:
            if i + length <= len(phonemes):
                substr = phonemes[i : i + length]
                if substr in PHONEME_TO_IDX:
                    indices.append(PHONEME_TO_IDX[substr])
                    i += length
                    found = True
                    break

        if not found:
            indices.append(0) # <blank>
            i += 1

    return indices


def indices_to_phonemes(indices: List[int]) -> str:
    """Convert list of indices back to phoneme string."""
    phonemes = []
    for idx in indices:
        if idx in IDX_TO_PHONEME:
            phoneme = IDX_TO_PHONEME[idx]
            # Skip blank token in reconstruction
            if phoneme != "<blank>":
                phonemes.append(phoneme)
    return "".join(phonemes)


def batch_text_to_phonemes(texts: List[str], languages: List[str]) -> List[str]:
    """Convert batch of texts to phonemes efficiently using phonemizer's batch processing.
    
    Groups texts by language and processes each group in batch for optimal performance.
    
    Args:
        texts: List of text strings to convert
        languages: List of language codes corresponding to each text
        
    Returns:
        List of phoneme strings in the same order as input texts
    """
    if not texts:
        return []
    
    # Group texts by language with their original indices
    language_groups = {}
    for idx, (text, lang) in enumerate(zip(texts, languages)):
        espeak_lang = get_espeak_lang(lang)
        if espeak_lang not in language_groups:
            language_groups[espeak_lang] = []
        language_groups[espeak_lang].append((idx, text))
    
    # Prepare result list
    all_phonemes = [None] * len(texts)

    # Process each language group in batch
    for espeak_lang, items in language_groups.items():
        indices, batch_texts = zip(*items)
        
        try:
            # Process entire batch at once
            batch_phonemes = phonemize(
                list(batch_texts),  # Convert tuple to list
                language=espeak_lang,
                backend="espeak",
                strip=True,
                preserve_punctuation=False,
                with_stress=True,
                language_switch="remove-flags",
            )
            
            # Place results back in correct positions
            for idx, phonemes in zip(indices, batch_phonemes):
                all_phonemes[idx] = phonemes
                
        except Exception as e:
            logger.error(f"Error batch phonemizing texts in {espeak_lang}: {e}")
            # Fallback to empty strings for this batch
            for idx in indices:
                all_phonemes[idx] = ""
    
    return all_phonemes


def create_phoneme_array(
    indices: List[int], max_length: Optional[int] = None
) -> jnp.ndarray:
    """Create JAX array from phoneme indices with left-padding to match audio."""
    if max_length is None:
        max_length = len(indices)

    # Pad with BLANK_IDX (0) for CTC using LEFT padding
    padded = np.full(max_length, BLANK_IDX, dtype=np.int32)
    actual_length = min(len(indices), max_length)
    # Place valid indices at the END (left-padding)
    padded[max_length - actual_length :] = indices[:actual_length]

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
        mask[i, max_length - actual_length :] = 0.0

    return jnp.array(mask)


def batch_text_to_phoneme_arrays(
    texts: List[str], languages: List[str]
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
    # Use efficient batch processing
    phonemes_list = batch_text_to_phonemes(texts, languages)
    
    # Convert phonemes to indices
    all_indices = []
    lengths = []
    for phonemes in phonemes_list:
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
    phoneme_indices = (
        jnp.stack(phoneme_arrays)
        if phoneme_arrays
        else jnp.zeros((0, max_length), dtype=jnp.int32)
    )

    # Create mask
    phoneme_mask = create_phoneme_mask(all_indices, max_length)

    return phoneme_indices, phoneme_mask, lengths


# Vocabulary size for model configuration
PHONEME_VOCAB_SIZE = len(IPA_PHONEMES)
