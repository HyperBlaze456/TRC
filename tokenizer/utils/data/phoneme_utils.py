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


def text_to_phonemes(text: str, language: str = "en-us", debug: bool = False) -> str:
    """Convert text to IPA phonemes using phonemizer with better debugging."""
    try:
        espeak_lang = get_espeak_lang(language)
        
        if debug:
            logger.info(f"Phonemizing text in {espeak_lang}: '{text[:50]}...'")
            # Log character composition
            for i, char in enumerate(text[:20]):  # Check first 20 chars
                logger.debug(f"  Char {i}: '{char}' (U+{ord(char):04X}) - {unicodedata.name(char, 'UNKNOWN')}")

        phonemes = phonemize(
            text,
            language=espeak_lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            with_stress=True,
            language_switch="remove-flags",
        )
        
        if debug:
            logger.info(f"Result phonemes: '{phonemes[:50]}...'")
            # Check for replacement characters in result
            for i, char in enumerate(phonemes[:50]):
                if char == '?':
                    logger.warning(f"  Replacement character '?' at position {i} - phonemizer couldn't process input")

        return phonemes
    except Exception as e:
        logger.error(f"Error phonemizing text in {language}: {e}")
        logger.error(f"  Text sample: '{text[:100]}...'")
        return ""


def phonemes_to_indices(phonemes: str, debug: bool = False) -> List[int]:
    """Convert phoneme string to list of indices with detailed debugging."""
    indices = []
    unknown_chars = set()
    unknown_count = 0

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
            char = phonemes[i]
            unknown_chars.add(char)
            unknown_count += 1
            
            # Detailed logging for unknown characters
            if char == '?':
                # This is likely a replacement character from phonemizer
                logger.warning(f"Found '?' at position {i} - this usually means phonemizer couldn't process a character")
            else:
                # Log the character details
                char_info = f"'{char}' (U+{ord(char):04X})"
                try:
                    char_name = unicodedata.name(char, 'UNKNOWN')
                    char_info += f" - {char_name}"
                except:
                    pass
                
                if debug or unknown_count <= 5:  # Log first 5 unknowns always
                    logger.warning(f"Unknown phoneme at position {i}: {char_info}")
                    # Show context
                    context_start = max(0, i - 5)
                    context_end = min(len(phonemes), i + 6)
                    context = phonemes[context_start:context_end]
                    logger.warning(f"  Context: '{context}' (position {i - context_start} in context)")
            
            i += 1
    
    if unknown_chars:
        logger.warning(f"Total unknown characters: {unknown_count}")
        logger.warning(f"Unique unknown characters: {unknown_chars}")
        for char in unknown_chars:
            try:
                logger.warning(f"  - '{char}' (U+{ord(char):04X}) - {unicodedata.name(char, 'UNKNOWN')}")
            except:
                logger.warning(f"  - '{char}' (U+{ord(char):04X})")

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
    texts: List[str], languages: List[str], debug: bool = False
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

    for idx, (text, lang) in enumerate(zip(texts, languages)):
        if debug and idx == 0:  # Debug first item in batch
            logger.info(f"\nProcessing item {idx} in language '{lang}':")
            logger.info(f"  Original text: '{text[:100]}...'")
        
        phonemes = text_to_phonemes(text, lang, debug=(debug and idx == 0))
        indices = phonemes_to_indices(phonemes, debug=(debug and idx == 0))
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
