#!/usr/bin/env python3
"""Test script to diagnose phonemization issues with German text."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer.utils.data.phoneme_utils import (
    text_to_phonemes, 
    phonemes_to_indices,
    PHONEME_TO_IDX,
    IPA_PHONEMES
)
import unicodedata
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_text(text, language="de"):
    """Analyze text for phonemization issues."""
    print(f"\n{'='*60}")
    print(f"Analyzing text in {language}:")
    print(f"Text: '{text}'")
    print(f"Length: {len(text)} characters")
    print(f"{'='*60}\n")
    
    # Analyze each character
    print("Character analysis:")
    for i, char in enumerate(text):
        try:
            char_name = unicodedata.name(char)
            category = unicodedata.category(char)
        except:
            char_name = "UNKNOWN"
            category = "?"
        
        print(f"  [{i:3d}] '{char}' U+{ord(char):04X} ({char_name}) Category: {category}")
    
    print("\nPhoneme conversion:")
    phonemes = text_to_phonemes(text, language, debug=True)
    print(f"\nResulting phonemes: '{phonemes}'")
    print(f"Phoneme length: {len(phonemes)} characters")
    
    print("\nPhoneme character analysis:")
    for i, char in enumerate(phonemes):
        if char in PHONEME_TO_IDX:
            print(f"  [{i:3d}] '{char}' U+{ord(char):04X} - ✓ Known phoneme (idx: {PHONEME_TO_IDX[char]})")
        else:
            try:
                char_name = unicodedata.name(char)
            except:
                char_name = "UNKNOWN"
            print(f"  [{i:3d}] '{char}' U+{ord(char):04X} ({char_name}) - ✗ UNKNOWN")
    
    print("\nConverting to indices:")
    indices = phonemes_to_indices(phonemes, debug=True)
    print(f"\nFinal indices: {indices[:50]}..." if len(indices) > 50 else f"\nFinal indices: {indices}")
    print(f"Number of indices: {len(indices)}")
    
    return phonemes, indices

def test_common_german_issues():
    """Test common problematic German text patterns."""
    
    test_cases = [
        # Basic German
        ("Hallo, wie geht es dir?", "de"),
        
        # German umlauts
        ("Über schöne Mädchen und große Häuser", "de"),
        
        # German eszett
        ("Die Straße ist groß", "de"),
        
        # Mixed case and punctuation
        ("Das ist SUPER! Wirklich toll...", "de"),
        
        # Numbers and special characters
        ("Ich habe 100€ dabei", "de"),
        
        # Potential language switch (English word in German text)
        ("Das ist ein Computer Problem", "de"),
        
        # Long compound word (typical in German)
        ("Donaudampfschifffahrtsgesellschaftskapitän", "de"),
    ]
    
    for text, lang in test_cases:
        analyze_text(text, lang)
        print("\n" + "="*60 + "\n")

def check_missing_phonemes():
    """Check which common German phonemes might be missing from our inventory."""
    
    # Common German-specific IPA phonemes
    german_phonemes = [
        'ʏ', 'ø', 'œ', 'ɐ', 'ʁ', 'ç', 'x', 'ʔ',  # German-specific sounds
        'pf', 'ts', 'ks',  # German affricates
        'aɪ', 'aʊ', 'ɔʏ',  # German diphthongs
        'ɐ̯',  # Non-syllabic schwa (German r-ending)
    ]
    
    print("\nChecking German-specific phonemes in our inventory:")
    missing = []
    for ph in german_phonemes:
        if ph in PHONEME_TO_IDX:
            print(f"  ✓ '{ph}' - Present (idx: {PHONEME_TO_IDX[ph]})")
        else:
            print(f"  ✗ '{ph}' - MISSING")
            missing.append(ph)
    
    if missing:
        print(f"\nMissing phonemes: {missing}")
        print("These should be added to IPA_PHONEMES list")
    
    return missing

if __name__ == "__main__":
    print("Testing German phonemization issues...")
    print("="*60)
    
    # Check our phoneme inventory
    print(f"\nCurrent phoneme inventory size: {len(IPA_PHONEMES)}")
    print(f"Blank token index: {PHONEME_TO_IDX.get('<blank>', 'NOT FOUND')}")
    
    # Check for missing German phonemes
    missing = check_missing_phonemes()
    
    # Test common German text patterns
    print("\n" + "="*60)
    print("Testing common German text patterns:")
    print("="*60)
    test_common_german_issues()
    
    print("\nDiagnostic complete!")
    print(f"If you see '?' characters, it means phonemizer couldn't handle certain inputs.")
    print(f"Missing phonemes from inventory: {len(missing)}")