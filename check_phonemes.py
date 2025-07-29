#!/usr/bin/env python3
from phonemizer import phonemize
import warnings

warnings.filterwarnings("ignore")

# Test comprehensive texts for all supported languages
test_texts = {
    "zh": "你好世界，这是测试。一二三四五六七八九十。北京上海广州深圳。今天天气很好。",
    "en-us": "Hello world! The quick brown fox jumps over the lazy dog. How are you today? I think this is great.",
    "ja": "こんにちは世界。これはテストです。ありがとうございます。おはようございます。今日は良い天気です。",
    "fr": "Bonjour le monde! Comment allez-vous? Je vais bien, merci. C'est magnifique! Au revoir!",
    "de": "Hallo Welt! Wie geht es dir? Mir geht es gut, danke. Das ist großartig! Auf Wiedersehen!",
    "ko": "발이 아픕니다. 간장 공장 공장장은 끄냥 깐 공장장이고 별다를 게 있는지? 뷁어는 잘 분리됩니까? 쌍시읏은 어떱니까?",
}

# Collect all phonemes from actual phonemizer output
all_phonemes = set()
for lang, text in test_texts.items():
    espeak_lang = {
        "zh": "cmn",
        "en-us": "en-us",
        "ja": "ja",
        "fr": "fr-fr",
        "de": "de",
        "ko": "ko",
    }[lang]

    phonemes = phonemize(
        text,
        language=espeak_lang,
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=True,
        language_switch="remove-flags",
    )

    for char in phonemes:
        all_phonemes.add(char)

print(f"Total unique phonemes from actual usage: {len(all_phonemes)}")
print("\nChecking for Latin vs IPA confusions:")

# Check specific confusable pairs
confusable_pairs = [
    ("g", "ɡ"),  # Latin g vs IPA g
    ("a", "ɑ"),  # Latin a vs IPA open back unrounded vowel
    (":", "ː"),  # Colon vs IPA length mark
    ("'", "ˈ"),  # Apostrophe vs IPA primary stress
]

for latin, ipa in confusable_pairs:
    latin_found = latin in all_phonemes
    ipa_found = ipa in all_phonemes
    print(
        f"  {repr(latin)} (U+{ord(latin):04X}): {'Found' if latin_found else 'Not found'}"
    )
    print(f"  {repr(ipa)} (U+{ord(ipa):04X}): {'Found' if ipa_found else 'Not found'}")
    if latin_found and ipa_found:
        print(f"  WARNING: Both versions found!")
    print()

# Now check our phoneme list
from tokenizer.utils.data.phoneme_utils import IPA_PHONEMES

print("\nChecking for duplicates in IPA_PHONEMES list:")
seen = {}
duplicates = []
for i, ph in enumerate(IPA_PHONEMES):
    if ph in seen:
        duplicates.append((ph, seen[ph], i))
    seen[ph] = i

if duplicates:
    print("DUPLICATES FOUND:")
    for ph, first_idx, second_idx in duplicates:
        print(f"  {repr(ph)} at indices {first_idx} and {second_idx}")
else:
    print("No duplicates found.")

print(f"\nChecking which symbols in our list are never used:")
unused = []
for ph in IPA_PHONEMES:
    if ph not in all_phonemes and ph != "<blank>":
        unused.append(ph)

if unused:
    print(f"Unused symbols ({len(unused)}):")
    # Show which are Latin vs IPA
    for ph in unused[:30]:
        info = f"  {repr(ph)} (U+{ord(ph):04X})"
        if ph == "g":
            info += " <- Latin g (not used by phonemizer)"
        elif ph == ":":
            info += " <- ASCII colon (not used)"
        print(info)
    if len(unused) > 30:
        print(f"  ... and {len(unused) - 30} more")

print("\nSymbols actually used by phonemizer that are NOT in our list:")
missing = []
for ph in all_phonemes:
    if ph not in IPA_PHONEMES:
        missing.append(ph)

if missing:
    print(f"MISSING SYMBOLS ({len(missing)}):")
    for ph in missing:
        print(f"  {repr(ph)} (U+{ord(ph):04X})")
else:
    print("None - all phonemizer output symbols are in our list.")

print("\nPhonemes actually output (sorted):")
sorted_phonemes = sorted(list(all_phonemes))
for ph in sorted_phonemes:
    print(f"{repr(ph)} ", end="")
print()
