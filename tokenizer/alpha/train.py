import sys
import os
from dotenv import load_dotenv
from huggingface_hub import login
# Add the TRC directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Dict, Any, Tuple, Optional
import time
from functools import partial
from dataclasses import dataclass

# Import our modules
from tokenizer.alpha.model import AudioTokenizer
from tokenizer.alpha.loss import (
    compute_generator_loss,
    compute_discriminator_loss,
    create_phoneme_vocabulary,
    extract_encoder_lengths
)
from tokenizer.alpha.components.discriminators import (
    MultiScaleDiscriminator,
    MultiPeriodDiscriminator,
    STFTDiscriminator
)

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
else:
    print("Warning: No HF_TOKEN found in environment")

# Try to import phonemizer for multilingual G2P
try:
    from phonemizer import phonemize

    PHONEMIZER_AVAILABLE = True
    print("Phonemizer available for multilingual G2P")
except ImportError:
    PHONEMIZER_AVAILABLE = False
    print("Warning: phonemizer not available. Using dummy phoneme targets.")

possible_phonemes = [ # Emilia has 'zh', 'en-us', 'ja', 'fr', 'ge', 'ko']
        "N", "a", "ai", "an", "au", "aı", "aŋ", "aɔ", "aː", "b", "bʲ", "d", "dz", "dʑ", "dʒ", "d͡ʑ",
        "e", "ei", "eɪ", "eː", "f", "g", "gʲ", "h", "i", "ia", "iau", "in", "iç", "iŋ", "iən", "iəu",
        "iɛ", "iʊŋ", "iː", "i̥", "j", "ja", "je", "jo", "ju", "jɛ", "jʌ", "k", "kʰ", "kʲ", "k͈", "l",
        "m", "mʲ", "n", "nʲ", "o", "ou", "oʊ", "oː", "o̯e", "p", "pf", "pʰ", "pʲ", "p͈", "r", "s", "s͈",
        "t", "ts", "tsʰ", "tɕ", "tɕʰ", "tʃ", "tʰ", "t͈", "t͡ɕ", "t͡ɕʰ", "t͡ɕ͈", "u", "ua", "uai", "uan",
        "uaŋ", "uŋ", "uəi", "uən", "uː", "u̥", "v", "w", "wa", "we", "wi", "wɛ", "wʌ", "x", "y", "yn",
        "yən", "yɛ", "yʊŋ", "yː", "z", "æ", "ç", "ð", "ø", "øː", "ı", "ŋ", "œ", "œ̃", "ɑ", "ɑɪ", "ɑʊ",
        "ɑ̃", "ɔ", "ɔø", "ɔɪ", "ɔ̃", "ɕ", "ə", "ən", "əu", "ɚ", "ɛ", "ɛɹ", "ɛː", "ɛ̃", "ɝ", "ɤ", "ɥ",
        "ɪ", "ɪɹ", "ɯ", "ɰi", "ɲ", "ɸ", "ɹ", "ɻ", "ɾ", "ɾʲ", "ʀ", "ʁ", "ʂ", "ʃ", "ʈʂ", "ʈʂʰ", "ʊ",
        "ʊŋ", "ʊɹ", "ʌ", "ʒ", "θ"
    ]

@dataclass
class TrainingState:
    """Training state containing models and optimizers."""
    generator: AudioTokenizer
    msd: MultiScaleDiscriminator  # Multi-Scale Discriminator
    mpd: MultiPeriodDiscriminator  # Multi-Period Discriminator
    stftd: STFTDiscriminator  # STFT Discriminator
    gen_optimizer: nnx.Optimizer
    msd_optimizer: nnx.Optimizer
    mpd_optimizer: nnx.Optimizer
    stftd_optimizer: nnx.Optimizer


def create_models_and_optimizers(
        config: Dict[str, Any],
        rngs: nnx.Rngs
) -> TrainingState:
    """Create models and optimizers."""

    # Create generator (AudioTokenizer)
    generator = AudioTokenizer(
        hidden_size=config['hidden_size'],
        encoder_depth=config['encoder_depth'],
        encoder_heads=config['encoder_heads'],
        phoneme_codebook_size=config['phoneme_codebook_size'],
        bsq_spherical_dim=config['bsq_spherical_dim'],
        decoder_output_48khz=config['decoder_output_48khz'],
        rngs=rngs,
    )

    # Create discriminators
    msd = MultiScaleDiscriminator(
        rates=[1, 2, 4],
        channels=[16, 64, 256, 1024, 1024, 1024],
        kernel_size=15,
        groups=[1, 4, 16, 64, 256, 1],
        rngs=rngs,
    )

    mpd = MultiPeriodDiscriminator(
        periods=[2, 3, 5, 7, 11],
        kernel_size=5,
        stride=3,
        rngs=rngs,
    )

    stftd = STFTDiscriminator(
        fft_sizes=[2048, 1024, 512],
        hop_lengths=None,  # Will use fft_size // 4
        win_lengths=None,  # Will use fft_size
        channels=[32, 64, 128, 256, 512],
        kernel_size=(3, 9),
        strides=(1, 2),
        padding="SAME",
        rngs=rngs,
    )

    # Create optimizers with different learning rates for different discriminators
    gen_optimizer = nnx.Optimizer(
        generator,
        optax.adamw(
            learning_rate=config['gen_learning_rate'],
            b1=0.8,
            b2=0.99,
            weight_decay=1e-4
        )
    )

    # Each discriminator can have different learning rate
    msd_optimizer = nnx.Optimizer(
        msd,
        optax.adamw(
            learning_rate=config.get('msd_learning_rate', config['disc_learning_rate']),
            b1=0.8,
            b2=0.99,
            weight_decay=1e-4
        )
    )

    mpd_optimizer = nnx.Optimizer(
        mpd,
        optax.adamw(
            learning_rate=config.get('mpd_learning_rate', config['disc_learning_rate']),
            b1=0.8,
            b2=0.99,
            weight_decay=1e-4
        )
    )

    stftd_optimizer = nnx.Optimizer(
        stftd,
        optax.adamw(
            learning_rate=config.get('stftd_learning_rate', config['disc_learning_rate']),
            b1=0.8,
            b2=0.99,
            weight_decay=1e-4
        )
    )

    return TrainingState(
        generator=generator,
        msd=msd,
        mpd=mpd,
        stftd=stftd,
        gen_optimizer=gen_optimizer,
        msd_optimizer=msd_optimizer,
        mpd_optimizer=mpd_optimizer,
        stftd_optimizer=stftd_optimizer,
    )


@partial(nnx.jit, static_argnums=(4,))
def train_generator_step(
        state: TrainingState,
        batch: Dict[str, jax.Array],
        phoneme_targets: Optional[jax.Array],
        target_lengths: Optional[jax.Array],
        loss_config: Dict[str, float],
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Single generator training step using nnx transforms."""

    audio = batch['audio']
    mask = batch['mask']  # Audio-level non-causal padding mask for losses
    encoder_mask = batch.get('encoder_mask', mask)  # Encoder-level non-causal mask
    encoder_causal_mask = batch.get('encoder_causal_mask', encoder_mask)  # Encoder-level causal mask

    # Define loss function for value_and_grad
    def loss_fn(generator):
        # Generator forward pass with encoder causal mask
        reconstructed, phoneme_indices, acoustic_codes, encoder_output = generator(audio, encoder_causal_mask)

        # Get quantizer intermediate outputs for commitment losses
        quantizer = generator.quantizer

        # Get phoneme quantized vectors from VQ codebook
        phoneme_quantized = quantizer.phoneme_vq.codebook[phoneme_indices]

        # Compute residual (what BSQ sees as input)
        residual = encoder_output - jax.lax.stop_gradient(phoneme_quantized)

        # Get the full quantized output from the quantizer
        full_quantized = quantizer.decode(phoneme_indices, acoustic_codes)

        # The BSQ quantized output is the difference
        residual_quantized = full_quantized - phoneme_quantized

        # Get discriminator outputs for generated audio from all discriminators
        # Multi-Scale Discriminator
        msd_outputs_fake, msd_features_fake = state.msd(reconstructed, training=False)
        msd_outputs_real, msd_features_real = state.msd(audio, training=False)

        # Multi-Period Discriminator
        mpd_outputs_fake, mpd_features_fake = state.mpd(reconstructed)
        mpd_outputs_real, mpd_features_real = state.mpd(audio)

        # STFT Discriminator
        stftd_outputs_fake, stftd_features_fake = state.stftd(reconstructed, training=False)
        stftd_outputs_real, stftd_features_real = state.stftd(audio, training=False)

        # Keep discriminator outputs separate for individual loss weighting
        all_disc_outputs_fake = {
            'msd': msd_outputs_fake,
            'mpd': mpd_outputs_fake,
            'stftd': stftd_outputs_fake
        }
        all_disc_features_real = {
            'msd': msd_features_real,
            'mpd': mpd_features_real,
            'stftd': stftd_features_real
        }
        all_disc_features_fake = {
            'msd': msd_features_fake,
            'mpd': mpd_features_fake,
            'stftd': stftd_features_fake
        }

        # Prepare encoder lengths if using phoneme targets
        if phoneme_targets is not None:
            encoder_lengths = extract_encoder_lengths(mask, generator.downsample_factor)
            phoneme_codebook = quantizer.phoneme_vq.codebook
        else:
            encoder_lengths = None
            phoneme_codebook = None

        # Compute generator loss with separate discriminator losses
        # First compute standard losses (reconstruction, quantizer, CTC)
        base_loss_config = {k: v for k, v in loss_config.items()
                            if not k.startswith(('msd_', 'mpd_', 'stftd_'))}
        base_loss_config['adversarial_weight'] = 0  # We'll add discriminator losses separately
        base_loss_config['feature_match_weight'] = 0

        # Dummy discriminator outputs for base loss computation
        dummy_disc_outputs = []
        dummy_disc_features = []

        base_loss, loss_dict = compute_generator_loss(
            pred_audio=reconstructed,
            target_audio=audio,
            encoder_output=encoder_output,
            phoneme_quantized=phoneme_quantized,
            residual=residual,
            residual_quantized=residual_quantized,
            phoneme_indices=phoneme_indices,
            phoneme_targets=phoneme_targets if phoneme_targets is not None else jnp.zeros((audio.shape[0], 1)),
            phoneme_codebook=phoneme_codebook if phoneme_codebook is not None else jnp.zeros((1, 1)),
            encoder_lengths=encoder_lengths if encoder_lengths is not None else jnp.ones(audio.shape[0]),
            target_lengths=target_lengths if target_lengths is not None else jnp.ones(audio.shape[0]),
            disc_outputs_fake=dummy_disc_outputs,
            disc_features_real=dummy_disc_features,
            disc_features_fake=dummy_disc_features,
            mask=mask,
            encoder_mask=encoder_mask,
            config=base_loss_config
        )

        # Compute separate adversarial and feature matching losses for each discriminator
        total_loss = base_loss

        # Multi-Scale Discriminator losses
        if loss_config.get('msd_adversarial_weight', 0) > 0:
            from tokenizer.alpha.loss import adversarial_g_loss, feature_matching_loss
            msd_adv_loss = adversarial_g_loss(all_disc_outputs_fake['msd'])
            msd_fm_loss = feature_matching_loss(
                all_disc_features_real['msd'],
                all_disc_features_fake['msd'],
                mask
            )
            loss_dict['msd_adversarial'] = msd_adv_loss
            loss_dict['msd_feature_match'] = msd_fm_loss
            total_loss += loss_config['msd_adversarial_weight'] * msd_adv_loss
            total_loss += loss_config.get('msd_feature_match_weight', 10.0) * msd_fm_loss

        # Multi-Period Discriminator losses
        if loss_config.get('mpd_adversarial_weight', 0) > 0:
            from tokenizer.alpha.loss import adversarial_g_loss, feature_matching_loss
            mpd_adv_loss = adversarial_g_loss(all_disc_outputs_fake['mpd'])
            mpd_fm_loss = feature_matching_loss(
                all_disc_features_real['mpd'],
                all_disc_features_fake['mpd'],
                mask
            )
            loss_dict['mpd_adversarial'] = mpd_adv_loss
            loss_dict['mpd_feature_match'] = mpd_fm_loss
            total_loss += loss_config['mpd_adversarial_weight'] * mpd_adv_loss
            total_loss += loss_config.get('mpd_feature_match_weight', 10.0) * mpd_fm_loss

        # STFT Discriminator losses
        if loss_config.get('stftd_adversarial_weight', 0) > 0:
            from tokenizer.alpha.loss import adversarial_g_loss, feature_matching_loss
            stftd_adv_loss = adversarial_g_loss(all_disc_outputs_fake['stftd'])
            stftd_fm_loss = feature_matching_loss(
                all_disc_features_real['stftd'],
                all_disc_features_fake['stftd'],
                mask
            )
            loss_dict['stftd_adversarial'] = stftd_adv_loss
            loss_dict['stftd_feature_match'] = stftd_fm_loss
            total_loss += loss_config['stftd_adversarial_weight'] * stftd_adv_loss
            total_loss += loss_config.get('stftd_feature_match_weight', 10.0) * stftd_fm_loss

        loss_dict['total'] = total_loss

        return total_loss, loss_dict

    # Compute gradients using nnx transforms
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_dict), grads = grad_fn(state.generator)

    # Update generator
    state.gen_optimizer.update(grads)

    return loss, loss_dict


@nnx.jit
def train_msd_step(
        state: TrainingState,
        batch: Dict[str, jax.Array],
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Multi-Scale Discriminator training step."""

    audio = batch['audio']
    encoder_mask = batch.get('encoder_mask', batch['mask'])

    # Define loss function for value_and_grad
    def loss_fn(msd):
        # Get generated audio (no gradients through generator)
        state.generator.eval()
        reconstructed, _, _, _ = state.generator(audio, encoder_mask)
        state.generator.train()

        # Multi-Scale Discriminator outputs
        real_outputs, real_features = msd(audio, training=True)
        fake_outputs, fake_features = msd(reconstructed, training=True)

        # Compute discriminator loss
        total_loss, loss_dict = compute_discriminator_loss(
            disc_outputs_real=real_outputs,
            disc_outputs_fake=fake_outputs,
            loss_type="hinge"
        )

        return total_loss, loss_dict

    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_dict), grads = grad_fn(state.msd)

    # Update MSD
    state.msd_optimizer.update(grads)

    return loss, loss_dict


@nnx.jit
def train_mpd_step(
        state: TrainingState,
        batch: Dict[str, jax.Array],
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Multi-Period Discriminator training step."""

    audio = batch['audio']
    encoder_mask = batch.get('encoder_mask', batch['mask'])

    # Define loss function for value_and_grad
    def loss_fn(mpd):
        # Get generated audio (no gradients through generator)
        state.generator.eval()
        reconstructed, _, _, _ = state.generator(audio, encoder_mask)
        state.generator.train()

        # Multi-Period Discriminator outputs
        real_outputs, real_features = mpd(audio)
        fake_outputs, fake_features = mpd(reconstructed)

        # Compute discriminator loss
        total_loss, loss_dict = compute_discriminator_loss(
            disc_outputs_real=real_outputs,
            disc_outputs_fake=fake_outputs,
            loss_type="hinge"
        )

        return total_loss, loss_dict

    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_dict), grads = grad_fn(state.mpd)

    # Update MPD
    state.mpd_optimizer.update(grads)

    return loss, loss_dict


@nnx.jit
def train_stftd_step(
        state: TrainingState,
        batch: Dict[str, jax.Array],
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """STFT Discriminator training step."""

    audio = batch['audio']
    encoder_mask = batch.get('encoder_mask', batch['mask'])

    # Define loss function for value_and_grad
    def loss_fn(stftd):
        # Get generated audio (no gradients through generator)
        state.generator.eval()
        reconstructed, _, _, _ = state.generator(audio, encoder_mask)
        state.generator.train()

        # STFT Discriminator outputs
        real_outputs, real_features = stftd(audio, training=True)
        fake_outputs, fake_features = stftd(reconstructed, training=True)

        # Compute discriminator loss
        total_loss, loss_dict = compute_discriminator_loss(
            disc_outputs_real=real_outputs,
            disc_outputs_fake=fake_outputs,
            loss_type="hinge"
        )

        return total_loss, loss_dict

    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_dict), grads = grad_fn(state.stftd)

    # Update STFTD
    state.stftd_optimizer.update(grads)

    return loss, loss_dict


def create_multilingual_g2p(possible_phonemes=None):
    """Create a multilingual G2P function using phonemizer with proper phoneme tokenization.
    
    Args:
        possible_phonemes: List of possible phoneme symbols for tokenization
        
    Returns:
        G2P function that converts text to phoneme sequences
    """
    if not PHONEMIZER_AVAILABLE:
        return None
    
    # Sort phonemes by length (longest first) for proper tokenization
    if possible_phonemes is None:
        possible_phonemes = globals()['possible_phonemes']
    
    # Create a sorted list for longest-match-first tokenization
    sorted_phonemes = sorted(possible_phonemes, key=len, reverse=True)

    def tokenize_phonemes(phoneme_string):
        """Tokenize phoneme string into individual phonemes using longest-match-first."""
        if not phoneme_string:
            return []
        
        tokens = []
        i = 0
        
        while i < len(phoneme_string):
            # Skip whitespace
            if phoneme_string[i] in ' \n\r\t':
                i += 1
                continue
                
            # Try to match longest phoneme first
            matched = False
            for phoneme in sorted_phonemes:
                if phoneme_string[i:i+len(phoneme)] == phoneme:
                    tokens.append(phoneme)
                    i += len(phoneme)
                    matched = True
                    break
            
            if not matched:
                # Unknown character, skip it
                i += 1
                
        return tokens if tokens else ['UNK']

    def g2p_function(text, locale='en-us'):
        """Convert text to phonemes using phonemizer with espeak backend.

        Args:
            text: Input text
            locale: Language locale (e.g., 'en-us', 'zh', 'ja', 'fr-fr', 'de', 'ko')

        Returns:
            List of phoneme symbols
        """
        try:
            # Use phonemizer for IPA conversion
            phoneme_string = phonemize(
                text,
                language=locale,
                backend='espeak',
                strip=True,
                preserve_punctuation=False,
                with_stress=False,
                njobs=1,
            )
            
            # Tokenize into individual phonemes
            return tokenize_phonemes(phoneme_string)
            
        except Exception as e:
            print(f"G2P error for locale {locale}: {e}")
            # Return UNK token on error
            return ['UNK']

    return g2p_function


def prepare_phoneme_targets_from_metadata(
        batch: Dict[str, Any],
        g2p_function,
        phoneme_vocab: Dict[str, int],
        max_phoneme_length: int = 200
) -> Tuple[jax.Array, jax.Array]:
    """Convert batch metadata to phoneme targets using G2P.

    Args:
        batch: Batch dictionary with metadata
        g2p_function: G2P conversion function
        phoneme_vocab: Phoneme to index mapping
        max_phoneme_length: Maximum phoneme sequence length

    Returns:
        phoneme_targets: Phoneme indices [B, max_phoneme_length]
        target_lengths: Actual phoneme sequence lengths [B]
    """
    batch_size = batch['audio'].shape[0]

    # Extract text and locale metadata
    texts = batch.get('meta_text', [None] * batch_size)
    locales = batch.get('meta_language', ['en-us'] * batch_size)
    
    # Map language codes to locales if needed
    locale_map = {
        'EN': 'en-us',
        'ZH': 'zh',
        'JA': 'ja',
        'FR': 'fr',
        'DE': 'de',
        'KO': 'ko',
    }

    # Convert texts to phonemes
    all_phoneme_indices = []
    all_lengths = []

    for i in range(batch_size):
        text = texts[i]
        locale = locales[i] if locales[i] else 'en-us'
        
        # Convert language code to locale if needed
        if locale in locale_map:
            locale = locale_map[locale]

        if text and g2p_function:
            # Convert text to phonemes using the locale
            phonemes = g2p_function(text, locale)

            # Convert phonemes to indices
            phoneme_indices = []
            for phoneme in phonemes[:max_phoneme_length]:
                idx = phoneme_vocab.get(phoneme, phoneme_vocab.get('UNK', 1))
                phoneme_indices.append(idx)

            all_phoneme_indices.append(phoneme_indices)
            all_lengths.append(len(phoneme_indices))
        else:
            # Fallback: use a dummy sequence with UNK tokens
            dummy_length = min(10, max_phoneme_length)
            all_phoneme_indices.append([phoneme_vocab.get('UNK', 1)] * dummy_length)
            all_lengths.append(dummy_length)

    # Pad sequences to max length (right padding for CTC)
    if all_lengths:
        max_length = min(max(all_lengths), max_phoneme_length)
    else:
        max_length = max_phoneme_length
        
    padded_targets = []

    for indices in all_phoneme_indices:
        if len(indices) < max_length:
            # Right pad with blank token (CTC blank is at index 0)
            indices = indices + [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        padded_targets.append(indices)

    # Convert to JAX arrays
    phoneme_targets = jnp.array(padded_targets, dtype=jnp.int32)
    target_lengths = jnp.array(all_lengths, dtype=jnp.int32)

    return phoneme_targets, target_lengths


def train_epoch(
        state: TrainingState,
        data_loader: Any,  # Can be SimpleAudioLoader or OptimizedAudioLoader
        loss_config: Dict[str, float],
        g2p_function: Optional[Any] = None,
        phoneme_vocab: Optional[Dict[str, int]] = None,
        disc_update_freq: int = 1,
) -> Dict[str, float]:
    """Train for one epoch."""

    epoch_metrics = {
        'gen_loss': 0.0,
        'msd_loss': 0.0,
        'mpd_loss': 0.0,
        'stftd_loss': 0.0,
        'l1_loss': 0.0,
        'l2_loss': 0.0,
        'spectral_loss': 0.0,
        'ctc_loss': 0.0,
        'msd_adversarial': 0.0,
        'mpd_adversarial': 0.0,
        'stftd_adversarial': 0.0,
        'msd_feature_match': 0.0,
        'mpd_feature_match': 0.0,
        'stftd_feature_match': 0.0,
        'num_batches': 0,
    }

    for batch_idx, batch in enumerate(data_loader):
        # Convert metadata to phoneme targets using G2P
        if g2p_function and phoneme_vocab:
            phoneme_targets, target_lengths = prepare_phoneme_targets_from_metadata(
                batch, g2p_function, phoneme_vocab
            )
        else:
            # If G2P not available, skip phoneme loss
            phoneme_targets, target_lengths = None, None

        # Train discriminators every disc_update_freq steps
        if batch_idx % disc_update_freq == 0:
            # Train Multi-Scale Discriminator
            msd_loss, msd_metrics = train_msd_step(state, batch)
            epoch_metrics['msd_loss'] = epoch_metrics.get('msd_loss', 0.0) + float(msd_loss)

            # Train Multi-Period Discriminator
            mpd_loss, mpd_metrics = train_mpd_step(state, batch)
            epoch_metrics['mpd_loss'] = epoch_metrics.get('mpd_loss', 0.0) + float(mpd_loss)

            # Train STFT Discriminator
            stftd_loss, stftd_metrics = train_stftd_step(state, batch)
            epoch_metrics['stftd_loss'] = epoch_metrics.get('stftd_loss', 0.0) + float(stftd_loss)

        # Train generator
        gen_loss, gen_metrics = train_generator_step(
            state, batch, phoneme_targets, target_lengths, loss_config
        )

        # Update metrics
        epoch_metrics['gen_loss'] += float(gen_loss)
        epoch_metrics['l1_loss'] += float(gen_metrics.get('l1', 0))
        epoch_metrics['l2_loss'] += float(gen_metrics.get('l2', 0))
        epoch_metrics['spectral_loss'] += float(gen_metrics.get('spectral_convergence', 0))
        epoch_metrics['ctc_loss'] += float(gen_metrics.get('ctc', 0))
        epoch_metrics['msd_adversarial'] += float(gen_metrics.get('msd_adversarial', 0))
        epoch_metrics['mpd_adversarial'] += float(gen_metrics.get('mpd_adversarial', 0))
        epoch_metrics['stftd_adversarial'] += float(gen_metrics.get('stftd_adversarial', 0))
        epoch_metrics['msd_feature_match'] += float(gen_metrics.get('msd_feature_match', 0))
        epoch_metrics['mpd_feature_match'] += float(gen_metrics.get('mpd_feature_match', 0))
        epoch_metrics['stftd_feature_match'] += float(gen_metrics.get('stftd_feature_match', 0))
        epoch_metrics['num_batches'] += 1

        # Print progress every 10 batches with timing info
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Gen Loss={gen_loss:.4f}, "
                  f"L1={gen_metrics.get('l1', 0):.4f}, "
                  f"CTC={gen_metrics.get('ctc', 0):.4f}, "
                  f"MSD Adv={gen_metrics.get('msd_adversarial', 0):.4f}")

        # Early exit for testing
        if batch_idx >= 50:  # Process only 50 batches for testing
            print("\nEarly exit after 50 batches for testing")
            break

    # Average metrics
    for key in epoch_metrics:
        if key != 'num_batches':
            epoch_metrics[key] /= max(epoch_metrics['num_batches'], 1)

    return epoch_metrics


def main():
    """Main training function."""

    # Configuration
    config = {
        # Model config
        'hidden_size': 512,
        'encoder_depth': 4,
        'encoder_heads': 8,
        'phoneme_codebook_size': 155,  # 152 phonemes + 3 special tokens (blank, unk, pad)
        'bsq_spherical_dim': 256,
        'decoder_output_48khz': False,

        # Training config
        'gen_learning_rate': 1e-4,
        'disc_learning_rate': 1e-4,
        # Individual discriminator learning rates (optional, falls back to disc_learning_rate)
        'msd_learning_rate': 1e-4,
        'mpd_learning_rate': 1e-4,
        'stftd_learning_rate': 5e-5,  # Often benefits from lower LR
        'batch_size': 4,
        'num_epochs': 10,
        'disc_update_freq': 2,

        # Data config
        'sample_rate': 24000,
        'max_duration_seconds': 5.0,
    }

    # Initialize RNGs
    rngs = nnx.Rngs(0)

    # Create models and optimizers
    print("Creating models...")
    state = create_models_and_optimizers(config, rngs)

    # Create optimized data loader
    print("Creating optimized Emilia data loader...")
    from tokenizer.utils.data.optimized_loader import create_optimized_emilia_loader
    data_loader = create_optimized_emilia_loader(
        split="train",
        sample_rate=config['sample_rate'],
        batch_size=config['batch_size'],
        max_duration_seconds=config['max_duration_seconds'],
    )

    # Setup multilingual G2P if available
    g2p_fn = create_multilingual_g2p(possible_phonemes)
    if g2p_fn:
        print("Multilingual G2P ready (phonemizer)")
        # Create phoneme vocabulary using the comprehensive phoneme list
        phoneme_vocab = create_phoneme_vocabulary(possible_phonemes)
        print(f"Phoneme vocabulary size: {len(phoneme_vocab)} "
              f"({len(possible_phonemes)} phonemes + 3 special tokens)")
    else:
        print("G2P not available, using dummy phoneme targets")
        phoneme_vocab = None

    # Loss weights configuration with separate weights for each discriminator
    loss_config = {
        'l1_weight': 1.0,
        'l2_weight': 1.0,
        'spectral_weight': 2.0,
        'log_mag_weight': 1.0,
        'ctc_weight': 10.0 if g2p_fn else 0.0,  # Enable if G2P available
        'phoneme_commit_weight': 0.1,
        'bsq_commit_weight': 1.0,
        # Multi-Scale Discriminator weights
        'msd_adversarial_weight': 0.1,  # Start small for stability
        'msd_feature_match_weight': 10.0,
        # Multi-Period Discriminator weights
        'mpd_adversarial_weight': 0.1,
        'mpd_feature_match_weight': 10.0,
        # STFT Discriminator weights
        'stftd_adversarial_weight': 0.05,  # Often smaller for STFT disc
        'stftd_feature_match_weight': 5.0,
    }

    # Training loop
    print("\nStarting training...")

    print("\nBeginning training epochs...")

    for epoch in range(config['num_epochs']):
        start_time = time.time()

        # Train one epoch
        metrics = train_epoch(
            state=state,
            data_loader=data_loader,
            loss_config=loss_config,
            g2p_function=g2p_fn,
            phoneme_vocab=phoneme_vocab,
            disc_update_freq=config['disc_update_freq'],
        )

        elapsed = time.time() - start_time

        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']} - "
              f"Time: {elapsed:.1f}s")
        print(f"  Gen Loss: {metrics['gen_loss']:.4f}")
        print(f"  Discriminator Losses:")
        print(f"    MSD: {metrics['msd_loss']:.4f}")
        print(f"    MPD: {metrics['mpd_loss']:.4f}")
        print(f"    STFTD: {metrics['stftd_loss']:.4f}")
        print(f"  Reconstruction Losses:")
        print(f"    L1: {metrics['l1_loss']:.4f}")
        print(f"    L2: {metrics['l2_loss']:.4f}")
        print(f"    Spectral: {metrics['spectral_loss']:.4f}")
        if metrics.get('ctc_loss', 0) > 0:
            print(f"    CTC: {metrics['ctc_loss']:.4f}")
        print(f"  Adversarial Losses:")
        print(f"    MSD: {metrics['msd_adversarial']:.4f}")
        print(f"    MPD: {metrics['mpd_adversarial']:.4f}")
        print(f"    STFTD: {metrics['stftd_adversarial']:.4f}")

    print("\nTraining completed!")

    # Clean up the data loader resources
    if hasattr(data_loader, 'close'):
        data_loader.close()


if __name__ == "__main__":
    main()