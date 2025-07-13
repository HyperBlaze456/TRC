import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from tqdm import tqdm

from tokenizer.alpha.model import AudioTokenizer
from tokenizer.alpha.components.discriminators.scale_discriminator import MultiScaleDiscriminator
from tokenizer.alpha.components.discriminators.stft_discriminator import STFTDiscriminator
from tokenizer.alpha.losses import (
    hinge_loss_generator,
    hinge_loss_discriminator,
    feature_matching_loss,
    multi_scale_spectrogram_loss,
    time_domain_loss,
    ctc_loss
)
from tokenizer.utils.metrics.logging import WandBLogger, TensorBoardLogger


class AudioTokenizerTrainer:
    """Trainer for AudioTokenizer with adversarial and reconstruction losses."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize random number generator
        self.rngs = nnx.Rngs(config.get('seed', 42))
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize logging
        self._init_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def _init_models(self):
        """Initialize generator and discriminators."""
        model_config = self.config['model']
        
        # Generator (AudioTokenizer)
        self.generator = AudioTokenizer(
            hidden_size=model_config['hidden_size'],
            encoder_depth=model_config['encoder_depth'],
            encoder_heads=model_config['encoder_heads'],
            phoneme_codebook_size=model_config['phoneme_codebook_size'],
            bsq_spherical_dim=model_config['bsq_spherical_dim'],
            decoder_output_48khz=model_config.get('output_48khz', False),
            rngs=self.rngs
        )
        
        # Multi-scale discriminator
        self.msd = MultiScaleDiscriminator(
            rates=[1, 2, 4],
            channels=[16, 64, 256, 1024, 1024, 1024],
            kernel_size=15,
            groups=[1, 4, 16, 64, 256, 1],
            rngs=self.rngs
        )
        
        # STFT discriminator
        self.stft_disc = STFTDiscriminator(
            fft_sizes=[2048, 1024, 512],
            channels=[32, 64, 128, 256, 512],
            rngs=self.rngs
        )
        
    def _init_optimizers(self):
        """Initialize optimizers for generator and discriminators."""
        opt_config = self.config['optimization']
        
        # Generator optimizer
        g_lr = opt_config['generator_lr']
        g_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=g_lr,
            warmup_steps=opt_config.get('warmup_steps', 1000),
            decay_steps=opt_config.get('total_steps', 100000),
            end_value=g_lr * 0.01
        )
        
        self.g_optimizer = nnx.Optimizer(
            self.generator,
            optax.chain(
                optax.clip_by_global_norm(opt_config.get('grad_clip', 1.0)),
                optax.adamw(
                    learning_rate=g_schedule,
                    b1=opt_config.get('beta1', 0.9),
                    b2=opt_config.get('beta2', 0.999),
                    weight_decay=opt_config.get('weight_decay', 0.01)
                )
            )
        )
        
        # Discriminator optimizer
        d_lr = opt_config['discriminator_lr']
        d_params = [self.msd, self.stft_disc]
        
        self.d_optimizer = nnx.Optimizer(
            d_params,
            optax.chain(
                optax.clip_by_global_norm(opt_config.get('grad_clip', 1.0)),
                optax.adamw(
                    learning_rate=d_lr,
                    b1=opt_config.get('beta1', 0.9),
                    b2=opt_config.get('beta2', 0.999),
                    weight_decay=opt_config.get('weight_decay', 0.01)
                )
            )
        )
        
    def _init_logging(self):
        """Initialize logging utilities."""
        log_config = self.config.get('logging', {})
        
        self.loggers = []
        
        if log_config.get('use_wandb', False):
            self.loggers.append(WandBLogger(
                project=log_config.get('wandb_project', 'audio-tokenizer'),
                name=log_config.get('run_name', f'run_{time.strftime("%Y%m%d_%H%M%S")}'),
                config=self.config
            ))
        
        if log_config.get('use_tensorboard', True):
            self.loggers.append(TensorBoardLogger(
                log_dir=str(self.log_dir / 'tensorboard')
            ))
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all loggers."""
        for logger in self.loggers:
            logger.log(metrics, step)
    
    @nnx.jit
    def train_discriminator_step(
        self,
        real_audio: jnp.ndarray,
        rng_key: jax.random.PRNGKey
    ) -> Tuple[Dict[str, float], jnp.ndarray]:
        """Single discriminator training step."""
        
        def d_loss_fn(d_params):
            # Generate fake audio
            with nnx.eval_mode():
                fake_audio, _, _, _ = self.generator(real_audio)
            
            # Multi-scale discriminator outputs
            msd_real, msd_real_feats = self.msd(real_audio, training=True)
            msd_fake, msd_fake_feats = self.msd(fake_audio, training=True)
            
            # STFT discriminator outputs
            stft_real = self.stft_disc(real_audio, training=True)
            stft_fake = self.stft_disc(fake_audio, training=True)
            
            # Compute losses
            msd_loss = hinge_loss_discriminator(msd_real, msd_fake)
            stft_loss = hinge_loss_discriminator(stft_real, stft_fake)
            
            total_d_loss = msd_loss + stft_loss
            
            metrics = {
                'd_loss': total_d_loss,
                'd_msd_loss': msd_loss,
                'd_stft_loss': stft_loss
            }
            
            return total_d_loss, metrics
        
        # Compute gradients and update
        (d_loss, metrics), grads = nnx.value_and_grad(d_loss_fn, has_aux=True)(
            nnx.state([self.msd, self.stft_disc])
        )
        
        self.d_optimizer.update(grads)
        
        return metrics, fake_audio
    
    @nnx.jit
    def train_generator_step(
        self,
        real_audio: jnp.ndarray,
        phoneme_targets: Optional[jnp.ndarray] = None,
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> Dict[str, float]:
        """Single generator training step."""
        loss_weights = self.config['loss_weights']
        
        def g_loss_fn(g_params):
            # Forward pass through generator
            fake_audio, phoneme_indices, acoustic_codes, encoder_output = self.generator(real_audio)
            
            # Adversarial losses
            msd_fake, msd_fake_feats = self.msd(fake_audio, training=True)
            stft_fake = self.stft_disc(fake_audio, training=True)
            
            adv_g_loss = (
                hinge_loss_generator(msd_fake) + 
                hinge_loss_generator(stft_fake)
            )
            
            # Feature matching loss
            with nnx.eval_mode():
                _, msd_real_feats = self.msd(real_audio, training=False)
            
            fm_loss = feature_matching_loss(msd_real_feats, msd_fake_feats)
            
            # Reconstruction losses
            spec_loss, spec_metrics = multi_scale_spectrogram_loss(
                real_audio, fake_audio,
                fft_sizes=[2048, 1024, 512, 256, 128],
                loss_type='l1'
            )
            
            time_loss = time_domain_loss(real_audio, fake_audio, loss_type='l1')
            
            # CTC loss (if phoneme targets provided)
            ctc_loss_value = jnp.array(0.0)
            if phoneme_targets is not None:
                # Assuming we have phoneme logits from somewhere
                # This is a placeholder - you'd need to add phoneme prediction to the model
                batch_size, seq_len = phoneme_indices.shape
                phoneme_logits = jnp.zeros((batch_size, seq_len, 100))  # Placeholder
                
                logit_lengths = jnp.full((batch_size,), seq_len)
                target_lengths = jnp.sum(phoneme_targets != 0, axis=1)
                
                ctc_loss_value = ctc_loss(
                    phoneme_logits, phoneme_targets,
                    logit_lengths, target_lengths
                )
            
            # Total generator loss
            total_g_loss = (
                loss_weights['adversarial'] * adv_g_loss +
                loss_weights['feature_matching'] * fm_loss +
                loss_weights['spectrogram'] * spec_loss +
                loss_weights['time_domain'] * time_loss +
                loss_weights.get('ctc', 0.0) * ctc_loss_value
            )
            
            metrics = {
                'g_loss': total_g_loss,
                'g_adv_loss': adv_g_loss,
                'g_fm_loss': fm_loss,
                'g_spec_loss': spec_loss,
                'g_time_loss': time_loss,
                'g_ctc_loss': ctc_loss_value,
                **spec_metrics
            }
            
            return total_g_loss, metrics
        
        # Compute gradients and update
        (g_loss, metrics), grads = nnx.value_and_grad(g_loss_fn, has_aux=True)(
            nnx.state(self.generator)
        )
        
        self.g_optimizer.update(grads)
        
        return metrics
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        epoch_metrics = {}
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {self.epoch}")):
            # Extract batch data
            real_audio = batch['audio']
            phoneme_targets = batch.get('phonemes', None)
            
            # Get RNG key for this step
            rng_key = self.rngs()
            
            # Discriminator step
            if self.global_step % self.config.get('d_steps_per_g_step', 1) == 0:
                d_metrics, fake_audio = self.train_discriminator_step(real_audio, rng_key)
                
                # Update epoch metrics
                for k, v in d_metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = []
                    epoch_metrics[k].append(float(v))
            
            # Generator step
            g_metrics = self.train_generator_step(real_audio, phoneme_targets, rng_key)
            
            # Update epoch metrics
            for k, v in g_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(float(v))
            
            # Log metrics
            if self.global_step % self.config.get('log_every', 100) == 0:
                step_metrics = {k: float(v) for k, v in {**d_metrics, **g_metrics}.items()}
                self._log_metrics(step_metrics, self.global_step)
            
            # Save checkpoint
            if self.global_step % self.config.get('checkpoint_every', 1000) == 0:
                self.save_checkpoint()
            
            self.global_step += 1
        
        # Compute epoch averages
        epoch_avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return epoch_avg_metrics
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.global_step}.nnx"
        
        checkpoint = {
            'generator': nnx.state(self.generator),
            'msd': nnx.state(self.msd),
            'stft_disc': nnx.state(self.stft_disc),
            'g_optimizer': nnx.state(self.g_optimizer),
            'd_optimizer': nnx.state(self.d_optimizer),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config
        }
        
        # Save using Flax serialization
        with open(checkpoint_path, 'wb') as f:
            nnx.experimental.save_state_dict(checkpoint, f)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = nnx.experimental.load_state_dict(f)
        
        # Restore model states
        nnx.update(self.generator, checkpoint['generator'])
        nnx.update(self.msd, checkpoint['msd'])
        nnx.update(self.stft_disc, checkpoint['stft_disc'])
        nnx.update(self.g_optimizer, checkpoint['g_optimizer'])
        nnx.update(self.d_optimizer, checkpoint['d_optimizer'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}, epoch {self.epoch}")
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs: int = 100):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Config: {json.dumps(self.config, indent=2)}")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)
            
            print(f"Epoch {epoch} - Train metrics:")
            for k, v in train_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Validation
            if val_dataloader is not None and epoch % self.config.get('val_every', 1) == 0:
                val_metrics = self.validate(val_dataloader)
                print(f"Epoch {epoch} - Validation metrics:")
                for k, v in val_metrics.items():
                    print(f"  {k}: {v:.4f}")
            
            # Save checkpoint at end of epoch
            self.save_checkpoint()
    
    @nnx.eval_mode
    def validate(self, dataloader):
        """Validation loop."""
        val_metrics = {}
        
        for batch in tqdm(dataloader, desc="Validation"):
            real_audio = batch['audio']
            
            # Generate audio
            fake_audio, _, _, _ = self.generator(real_audio)
            
            # Compute reconstruction losses only
            spec_loss, spec_metrics = multi_scale_spectrogram_loss(
                real_audio, fake_audio,
                fft_sizes=[2048, 1024, 512, 256, 128],
                loss_type='l1'
            )
            
            time_loss = time_domain_loss(real_audio, fake_audio, loss_type='l1')
            
            # Update metrics
            metrics = {
                'val_spec_loss': float(spec_loss),
                'val_time_loss': float(time_loss),
                **{f'val_{k}': float(v) for k, v in spec_metrics.items()}
            }
            
            for k, v in metrics.items():
                if k not in val_metrics:
                    val_metrics[k] = []
                val_metrics[k].append(v)
        
        # Compute averages
        val_avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        return val_avg_metrics


def create_default_config():
    """Create default training configuration."""
    return {
        'seed': 42,
        'model': {
            'hidden_size': 512,
            'encoder_depth': 4,
            'encoder_heads': 8,
            'phoneme_codebook_size': 100,
            'bsq_spherical_dim': 256,
            'output_48khz': False
        },
        'optimization': {
            'generator_lr': 1e-4,
            'discriminator_lr': 1e-4,
            'warmup_steps': 1000,
            'total_steps': 100000,
            'grad_clip': 1.0,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.01
        },
        'loss_weights': {
            'adversarial': 1.0,
            'feature_matching': 10.0,
            'spectrogram': 45.0,
            'time_domain': 1.0,
            'ctc': 0.0  # Set to > 0 if using phoneme targets
        },
        'training': {
            'd_steps_per_g_step': 1,
            'log_every': 100,
            'checkpoint_every': 1000,
            'val_every': 1
        },
        'logging': {
            'use_wandb': False,
            'use_tensorboard': True,
            'wandb_project': 'audio-tokenizer',
            'run_name': f'run_{time.strftime("%Y%m%d_%H%M%S")}'
        }
    }


if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    
    # Initialize trainer
    trainer = AudioTokenizerTrainer(config)
    
    # Create dummy dataloader for testing
    # In practice, you would load real audio data here
    class DummyDataLoader:
        def __init__(self, num_batches=10, batch_size=4, seq_len=24000):
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.seq_len = seq_len
        
        def __iter__(self):
            for _ in range(self.num_batches):
                yield {
                    'audio': jnp.ones((self.batch_size, self.seq_len, 1)),
                    'phonemes': None
                }
        
        def __len__(self):
            return self.num_batches
    
    # Create dummy data
    train_loader = DummyDataLoader(num_batches=100)
    val_loader = DummyDataLoader(num_batches=10)
    
    # Train
    trainer.train(train_loader, val_loader, num_epochs=1)
