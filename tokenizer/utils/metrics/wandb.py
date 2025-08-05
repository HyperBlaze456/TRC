import wandb
import jax
import time
from typing import Dict, Any, Optional

def init_wandb(config: Any, project: str = "speech-tokenizer", run_name: Optional[str] = None):
    """Initialize wandb logging.
    
    Args:
        config: Training configuration object
        project: WandB project name
        run_name: Optional run name, auto-generated if None
    
    Returns:
        The wandb run object
    """
    # Create config dict from dataclass
    config_dict = {
        k: v for k, v in config.__dict__.items() if not k.startswith('_')
    }
    
    # Initialize wandb
    run = wandb.init(
        project=project,
        name=run_name or f"run_{int(time.time())}",
        config=config_dict
    )
    
    return run

def log_metrics(metrics: Dict[str, Any], step: int):
    """Log metrics to wandb.
    
    Args:
        metrics: Dictionary of metric names to values
        step: Current training step
    """
    # If metrics contain JAX arrays, convert to Python scalars
    logged_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):  # JAX array
            logged_metrics[key] = value.item()
        else:
            logged_metrics[key] = value
    
    wandb.log(logged_metrics, step=step)

def log_discriminator_metrics(disc_losses: Dict[str, jax.Array], step: int):
    """Log discriminator metrics to wandb.
    
    Args:
        disc_losses: Dictionary of discriminator losses
        step: Current training step
    """
    metrics = {f"disc/{k}": v for k, v in disc_losses.items()}
    log_metrics(metrics, step)

def log_generator_metrics(gen_losses: Dict[str, jax.Array], step: int):
    """Log generator metrics to wandb.
    
    Args:
        gen_losses: Dictionary of generator losses
        step: Current training step
    """
    metrics = {f"gen/{k}": v for k, v in gen_losses.items()}
    log_metrics(metrics, step)

def finish_wandb():
    """Finish wandb run."""
    wandb.finish()