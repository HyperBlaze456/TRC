import wandb
import jax
import time
from typing import Dict, Any, Optional

def init_wandb(config: Any, project: str = "speech-tokenizer", run_name: Optional[str] = None):
    """Initialize wandb logging with real-time tracking support.
    
    Args:
        config: Training configuration object
        project: WandB project name
        run_name: Optional run name, auto-generated if None
    
    Returns:
        The wandb run object
    """
    # Check if logged in
    try:
        if not wandb.api.api_key:
            print("\n⚠️  W&B API key not found. Please run 'wandb login' or set WANDB_API_KEY environment variable.")
            print("    You can get your API key from: https://wandb.ai/authorize\n")
            wandb.login()
    except:
        print("\n⚠️  Attempting W&B login...")
        wandb.login()
    
    # Create config dict from dataclass
    config_dict = {
        k: v for k, v in config.__dict__.items() if not k.startswith('_')
    }
    
    # Initialize wandb with explicit settings for real-time tracking
    run = wandb.init(
        entity="hyperblaze",
        project=project,
        name=run_name or f"run_{int(time.time())}",
        config=config_dict,
        mode="online",  # Force online mode for real-time updates
        reinit=True,
        settings=wandb.Settings()
    )
    
    # Print the run URL immediately
    print(f"\n🔗 W&B Run URL: {run.get_url()}")
    print(f"   Project: {project}")
    print(f"   Run Name: {run.name}\n")
    
    return run

def log_metrics(metrics: Dict[str, Any], step: int, commit: bool = True):
    """Log metrics to wandb with real-time updates.
    
    Args:
        metrics: Dictionary of metric names to values
        step: Current training step
        commit: Whether to commit immediately (for real-time updates)
    """
    # If metrics contain JAX arrays, convert to Python scalars
    logged_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):  # JAX array
            logged_metrics[key] = value.item()
        else:
            logged_metrics[key] = value
    
    # Add step to metrics for clarity
    logged_metrics['training/step'] = step
    
    wandb.log(logged_metrics, step=step, commit=commit)

def log_discriminator_metrics(disc_losses: Dict[str, jax.Array], step: int):
    """Log discriminator metrics to wandb.
    
    Args:
        disc_losses: Dictionary of discriminator losses
        step: Current training step
    """
    metrics = {f"disc/{k}": v for k, v in disc_losses.items()}
    log_metrics(metrics, step, commit=True)  # Commit immediately

def log_generator_metrics(gen_losses: Dict[str, jax.Array], step: int):
    """Log generator metrics to wandb.
    
    Args:
        gen_losses: Dictionary of generator losses
        step: Current training step
    """
    metrics = {f"gen/{k}": v for k, v in gen_losses.items()}
    log_metrics(metrics, step, commit=True)  # Commit immediately

def finish_wandb():
    """Finish wandb run."""
    print("\n📊 Finishing W&B run...")
    wandb.finish()
    print("✅ W&B run finished successfully\n")