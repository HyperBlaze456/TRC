import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional, Tuple
from .dit import DiT


def create_attention_mask(mask: jax.Array, causal: bool = False) -> jax.Array:
    """Convert padding mask to attention mask.

    Args:
        mask: Padding mask [B, N, D] where 1 = valid, 0 = padding
        causal: If True, add causal masking on top of padding mask

    Returns:
        Attention mask [B, 1, N, N] where True = attend, False = mask out
    """
    B, N, D = mask.shape
    # Take any feature dimension to get sequence mask [B, N]
    seq_mask = mask[:, :, 0]

    # Create attention mask [B, N, N]
    # This allows position i to attend to position j only if both are valid
    attn_mask = seq_mask[:, :, None] & seq_mask[:, None, :]

    if causal:
        # Add causal mask: position i can only attend to positions <= i
        causal_mask = jnp.tril(jnp.ones((N, N), dtype=bool))
        attn_mask = attn_mask & causal_mask[None, :, :]

    # Add head dimension [B, 1, N, N]
    return attn_mask[:, None, :, :]


class ConditionalCFM(nnx.Module):
    """Base Conditional Flow Matching class for continuous normalizing flows."""

    def __init__(
        self,
        n_feats: int,
        sigma_min: float = 1e-4,
        t_scheduler: str = "linear",
        training_cfg_rate: float = 0.0,
        inference_cfg_rate: float = 0.0,
    ):
        self.n_feats = n_feats
        self.sigma_min = sigma_min
        self.t_scheduler = t_scheduler
        self.training_cfg_rate = training_cfg_rate
        self.inference_cfg_rate = inference_cfg_rate

    def sample_time(self, key: jax.random.PRNGKey, shape: Tuple[int, ...]) -> jax.Array:
        """Sample time values with optional cosine scheduling."""
        t = jax.random.uniform(key, shape, minval=0.0, maxval=1.0)
        if self.t_scheduler == "cosine":
            t = 1 - jnp.cos(t * 0.5 * jnp.pi)
        return t

    def get_time_schedule(self, n_timesteps: int) -> jax.Array:
        """Get time schedule for inference."""
        t_span = jnp.linspace(0, 1, n_timesteps + 1)
        if self.t_scheduler == "cosine":
            t_span = 1 - jnp.cos(t_span * 0.5 * jnp.pi)
        return t_span

    def interpolate(
        self, x0: jax.Array, x1: jax.Array, t: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Interpolate between noise and data for flow matching. We use linear interpolation for training

        Args:
            x0: Initial noise sample
            x1: Target data
            t: Time values

        Returns:
            y: Interpolated sample
            u: Target velocity
        """
        # Expand t to match dimensions
        t = jnp.expand_dims(t, axis=-1)  # [B, 1, 1]

        # Linear interpolation with sigma_min
        y = (1 - (1 - self.sigma_min) * t) * x0 + t * x1
        u = x1 - (1 - self.sigma_min) * x0

        return y, u


class DiTCFM(ConditionalCFM):
    """Conditional Flow Matching with DiT as the velocity estimator."""

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        rngs: nnx.Rngs,
        n_feats: int = 80,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        patch_size: int = 1,
        spk_dim: Optional[int] = None,
        sigma_min: float = 1e-4,
        t_scheduler: str = "linear",
        training_cfg_rate: float = 0.1,
        inference_cfg_rate: float = 1.0,
        use_causal_mask: bool = False,
    ):
        super().__init__(
            n_feats=n_feats,
            sigma_min=sigma_min,
            t_scheduler=t_scheduler,
            training_cfg_rate=training_cfg_rate,
            inference_cfg_rate=inference_cfg_rate,
        )

        self.use_causal_mask = use_causal_mask

        # Initialize DiT as velocity estimator
        self.estimator = DiT(
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            rngs=rngs,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            patch_size=patch_size,
            in_channels=n_feats,
            out_channels=n_feats,
            spk_dim=spk_dim,
            learn_sigma=False,  # We don't need sigma for velocity prediction
        )

    def compute_loss(
        self,
        x1: jax.Array,
        mask: jax.Array,
        mu: jax.Array,
        key: jax.random.PRNGKey,
        spks: Optional[jax.Array] = None,
        cond: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute conditional flow matching loss.

        Args:
            x1: Target data [B, N, D]
            mask: Valid positions mask [B, N, D]
            mu: Conditioning features (reference) [B, N, D]
            key: PRNG key for randomness
            spks: Speaker embeddings [B, spk_dim]
            cond: Additional conditioning. In future implementation, this could be added as cross attn

        Returns:
            loss: CFM loss value
            y: Interpolated sample for monitoring
        """
        B, N, D = x1.shape

        # Split keys
        key_t, key_z, key_cfg = jax.random.split(key, 3)

        # Sample random time
        t = self.sample_time(key_t, (B, 1))  # [B, 1]

        # Sample noise
        z = jax.random.normal(key_z, x1.shape)

        # Interpolate between noise and data
        y, u = self.interpolate(z, x1, t)

        # Apply classifier-free guidance dropout during training
        if self.training_cfg_rate > 0:
            cfg_mask = jax.random.uniform(key_cfg, (B,)) > self.training_cfg_rate
            cfg_mask_3d = jnp.expand_dims(cfg_mask, axis=(1, 2))  # [B, 1, 1]
            cfg_mask_2d = jnp.expand_dims(cfg_mask, axis=1)  # [B, 1]
            mu = mu * cfg_mask_3d
            if spks is not None:
                spks = spks * cfg_mask_2d  # [B, spk_dim]

        # Predict velocity using DiT
        # Reshape t for DiT: [B, 1] -> [B, hidden_size]
        t_expanded = jnp.broadcast_to(t, (B, self.estimator.hidden_size))

        # Convert padding mask to attention mask if needed
        attn_mask = (
            create_attention_mask(mask.astype(bool), causal=self.use_causal_mask)
            if mask is not None
            else None
        )

        pred = self.estimator(y, t_expanded, ref=mu, spk_emb=spks, mask=attn_mask)

        # Compute MSE loss with masking
        # mask is already [B, N, D], matching pred and u shapes
        loss = jnp.sum((pred - u) ** 2 * mask) / jnp.sum(mask)

        return loss, y

    def solve_euler(
        self,
        x: jax.Array,
        t_span: jax.Array,
        mu: jax.Array,
        mask: jax.Array,
        spks: Optional[jax.Array] = None,
        cond: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Solve ODE using Euler method.

        Args:
            x: Initial noise [B, N, D]
            t_span: Time steps for integration
            mu: Conditioning features [B, N, D]
            mask: Valid positions mask [B, N, D]
            spks: Speaker embeddings [B, spk_dim]
            cond: This just exists, not even used

        Returns:
            Final integrated sample
        """
        B, N, D = x.shape
        t = t_span[0]

        for step in range(1, len(t_span)):
            dt = t_span[step] - t

            # Prepare time for DiT
            t_expanded = jnp.full((B, self.estimator.hidden_size), t)

            # Convert padding mask to attention mask if needed
            attn_mask = (
                create_attention_mask(mask.astype(bool), causal=self.use_causal_mask)
                if mask is not None
                else None
            )

            # For CFG, we need to run two forward passes
            if self.inference_cfg_rate > 0:
                # Conditional prediction
                v_cond = self.estimator(
                    x, t_expanded, ref=mu, spk_emb=spks, mask=attn_mask
                )

                # Unconditional prediction (zero conditioning)
                v_uncond = self.estimator(
                    x,
                    t_expanded,
                    ref=jnp.zeros_like(mu),
                    spk_emb=jnp.zeros_like(spks) if spks is not None else None,
                    mask=attn_mask,
                )

                # Apply CFG
                v = (
                    1.0 + self.inference_cfg_rate
                ) * v_cond - self.inference_cfg_rate * v_uncond
            else:
                # Standard prediction
                v = self.estimator(x, t_expanded, ref=mu, spk_emb=spks, mask=attn_mask)

            # Euler step
            x = x + dt * v
            t = t_span[step]

        return x

    def __call__(
        self,
        mu: jax.Array,
        mask: jax.Array,
        n_timesteps: int,
        key: jax.random.PRNGKey,
        temperature: float = 1.0,
        spks: Optional[jax.Array] = None,
        cond: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Generate samples using flow matching.

        Args:
            mu: Conditioning features [B, N, D]
            mask: Valid positions mask [B, N, D]
            n_timesteps: Number of integration steps
            key: PRNG key for initial noise
            temperature: Noise temperature scaling
            spks: Speaker embeddings [B, spk_dim]
            cond: Additional conditioning (unused)

        Returns:
            Generated samples [B, N, D]
        """
        # Sample initial noise
        z = jax.random.normal(key, mu.shape) * temperature

        # Get time schedule
        t_span = self.get_time_schedule(n_timesteps)

        # Solve ODE
        sample = self.solve_euler(z, t_span, mu, mask, spks, cond)

        return sample
