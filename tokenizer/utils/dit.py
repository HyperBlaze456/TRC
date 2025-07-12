from flax import nnx
import jax
import jax.numpy as jnp
from .norm import AdaRMSZero
from .embeddings import RotaryPositionalEmbedding
from .attention import MultiHeadAttentionWithRoPE


class DiTBlock(nnx.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            rngs:nnx.Rngs,
            mlp_ratio = 4.0,
            dropout = 0.0,
            rope = None
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.norm1 = AdaRMSZero(hidden_size, rngs)
        self.norm2 = AdaRMSZero(hidden_size, rngs)
        
        # Always use 2x hidden size for concatenation
        self.cond_proj_in = nnx.Linear(hidden_size * 2, hidden_size, rngs=rngs)
        self.cond_proj_out = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        
        self.attn = MultiHeadAttentionWithRoPE(
            num_heads=num_heads,
            in_features=hidden_size,
            rngs=rngs,
            qkv_features=hidden_size,
            out_features=hidden_size,
            dropout_rate=dropout,
            rope=rope
        )

        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, int(hidden_size * mlp_ratio), rngs=rngs),
            nnx.gelu,
            nnx.Linear(int(hidden_size * mlp_ratio), hidden_size, rngs=rngs),
        )

    def __call__(self, x, t, ref, spk_emb, mask=None, rope_cos_sin=None):
        residual = x
        
        x, gate_msa = self.norm1(x, t)
        
        # Always concatenate (cond will be zeros if not provided)
        x_cond = jnp.concatenate([x, spk_emb], axis=-1)
        x = self.cond_proj_in(x_cond)
        
        x = self.attn(x, x, x, mask=mask, rope_cos_sin=rope_cos_sin)
        x = self.cond_proj_out(x)
        
        x = gate_msa * x + residual
        
        residual = x
        x, gate_mlp = self.norm2(x, t)
        x = self.mlp(x)
        x = gate_mlp * x + residual
        
        return x


class DiT(nnx.Module):
    def __init__(
            self,
            hidden_size,
            depth,
            num_heads,
            rngs: nnx.Rngs,
            mlp_ratio=4.0,
            dropout=0.0,
            patch_size=1,
            in_channels=None,
            out_channels=None,
            spk_dim=None,
            learn_sigma=True,
            use_pos_emb=False,
            max_seq_len=5000
    ):
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.learn_sigma = learn_sigma
        self.patch_size = patch_size
        self.cond_dim = spk_dim
        self.use_pos_emb = use_pos_emb
        
        # Initialize RoPE if enabled
        if use_pos_emb:
            self.rope = RotaryPositionalEmbedding(
                dim=hidden_size // num_heads,
                max_seq_len=max_seq_len
            )
        else:
            self.rope = None
        
        if in_channels is not None:
            # x_embedder now expects concatenated input [x, ref]
            self.x_embedder = nnx.Linear(in_channels * patch_size * 2, hidden_size, rngs=rngs)
        else:
            self.x_embedder = nnx.Linear(hidden_size * 2, hidden_size, rngs=rngs)
        
        self.t_embedder = nnx.Sequential(
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        )
        
        if spk_dim is not None:
            self.spk_embedder = nnx.Sequential(
                nnx.Linear(spk_dim, hidden_size, rngs=rngs),
                nnx.silu,
                nnx.Linear(hidden_size, hidden_size, rngs=rngs)
            )
        
        self.blocks = [
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                rngs=rngs,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                rope=self.rope
            ) for _ in range(depth)
        ]
        # Maybe remove this?
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels or in_channels, rngs, learn_sigma)
    
    def __call__(self, x, t, ref = None, spk_emb=None, mask=None):
        """
        DiT model for denoising one step.
        :param x: A noised value [B, N, D]
        :param t: On what timestep is the noise value at
        :param ref: Reference token that has the same shape as X. Main steering embeds.
        :param spk_emb: Optional speaker embedding. [B, own_dim], projected to [B, D] and broadcast to [B, 1, D]
        :param mask: Optional attention mask [B, N, N] or [B, 1, N, N] for causal/padding masking
        :return: One denoised value, or ground truth according to the using module's task.
        """
        B, N, D = x.shape
        if ref is None:
            # CFG support w/ zeroed cond
            ref = jnp.zeros((B, N, D), dtype=x.dtype)

        # Concatenate x and ref directly (no linear pass)
        x = jnp.concatenate([x, ref], axis=-1)
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        
        if spk_emb is not None:
            # spk_emb is shape [B, spk_dim], need to broadcast to [B, N, hidden_size]
            spk_emb = self.spk_embedder(spk_emb)  # [B, hidden_size], reshaping linear
            spk_emb = jnp.expand_dims(spk_emb, axis=1)  # [B, 1, hidden_size]
            spk_emb = jnp.broadcast_to(spk_emb, (B, N, self.hidden_size))  # [B, N, hidden_size]
        else:
            # Use zeros for unconditional generation (CFG support)
            spk_emb = jnp.zeros((B, N, self.hidden_size), dtype=x.dtype)
        
        # Prepare positional embeddings if enabled
        if self.use_pos_emb:
            cos, sin = self.rope(N)
            rope_cos_sin = (cos, sin)
        else:
            rope_cos_sin = None
        
        for block in self.blocks:
            x = block(x, t, ref, spk_emb, mask, rope_cos_sin)
        
        x = self.final_layer(x, t)
        return x


class FinalLayer(nnx.Module):
    def __init__(self, hidden_size, patch_size, out_channels, rngs: nnx.Rngs, learn_sigma=True):
        self.norm_final = AdaRMSZero(hidden_size, rngs)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.learn_sigma = learn_sigma
        jax.debug.print(str(type(patch_size)))
        out_features = patch_size * out_channels
        if learn_sigma:
            out_features *= 2
            
        self.linear = nnx.Linear(
            hidden_size, 
            out_features,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs
        )
    
    def __call__(self, x, c):
        x, gate = self.norm_final(x, c)
        x = self.linear(x * gate)
        return x