from flax import nnx
import jax
import jax.numpy as jnp

class VectorQuantizer(nnx.Module):
    def __init__(self, codebook_size: int, dim: int, rngs: nnx.Rngs):
        self.dim = dim
        self.codebook = nnx.Param(
            value=jax.random.uniform(key=rngs.params(), shape=(codebook_size, dim))
        )
    def __call__(self, z):
        # Allows batch, [N, D] or [B, N, D]?

        diff_squared = jnp.square(
            z[:, :, None, :]
            - self.codebook[None, None, :, :]
        )
        l2_distance = jnp.sum(diff_squared, axis=-1)
        jax.debug.print(str(l2_distance.shape))
        codebook_indicies = jnp.argmin(l2_distance, axis=-1)

        z_q = self.codebook[codebook_indicies]
        jax.debug.print(str(z_q.shape))

        z_q = z + jax.lax.stop_gradient(z_q - z)
        return z_q, codebook_indicies


class ResidualVQ(nnx.Module):
    def __init__(self, codebook_size: int, dim: int, rngs: nnx.Rngs, num_codebooks: int = 1):
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.codebooks = []
        
        # Create multiple VectorQuantizer layers
        for i in range(num_codebooks):
            self.codebooks.append(VectorQuantizer(codebook_size, dim, rngs))
    
    def __call__(self, z):
        # z shape: [B, N, D]
        residual = z
        all_indices = []
        all_quantized = []
        
        for i, vq in enumerate(self.codebooks):
            # Quantize the current residual
            z_q, indices = vq(residual)
            all_indices.append(indices)
            all_quantized.append(z_q)
            
            # Calculate residual for next layer
            if i < self.num_codebooks - 1:
                residual = residual - z_q
        
        # Stack indices along a new dimension: [B, N, num_codebooks]
        all_indices = jnp.stack(all_indices, axis=-1)
        
        # Sum all quantized vectors to get final output
        z_q_final = sum(all_quantized)
        
        # Apply straight-through estimator
        z_q_final = z + jax.lax.stop_gradient(z_q_final - z)
        
        return z_q_final, all_indices


class BinarySphericalQuantizer(nnx.Module):
    """Binary Spherical Quantizer (BSQ) for efficient tokenization.
    
    BSQ projects high-dimensional embeddings to a lower-dimensional hypersphere
    and then applies binary quantization. This creates an implicit codebook
    whose effective vocabulary grows exponentially with the spherical dimension.
    
    Based on "Image and Video Tokenization with Binary Spherical Quantization"
    """
    
    def __init__(
        self,
        input_dim: int,
        spherical_dim: int,
        rngs: nnx.Rngs,
        temperature: float = 1.0,
        use_straight_through: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of input embeddings
            spherical_dim: Dimension of the hypersphere (lower than input_dim)
            rngs: Random number generators
            temperature: Temperature for quantization sharpness
            use_straight_through: Whether to use straight-through gradient estimator
        """
        self.input_dim = input_dim
        self.spherical_dim = spherical_dim
        self.temperature = temperature
        self.use_straight_through = use_straight_through
        
        # Projection layers
        self.proj_sphere = nnx.Linear(input_dim, spherical_dim, rngs=rngs)
        self.sphere_restore = nnx.Linear(spherical_dim, input_dim, rngs=rngs)
        
    def _normalize_to_sphere(self, x: jnp.ndarray) -> jnp.ndarray:
        """Normalize vectors to unit hypersphere."""
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    
    def _binary_quantize(self, x: jnp.ndarray):
        """Apply binary quantization with straight-through estimator.
        
        Returns:
            quantized: Binary quantized values {-1, 1}
            codes: Binary codes {0, 1} for indexing
        """
        # Convert to binary {-1, 1}
        if self.use_straight_through:
            # Straight-through estimator: forward pass quantizes, backward pass flows through
            codes = (x > 0).astype(jnp.float32)
            quantized = 2 * codes - 1
            # Stop gradient and add back for straight-through
            quantized = jax.lax.stop_gradient(quantized - x) + x
        else:
            # Soft quantization using tanh
            quantized = jnp.tanh(x / self.temperature)
            codes = (quantized > 0).astype(jnp.float32)
        
        return quantized, codes
    
    def encode(self, x: jnp.ndarray):
        """Encode input to binary codes.
        
        Args:
            x: Input embeddings [B, T, D]
            
        Returns:
            codes: Binary codes [B, T, spherical_dim]
            spherical_embeddings: Normalized spherical embeddings before quantization
        """
        # Project to lower-dimensional space
        spherical = self.proj_sphere(x)
        
        # Normalize to unit hypersphere
        spherical_normalized = self._normalize_to_sphere(spherical)
        
        # Binary quantization
        _, codes = self._binary_quantize(spherical_normalized)
        
        return codes, spherical_normalized
    
    def decode(self, codes: jnp.ndarray) -> jnp.ndarray:
        """Decode binary codes back to embeddings.
        
        Args:
            codes: Binary codes [B, T, spherical_dim]
            
        Returns:
            Reconstructed embeddings [B, T, D]
        """
        # Convert binary codes {0, 1} to {-1, 1}
        quantized = 2 * codes - 1
        
        # Project back to original dimension
        reconstructed = self.sphere_restore(quantized)
        
        return reconstructed
    
    def __call__(self, x: jnp.ndarray):
        """Forward pass with quantization.
        
        Args:
            x: Input embeddings [B, T, D]
            
        Returns:
            reconstructed: Reconstructed embeddings [B, T, D]
            codes: Binary codes [B, T, spherical_dim]
        """
        # Project to sphere
        spherical = self.proj_sphere(x)
        spherical_normalized = self._normalize_to_sphere(spherical)
        
        # Binary quantization
        quantized, codes = self._binary_quantize(spherical_normalized)
        
        # Reconstruct
        reconstructed = self.sphere_restore(quantized)
        
        # Straight-through estimator for the whole reconstruction
        reconstructed = x + jax.lax.stop_gradient(reconstructed - x)
        
        return reconstructed, codes


class PhonemeBSQQuantizer(nnx.Module):
    """Specialized quantizer combining VQ for phoneme and BSQ for acoustic residual.
    
    First applies VQ for phoneme classification (small codebook), then uses BSQ
    on the residual to capture remaining acoustic features.
    """
    
    def __init__(
        self,
        input_dim: int,
        phoneme_codebook_size: int,
        spherical_dim: int,
        rngs: nnx.Rngs,
        temperature: float = 1.0,
    ):
        """
        Args:
            input_dim: Dimension of input embeddings
            phoneme_codebook_size: Size of VQ codebook for phoneme classification
            spherical_dim: Dimension for BSQ hypersphere
            rngs: Random number generators
            temperature: Temperature for BSQ quantization
        """
        self.input_dim = input_dim
        
        # First layer: VQ for phoneme classification
        self.phoneme_vq = VectorQuantizer(
            codebook_size=phoneme_codebook_size,
            dim=input_dim,
            rngs=rngs
        )
        
        # Second layer: BSQ for acoustic residual
        self.acoustic_bsq = BinarySphericalQuantizer(
            input_dim=input_dim,
            spherical_dim=spherical_dim,
            rngs=rngs,
            temperature=temperature,
            use_straight_through=True
        )
    
    def __call__(self, x: jnp.ndarray):
        """Forward pass with two-stage quantization.
        
        Args:
            x: Input embeddings [B, T, D]
            
        Returns:
            reconstructed: Reconstructed embeddings [B, T, D]
            phoneme_indices: Phoneme codebook indices [B, T]
            acoustic_codes: Binary codes for acoustic features [B, T, spherical_dim]
        """
        # Stage 1: VQ for phoneme classification
        phoneme_quantized, phoneme_indices = self.phoneme_vq(x)
        
        # Calculate residual after phoneme quantization
        residual = x - jax.lax.stop_gradient(phoneme_quantized)
        
        # Stage 2: BSQ for acoustic residual
        acoustic_quantized, acoustic_codes = self.acoustic_bsq(residual)
        
        # Combine both quantizations
        reconstructed = phoneme_quantized + acoustic_quantized
        
        # Apply straight-through estimator
        reconstructed = x + jax.lax.stop_gradient(reconstructed - x)
        
        return reconstructed, phoneme_indices, acoustic_codes
    
    def encode(self, x: jnp.ndarray):
        """Encode input to discrete codes.
        
        Args:
            x: Input embeddings [B, T, D]
            
        Returns:
            phoneme_indices: Phoneme codebook indices [B, T]
            acoustic_codes: Binary codes for acoustic features [B, T, spherical_dim]
        """
        # Get phoneme codes
        phoneme_quantized, phoneme_indices = self.phoneme_vq(x)
        
        # Get acoustic codes from residual
        residual = x - phoneme_quantized
        acoustic_codes, _ = self.acoustic_bsq.encode(residual)
        
        return phoneme_indices, acoustic_codes
    
    def decode(self, phoneme_indices: jnp.ndarray, acoustic_codes: jnp.ndarray):
        """Decode discrete codes back to embeddings.
        
        Args:
            phoneme_indices: Phoneme codebook indices [B, T]
            acoustic_codes: Binary codes for acoustic features [B, T, spherical_dim]
            
        Returns:
            Reconstructed embeddings [B, T, D]
        """
        # Decode phoneme part
        phoneme_embeddings = self.phoneme_vq.codebook[phoneme_indices]
        
        # Decode acoustic part
        acoustic_embeddings = self.acoustic_bsq.decode(acoustic_codes)
        
        # Combine
        reconstructed = phoneme_embeddings + acoustic_embeddings
        
        return reconstructed
