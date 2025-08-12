"""
Masking utilities for MaskGCT S2A model
"""

import torch
import numpy as np


def get_mask_layer_schedule(schedule_type="linear", num_layers=12):
    """
    Get masking schedule for different layers
    
    Args:
        schedule_type: Type of schedule ("linear", "cosine", "arccos")
        num_layers: Number of quantizer layers
        
    Returns:
        List of masking probabilities for each layer
    """
    if schedule_type == "linear":
        return np.linspace(0.0, 1.0, num_layers).tolist()
    elif schedule_type == "cosine":
        return [0.5 * (1 + np.cos(np.pi * i / (num_layers - 1))) for i in range(num_layers)]
    elif schedule_type == "arccos":
        return [np.arccos(1 - 2 * i / (num_layers - 1)) / np.pi for i in range(num_layers)]
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def create_random_mask(shape, mask_prob, device="cpu"):
    """
    Create random mask for training
    
    Args:
        shape: Shape of the mask tensor
        mask_prob: Probability of masking
        device: Device to create mask on
        
    Returns:
        Boolean mask tensor
    """
    return torch.rand(shape, device=device) < mask_prob


def create_causal_mask(seq_len, device="cpu"):
    """
    Create causal attention mask
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        Causal mask tensor
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def create_padding_mask(lengths, max_len, device="cpu"):
    """
    Create padding mask from sequence lengths
    
    Args:
        lengths: Actual lengths of sequences
        max_len: Maximum sequence length
        device: Device to create mask on
        
    Returns:
        Padding mask tensor
    """
    batch_size = len(lengths)
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len)
    mask = mask >= lengths.unsqueeze(1)
    return mask


def apply_mask_to_sequence(sequence, mask, mask_token_id):
    """
    Apply mask to sequence by replacing masked positions with mask token
    
    Args:
        sequence: Input sequence tensor
        mask: Boolean mask tensor
        mask_token_id: ID of mask token
        
    Returns:
        Masked sequence
    """
    masked_sequence = sequence.clone()
    masked_sequence[mask] = mask_token_id
    return masked_sequence


def compute_masked_loss(logits, targets, mask, ignore_index=-100):
    """
    Compute loss only on masked positions
    
    Args:
        logits: Model predictions
        targets: Ground truth targets
        mask: Boolean mask indicating positions to compute loss
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Masked loss value
    """
    # Set non-masked positions to ignore_index
    masked_targets = targets.clone()
    masked_targets[~mask] = ignore_index
    
    # Compute cross entropy loss
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        masked_targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean"
    )
    
    return loss


def get_layer_mask_prob(layer_idx, schedule, base_prob=0.3):
    """
    Get masking probability for a specific layer
    
    Args:
        layer_idx: Index of the layer
        schedule: Masking schedule values
        base_prob: Base masking probability
        
    Returns:
        Masking probability for the layer
    """
    if layer_idx >= len(schedule):
        return base_prob
    
    return base_prob + (1 - base_prob) * schedule[layer_idx]


def create_structured_mask(shape, mask_prob, min_span=1, max_span=10, device="cpu"):
    """
    Create structured mask with continuous spans
    
    Args:
        shape: Shape of the mask tensor (batch_size, seq_len)
        mask_prob: Overall probability of masking
        min_span: Minimum span length
        max_span: Maximum span length
        device: Device to create mask on
        
    Returns:
        Structured boolean mask tensor
    """
    batch_size, seq_len = shape
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        num_masked = int(mask_prob * seq_len)
        masked_so_far = 0
        
        while masked_so_far < num_masked:
            span_len = torch.randint(min_span, min(max_span + 1, num_masked - masked_so_far + 1), (1,)).item()
            start_idx = torch.randint(0, seq_len - span_len + 1, (1,)).item()
            
            mask[b, start_idx:start_idx + span_len] = True
            masked_so_far += span_len
            
            if masked_so_far >= num_masked:
                break
    
    return mask