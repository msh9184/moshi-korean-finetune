"""
Loss computation for Moshi finetuning.

IMPORTANT: The model intentionally fills invalid positions with NaN values
(see lm.py _undelay_sequence with fill_value=float('NaN')). This is a safety
mechanism to ensure invalid positions are properly handled.

The loss computation MUST only compute loss on valid (masked) positions,
not the entire logits tensor. Invalid positions are excluded by the mask.

For Full Duplex mode (dep_q=16), the audio codebooks are structured as:
- Codebooks 0-7: Moshi audio (8 codebooks)
- Codebooks 8-15: User audio (8 codebooks)

The first codebook of each speaker (0 for Moshi, 8 for User) is the semantic
codebook which carries the most important information and gets higher weight.

Loss Calculation Method (J-Moshi Style):
========================================
This implementation follows J-Moshi's token-count based normalization:

    audio_weight = N_semantic * w_s + N_acoustic * w_a
    semantic_scale = w_s / audio_weight
    acoustic_scale = w_a / audio_weight
    audio_loss = loss_semantic.sum() * semantic_scale + loss_acoustic.sum() * acoustic_scale

This ensures that the loss is properly normalized by the actual number of
valid tokens, making training more stable across varying sequence lengths.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.nn import functional as F

logger = logging.getLogger("loss")

# Flag to log debug info only once
_first_call_logged = {"audio": False, "text": False, "user_stream": False}


@dataclass
class AudioLossResult:
    """
    Result of per-speaker audio loss computation.

    Attributes:
        total_loss: Combined weighted loss for backpropagation
        moshi_semantic_loss: Mean loss for Moshi's semantic codebook (for logging)
        moshi_acoustic_loss: Mean loss for Moshi's acoustic codebooks (for logging)
        user_semantic_loss: Mean loss for User's semantic codebook (for logging)
        user_acoustic_loss: Mean loss for User's acoustic codebooks (for logging)
        moshi_total_loss: Combined Moshi audio loss (for logging)
        user_total_loss: Combined User audio loss (for logging, None if no user stream)
    """
    total_loss: torch.Tensor
    moshi_semantic_loss: torch.Tensor
    moshi_acoustic_loss: torch.Tensor
    moshi_total_loss: torch.Tensor
    user_semantic_loss: Optional[torch.Tensor] = None
    user_acoustic_loss: Optional[torch.Tensor] = None
    user_total_loss: Optional[torch.Tensor] = None


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
    prompt_mask: Optional[torch.Tensor] = None,
):
    """
    Compute masked loss with weighted averaging.

    Args:
        logits: Model output logits [batch, K, T, vocab] or [batch, 1, T, vocab]
                NOTE: Invalid positions contain NaN by design!
        target: Target token IDs [batch, K, T] or [batch, 1, T]
        target_mask: Valid target position mask [batch, K, T] or [batch, 1, T]
        mode: "audio" or "text"
        first_codebook_weight_multiplier: First codebook weight (audio mode)
        text_padding_weight: Text padding token weight (text mode)
        text_padding_ids: Set of token IDs to treat as padding
        prompt_mask: Optional mask indicating prompt positions [B, T] to exclude from loss.
                     When audio prompting is enabled, this prevents the model from learning
                     to predict the prompt region (which would be "cheating").

    Returns:
        Weighted average loss value
    """
    global _first_call_logged

    # Get dimensions
    vocab_size = logits.size(-1)

    # =================================================================
    # CRITICAL FIX: Combine prompt_mask with target_mask
    # =================================================================
    # When audio prompting is enabled, prompt_mask indicates which
    # positions are prompt (True) vs training target (False).
    # We must exclude prompt positions from loss computation to prevent
    # the model from "cheating" by learning to predict the prompt.
    # =================================================================
    if prompt_mask is not None:
        if prompt_mask.dim() == 2 and target_mask.dim() == 3:
            # prompt_mask: [B, T] -> [B, 1, T] for broadcasting across codebooks
            prompt_mask_expanded = prompt_mask.unsqueeze(1)
            target_mask = target_mask & ~prompt_mask_expanded
        elif prompt_mask.dim() == target_mask.dim():
            # Same dimensionality - direct combination
            target_mask = target_mask & ~prompt_mask

    # Flatten for processing: [B, K, T, V] -> [B*K*T, V] and [B, K, T] -> [B*K*T]
    logits_flat = logits.reshape(-1, vocab_size).float()
    target_flat = target.reshape(-1)
    mask_flat = target_mask.reshape(-1)

    # Count valid positions
    num_valid = mask_flat.sum().item()
    num_total = mask_flat.numel()

    # One-time debug logging
    if not _first_call_logged[mode]:
        logger.info(
            f"[LOSS DEBUG] {mode} mode: "
            f"logits_flat={logits_flat.shape}, "
            f"valid={num_valid}/{num_total} ({100*num_valid/max(num_total,1):.1f}%), "
            f"vocab_size={vocab_size}"
        )
        # Check how many NaN in logits (expected for masked positions)
        total_nan = torch.isnan(logits_flat).sum().item()
        logger.info(
            f"[LOSS DEBUG] {mode} mode: total NaN in logits: {total_nan} "
            f"(expected for {num_total - num_valid} masked positions)"
        )
        _first_call_logged[mode] = True

    if num_valid == 0:
        logger.warning(
            f"No valid positions in {mode} mode! "
            f"mask sum: {num_valid}, total: {num_total}"
        )
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

    # CRITICAL: Only select VALID positions for loss computation
    # The model fills invalid positions with NaN by design!
    valid_logits = logits_flat[mask_flat]  # [num_valid, vocab_size]
    valid_targets = target_flat[mask_flat].long()  # [num_valid] - ensure long for cross_entropy

    # Check for NaN/Inf only in VALID positions
    if torch.isnan(valid_logits).any() or torch.isinf(valid_logits).any():
        nan_count = torch.isnan(valid_logits).sum().item()
        inf_count = torch.isinf(valid_logits).sum().item()
        total_valid = valid_logits.numel()
        logger.error(
            f"NaN/Inf in VALID positions for {mode} mode! "
            f"NaN: {nan_count}/{total_valid}, Inf: {inf_count}/{total_valid}, "
            f"This indicates a real model issue, not masked positions."
        )
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

    # Check for out-of-range target indices
    max_target = valid_targets.max().item()
    min_target = valid_targets.min().item()
    if max_target >= vocab_size or min_target < 0:
        logger.error(
            f"Invalid target indices in {mode} mode! "
            f"target range: [{min_target}, {max_target}], vocab_size: {vocab_size}"
        )
        # Clamp to valid range
        valid_targets = valid_targets.clamp(0, vocab_size - 1)

    # Compute per-token cross entropy ONLY on valid positions
    per_token_loss = F.cross_entropy(valid_logits, valid_targets, reduction="none")

    # Create weights for valid positions only
    # Start with all 1s for valid positions
    valid_weights = torch.ones(num_valid, device=logits.device, dtype=torch.float32)

    # Apply mode-specific weighting
    if mode == "audio":
        # For audio mode, weight first codebook differently
        if target_mask.dim() == 3:
            B, K, T = target_mask.shape
            # Create position indices to identify first codebook
            mask_3d = target_mask  # [B, K, T]
            weights_3d = torch.ones_like(mask_3d, dtype=torch.float32)
            weights_3d[:, 0] *= first_codebook_weight_multiplier
            valid_weights = weights_3d.reshape(-1)[mask_flat]
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            valid_weights = torch.where(
                valid_targets == id,
                valid_weights * text_padding_weight,
                valid_weights
            )

    # Apply weighting
    weighted_loss = per_token_loss * valid_weights

    # Compute weight sum
    weight_sum = valid_weights.sum()

    if weight_sum == 0:
        logger.warning(
            f"Zero weight sum in {mode} loss computation. "
            f"All valid positions have zero weight."
        )
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

    # Compute final loss
    mb_loss = weighted_loss.sum() / weight_sum

    return mb_loss


def _compute_codebook_loss_sum_and_count(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """
    Compute loss sum and valid token count for a codebook (J-Moshi style).

    This function returns the raw sum of losses and the count of valid tokens,
    allowing the caller to perform proper weighted normalization.

    Args:
        logits: [B, K, T, V] - model predictions
        target: [B, K, T] - target token IDs
        mask: [B, K, T] - valid position mask

    Returns:
        Tuple of (loss_sum, valid_count)
    """
    vocab_size = logits.size(-1)

    logits_flat = logits.reshape(-1, vocab_size).float()
    target_flat = target.reshape(-1)
    mask_flat = mask.reshape(-1)

    num_valid = mask_flat.sum().item()
    if num_valid == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True), 0

    valid_logits = logits_flat[mask_flat]
    valid_targets = target_flat[mask_flat].long()

    # Check for NaN/Inf
    if torch.isnan(valid_logits).any() or torch.isinf(valid_logits).any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True), 0

    # Clamp targets to valid range
    valid_targets = valid_targets.clamp(0, vocab_size - 1)

    # Compute per-token loss and sum (NOT mean!)
    per_token_loss = F.cross_entropy(valid_logits, valid_targets, reduction="none")
    loss_sum = per_token_loss.sum()

    return loss_sum, int(num_valid)


def _compute_codebook_loss_mean(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean loss for a codebook (for logging purposes).

    Args:
        logits: [B, K, T, V] - model predictions
        target: [B, K, T] - target token IDs
        mask: [B, K, T] - valid position mask

    Returns:
        Mean loss value
    """
    loss_sum, count = _compute_codebook_loss_sum_and_count(logits, target, mask)
    if count == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
    return loss_sum / count


def compute_audio_loss_per_speaker(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dep_q: int,
    semantic_weight: float = 100.0,
    acoustic_weight: float = 1.0,
    user_semantic_weight: Optional[float] = None,
    user_acoustic_weight: Optional[float] = None,
    prompt_mask: Optional[torch.Tensor] = None,
) -> AudioLossResult:
    """
    Compute audio loss with per-speaker and per-codebook breakdown (J-Moshi Style).

    This function implements J-Moshi's token-count based normalization:
        audio_weight = N_semantic * w_s + N_acoustic * w_a
        semantic_scale = w_s / audio_weight
        acoustic_scale = w_a / audio_weight
        audio_loss = loss_semantic.sum() * semantic_scale + loss_acoustic.sum() * acoustic_scale

    This ensures proper normalization by actual token counts.

    Args:
        logits: Model output [B, K, T, V] where K = dep_q
        target: Target tokens [B, K, T]
        mask: Valid position mask [B, K, T]
        dep_q: Number of audio codebooks (8 for mono, 16 for stereo)
        semantic_weight: Weight for semantic (first) codebook (default: 100.0)
        acoustic_weight: Weight for acoustic codebooks (default: 1.0)
        user_semantic_weight: Weight for user semantic (default: same as semantic_weight)
        user_acoustic_weight: Weight for user acoustic (default: same as acoustic_weight)
        prompt_mask: Optional mask indicating prompt positions [B, T] to exclude from loss

    Returns:
        AudioLossResult with breakdown of all loss components
    """
    global _first_call_logged

    # Use same weights for user if not specified
    if user_semantic_weight is None:
        user_semantic_weight = semantic_weight
    if user_acoustic_weight is None:
        user_acoustic_weight = acoustic_weight

    B, K, T = target.shape
    has_user_stream = (dep_q == 16)

    # =================================================================
    # CRITICAL FIX: Apply prompt_mask to exclude prompt positions
    # =================================================================
    # When audio prompting is enabled, prompt_mask indicates which
    # positions are prompt (True) vs training target (False).
    # We must exclude prompt positions from loss computation.
    # =================================================================
    if prompt_mask is not None:
        # prompt_mask: [B, T] -> [B, 1, T] for broadcasting across codebooks
        prompt_mask_expanded = prompt_mask.unsqueeze(1) if prompt_mask.dim() == 2 else prompt_mask
        mask = mask & ~prompt_mask_expanded

    # One-time debug logging
    if not _first_call_logged.get("user_stream", False):
        logger.info(
            f"[AUDIO LOSS] J-Moshi style token-count based normalization"
        )
        logger.info(
            f"[AUDIO LOSS] dep_q={dep_q}, has_user_stream={has_user_stream}, "
            f"logits={logits.shape}, target={target.shape}, mask={mask.shape}"
        )
        logger.info(
            f"[AUDIO LOSS] Weights: semantic={semantic_weight}, acoustic={acoustic_weight}, "
            f"user_semantic={user_semantic_weight}, user_acoustic={user_acoustic_weight}"
        )
        _first_call_logged["user_stream"] = True

    # =========================================================================
    # Moshi codebooks (always present)
    # Semantic: codebook 0, Acoustic: codebooks 1-7
    # =========================================================================
    moshi_semantic_logits = logits[:, 0:1]  # [B, 1, T, V]
    moshi_semantic_target = target[:, 0:1]  # [B, 1, T]
    moshi_semantic_mask = mask[:, 0:1]      # [B, 1, T]

    moshi_acoustic_logits = logits[:, 1:8]  # [B, 7, T, V]
    moshi_acoustic_target = target[:, 1:8]  # [B, 7, T]
    moshi_acoustic_mask = mask[:, 1:8]      # [B, 7, T]

    # Compute Moshi loss sums and counts
    moshi_semantic_loss_sum, moshi_semantic_count = _compute_codebook_loss_sum_and_count(
        moshi_semantic_logits, moshi_semantic_target, moshi_semantic_mask
    )
    moshi_acoustic_loss_sum, moshi_acoustic_count = _compute_codebook_loss_sum_and_count(
        moshi_acoustic_logits, moshi_acoustic_target, moshi_acoustic_mask
    )

    # =========================================================================
    # User stream losses (only if dep_q=16)
    # =========================================================================
    if has_user_stream and K < 16:
        logger.warning(
            f"[AUDIO LOSS] dep_q={dep_q} expects 16 codebooks but got K={K}. "
            f"Falling back to Moshi-only mode (no user stream loss)."
        )

    user_semantic_loss_sum = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    user_acoustic_loss_sum = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    user_semantic_count = 0
    user_acoustic_count = 0

    if has_user_stream and K >= 16:
        # User Semantic: codebook 8, User Acoustic: codebooks 9-15
        user_semantic_logits = logits[:, 8:9]   # [B, 1, T, V]
        user_semantic_target = target[:, 8:9]   # [B, 1, T]
        user_semantic_mask = mask[:, 8:9]       # [B, 1, T]

        user_acoustic_logits = logits[:, 9:16]  # [B, 7, T, V]
        user_acoustic_target = target[:, 9:16]  # [B, 7, T]
        user_acoustic_mask = mask[:, 9:16]      # [B, 7, T]

        user_semantic_loss_sum, user_semantic_count = _compute_codebook_loss_sum_and_count(
            user_semantic_logits, user_semantic_target, user_semantic_mask
        )
        user_acoustic_loss_sum, user_acoustic_count = _compute_codebook_loss_sum_and_count(
            user_acoustic_logits, user_acoustic_target, user_acoustic_mask
        )

    # =========================================================================
    # J-Moshi Style: Token-count based normalization
    # =========================================================================
    # audio_weight = N_semantic * w_s + N_acoustic * w_a
    audio_weight = (
        moshi_semantic_count * semantic_weight +
        moshi_acoustic_count * acoustic_weight
    )
    if has_user_stream and K >= 16:
        audio_weight += (
            user_semantic_count * user_semantic_weight +
            user_acoustic_count * user_acoustic_weight
        )

    # Compute scales
    if audio_weight > 0:
        semantic_scale = semantic_weight / audio_weight
        acoustic_scale = acoustic_weight / audio_weight
        if has_user_stream and K >= 16:
            user_semantic_scale = user_semantic_weight / audio_weight
            user_acoustic_scale = user_acoustic_weight / audio_weight
        else:
            user_semantic_scale = 0.0
            user_acoustic_scale = 0.0
    else:
        semantic_scale = 0.0
        acoustic_scale = 0.0
        user_semantic_scale = 0.0
        user_acoustic_scale = 0.0

    # Compute total audio loss (J-Moshi style)
    total_loss = (
        moshi_semantic_loss_sum * semantic_scale +
        moshi_acoustic_loss_sum * acoustic_scale
    )
    if has_user_stream and K >= 16:
        total_loss = total_loss + (
            user_semantic_loss_sum * user_semantic_scale +
            user_acoustic_loss_sum * user_acoustic_scale
        )

    # =========================================================================
    # Compute mean losses for logging (not for backprop)
    # =========================================================================
    moshi_semantic_loss_mean = (
        moshi_semantic_loss_sum / moshi_semantic_count
        if moshi_semantic_count > 0
        else torch.tensor(0.0, device=logits.device)
    )
    moshi_acoustic_loss_mean = (
        moshi_acoustic_loss_sum / moshi_acoustic_count
        if moshi_acoustic_count > 0
        else torch.tensor(0.0, device=logits.device)
    )

    # Moshi total for logging (using J-Moshi style within Moshi only)
    moshi_audio_weight = (
        moshi_semantic_count * semantic_weight +
        moshi_acoustic_count * acoustic_weight
    )
    if moshi_audio_weight > 0:
        moshi_total_loss = (
            moshi_semantic_loss_sum * semantic_weight +
            moshi_acoustic_loss_sum * acoustic_weight
        ) / moshi_audio_weight
    else:
        moshi_total_loss = torch.tensor(0.0, device=logits.device)

    # User losses for logging
    if has_user_stream and K >= 16:
        user_semantic_loss_mean = (
            user_semantic_loss_sum / user_semantic_count
            if user_semantic_count > 0
            else torch.tensor(0.0, device=logits.device)
        )
        user_acoustic_loss_mean = (
            user_acoustic_loss_sum / user_acoustic_count
            if user_acoustic_count > 0
            else torch.tensor(0.0, device=logits.device)
        )

        # User total for logging
        user_audio_weight = (
            user_semantic_count * user_semantic_weight +
            user_acoustic_count * user_acoustic_weight
        )
        if user_audio_weight > 0:
            user_total_loss = (
                user_semantic_loss_sum * user_semantic_weight +
                user_acoustic_loss_sum * user_acoustic_weight
            ) / user_audio_weight
        else:
            user_total_loss = torch.tensor(0.0, device=logits.device)
    else:
        user_semantic_loss_mean = None
        user_acoustic_loss_mean = None
        user_total_loss = None

    return AudioLossResult(
        total_loss=total_loss,
        moshi_semantic_loss=moshi_semantic_loss_mean,
        moshi_acoustic_loss=moshi_acoustic_loss_mean,
        moshi_total_loss=moshi_total_loss,
        user_semantic_loss=user_semantic_loss_mean,
        user_acoustic_loss=user_acoustic_loss_mean,
        user_total_loss=user_total_loss,
    )
