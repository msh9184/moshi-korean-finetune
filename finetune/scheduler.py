"""
Learning rate scheduler implementations for K-Moshi training.

Supports multiple scheduler types:
- onecycle: PyTorch OneCycleLR (original moshi-finetune)
- cosine_warmup: Cosine annealing with linear warmup (recommended)
- warmup_linear: Linear warmup then constant (J-Moshi DeepSpeed style)
- cosine_restarts: Cosine annealing with warm restarts
"""

import logging
import math
from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR, CosineAnnealingWarmRestarts

logger = logging.getLogger(__name__)


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.

    Learning rate starts at 0, linearly increases to max_lr over warmup_steps,
    then follows cosine decay to min_lr over remaining steps.

    This is the recommended scheduler for K-Moshi training.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        warmup_steps: int = 500,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

        # Store base learning rates before calling super().__init__
        # This is needed because _LRScheduler calls get_lr() in __init__
        self._base_lrs = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self._base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            progress = min(1.0, progress)  # Clamp to [0, 1]

            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self._base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """
    Linear warmup scheduler (J-Moshi DeepSpeed style).

    Learning rate starts at 0, linearly increases to max_lr over warmup_steps,
    then stays constant at max_lr.

    This matches the WarmupLR scheduler used in J-Moshi's DeepSpeed config.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 500,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self._base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self._base_lrs]
        else:
            # Constant after warmup
            return self._base_lrs


def get_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    max_steps: int,
    warmup_steps: int = 500,
    min_lr: float = 1e-7,
    pct_start: float = 0.05,
    t_0: int = 1000,
    t_mult: int = 2,
) -> _LRScheduler:
    """
    Factory function to create a learning rate scheduler.

    Args:
        scheduler_type: Type of scheduler ("onecycle", "cosine_warmup", "warmup_linear", "cosine_restarts")
        optimizer: PyTorch optimizer
        max_steps: Total number of training steps
        warmup_steps: Number of warmup steps (for cosine_warmup, warmup_linear)
        min_lr: Minimum learning rate (for cosine schedulers)
        pct_start: Percentage of steps for warmup (for onecycle)
        t_0: Initial period for cosine_restarts
        t_mult: Period multiplier for cosine_restarts

    Returns:
        Learning rate scheduler
    """
    logger.info(f"Creating scheduler: type={scheduler_type}, max_steps={max_steps}, warmup_steps={warmup_steps}")

    if scheduler_type == "onecycle":
        # Original moshi-finetune scheduler
        max_lr = [group['lr'] for group in optimizer.param_groups]
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max_steps,
            pct_start=pct_start,
        )
        logger.info(f"OneCycleLR: max_lr={max_lr}, pct_start={pct_start}")

    elif scheduler_type == "cosine_warmup":
        # Recommended scheduler for K-Moshi
        scheduler = CosineWarmupScheduler(
            optimizer,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
        )
        logger.info(f"CosineWarmupScheduler: warmup_steps={warmup_steps}, min_lr={min_lr}")

    elif scheduler_type == "warmup_linear":
        # J-Moshi DeepSpeed style
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
        )
        logger.info(f"WarmupLinearScheduler: warmup_steps={warmup_steps}")

    elif scheduler_type == "cosine_restarts":
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,
            T_mult=t_mult,
            eta_min=min_lr,
        )
        logger.info(f"CosineAnnealingWarmRestarts: T_0={t_0}, T_mult={t_mult}, eta_min={min_lr}")

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def get_two_rate_optimizer(
    model: torch.nn.Module,
    tempformer_lr: float,
    depformer_lr: float,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-5,
) -> torch.optim.AdamW:
    """
    Create AdamW optimizer with separate learning rates for TempFormer and DepFormer.

    This follows the J-Moshi approach of using different learning rates for
    the temporal transformer (main transformer) and depth transformer (depformer).

    Args:
        model: Moshi LMModel
        tempformer_lr: Learning rate for temporal transformer parameters
        depformer_lr: Learning rate for depth transformer (depformer) parameters
        weight_decay: L2 regularization weight
        betas: AdamW beta parameters
        eps: Numerical stability epsilon

    Returns:
        AdamW optimizer with parameter groups
    """
    # Separate parameters into tempformer and depformer groups
    tempformer_params = []
    depformer_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "depformer" in name.lower():
            depformer_params.append(param)
        else:
            tempformer_params.append(param)

    logger.info(f"Parameter groups: tempformer={len(tempformer_params)}, depformer={len(depformer_params)}")

    # Create parameter groups with different learning rates
    param_groups = [
        {
            "params": tempformer_params,
            "lr": tempformer_lr,
            "name": "tempformer",
        },
        {
            "params": depformer_params,
            "lr": depformer_lr,
            "name": "depformer",
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        # CRITICAL: foreach=False is required for FSDP resume compatibility
        # The default foreach=True uses multi-tensor operations that require
        # all tensors to be on the same device/dtype, which can break after
        # loading optimizer state from checkpoint in FSDP distributed training.
        foreach=False,
    )

    logger.info(
        f"Created two-rate optimizer: "
        f"tempformer_lr={tempformer_lr}, depformer_lr={depformer_lr}, "
        f"weight_decay={weight_decay}, betas={betas}"
    )

    return optimizer


def get_current_lr(scheduler: _LRScheduler) -> dict:
    """
    Get current learning rates from scheduler.

    Returns a dictionary with learning rates for each parameter group.
    For two-rate optimizer, returns {'tempformer': lr1, 'depformer': lr2}.
    For single-rate optimizer, returns {'lr': lr}.
    """
    lrs = scheduler.get_last_lr()

    # Check if we have named parameter groups
    if hasattr(scheduler.optimizer, 'param_groups'):
        groups = scheduler.optimizer.param_groups
        if len(groups) > 0 and 'name' in groups[0]:
            return {g['name']: lr for g, lr in zip(groups, lrs)}

    # Fallback: return as list or single value
    if len(lrs) == 1:
        return {"lr": lrs[0]}
    else:
        return {f"lr_{i}": lr for i, lr in enumerate(lrs)}
