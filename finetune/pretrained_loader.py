"""
Stage-Based Pretrained Model Loading for K-Moshi.

This module provides functionality to load weights from a previous training stage
while initializing new parameters (e.g., speaker_conditioner) from scratch.

Key Features:
    - Partial weight loading (only matching keys)
    - New parameter preservation (random init preserved)
    - FSDP compatibility
    - Automatic checkpoint discovery ("best"/"last" keywords)

Usage:
    from finetune.pretrained_loader import load_pretrained_weights

    # In train.py, after model creation
    loaded, skipped, new_params = load_pretrained_weights(
        model=model,
        args=args.pretrained,
        run_dir=Path(args.run_dir).parent,
    )

See docs/STAGE_PRETRAINED_LOADING_DESIGN.md for detailed documentation.

Author: K-Moshi Development Team
Date: 2026-01-23
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import safetensors.torch
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .args import PretrainedModelArgs
from .distributed import get_rank, get_world_size, is_distributed

logger = logging.getLogger("pretrained_loader")


def main_logger_info(message: str) -> None:
    """Log message only on rank 0."""
    if get_rank() == 0:
        logger.info(message)


def main_logger_warning(message: str) -> None:
    """Log warning only on rank 0."""
    if get_rank() == 0:
        logger.warning(message)


def load_pretrained_weights(
    model: torch.nn.Module,
    args: PretrainedModelArgs,
    run_dir: Optional[Path] = None,
) -> Tuple[int, int, Set[str]]:
    """
    Load pretrained weights from a previous training stage.

    This function performs PARTIAL weight loading:
    - Loads weights that exist in both model and checkpoint
    - Skips weights that only exist in checkpoint (removed modules)
    - Leaves newly added parameters with their initial values

    IMPORTANT: This function should be called BEFORE FSDP wrapping for
    correct parameter loading. If model is already FSDP-wrapped, it will
    attempt to load via FSDP's state_dict API.

    Args:
        model: Target model to load weights into
        args: Pretrained model loading configuration
        run_dir: Current run directory for relative path resolution

    Returns:
        Tuple of (loaded_count, skipped_count, new_params):
        - loaded_count: Number of parameters successfully loaded
        - skipped_count: Number of checkpoint params not in model
        - new_params: Set of model param names not in checkpoint

    Raises:
        FileNotFoundError: If checkpoint file cannot be found
        RuntimeError: If strict=True and unexpected keys are found
    """
    if not args.enabled:
        return 0, 0, set()

    # Resolve checkpoint path
    ckpt_path = _resolve_checkpoint_path(args, run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Could not find pretrained checkpoint. "
            f"path={args.path}, checkpoint_dir={args.checkpoint_dir}"
        )

    main_logger_info(f"[PRETRAINED] Loading weights from: {ckpt_path}")

    # Load checkpoint weights to CPU first
    ckpt_state_dict = safetensors.torch.load_file(str(ckpt_path), device="cpu")
    main_logger_info(f"[PRETRAINED] Loaded checkpoint with {len(ckpt_state_dict)} keys")

    # Get model state dict (handle FSDP if already wrapped)
    is_fsdp = isinstance(model, FSDP)

    if is_fsdp:
        # For FSDP-wrapped models, use FSDP state dict API
        return _load_fsdp_partial(model, ckpt_state_dict, args)
    else:
        # For unwrapped models (before FSDP wrapping), load directly
        return _load_direct_partial(model, ckpt_state_dict, args)


def _load_direct_partial(
    model: torch.nn.Module,
    ckpt_state_dict: Dict[str, torch.Tensor],
    args: PretrainedModelArgs,
) -> Tuple[int, int, Set[str]]:
    """
    Load weights directly into model (before FSDP wrapping).

    This is the preferred method - load weights before FSDP wrapping.
    """
    # Get current model state dict
    model_state_dict = dict(model.state_dict())

    # Compute key differences
    result = _compute_key_differences(
        ckpt_keys=set(ckpt_state_dict.keys()),
        model_keys=set(model_state_dict.keys()),
        expected_new_modules=args.expected_new_modules,
        strict=args.strict,
        verbose=args.verbose,
    )

    common_keys, ckpt_only_keys, model_only_keys = result

    # Create partial state dict with matching shapes only
    partial_state_dict = {}
    shape_mismatches = []

    for key in common_keys:
        ckpt_tensor = ckpt_state_dict[key]
        model_tensor = model_state_dict[key]

        if ckpt_tensor.shape != model_tensor.shape:
            shape_mismatches.append(
                f"{key}: ckpt={ckpt_tensor.shape} vs model={model_tensor.shape}"
            )
            continue

        partial_state_dict[key] = ckpt_tensor

    if shape_mismatches and args.verbose:
        main_logger_warning(
            f"[PRETRAINED] Shape mismatches (skipped): {shape_mismatches[:5]}..."
        )

    # Load the partial state dict with strict=False
    missing, unexpected = model.load_state_dict(partial_state_dict, strict=False)

    # Missing keys are expected (new modules + shape mismatches)
    # Unexpected keys should not occur (we only pass common keys)
    if unexpected:
        main_logger_warning(
            f"[PRETRAINED] Unexpected keys during load: {unexpected[:5]}..."
        )

    main_logger_info(
        f"[PRETRAINED] Successfully loaded {len(partial_state_dict)} parameters"
    )

    return len(partial_state_dict), len(ckpt_only_keys), model_only_keys


def _load_fsdp_partial(
    model: FSDP,
    ckpt_state_dict: Dict[str, torch.Tensor],
    args: PretrainedModelArgs,
) -> Tuple[int, int, Set[str]]:
    """
    Load weights into FSDP-wrapped model.

    This requires special handling via FSDP's state_dict_type context manager.
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    main_logger_info("[PRETRAINED] Loading into FSDP-wrapped model")

    # Get full state dict from FSDP model for key comparison
    full_state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=False  # All ranks need the keys for comparison
    )

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        model_state_dict = dict(model.state_dict())

    # Compute key differences
    result = _compute_key_differences(
        ckpt_keys=set(ckpt_state_dict.keys()),
        model_keys=set(model_state_dict.keys()),
        expected_new_modules=args.expected_new_modules,
        strict=args.strict,
        verbose=args.verbose,
    )

    common_keys, ckpt_only_keys, model_only_keys = result

    # Create partial state dict with matching shapes
    partial_state_dict = {}
    shape_mismatches = []

    for key in common_keys:
        ckpt_tensor = ckpt_state_dict[key]
        model_tensor = model_state_dict[key]

        if ckpt_tensor.shape != model_tensor.shape:
            shape_mismatches.append(
                f"{key}: ckpt={ckpt_tensor.shape} vs model={model_tensor.shape}"
            )
            continue

        partial_state_dict[key] = ckpt_tensor

    if shape_mismatches and args.verbose:
        main_logger_warning(
            f"[PRETRAINED] Shape mismatches (skipped): {shape_mismatches[:5]}..."
        )

    # Load via FSDP state dict API
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        model.load_state_dict(partial_state_dict, strict=False)

    # Synchronize across ranks
    if is_distributed():
        dist.barrier()

    main_logger_info(
        f"[PRETRAINED] Successfully loaded {len(partial_state_dict)} parameters (FSDP)"
    )

    return len(partial_state_dict), len(ckpt_only_keys), model_only_keys


def _compute_key_differences(
    ckpt_keys: Set[str],
    model_keys: Set[str],
    expected_new_modules: list,
    strict: bool,
    verbose: bool,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Compute the differences between checkpoint and model keys.

    Returns:
        Tuple of (common_keys, ckpt_only_keys, model_only_keys)
    """
    # Keys in both
    common_keys = ckpt_keys & model_keys

    # Keys only in checkpoint (will be skipped)
    ckpt_only_keys = ckpt_keys - model_keys

    # Keys only in model (new parameters)
    model_only_keys = model_keys - ckpt_keys

    # Filter out expected new modules
    expected_new_keys = set()
    unexpected_new_keys = set()

    for key in model_only_keys:
        is_expected = any(
            key.startswith(prefix) or f".{prefix}." in key
            for prefix in expected_new_modules
        )
        if is_expected:
            expected_new_keys.add(key)
        else:
            unexpected_new_keys.add(key)

    # Log statistics
    if verbose:
        main_logger_info(f"[PRETRAINED] Checkpoint keys: {len(ckpt_keys)}")
        main_logger_info(f"[PRETRAINED] Model keys: {len(model_keys)}")
        main_logger_info(f"[PRETRAINED] Common keys (to load): {len(common_keys)}")
        main_logger_info(f"[PRETRAINED] Checkpoint-only keys (skipped): {len(ckpt_only_keys)}")
        main_logger_info(f"[PRETRAINED] Expected new params: {len(expected_new_keys)}")
        main_logger_info(f"[PRETRAINED] Unexpected new params: {len(unexpected_new_keys)}")

        if ckpt_only_keys:
            main_logger_warning(
                f"[PRETRAINED] Skipping {len(ckpt_only_keys)} checkpoint keys not in model: "
                f"{sorted(list(ckpt_only_keys))[:5]}..."
            )

        if expected_new_keys:
            main_logger_info(
                f"[PRETRAINED] New modules (randomly initialized): "
                f"{sorted(list(expected_new_keys))[:5]}..."
            )

    # Handle unexpected new keys
    if unexpected_new_keys:
        msg = (
            f"[PRETRAINED] Found {len(unexpected_new_keys)} unexpected new parameters "
            f"not in checkpoint and not in expected_new_modules: "
            f"{sorted(list(unexpected_new_keys))[:5]}..."
        )
        if strict:
            raise RuntimeError(msg)
        else:
            main_logger_warning(msg)

    return common_keys, ckpt_only_keys, model_only_keys


def _resolve_checkpoint_path(
    args: PretrainedModelArgs,
    run_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Resolve checkpoint path from configuration.

    Supports:
        - Absolute path: "/path/to/checkpoint.safetensors"
        - Keywords: "best", "last" (searches in checkpoint_dir)
        - Relative path (to checkpoint_dir or run_dir parent)
    """
    if args.path is None:
        return None

    # Handle "best" or "last" keywords
    if args.path.lower() in ("best", "last"):
        if args.checkpoint_dir is None:
            logger.error(
                f"checkpoint_dir must be specified when path='{args.path}'"
            )
            return None

        ckpt_dir = Path(args.checkpoint_dir)
        if not ckpt_dir.is_absolute() and run_dir:
            ckpt_dir = run_dir / args.checkpoint_dir

        main_logger_info(f"[PRETRAINED] Searching for '{args.path}' in {ckpt_dir}")
        return _find_checkpoint_by_tag(ckpt_dir, args.path.lower())

    # Handle absolute path
    path = Path(args.path)
    if path.is_absolute():
        if path.exists():
            return path
        else:
            logger.error(f"Absolute path does not exist: {path}")
            return None

    # Handle relative path (to checkpoint_dir first, then run_dir)
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        if not ckpt_dir.is_absolute() and run_dir:
            ckpt_dir = run_dir / args.checkpoint_dir
        candidate = ckpt_dir / args.path
        if candidate.exists():
            return candidate

    if run_dir:
        candidate = run_dir / args.path
        if candidate.exists():
            return candidate

    logger.error(
        f"Could not resolve relative path: {args.path}, "
        f"checkpoint_dir={args.checkpoint_dir}, run_dir={run_dir}"
    )
    return None


def _find_checkpoint_by_tag(ckpt_dir: Path, tag: str) -> Optional[Path]:
    """
    Find checkpoint by 'best' or 'last' symlink or by parsing filenames.

    Priority:
        1. Look for *.{tag}.safetensors symlink and resolve it
        2. Fallback: find highest step checkpoint

    Args:
        ckpt_dir: Directory containing checkpoints
        tag: "best" or "last"

    Returns:
        Path to checkpoint file, or None if not found
    """
    if not ckpt_dir.exists():
        logger.error(f"Checkpoint directory does not exist: {ckpt_dir}")
        return None

    # Look for symlink first
    for f in ckpt_dir.iterdir():
        if f.name.endswith(f".{tag}.safetensors"):
            if f.is_symlink():
                target = f.resolve()
                if target.exists():
                    main_logger_info(f"[PRETRAINED] Found {tag} symlink -> {target.name}")
                    return target
                else:
                    main_logger_warning(f"Broken symlink: {f.name} -> {target}")
            elif f.exists():
                main_logger_info(f"[PRETRAINED] Found {tag} file: {f.name}")
                return f

    # Fallback: find by parsing filenames
    # Pattern: {prefix}.{metric_type}-{value}.step-{step}.safetensors
    pattern = re.compile(
        r"^(?P<prefix>.+)\.(?P<metric_type>[\w_]+)-(?P<metric_value>[\d.]+)"
        r"\.step-(?P<step>\d+)\.safetensors$"
    )

    checkpoints = []
    for f in ckpt_dir.iterdir():
        if f.suffix == ".safetensors" and not f.is_symlink():
            # Skip best/last tagged files (they're symlinks or copies)
            if f.name.endswith((".best.safetensors", ".last.safetensors")):
                continue

            match = pattern.match(f.name)
            if match:
                step = int(match.group("step"))
                metric_value = float(match.group("metric_value"))
                checkpoints.append({
                    "path": f,
                    "step": step,
                    "metric_value": metric_value,
                })

    if not checkpoints:
        logger.error(f"No valid checkpoints found in {ckpt_dir}")
        return None

    # Sort by step (descending)
    checkpoints.sort(key=lambda x: x["step"], reverse=True)

    if tag == "last":
        # Return highest step
        result = checkpoints[0]["path"]
    elif tag == "best":
        # Return lowest metric (assuming lower is better)
        checkpoints.sort(key=lambda x: x["metric_value"])
        result = checkpoints[0]["path"]
    else:
        result = checkpoints[0]["path"]

    main_logger_info(
        f"[PRETRAINED] Found {tag} checkpoint by filename parsing: {result.name}"
    )
    return result


def verify_pretrained_loading(
    model: torch.nn.Module,
    loaded_count: int,
    new_params: Set[str],
) -> bool:
    """
    Verify that pretrained loading was successful.

    Checks:
        - At least some parameters were loaded
        - New parameters are non-zero (have been initialized)

    Returns:
        True if verification passes, False otherwise
    """
    if loaded_count == 0:
        main_logger_warning("[PRETRAINED] No parameters were loaded!")
        return False

    # Check that new parameters are properly initialized (non-zero)
    for name, param in model.named_parameters():
        if name in new_params:
            if param.numel() > 0:
                # Check for all-zero (might indicate failed init)
                if (param == 0).all():
                    main_logger_warning(
                        f"[PRETRAINED] New parameter {name} is all zeros - "
                        "verify initialization is correct"
                    )

    main_logger_info("[PRETRAINED] Loading verification passed")
    return True
