"""
Checkpoint saving and resume utilities for distributed training.

This module provides comprehensive checkpoint management with:
- Metric-based checkpoint naming for easy model selection
- Automatic best/last model tracking with symlinks
- Full resume support (model + optimizer + scheduler + training state + RNG)
- Configurable retention policy

═══════════════════════════════════════════════════════════════════════════════
CHECKPOINT FILE STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

runs/{run_name}/checkpoints/
├── config.json                                           # Model config (shared)
├── checkpoint.eval_loss-5.430.step-000040.safetensors   # Model weights
├── checkpoint.eval_loss-4.472.step-000050.safetensors
├── checkpoint.eval_loss-3.570.step-000080.safetensors
├── checkpoint.eval_loss-3.570.step-000080.best.safetensors  # Symlink → best
├── checkpoint.eval_loss-3.570.step-000080.last.safetensors  # Symlink → latest
├── training_state.step-000080.last.pt                   # Training state (last)
└── training_state.step-000080.best.pt                   # Training state (best)

═══════════════════════════════════════════════════════════════════════════════
NAMING FORMAT
═══════════════════════════════════════════════════════════════════════════════

Model weights: {name_prefix}.{metric_type}-{value:.3f}.step-{step:06d}.safetensors
Training state: training_state.step-{step:06d}.{best|last}.pt

Supports both FSDP and DDP wrapped models.
"""

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import safetensors.torch
import torch
from moshi.models.lm import LMModel
from moshi.modules.lora import LoRALinear
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel

from .args import CheckpointArgs
from .distributed import get_rank, get_world_size, is_distributed
from .utils import TrainState

logger = logging.getLogger("checkpointing")


def main_logger_info(message: str) -> None:
    """Log message only on rank 0."""
    if get_rank() == 0:
        logger.info(message)


def safe_barrier() -> None:
    """Synchronize all processes if distributed is initialized."""
    if dist.is_initialized():
        dist.barrier()


def get_save_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


class CheckpointManager:
    """
    Comprehensive checkpoint manager with resume support.

    Handles:
    - Model weights saving (safetensors format)
    - Training state saving (optimizer, scheduler, RNG)
    - Best/Last model tracking with symlinks
    - Automatic old checkpoint cleanup
    - Resume from checkpoint

    Example usage:
        # Initialize
        ckpt_manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=train_state,
            config=model_config,
            args=checkpoint_args,
            run_dir=run_dir,
            full_finetuning=False,
        )

        # Check for resume
        if ckpt_manager.should_resume():
            ckpt_manager.load_checkpoint()

        # Save during training
        ckpt_manager.save_checkpoint(metric_value=eval_loss)
    """

    # Regex pattern for parsing checkpoint filenames
    CKPT_PATTERN = re.compile(
        r"^(?P<prefix>.+)\.(?P<metric_type>[\w_]+)-(?P<metric_value>[\d.]+)"
        r"\.step-(?P<step>\d+)(?:\.(?P<tag>best|last))?\.safetensors$"
    )

    def __init__(
        self,
        model: Union[FullyShardedDataParallel, DistributedDataParallel, LMModel],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,  # LR scheduler
        state: TrainState,
        config: dict,
        args: CheckpointArgs,
        run_dir: Path | str,
        full_finetuning: bool = False,
    ):
        """
        Initialize CheckpointManager.

        Args:
            model: The model (FSDP, DDP, or raw LMModel)
            optimizer: The optimizer
            scheduler: The LR scheduler
            state: TrainState instance
            config: Model configuration dictionary
            args: CheckpointArgs configuration
            run_dir: Directory to save checkpoints
            full_finetuning: Whether this is full finetuning (vs LoRA)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = state
        self.config = config
        self.args = args
        self.run_dir = Path(run_dir)
        self.full_finetuning = full_finetuning
        self.rank = get_rank()

        # Determine model wrapper type
        self._is_fsdp = isinstance(model, FullyShardedDataParallel)
        self._is_ddp = isinstance(model, DistributedDataParallel)

        # Create checkpoint directory
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Track saved checkpoints for cleanup
        self._saved_checkpoints: list[Path] = []

        # Initialize metric tracking from state
        self.state.metric_type = args.metric_type

        # Initialize checkpoint state (repair symlinks, enforce max_keep)
        if self.rank == 0:
            self._initialize_checkpoint_state()

    @property
    def ckpt_dir(self) -> Path:
        """Checkpoint directory path."""
        return self.run_dir / "checkpoints"

    # =========================================================================
    # FILE NAMING
    # =========================================================================

    def _format_checkpoint_name(
        self,
        step: int,
        metric_value: float,
        tag: Optional[str] = None,
    ) -> str:
        """
        Format checkpoint filename.

        Args:
            step: Training step
            metric_value: Metric value for this checkpoint
            tag: Optional tag ("best" or "last")

        Returns:
            Formatted filename (without path)
        """
        name = (
            f"{self.args.name_prefix}."
            f"{self.args.metric_type}-{metric_value:.3f}."
            f"step-{step:06d}"
        )
        if tag:
            name += f".{tag}"
        name += ".safetensors"
        return name

    def _format_training_state_name(self, step: int, tag: str) -> str:
        """Format training state filename."""
        return f"training_state.step-{step:06d}.{tag}.pt"

    def _parse_checkpoint_name(self, filename: str) -> Optional[dict]:
        """
        Parse checkpoint filename to extract metadata.

        Args:
            filename: Checkpoint filename

        Returns:
            Dictionary with prefix, metric_type, metric_value, step, tag
            or None if pattern doesn't match
        """
        match = self.CKPT_PATTERN.match(filename)
        if match:
            return {
                "prefix": match.group("prefix"),
                "metric_type": match.group("metric_type"),
                "metric_value": float(match.group("metric_value")),
                "step": int(match.group("step")),
                "tag": match.group("tag"),
            }
        return None

    # =========================================================================
    # RESUME DETECTION
    # =========================================================================

    def should_resume(self) -> bool:
        """
        Check if training should resume from checkpoint.

        Returns:
            True if resume should occur, False otherwise
        """
        if not self.args.enabled:
            return False

        # Explicit resume path takes priority
        if self.args.resume_from:
            resume_path = self._resolve_resume_path(self.args.resume_from)
            if resume_path and resume_path.exists():
                main_logger_info(f"[RESUME] Explicit checkpoint found: {resume_path}")
                return True
            else:
                main_logger_info(f"[RESUME] Explicit checkpoint not found: {self.args.resume_from}")
                return False

        # Check for automatic resume
        if self.args.resume_if_exist:
            last_ckpt = self.find_last_checkpoint()
            if last_ckpt:
                main_logger_info(f"[RESUME] Last checkpoint found: {last_ckpt}")
                return True

        return False

    def _resolve_resume_path(self, path_str: str) -> Optional[Path]:
        """Resolve resume path (absolute or relative to ckpt_dir)."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        else:
            return self.ckpt_dir / path

    def find_last_checkpoint(self) -> Optional[Path]:
        """
        Find the last saved checkpoint.

        Looks for:
        1. *.last.safetensors symlink
        2. Highest step checkpoint if no symlink

        Returns:
            Path to last checkpoint, or None if not found
        """
        if not self.ckpt_dir.exists():
            return None

        # Try to find .last symlink
        for f in self.ckpt_dir.iterdir():
            if f.name.endswith(".last.safetensors"):
                # Resolve symlink to actual file
                if f.is_symlink():
                    target = f.resolve()
                    if target.exists():
                        return target
                elif f.exists():
                    return f

        # Fallback: find highest step checkpoint
        checkpoints = []
        for f in self.ckpt_dir.iterdir():
            if f.suffix == ".safetensors" and not f.name.endswith((".best.safetensors", ".last.safetensors")):
                info = self._parse_checkpoint_name(f.name)
                if info:
                    checkpoints.append((info["step"], f))

        if checkpoints:
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            return checkpoints[0][1]

        return None

    def find_best_checkpoint(self) -> Optional[Path]:
        """Find the best checkpoint by metric."""
        if not self.ckpt_dir.exists():
            return None

        # Try to find .best symlink
        for f in self.ckpt_dir.iterdir():
            if f.name.endswith(".best.safetensors"):
                if f.is_symlink():
                    target = f.resolve()
                    if target.exists():
                        return target
                elif f.exists():
                    return f

        return None

    # =========================================================================
    # CHECKPOINT STATE MANAGEMENT (Symlink Repair & Enforcement)
    # =========================================================================

    def _initialize_checkpoint_state(self) -> None:
        """
        Initialize checkpoint state on startup.

        This method is called during __init__ on rank 0 only and performs:
        1. Scan all existing checkpoints
        2. Repair broken or missing symlinks (safetensors)
        3. Repair broken or missing training_state files (.pt)
        4. Enforce max_keep retention policy
        5. Log checkpoint state summary

        This ensures consistent state after training interruptions or restarts.
        """
        if not self.ckpt_dir.exists():
            return

        main_logger_info("[CHECKPOINT] Initializing checkpoint state...")

        # Scan all existing checkpoints
        checkpoints = self._scan_all_checkpoints()

        if not checkpoints:
            main_logger_info("[CHECKPOINT] No existing checkpoints found")
            return

        main_logger_info(f"[CHECKPOINT] Found {len(checkpoints)} existing checkpoints")

        # Repair safetensors symlinks (best/last)
        self._repair_symlinks(checkpoints)

        # Repair training_state files (best/last)
        self._repair_training_states(checkpoints)

        # Enforce max_keep
        self._enforce_max_keep(checkpoints)

        main_logger_info("[CHECKPOINT] Checkpoint state initialization complete")

        # Log summary
        self._log_checkpoint_summary(checkpoints)

    def _scan_all_checkpoints(self) -> list[dict]:
        """
        Scan all checkpoints in the checkpoint directory.

        Returns:
            List of checkpoint info dictionaries, each containing:
            - path: Path to the checkpoint file
            - prefix: Filename prefix
            - metric_type: Type of metric (e.g., "eval_loss")
            - metric_value: Numeric metric value
            - step: Training step number
            - tag: Optional tag ("best", "last", or None)
            - is_symlink: Whether this is a symlink

        Sorted by step (newest first), then by metric value.
        """
        if not self.ckpt_dir.exists():
            return []

        checkpoints = []

        for f in self.ckpt_dir.iterdir():
            # Skip non-safetensors files
            if f.suffix != ".safetensors":
                continue

            # Skip symlinks for the main checkpoint list
            # (we'll handle them separately)
            is_symlink = f.is_symlink()

            # Parse filename
            info = self._parse_checkpoint_name(f.name)
            if info is None:
                continue

            # Add path and symlink info
            info["path"] = f
            info["is_symlink"] = is_symlink

            # Skip .best and .last symlinks from main list
            if info["tag"] in ("best", "last"):
                continue

            checkpoints.append(info)

        # Sort by step (descending), then by metric value (ascending for loss)
        # This puts newest first, and best metric first within same step
        checkpoints.sort(
            key=lambda x: (-x["step"], x["metric_value"])
        )

        return checkpoints

    def _find_best_by_metric(
        self,
        checkpoints: list[dict],
        metric_best: str = "min",
    ) -> Optional[dict]:
        """
        Find the best checkpoint by metric value.

        Args:
            checkpoints: List of checkpoint info dictionaries
            metric_best: "min" for lower is better, "max" for higher is better

        Returns:
            Best checkpoint info dict, or None if no checkpoints
        """
        if not checkpoints:
            return None

        if metric_best == "min":
            return min(checkpoints, key=lambda x: x["metric_value"])
        else:
            return max(checkpoints, key=lambda x: x["metric_value"])

    def _find_latest_by_step(self, checkpoints: list[dict]) -> Optional[dict]:
        """
        Find the latest checkpoint by step number.

        Args:
            checkpoints: List of checkpoint info dictionaries

        Returns:
            Latest checkpoint info dict, or None if no checkpoints
        """
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda x: x["step"])

    def _repair_symlinks(self, checkpoints: list[dict]) -> None:
        """
        Repair broken or missing best/last symlinks.

        This method ensures:
        1. Broken symlinks (pointing to deleted targets) are removed
        2. Missing "last" symlink is created pointing to highest step checkpoint
        3. Missing "best" symlink is created pointing to best metric checkpoint
        4. Symlinks point to correct targets based on current checkpoint state

        Args:
            checkpoints: List of checkpoint info dictionaries (non-symlink only)
        """
        if not checkpoints:
            return

        # Clean up orphan symlinks first
        self._cleanup_orphan_symlinks()

        # Find current best and latest checkpoints
        best_ckpt = self._find_best_by_metric(checkpoints, self.args.metric_best)
        latest_ckpt = self._find_latest_by_step(checkpoints)

        # Check existing symlinks
        existing_best = None
        existing_last = None
        best_target_step = None
        last_target_step = None

        for f in self.ckpt_dir.iterdir():
            if not f.is_symlink():
                continue

            if f.name.endswith(".best.safetensors"):
                if f.exists():  # Symlink target exists
                    target = f.resolve()
                    target_info = self._parse_checkpoint_name(target.name)
                    if target_info:
                        best_target_step = target_info["step"]
                existing_best = f

            elif f.name.endswith(".last.safetensors"):
                if f.exists():  # Symlink target exists
                    target = f.resolve()
                    target_info = self._parse_checkpoint_name(target.name)
                    if target_info:
                        last_target_step = target_info["step"]
                existing_last = f

        # Repair "last" symlink
        if latest_ckpt:
            needs_last_repair = False

            if existing_last is None:
                needs_last_repair = True
                main_logger_info("[CHECKPOINT] Missing 'last' symlink, creating...")
            elif not existing_last.exists():
                needs_last_repair = True
                main_logger_info("[CHECKPOINT] Broken 'last' symlink, repairing...")
                existing_last.unlink(missing_ok=True)
            elif last_target_step != latest_ckpt["step"]:
                needs_last_repair = True
                main_logger_info(
                    f"[CHECKPOINT] 'last' symlink outdated "
                    f"(points to step {last_target_step}, latest is {latest_ckpt['step']}), updating..."
                )
                existing_last.unlink(missing_ok=True)

            if needs_last_repair:
                self._create_symlink_atomic(
                    latest_ckpt["path"],
                    tag="last",
                    step=latest_ckpt["step"],
                    metric_value=latest_ckpt["metric_value"],
                )

        # Repair "best" symlink
        if best_ckpt:
            needs_best_repair = False

            if existing_best is None:
                needs_best_repair = True
                main_logger_info("[CHECKPOINT] Missing 'best' symlink, creating...")
            elif not existing_best.exists():
                needs_best_repair = True
                main_logger_info("[CHECKPOINT] Broken 'best' symlink, repairing...")
                existing_best.unlink(missing_ok=True)
            elif best_target_step is not None:
                # Check if current best symlink points to the actual best checkpoint
                current_best_info = None
                for ckpt in checkpoints:
                    if ckpt["step"] == best_target_step:
                        current_best_info = ckpt
                        break

                if current_best_info:
                    # Compare metrics - update if actual best is different from symlink target
                    if self.args.metric_best == "min":
                        if best_ckpt["metric_value"] < current_best_info["metric_value"]:
                            needs_best_repair = True
                            main_logger_info(
                                f"[CHECKPOINT] Found better checkpoint: "
                                f"step {best_ckpt['step']} ({best_ckpt['metric_value']:.4f}) < "
                                f"step {current_best_info['step']} ({current_best_info['metric_value']:.4f})"
                            )
                    else:
                        if best_ckpt["metric_value"] > current_best_info["metric_value"]:
                            needs_best_repair = True
                            main_logger_info(
                                f"[CHECKPOINT] Found better checkpoint: "
                                f"step {best_ckpt['step']} ({best_ckpt['metric_value']:.4f}) > "
                                f"step {current_best_info['step']} ({current_best_info['metric_value']:.4f})"
                            )
                else:
                    # Symlink points to a checkpoint that was deleted (not in checkpoints list)
                    needs_best_repair = True
                    main_logger_info(
                        f"[CHECKPOINT] 'best' symlink points to deleted checkpoint (step {best_target_step}), repairing..."
                    )

            if needs_best_repair:
                # Remove old best symlink
                for f in self.ckpt_dir.iterdir():
                    if f.is_symlink() and f.name.endswith(".best.safetensors"):
                        f.unlink(missing_ok=True)

                self._create_symlink_atomic(
                    best_ckpt["path"],
                    tag="best",
                    step=best_ckpt["step"],
                    metric_value=best_ckpt["metric_value"],
                )

            # =====================================================================
            # CRITICAL: Always update state with best metric
            # =====================================================================
            # This ensures that when training resumes, the state knows the current
            # best metric value for correct comparison in update_best()
            # Without this, the state might have stale/wrong best_metric after resume,
            # causing new checkpoints with better metrics to not be recognized as best.
            self.state.best_metric = best_ckpt["metric_value"]
            self.state.best_step = best_ckpt["step"]
            main_logger_info(
                f"[CHECKPOINT] State synced: best_metric={best_ckpt['metric_value']:.4f}, "
                f"best_step={best_ckpt['step']}"
            )

    def _sync_best_metric_from_checkpoints(self) -> None:
        """
        Sync best_metric from actual checkpoints after loading training state.

        This is called after load_checkpoint() to ensure the state has the correct
        best_metric value. The training_state file may have an outdated best_metric
        if newer checkpoints were saved after it was created.

        This function:
        1. Scans all existing checkpoints
        2. Finds the best checkpoint by metric value
        3. Updates state.best_metric and state.best_step
        """
        if not self.ckpt_dir.exists():
            return

        checkpoints = self._scan_all_checkpoints()
        if not checkpoints:
            return

        # Find the best checkpoint by actual metric value
        best_ckpt = self._find_best_by_metric(checkpoints, self.args.metric_best)
        if best_ckpt is None:
            return

        # Compare with current state
        old_best_metric = self.state.best_metric
        old_best_step = self.state.best_step

        # Update state with actual best
        self.state.best_metric = best_ckpt["metric_value"]
        self.state.best_step = best_ckpt["step"]

        if old_best_metric != best_ckpt["metric_value"] or old_best_step != best_ckpt["step"]:
            main_logger_info(
                f"[RESUME] Best metric synced from checkpoints: "
                f"best_metric={best_ckpt['metric_value']:.4f} (was {old_best_metric}), "
                f"best_step={best_ckpt['step']} (was {old_best_step})"
            )
        else:
            main_logger_info(
                f"[RESUME] Best metric confirmed: "
                f"best_metric={best_ckpt['metric_value']:.4f}, best_step={best_ckpt['step']}"
            )

    def _cleanup_orphan_symlinks(self) -> None:
        """Remove symlinks pointing to non-existent targets."""
        if not self.ckpt_dir.exists():
            return

        for f in self.ckpt_dir.iterdir():
            if f.is_symlink() and not f.exists():
                main_logger_info(f"[CHECKPOINT] Removing orphan symlink: {f.name}")
                f.unlink(missing_ok=True)

    def _scan_training_states(self) -> dict[str, list[dict]]:
        """
        Scan all training_state files in the checkpoint directory.

        Returns:
            Dictionary with keys "last" and "best", each containing a list of
            training state info dictionaries with step and path.
        """
        result = {"last": [], "best": []}

        if not self.ckpt_dir.exists():
            return result

        # Pattern: training_state.step-XXXXXX.{last|best}.pt
        pattern = re.compile(r"^training_state\.step-(\d+)\.(last|best)\.pt$")

        for f in self.ckpt_dir.iterdir():
            if f.is_symlink():
                continue

            match = pattern.match(f.name)
            if match:
                step = int(match.group(1))
                tag = match.group(2)
                result[tag].append({"step": step, "path": f})

        # Sort by step (newest first)
        for tag in result:
            result[tag].sort(key=lambda x: -x["step"])

        return result

    def _repair_training_states(self, checkpoints: list[dict]) -> None:
        """
        Repair missing or inconsistent training_state files.

        This ensures training_state.*.last.pt and training_state.*.best.pt files
        exist and are consistent with the corresponding safetensors symlinks.

        Key behaviors:
        1. If training_state.*.last.pt is missing for the latest checkpoint,
           create it or copy from an existing training_state file
        2. If training_state.*.best.pt is missing for the best checkpoint,
           create it or copy from an existing training_state file
        3. Clean up orphaned training_state files (pointing to non-existent checkpoints)

        Args:
            checkpoints: List of checkpoint info dictionaries from _scan_all_checkpoints()
        """
        if not checkpoints:
            return

        main_logger_info("[CHECKPOINT] Checking training_state files...")

        # Find best and latest checkpoints
        best_ckpt = self._find_best_by_metric(checkpoints, self.args.metric_best)
        latest_ckpt = self._find_latest_by_step(checkpoints)

        # Scan existing training_state files
        training_states = self._scan_training_states()

        # Get all valid checkpoint steps for orphan detection
        valid_steps = {ckpt["step"] for ckpt in checkpoints}

        # =====================================================================
        # Repair "last" training_state
        # =====================================================================
        if latest_ckpt:
            latest_step = latest_ckpt["step"]
            expected_last_name = self._format_training_state_name(latest_step, "last")
            expected_last_path = self.ckpt_dir / expected_last_name

            if not expected_last_path.exists():
                # Try to find any existing training_state for this step
                source_state = self._find_training_state_for_step(latest_step)

                if source_state:
                    # Copy existing training_state with updated tag
                    self._copy_training_state(source_state, expected_last_path, "last")
                    main_logger_info(
                        f"[CHECKPOINT] Created training_state 'last' for step {latest_step} "
                        f"(copied from {source_state.name})"
                    )
                else:
                    # Create minimal training_state without optimizer/scheduler
                    self._create_minimal_training_state(
                        latest_step,
                        latest_ckpt["metric_value"],
                        "last",
                    )
                    main_logger_info(
                        f"[CHECKPOINT] Created minimal training_state 'last' for step {latest_step}"
                    )

        # =====================================================================
        # Repair "best" training_state
        # =====================================================================
        if best_ckpt:
            best_step = best_ckpt["step"]
            expected_best_name = self._format_training_state_name(best_step, "best")
            expected_best_path = self.ckpt_dir / expected_best_name

            if not expected_best_path.exists():
                # Try to find any existing training_state for this step
                source_state = self._find_training_state_for_step(best_step)

                if source_state:
                    # Copy existing training_state with updated tag
                    self._copy_training_state(source_state, expected_best_path, "best")
                    main_logger_info(
                        f"[CHECKPOINT] Created training_state 'best' for step {best_step} "
                        f"(copied from {source_state.name})"
                    )
                else:
                    # Create minimal training_state without optimizer/scheduler
                    self._create_minimal_training_state(
                        best_step,
                        best_ckpt["metric_value"],
                        "best",
                    )
                    main_logger_info(
                        f"[CHECKPOINT] Created minimal training_state 'best' for step {best_step}"
                    )

        # =====================================================================
        # Clean up orphaned training_state files
        # =====================================================================
        self._cleanup_orphan_training_states(valid_steps, latest_ckpt, best_ckpt)

        main_logger_info("[CHECKPOINT] Training state files check complete")

    def _find_training_state_for_step(self, step: int) -> Optional[Path]:
        """
        Find any existing training_state file for a given step.

        Prefers 'last' over 'best' if both exist.

        Args:
            step: Training step number

        Returns:
            Path to training_state file if found, None otherwise
        """
        # Try 'last' first
        last_path = self.ckpt_dir / self._format_training_state_name(step, "last")
        if last_path.exists() and not last_path.is_symlink():
            return last_path

        # Try 'best'
        best_path = self.ckpt_dir / self._format_training_state_name(step, "best")
        if best_path.exists() and not best_path.is_symlink():
            return best_path

        return None

    def _copy_training_state(
        self,
        source_path: Path,
        dest_path: Path,
        new_tag: str,
    ) -> None:
        """
        Copy a training_state file with updated metadata.

        Args:
            source_path: Source training_state file
            dest_path: Destination path
            new_tag: New tag ("last" or "best") for metadata update
        """
        try:
            # Load source state
            state_data = torch.load(source_path, map_location="cpu")

            # Update metadata timestamp
            if "metadata" in state_data:
                state_data["metadata"]["timestamp"] = datetime.now().isoformat()

            # Save to destination
            torch.save(state_data, dest_path)

        except Exception as e:
            logger.warning(f"Failed to copy training_state {source_path} -> {dest_path}: {e}")

    def _create_minimal_training_state(
        self,
        step: int,
        metric_value: float,
        tag: str,
    ) -> Path:
        """
        Create a minimal training_state file without optimizer/scheduler state.

        This is used when no existing training_state can be found for a checkpoint.
        The resulting file can be used for basic resume (step tracking) but will
        require fresh optimizer/scheduler initialization.

        Args:
            step: Training step number
            metric_value: Metric value for this checkpoint
            tag: "last" or "best"

        Returns:
            Path to created training_state file
        """
        state_name = self._format_training_state_name(step, tag)
        state_path = self.ckpt_dir / state_name

        # Build minimal training state from current state
        minimal_state = {
            "step": step,
            "train_state": {
                "step": step,
                "best_metric": self.state.best_metric if tag == "best" else metric_value,
                "best_step": step if tag == "best" else self.state.best_step,
                "metric_type": self.args.metric_type,
                # Include essential fields with defaults
                "train_samples": getattr(self.state, "train_samples", 0),
                "train_loss": getattr(self.state, "train_loss", 0.0),
                "this_eval_loss": metric_value,
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "world_size": get_world_size(),
                "metric_type": self.args.metric_type,
                "metric_value": metric_value,
                "model_file": self._format_checkpoint_name(step, metric_value),
                "is_minimal": True,  # Flag indicating this is a minimal state
            },
            # No optimizer_state_dict - will trigger fresh optimizer on resume
            # No scheduler_state_dict - will trigger fresh scheduler on resume
            # No rng_state - will use current RNG state on resume
        }

        torch.save(minimal_state, state_path)
        return state_path

    def _cleanup_orphan_training_states(
        self,
        valid_steps: set[int],
        latest_ckpt: Optional[dict],
        best_ckpt: Optional[dict],
    ) -> None:
        """
        Clean up orphaned training_state files.

        Removes training_state files that:
        1. Point to non-existent checkpoint steps
        2. Are old 'last' files not matching current latest step
        3. Are old 'best' files not matching current best step

        Protected files:
        - Current latest step's training_state.*.last.pt
        - Current best step's training_state.*.best.pt

        Args:
            valid_steps: Set of valid checkpoint step numbers
            latest_ckpt: Latest checkpoint info dict
            best_ckpt: Best checkpoint info dict
        """
        if not self.ckpt_dir.exists():
            return

        protected_files = set()

        # Protect current last
        if latest_ckpt:
            protected_files.add(
                self._format_training_state_name(latest_ckpt["step"], "last")
            )

        # Protect current best
        if best_ckpt:
            protected_files.add(
                self._format_training_state_name(best_ckpt["step"], "best")
            )

        # Also protect training_state files for the same step (cross-protection)
        # e.g., if step 100 is both best and last, protect both .best.pt and .last.pt
        if latest_ckpt:
            protected_files.add(
                self._format_training_state_name(latest_ckpt["step"], "best")
            )
        if best_ckpt:
            protected_files.add(
                self._format_training_state_name(best_ckpt["step"], "last")
            )

        # Pattern for training_state files
        pattern = re.compile(r"^training_state\.step-(\d+)\.(last|best)\.pt$")

        deleted_count = 0
        for f in self.ckpt_dir.iterdir():
            if f.is_symlink():
                continue

            match = pattern.match(f.name)
            if not match:
                continue

            # Skip protected files
            if f.name in protected_files:
                continue

            step = int(match.group(1))
            tag = match.group(2)

            should_delete = False

            # Delete if checkpoint doesn't exist
            if step not in valid_steps:
                should_delete = True
                reason = "orphaned (checkpoint deleted)"

            # Delete old 'last' files (not matching current latest)
            elif tag == "last" and latest_ckpt and step != latest_ckpt["step"]:
                should_delete = True
                reason = f"outdated last (current={latest_ckpt['step']})"

            # Delete old 'best' files (not matching current best)
            elif tag == "best" and best_ckpt and step != best_ckpt["step"]:
                should_delete = True
                reason = f"outdated best (current={best_ckpt['step']})"

            if should_delete:
                try:
                    f.unlink()
                    deleted_count += 1
                    main_logger_info(
                        f"[CHECKPOINT] Deleted {reason} training_state: {f.name}"
                    )
                except OSError as e:
                    logger.warning(f"Failed to delete training_state {f.name}: {e}")

        if deleted_count > 0:
            main_logger_info(
                f"[CHECKPOINT] Cleaned up {deleted_count} orphaned training_state files"
            )

    def _create_symlink_atomic(
        self,
        target_path: Path,
        tag: str,
        step: int,
        metric_value: float,
    ) -> Path:
        """
        Create a symlink atomically to prevent race conditions.

        Args:
            target_path: Path to the target checkpoint file
            tag: "best" or "last"
            step: Step number for naming
            metric_value: Metric value for naming

        Returns:
            Path to created symlink
        """
        # Format symlink name
        link_name = self._format_checkpoint_name(step, metric_value, tag=tag)
        link_path = self.ckpt_dir / link_name

        # Create temporary symlink first
        tmp_link = self.ckpt_dir / f".tmp.{link_name}"

        try:
            # Remove any existing temp link
            tmp_link.unlink(missing_ok=True)

            # Create symlink to relative path for portability
            tmp_link.symlink_to(target_path.name)

            # Atomic rename
            tmp_link.rename(link_path)

            main_logger_info(f"[CHECKPOINT] Created symlink: {link_name} -> {target_path.name}")

        except OSError as e:
            logger.warning(f"Failed to create symlink {link_name}: {e}")
            tmp_link.unlink(missing_ok=True)
            raise

        return link_path

    def _enforce_max_keep(self, checkpoints: list[dict]) -> None:
        """
        Enforce max_keep retention policy.

        Removes oldest checkpoints exceeding max_keep limit while preserving:
        - The checkpoint pointed to by "best" symlink
        - The checkpoint pointed to by "last" symlink

        Args:
            checkpoints: List of checkpoint info dictionaries
        """
        if self.args.max_keep is None:
            return

        if len(checkpoints) <= self.args.max_keep:
            return

        main_logger_info(
            f"[CHECKPOINT] Enforcing max_keep={self.args.max_keep}, "
            f"current count={len(checkpoints)}"
        )

        # Find protected checkpoints (best and last targets)
        protected_steps = set()

        for f in self.ckpt_dir.iterdir():
            if f.is_symlink() and f.exists():
                if f.name.endswith((".best.safetensors", ".last.safetensors")):
                    target = f.resolve()
                    target_info = self._parse_checkpoint_name(target.name)
                    if target_info:
                        protected_steps.add(target_info["step"])

        # Sort by step (oldest first for deletion)
        checkpoints_sorted = sorted(checkpoints, key=lambda x: x["step"])

        # Calculate how many to delete
        to_delete_count = len(checkpoints) - self.args.max_keep

        deleted_count = 0
        for ckpt in checkpoints_sorted:
            if deleted_count >= to_delete_count:
                break

            # Don't delete protected checkpoints
            if ckpt["step"] in protected_steps:
                main_logger_info(
                    f"[CHECKPOINT] Keeping protected checkpoint: step {ckpt['step']}"
                )
                continue

            # Delete checkpoint file
            try:
                ckpt["path"].unlink()
                main_logger_info(f"[CHECKPOINT] Deleted old checkpoint: {ckpt['path'].name}")
                deleted_count += 1

                # Also delete associated training state files
                for tag in ["last", "best"]:
                    state_name = self._format_training_state_name(ckpt["step"], tag)
                    state_path = self.ckpt_dir / state_name
                    if state_path.exists() and not state_path.is_symlink():
                        # Only delete if not current best or last step
                        if ckpt["step"] not in protected_steps:
                            state_path.unlink(missing_ok=True)
                            main_logger_info(f"[CHECKPOINT] Deleted training state: {state_name}")

            except OSError as e:
                logger.warning(f"Failed to delete checkpoint {ckpt['path']}: {e}")

        main_logger_info(
            f"[CHECKPOINT] Cleanup complete: deleted {deleted_count} checkpoints"
        )

    def _log_checkpoint_summary(self, checkpoints: list[dict]) -> None:
        """Log a summary of checkpoint state including training_state files."""
        if not checkpoints:
            return

        best_ckpt = self._find_best_by_metric(checkpoints, self.args.metric_best)
        latest_ckpt = self._find_latest_by_step(checkpoints)

        summary_lines = [
            "[CHECKPOINT] ═══════════════════════════════════════════════",
            f"[CHECKPOINT] Checkpoint Summary:",
            f"[CHECKPOINT]   Total checkpoints: {len(checkpoints)}",
            f"[CHECKPOINT]   max_keep: {self.args.max_keep}",
        ]

        if latest_ckpt:
            summary_lines.append(
                f"[CHECKPOINT]   Latest: step {latest_ckpt['step']} "
                f"({latest_ckpt['metric_type']}={latest_ckpt['metric_value']:.4f})"
            )

        if best_ckpt:
            summary_lines.append(
                f"[CHECKPOINT]   Best: step {best_ckpt['step']} "
                f"({best_ckpt['metric_type']}={best_ckpt['metric_value']:.4f})"
            )

        # Check symlink status
        best_link_ok = False
        last_link_ok = False
        for f in self.ckpt_dir.iterdir():
            if f.is_symlink():
                if f.name.endswith(".best.safetensors") and f.exists():
                    best_link_ok = True
                elif f.name.endswith(".last.safetensors") and f.exists():
                    last_link_ok = True

        summary_lines.append(
            f"[CHECKPOINT]   Symlinks: best={'✓' if best_link_ok else '✗'}, "
            f"last={'✓' if last_link_ok else '✗'}"
        )

        # Check training_state file status
        best_state_ok = False
        last_state_ok = False

        if latest_ckpt:
            last_state_path = self.ckpt_dir / self._format_training_state_name(
                latest_ckpt["step"], "last"
            )
            last_state_ok = last_state_path.exists() and not last_state_path.is_symlink()

        if best_ckpt:
            best_state_path = self.ckpt_dir / self._format_training_state_name(
                best_ckpt["step"], "best"
            )
            best_state_ok = best_state_path.exists() and not best_state_path.is_symlink()

        summary_lines.append(
            f"[CHECKPOINT]   Training states: best={'✓' if best_state_ok else '✗'}, "
            f"last={'✓' if last_state_ok else '✗'}"
        )

        summary_lines.append(
            "[CHECKPOINT] ═══════════════════════════════════════════════"
        )

        for line in summary_lines:
            main_logger_info(line)

    def repair_checkpoint_state(self) -> None:
        """
        Public method to manually repair checkpoint state.

        This can be called externally to force a repair of symlinks,
        training_state files, and enforcement of max_keep policy.
        Useful for recovery scenarios.

        Only executes on rank 0.
        """
        if self.rank != 0:
            return

        main_logger_info("[CHECKPOINT] Manual checkpoint state repair requested...")
        checkpoints = self._scan_all_checkpoints()

        if checkpoints:
            self._repair_symlinks(checkpoints)
            self._repair_training_states(checkpoints)
            self._enforce_max_keep(checkpoints)
            self._log_checkpoint_summary(checkpoints)

        main_logger_info("[CHECKPOINT] Manual repair complete")

    def get_checkpoint_stats(self) -> dict:
        """
        Get statistics about current checkpoint state.

        Returns:
            Dictionary with checkpoint statistics
        """
        checkpoints = self._scan_all_checkpoints()

        if not checkpoints:
            return {
                "count": 0,
                "best_step": None,
                "best_metric": None,
                "latest_step": None,
                "latest_metric": None,
                "symlinks_ok": False,
            }

        best_ckpt = self._find_best_by_metric(checkpoints, self.args.metric_best)
        latest_ckpt = self._find_latest_by_step(checkpoints)

        # Check symlinks
        best_link_ok = False
        last_link_ok = False
        for f in self.ckpt_dir.iterdir():
            if f.is_symlink():
                if f.name.endswith(".best.safetensors") and f.exists():
                    best_link_ok = True
                elif f.name.endswith(".last.safetensors") and f.exists():
                    last_link_ok = True

        return {
            "count": len(checkpoints),
            "best_step": best_ckpt["step"] if best_ckpt else None,
            "best_metric": best_ckpt["metric_value"] if best_ckpt else None,
            "latest_step": latest_ckpt["step"] if latest_ckpt else None,
            "latest_metric": latest_ckpt["metric_value"] if latest_ckpt else None,
            "symlinks_ok": best_link_ok and last_link_ok,
            "best_link_ok": best_link_ok,
            "last_link_ok": last_link_ok,
            "max_keep": self.args.max_keep,
        }

    # =========================================================================
    # MODEL STATE RETRIEVAL
    # =========================================================================

    @staticmethod
    def get_non_lora_states(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Filter out LoRA-specific keys from state dict."""
        return {
            k: v
            for k, v in state_dict.items()
            if not any(l_key in k for l_key in ["lora", "frozen"])
        }

    def _get_unwrapped_model(self) -> LMModel:
        """Get the underlying LMModel from wrapped distributed model."""
        if self._is_ddp:
            return self.model.module
        elif self._is_fsdp:
            return self.model
        else:
            return self.model

    @torch.no_grad()
    def _retrieve_model_states(
        self,
        save_only_lora: bool,
        save_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """
        Retrieve model states for checkpointing.

        For FSDP: Uses summon_full_params to gather sharded parameters
        For DDP: Directly accesses model.module

        Args:
            save_only_lora: If True, only save LoRA adapter weights
            save_dtype: Data type for saved tensors

        Returns:
            Dictionary of parameter names to tensors
        """
        assert not (save_only_lora and self.full_finetuning), (
            "Cannot save LoRA checkpoint as LoRA training is not enabled."
        )

        # Remove all potential hooks from previous saves
        for module in self.model.modules():
            if isinstance(module, LoRALinear) and hasattr(module, "_merge_lora_handle"):
                module._merge_lora_handle.remove()

        # Route to appropriate handler
        if self._is_ddp:
            states = self._retrieve_ddp_states(save_only_lora, save_dtype)
        elif self._is_fsdp:
            states = self._retrieve_fsdp_states(save_only_lora, save_dtype)
        else:
            states = self._retrieve_single_gpu_states(save_only_lora, save_dtype)

        states = dict(sorted(states.items()))
        return states

    @torch.no_grad()
    def _retrieve_ddp_states(
        self,
        save_only_lora: bool,
        save_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Retrieve states from DDP wrapped model."""
        inner_model = self.model.module

        if save_only_lora:
            states = {}
            for name, param in inner_model.named_parameters():
                if param.requires_grad and "lora" in name:
                    states[name] = param.clone().to(dtype=save_dtype)
            return states
        else:
            def merge_lora(m, destination, prefix, *args):
                weight = m.merge_weight()
                destination[prefix + "weight"] = weight

            for module in inner_model.modules():
                if isinstance(module, LoRALinear):
                    module._merge_lora_handle = module._register_state_dict_hook(merge_lora)

            states = self.get_non_lora_states(inner_model.state_dict())
            states = {k: v.clone().to(dtype=save_dtype) for k, v in states.items()}
            return states

    @torch.no_grad()
    def _retrieve_single_gpu_states(
        self,
        save_only_lora: bool,
        save_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Retrieve states from unwrapped model (single GPU case)."""
        if save_only_lora:
            states = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and "lora" in name:
                    states[name] = param.clone().to(dtype=save_dtype)
            return states
        else:
            def merge_lora(m, destination, prefix, *args):
                weight = m.merge_weight()
                destination[prefix + "weight"] = weight

            for module in self.model.modules():
                if isinstance(module, LoRALinear):
                    module._merge_lora_handle = module._register_state_dict_hook(merge_lora)

            states = self.get_non_lora_states(self.model.state_dict())
            states = {k: v.clone().to(dtype=save_dtype) for k, v in states.items()}
            return states

    @torch.no_grad()
    def _retrieve_fsdp_states(
        self,
        save_only_lora: bool,
        save_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Retrieve states from FSDP wrapped model."""
        offload_to_cpu = get_world_size() > 1

        if save_only_lora:
            def is_trainable_fsdp(module):
                is_fsdp = isinstance(module, FullyShardedDataParallel) or get_world_size() == 1
                all_params_have_grads = is_fsdp and all(p.requires_grad for p in module.parameters())
                is_leaf_node = is_fsdp and (
                    get_world_size() == 1 or len(list(module.module.children())) == 0
                )
                return is_fsdp and all_params_have_grads and is_leaf_node

            modules = {k: m for k, m in self.model.named_modules() if is_trainable_fsdp(m)}

            states = {}
            for key, module in modules.items():
                parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", ""
                )
                if get_world_size() > 1:
                    with module.summon_full_params(module, writeback=True, offload_to_cpu=offload_to_cpu):
                        states.update({
                            f"{parent_prefix}.{k}": v.to(dtype=save_dtype)
                            for k, v in module.state_dict().items()
                        })
                else:
                    states.update({
                        f"{parent_prefix}.{k}": v.clone().to(dtype=save_dtype)
                        for k, v in module.state_dict().items()
                    })
        else:
            def merge_lora(m, destination, prefix, *args):
                weight = m.merge_weight()
                destination[prefix + "weight"] = weight

            for module in self.model.modules():
                if isinstance(module, LoRALinear):
                    module._merge_lora_handle = module._register_state_dict_hook(merge_lora)

            if get_world_size() > 1:
                with self.model.summon_full_params(self.model, writeback=True, offload_to_cpu=offload_to_cpu):
                    states = self.get_non_lora_states(self.model.state_dict())
                    states = {k: v.to(dtype=save_dtype) for k, v in states.items()}
            else:
                states = self.get_non_lora_states(self.model.state_dict())
                states = {k: v.clone().to(dtype=save_dtype) for k, v in states.items()}

        return states

    # =========================================================================
    # SAVE CHECKPOINT
    # =========================================================================

    @torch.no_grad()
    def save_checkpoint(
        self,
        metric_value: float,
        force_save: bool = False,
    ) -> Optional[Path]:
        """
        Save checkpoint with metric-based naming.

        Args:
            metric_value: Current metric value (e.g., eval_loss)
            force_save: Force save even if not at save_freq

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        if not self.args.enabled:
            return None

        # Check if we should save at this step
        step = self.state.step
        should_save = force_save or (
            self.args.save_freq > 0 and step > 0 and step % self.args.save_freq == 0
        )

        if not should_save:
            return None

        # Determine if this is the best model
        is_new_best = self.state.update_best(metric_value, self.args.metric_best)

        main_logger_info(f"[CHECKPOINT] Saving at step {step}, metric={metric_value:.4f}, is_best={is_new_best}")

        # Get save dtype
        save_dtype = get_save_dtype(self.args.save_dtype)
        save_only_lora = self.args.save_adapters_only and not self.full_finetuning

        # Create checkpoint filename
        ckpt_name = self._format_checkpoint_name(step, metric_value)
        ckpt_path = self.ckpt_dir / ckpt_name
        tmp_path = self.ckpt_dir / f"tmp.{ckpt_name}"

        # =====================================================================
        # PHASE 1: Gather states (ALL RANKS must participate in collectives)
        # =====================================================================

        # Retrieve model states - ALL ranks participate (uses FSDP.summon_full_params)
        with torch.no_grad():
            model_states = self._retrieve_model_states(save_only_lora, save_dtype)

        # Gather optimizer state - ALL ranks must participate (uses FSDP.optim_state_dict)
        optim_state = None
        if self.args.save_optimizer and self.optimizer is not None:
            if self._is_fsdp:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                # Get full optimizer state dict - ALL ranks participate, result on rank 0 only
                full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                    optim_state = FSDP.optim_state_dict(self.model, self.optimizer)
                main_logger_info("[CHECKPOINT] Optimizer state gathered from FSDP")
            else:
                optim_state = self.optimizer.state_dict()

        # =====================================================================
        # PHASE 2: Synchronize all ranks
        # =====================================================================
        if is_distributed():
            safe_barrier()

        # =====================================================================
        # PHASE 3: Only rank 0 writes files to disk
        # =====================================================================
        if self.rank == 0:
            # Save model weights
            safetensors.torch.save_file(model_states, tmp_path)

            # Atomic rename
            tmp_path.rename(ckpt_path)

            # Save config.json (overwrite each time)
            config_path = self.ckpt_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=4)

            # Save training state (optimizer state is already gathered)
            self._save_training_state_rank0(step, metric_value, "last", optim_state)
            if is_new_best:
                self._save_training_state_rank0(step, metric_value, "best", optim_state)

            # Update symlinks
            self._update_symlinks(ckpt_path, is_new_best)

            # Track saved checkpoint
            self._saved_checkpoints.append(ckpt_path)

            # Cleanup old checkpoints
            if self.args.max_keep is not None:
                self._cleanup_old_checkpoints()

            main_logger_info(f"[CHECKPOINT] Saved: {ckpt_path.name}")

        # =====================================================================
        # PHASE 4: Final synchronization
        # =====================================================================
        if is_distributed():
            safe_barrier()

        return ckpt_path

    def _save_training_state_rank0(
        self,
        step: int,
        metric_value: float,
        tag: str,
        optim_state: Optional[dict] = None,
    ) -> Path:
        """
        Save training state on rank 0 only.

        This function should ONLY be called on rank 0.
        The optimizer state should be pre-gathered using FSDP.optim_state_dict()
        BEFORE calling this function (in save_checkpoint where all ranks participate).

        Args:
            step: Current step
            metric_value: Current metric value
            tag: "last" or "best"
            optim_state: Pre-gathered optimizer state dict (from FSDP collective)

        Returns:
            Path to saved training state
        """
        state_name = self._format_training_state_name(step, tag)
        state_path = self.ckpt_dir / state_name

        training_state = {
            "step": step,
            "train_state": self.state.to_dict(),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "world_size": get_world_size(),
                "metric_type": self.args.metric_type,
                "metric_value": metric_value,
                "model_file": self._format_checkpoint_name(step, metric_value),
            },
        }

        # Save pre-gathered optimizer state (no collective operations here!)
        if self.args.save_optimizer and optim_state is not None:
            training_state["optimizer_state_dict"] = optim_state

        # Save scheduler state (not a collective, safe on rank 0 only)
        if self.args.save_scheduler and self.scheduler is not None:
            training_state["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save RNG state for reproducibility (not a collective, safe on rank 0 only)
        if self.args.save_rng_state:
            training_state["rng_state"] = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }

        # Save to file
        torch.save(training_state, state_path)
        main_logger_info(f"[CHECKPOINT] Training state saved: {state_path.name}")

        return state_path

    def _update_symlinks(self, ckpt_path: Path, is_new_best: bool):
        """
        Update best/last symlinks to point to current checkpoint.

        This method uses atomic symlink creation for safety and ensures:
        1. Old symlinks are properly removed before creating new ones
        2. Training state files are cleaned up consistently
        3. Best symlink is only updated when there's a new best model

        Args:
            ckpt_path: Path to the newly saved checkpoint
            is_new_best: Whether this is a new best checkpoint
        """
        step = self.state.step

        # Get metric value from checkpoint filename for consistency
        info = self._parse_checkpoint_name(ckpt_path.name)
        metric_value = info["metric_value"] if info else 0.0

        # =====================================================================
        # Update "last" symlink
        # =====================================================================

        # Remove all old .last symlinks first
        for f in self.ckpt_dir.iterdir():
            if f.name.endswith(".last.safetensors"):
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass

        # Create new last symlink atomically
        try:
            self._create_symlink_atomic(
                target_path=ckpt_path,
                tag="last",
                step=step,
                metric_value=metric_value,
            )
        except OSError as e:
            logger.warning(f"Failed to create 'last' symlink: {e}")

        # Remove old training_state.*.last.pt files (keep only current step)
        current_state_name = self._format_training_state_name(step, "last")
        for f in self.ckpt_dir.iterdir():
            if f.name.startswith("training_state.") and f.name.endswith(".last.pt"):
                if f.name != current_state_name and not f.is_symlink():
                    try:
                        f.unlink(missing_ok=True)
                    except OSError:
                        pass

        # =====================================================================
        # Update "best" symlink (only if new best)
        # =====================================================================

        if is_new_best:
            best_metric = self.state.best_metric if self.state.best_metric is not None else metric_value

            # Remove all old .best symlinks first
            for f in self.ckpt_dir.iterdir():
                if f.name.endswith(".best.safetensors"):
                    try:
                        f.unlink(missing_ok=True)
                    except OSError:
                        pass

            # Create new best symlink atomically
            try:
                self._create_symlink_atomic(
                    target_path=ckpt_path,
                    tag="best",
                    step=step,
                    metric_value=best_metric,
                )
            except OSError as e:
                logger.warning(f"Failed to create 'best' symlink: {e}")

            # Remove old training_state.*.best.pt files (keep only current step)
            current_best_state_name = self._format_training_state_name(step, "best")
            for f in self.ckpt_dir.iterdir():
                if f.name.startswith("training_state.") and f.name.endswith(".best.pt"):
                    if f.name != current_best_state_name and not f.is_symlink():
                        try:
                            f.unlink(missing_ok=True)
                        except OSError:
                            pass

            # Update best step in state
            self.state.best_step = step

    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints keeping only max_keep most recent.

        This method:
        1. Identifies all non-symlink checkpoint files
        2. Protects checkpoints pointed to by best/last symlinks
        3. Deletes oldest checkpoints exceeding max_keep
        4. Cleans up associated training state files
        """
        if self.args.max_keep is None:
            return

        # Get all regular checkpoints (not symlinks, not best/last tagged)
        checkpoints = []
        for f in self.ckpt_dir.iterdir():
            if f.suffix == ".safetensors" and not f.is_symlink():
                if not f.name.endswith((".best.safetensors", ".last.safetensors")):
                    info = self._parse_checkpoint_name(f.name)
                    if info:
                        checkpoints.append((info["step"], info["metric_value"], f))

        if len(checkpoints) <= self.args.max_keep:
            return

        # Find protected checkpoints (targets of best/last symlinks)
        protected_steps = set()
        for f in self.ckpt_dir.iterdir():
            if f.is_symlink() and f.exists():
                if f.name.endswith((".best.safetensors", ".last.safetensors")):
                    target = f.resolve()
                    target_info = self._parse_checkpoint_name(target.name)
                    if target_info:
                        protected_steps.add(target_info["step"])

        # Also protect current step and best step from state
        protected_steps.add(self.state.step)
        if self.state.best_step is not None:
            protected_steps.add(self.state.best_step)

        # Sort by step (newest first)
        checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Determine candidates for deletion (beyond max_keep)
        to_check = checkpoints[self.args.max_keep:]

        deleted_count = 0
        for step, metric_value, ckpt_path in to_check:
            # Never delete protected checkpoints
            if step in protected_steps:
                main_logger_info(
                    f"[CHECKPOINT] Keeping protected checkpoint: {ckpt_path.name} (step {step})"
                )
                continue

            try:
                ckpt_path.unlink()
                deleted_count += 1
                main_logger_info(f"[CHECKPOINT] Deleted old checkpoint: {ckpt_path.name}")

                # Also delete associated training state files
                for tag in ["last", "best"]:
                    state_name = self._format_training_state_name(step, tag)
                    state_path = self.ckpt_dir / state_name
                    if state_path.exists() and not state_path.is_symlink():
                        state_path.unlink(missing_ok=True)
                        main_logger_info(f"[CHECKPOINT] Deleted training state: {state_name}")

            except OSError as e:
                logger.warning(f"Failed to delete checkpoint {ckpt_path}: {e}")

        if deleted_count > 0:
            main_logger_info(f"[CHECKPOINT] Cleanup: deleted {deleted_count} old checkpoints")

    # =========================================================================
    # LOAD CHECKPOINT
    # =========================================================================

    def load_checkpoint(self, ckpt_path: Optional[Path] = None) -> Tuple[int, bool]:
        """
        Load checkpoint and restore all states.

        Args:
            ckpt_path: Explicit checkpoint path (optional).
                      If None, finds last checkpoint automatically.

        Returns:
            Tuple of (restored_step, success)
        """
        # Find checkpoint to load
        if ckpt_path is None:
            if self.args.resume_from:
                ckpt_path = self._resolve_resume_path(self.args.resume_from)
            else:
                ckpt_path = self.find_last_checkpoint()

        if ckpt_path is None or not ckpt_path.exists():
            main_logger_info("[RESUME] No checkpoint found to resume from")
            return 0, False

        main_logger_info(f"[RESUME] Loading checkpoint: {ckpt_path}")

        # Parse checkpoint info
        info = self._parse_checkpoint_name(ckpt_path.name)
        if info is None:
            logger.error(f"Could not parse checkpoint filename: {ckpt_path.name}")
            return 0, False

        step = info["step"]

        # Load model weights
        self._load_model_weights(ckpt_path)

        # Find and load training state
        training_state_loaded = self._load_training_state(step)

        if training_state_loaded:
            main_logger_info(f"[RESUME] Restored to step {self.state.step}")
        else:
            # If no training state, just restore step
            self.state.step = step
            main_logger_info(f"[RESUME] Model weights loaded, step set to {step} (no training state found)")

        # =====================================================================
        # CRITICAL: Sync best_metric from actual checkpoints after loading
        # =====================================================================
        # The training_state file contains the best_metric from when it was saved,
        # but newer checkpoints may have been saved since then (or the best may
        # have changed). We need to re-scan and find the actual best checkpoint.
        if self.rank == 0:
            self._sync_best_metric_from_checkpoints()

        # Synchronize all ranks after loading
        safe_barrier()

        return self.state.step, True

    def _load_model_weights(self, ckpt_path: Path):
        """Load model weights from safetensors file."""
        main_logger_info(f"[RESUME] Loading model weights from {ckpt_path.name}")

        # Load weights
        state_dict = safetensors.torch.load_file(ckpt_path, device="cpu")

        # Get unwrapped model for loading
        if self._is_ddp:
            target_model = self.model.module
        elif self._is_fsdp:
            # For FSDP, we need special handling
            # The weights should be loaded after FSDP wrapping
            target_model = self.model
        else:
            target_model = self.model

        # Check if these are LoRA weights or full weights
        has_lora = any("lora" in k for k in state_dict.keys())

        if has_lora:
            # LoRA weights - load only trainable params
            missing, unexpected = target_model.load_state_dict(state_dict, strict=False)
            if missing:
                # Filter out non-LoRA missing keys (expected)
                lora_missing = [k for k in missing if "lora" in k]
                if lora_missing:
                    logger.warning(f"Missing LoRA keys: {lora_missing[:5]}...")
        else:
            # Full weights
            if self._is_fsdp:
                # For FSDP with full weights, need summon_full_params
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                    self.model.load_state_dict(state_dict)
            else:
                target_model.load_state_dict(state_dict)

        main_logger_info("[RESUME] Model weights loaded successfully")

    def _load_training_state(self, step: int) -> bool:
        """
        Load training state (optimizer, scheduler, RNG).

        Args:
            step: Step number to look for

        Returns:
            True if training state was found and loaded
        """
        # Try to find training state file
        state_path = None

        # First try exact step with .last tag
        for tag in ["last", "best"]:
            candidate = self.ckpt_dir / self._format_training_state_name(step, tag)
            if candidate.exists():
                state_path = candidate
                break

        # Fallback: find any training state with matching step
        if state_path is None:
            for f in self.ckpt_dir.iterdir():
                if f.name.startswith(f"training_state.step-{step:06d}") and f.suffix == ".pt":
                    state_path = f
                    break

        if state_path is None:
            logger.warning(f"No training state found for step {step}")
            return False

        main_logger_info(f"[RESUME] Loading training state from {state_path.name}")

        # Load training state
        training_state = torch.load(state_path, map_location="cpu")

        # Restore TrainState
        if "train_state" in training_state:
            self.state = TrainState.from_dict(
                training_state["train_state"],
                max_steps=self.state.max_steps,
            )
            main_logger_info(f"[RESUME] TrainState restored: {self.state}")

        # Restore optimizer state
        if self.args.save_optimizer and "optimizer_state_dict" in training_state:
            try:
                if self._is_fsdp:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    from torch.distributed.fsdp import FullOptimStateDictConfig, StateDictType

                    optim_state = training_state["optimizer_state_dict"]

                    # IMPORTANT: Use offload_to_cpu=False for loading!
                    # When loading, we need tensors on GPU to match model parameters.
                    # Using offload_to_cpu=True causes device mismatch error in optimizer.step()
                    full_optim_state_dict_config = FullOptimStateDictConfig(
                        offload_to_cpu=False,  # Changed from True - tensors should be on GPU
                        rank0_only=False
                    )
                    with FSDP.state_dict_type(
                        self.model, StateDictType.FULL_STATE_DICT,
                        optim_state_dict_config=full_optim_state_dict_config
                    ):
                        optim_state = FSDP.optim_state_dict_to_load(
                            self.model, self.optimizer, optim_state
                        )
                        self.optimizer.load_state_dict(optim_state)

                    # Verify and fix optimizer state device AND dtype
                    self._verify_optimizer_state()
                else:
                    self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
                main_logger_info("[RESUME] Optimizer state restored")
            except Exception as e:
                logger.warning(f"[RESUME] Failed to restore optimizer state: {e}")
                logger.warning("[RESUME] Training will continue with fresh optimizer state")

        # Restore scheduler state
        if self.args.save_scheduler and "scheduler_state_dict" in training_state:
            try:
                self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
                main_logger_info("[RESUME] Scheduler state restored")
            except Exception as e:
                logger.warning(f"[RESUME] Failed to restore scheduler state: {e}")
                logger.warning("[RESUME] Training will continue with fresh scheduler state")

        # Restore RNG state
        if self.args.save_rng_state and "rng_state" in training_state:
            try:
                rng_state = training_state["rng_state"]
                if "torch" in rng_state:
                    torch.set_rng_state(rng_state["torch"])
                if "cuda" in rng_state and rng_state["cuda"] is not None:
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
                main_logger_info("[RESUME] RNG state restored")
            except Exception as e:
                logger.warning(f"[RESUME] Failed to restore RNG state: {e}")
                logger.warning("[RESUME] Training will continue with fresh RNG state")

        return True

    def _verify_optimizer_state(self) -> None:
        """
        Verify and fix optimizer state device AND dtype placement after loading.

        After loading optimizer state from a checkpoint, tensors might have:
        1. Wrong device (CPU instead of GPU, or wrong GPU)
        2. Wrong dtype (should be float32 for exp_avg/exp_avg_sq, not bfloat16!)

        CRITICAL: AdamW optimizer states (exp_avg, exp_avg_sq) must ALWAYS be float32!
        - Gradients are computed in float32 during backward pass
        - exp_avg.lerp_(grad, ...) requires both tensors to have same dtype
        - Using bfloat16 for optimizer states causes numerical instability

        This method checks and fixes both issues to prevent RuntimeError:
        "expected dtype c10::BFloat16 for `end` but got dtype float"

        IMPORTANT: For FSDP, we iterate directly over optimizer.state items,
        because the param objects in param_groups might not match the keys
        in optimizer.state after optim_state_dict_to_load().
        """
        if self.optimizer is None:
            return

        device_fixes = 0
        dtype_fixes = 0
        total_state_tensors = 0
        params_found = 0

        # FSDP-compatible approach: iterate directly over optimizer.state
        # The keys in optimizer.state are the actual parameter tensors
        for param, state in self.optimizer.state.items():
            if not isinstance(param, torch.Tensor):
                continue

            params_found += 1
            param_device = param.device

            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue

                total_state_tensors += 1

                # Skip "step" tensor - it can have different dtype (usually int64 or float32)
                if key == "step":
                    continue

                needs_fix = False
                target_device = param_device

                # CRITICAL: AdamW optimizer states (exp_avg, exp_avg_sq) must be float32!
                # This is because:
                # 1. Gradients are computed in float32 during backward (even with bf16 params)
                # 2. exp_avg.lerp_(grad, beta) requires matching dtypes
                # 3. float32 provides numerical stability for momentum/variance tracking
                #
                # DO NOT match param_dtype - that would break if params are bfloat16!
                target_dtype = torch.float32

                # Check device mismatch
                if value.device != target_device:
                    device_fixes += 1
                    needs_fix = True

                # Check dtype mismatch (must be float32, not bfloat16!)
                if value.dtype != target_dtype:
                    dtype_fixes += 1
                    needs_fix = True

                # Apply fix if needed
                if needs_fix:
                    state[key] = value.to(device=target_device, dtype=target_dtype)

        # Always log state info for debugging FSDP resume issues
        main_logger_info(
            f"[RESUME] Optimizer state verification: "
            f"params={params_found}, tensors={total_state_tensors}, "
            f"device_fixes={device_fixes}, dtype_fixes={dtype_fixes}"
        )

        if dtype_fixes > 0:
            main_logger_info(
                f"[RESUME] Converted {dtype_fixes} optimizer state tensors to float32 "
                f"(required for AdamW gradient compatibility)"
            )


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# Keep the old Checkpointer class for backward compatibility
# This wraps the new CheckpointManager with the old interface

class Checkpointer:
    """
    Legacy checkpointer class for backward compatibility.

    This class provides the old interface while using CheckpointManager internally.
    For new code, use CheckpointManager directly.
    """

    def __init__(
        self,
        model: Union[FullyShardedDataParallel, DistributedDataParallel, LMModel],
        state: TrainState,
        run_dir: Path | str,
        config: dict,
        optimizer: torch.optim.Optimizer | None = None,
        num_ckpt_keep: int | None = None,
        full_finetuning: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.state = state
        self.run_dir = Path(run_dir)
        self.rank = get_rank()
        self.num_ckpt_keep = num_ckpt_keep
        self.full_finetuning = full_finetuning
        self.config = config

        self._is_fsdp = isinstance(model, FullyShardedDataParallel)
        self._is_ddp = isinstance(model, DistributedDataParallel)

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def dst_dir(self) -> Path:
        return self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "consolidated"

    @staticmethod
    def consolidated_path(ckpt_dir: Path, save_only_lora: bool = False) -> Path:
        suffix = "safetensors"
        prefix = "lora" if save_only_lora else "consolidated"
        return ckpt_dir / f"{prefix}.{suffix}"

    @staticmethod
    def _tmp(ckpt_dir: Path) -> Path:
        return ckpt_dir.with_name(f"tmp.{ckpt_dir.name}")

    def delete_old_ckpts(self) -> list[Path]:
        all_saved_ckpts = [d for d in self.ckpt_dir.iterdir() if d.is_dir()]
        all_saved_ckpts.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        ckpts_to_delete = all_saved_ckpts[self.num_ckpt_keep:]

        for ckpt_to_delete in ckpts_to_delete:
            try:
                shutil.rmtree(ckpt_to_delete)
                main_logger_info(f"Deleted ckpt: {ckpt_to_delete}")
            except OSError as e:
                main_logger_info(f"Error deleting directory {ckpt_to_delete}: {e}")

        return ckpts_to_delete

    def write_params_info(self, tmp_dst: Path):
        params_path = tmp_dst / "config.json"
        with open(params_path, "w") as f:
            f.write(json.dumps(self.config, indent=4))

    @staticmethod
    def get_non_lora_states(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            k: v for k, v in state_dict.items()
            if not any(l_key in k for l_key in ["lora", "frozen"])
        }

    def _get_unwrapped_model(self) -> LMModel:
        if self._is_ddp:
            return self.model.module
        elif self._is_fsdp:
            return self.model
        else:
            return self.model

    @torch.no_grad()
    def retrieve_save_states(self, save_only_lora: bool, save_dtype: torch.dtype) -> dict[str, torch.Tensor]:
        # Delegate to CheckpointManager's implementation
        manager = CheckpointManager.__new__(CheckpointManager)
        manager.model = self.model
        manager.full_finetuning = self.full_finetuning
        manager._is_fsdp = self._is_fsdp
        manager._is_ddp = self._is_ddp
        return manager._retrieve_model_states(save_only_lora, save_dtype)

    @torch.no_grad()
    def save_checkpoint(self, save_only_lora: bool, dtype: torch.dtype = torch.float16):
        if self.full_finetuning:
            assert not save_only_lora, "Cannot save LoRA checkpoint in full finetuning"

        tmp_dst = self._tmp(self.dst_dir)
        main_logger_info(f"Dumping checkpoint in {self.dst_dir} using tmp name: {tmp_dst.name}")

        assert not self.dst_dir.exists(), f"dst exists {self.dst_dir}"
        tmp_dst.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            states = self.retrieve_save_states(save_only_lora, dtype)

        safe_barrier()

        if self.rank == 0:
            safetensors.torch.save_file(
                states, self.consolidated_path(tmp_dst, save_only_lora=save_only_lora)
            )
            self.write_params_info(tmp_dst)
            assert not self.dst_dir.exists(), f"should not happen! {self.dst_dir}"
            tmp_dst.rename(self.dst_dir)

            logger.info(f"Done dumping checkpoint in {self.dst_dir} for step: {self.state.step}")

            if self.num_ckpt_keep is not None:
                ckpts_to_delete = self.delete_old_ckpts()
                logger.info(f"Done deleting checkpoints {', '.join([str(c) for c in ckpts_to_delete])}")

        main_logger_info("Done!")
