"""
Metrics Logger for K-Moshi Training.

Provides unified logging interface with:
- TensorBoard (primary, always enabled)
- Weights & Biases (optional, disabled by default)
- JSONL file logging (always enabled)

This module handles import errors gracefully to avoid
TensorFlow/TensorBoard conflicts.
"""

import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Suppress TensorFlow warnings before any imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from finetune.args import TrainArgs, WandbArgs
from finetune.utils import TrainState

logger = logging.getLogger("metrics_logger")

GB = 1024**3

# Lazy load SummaryWriter to avoid TensorFlow import issues
_SummaryWriter = None
_TENSORBOARD_AVAILABLE = None


def _get_summary_writer():
    """Lazy load TensorBoard SummaryWriter with error handling."""
    global _SummaryWriter, _TENSORBOARD_AVAILABLE

    if _TENSORBOARD_AVAILABLE is not None:
        if _TENSORBOARD_AVAILABLE:
            return _SummaryWriter
        else:
            return None

    try:
        # Try PyTorch's built-in TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        _SummaryWriter = SummaryWriter
        _TENSORBOARD_AVAILABLE = True
        logger.debug("TensorBoard initialized successfully (PyTorch)")
        return _SummaryWriter
    except ImportError as e:
        logger.warning(f"PyTorch TensorBoard not available: {e}")
        try:
            # Fallback to standalone tensorboard
            from tensorboard import SummaryWriter
            _SummaryWriter = SummaryWriter
            _TENSORBOARD_AVAILABLE = True
            logger.debug("TensorBoard initialized successfully (standalone)")
            return _SummaryWriter
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            _TENSORBOARD_AVAILABLE = False
            return None
    except Exception as e:
        logger.warning(f"TensorBoard initialization failed: {e}")
        _TENSORBOARD_AVAILABLE = False
        return None


def get_train_logs(
    state: TrainState,
    loss: float,
    num_real_tokens: int,
    lr: float | dict[str, float],
    peak_allocated_mem: float,
    allocated_mem: float,
    train_args: TrainArgs,
    text_loss: Optional[float] = None,
    audio_loss: Optional[float] = None,
    codebook_losses: Optional[list] = None,
) -> dict[str, float | int]:
    """Create training log dictionary with all metrics.

    Args:
        state: Current training state
        loss: Current loss value
        num_real_tokens: Number of real (non-padding) tokens
        lr: Learning rate - either a single float or dict with
            'tempformer' and 'depformer' keys for two-rate optimizer
        peak_allocated_mem: Peak GPU memory allocated (bytes)
        allocated_mem: Current GPU memory allocated (bytes)
        train_args: Training arguments
        text_loss: Optional text stream loss component
        audio_loss: Optional audio stream loss component
        codebook_losses: Optional per-codebook loss list
    """
    metrics = {
        "step": state.step,
        "loss": loss,
        "prob_real_tokens": num_real_tokens / max(state.this_step_tokens, 1),
        "percent_done": 100 * state.step / max(train_args.max_steps, 1),
        "peak_allocated_mem": peak_allocated_mem / GB,
        "allocated_mem": allocated_mem / GB,
        "wps": state.wps,
        "avg_wps": state.avg_wps,
        "eta_in_seconds": state.eta,
    }

    # Handle learning rate - supports both single and two-rate optimizer
    if isinstance(lr, dict):
        # Two-rate optimizer: log both learning rates
        metrics["lr"] = lr.get("tempformer", lr.get("lr", 0.0))
        if "tempformer" in lr:
            metrics["lr_tempformer"] = lr["tempformer"]
        if "depformer" in lr:
            metrics["lr_depformer"] = lr["depformer"]
    else:
        # Single-rate optimizer
        metrics["lr"] = lr

    # Add component losses if provided
    if text_loss is not None:
        metrics["text_loss"] = text_loss
    if audio_loss is not None:
        metrics["audio_loss"] = audio_loss
    if codebook_losses is not None:
        for i, cb_loss in enumerate(codebook_losses):
            metrics[f"codebook_{i}_loss"] = cb_loss

    return metrics


def get_eval_logs(
    step: int,
    train_loss: float,
    perplexity: float | None = None,
    eval_loss: float | None = None,
    text_eval_loss: float | None = None,
    audio_eval_loss: float | None = None,
) -> dict[str, float | int]:
    """Create evaluation log dictionary."""
    eval_dict = {"step": step, "train_loss": train_loss}

    if perplexity is not None:
        eval_dict["perplexity"] = perplexity
    if eval_loss is not None:
        eval_dict["eval_loss"] = eval_loss
    if text_eval_loss is not None:
        eval_dict["text_eval_loss"] = text_eval_loss
    if audio_eval_loss is not None:
        eval_dict["audio_eval_loss"] = audio_eval_loss

    return eval_dict


def train_log_msg(state: TrainState, logs: dict[str, float | int], loss: float) -> str:
    """Format training log message for console output."""
    metrics: dict[str, float | int | datetime] = dict(logs)
    metrics.pop("eta_in_seconds", None)

    metrics["eta"] = datetime.now() + timedelta(seconds=state.eta)
    metrics["step"] = state.step
    metrics["loss"] = loss

    parts = []
    format_specs = [
        ("step", "06", None),
        ("percent_done", "03.1f", "done (%)"),
        ("loss", ".3f", None),
        ("lr", ".1e", None),
        ("peak_allocated_mem", ".1f", "peak_alloc_mem (GB)"),
        ("allocated_mem", ".1f", "alloc_mem (GB)"),
        ("wps", ".1f", "words_per_second"),
        ("avg_wps", ".1f", "avg_words_per_second"),
        ("eta", "%Y-%m-%d %H:%M:%S", "ETA"),
    ]

    for key, fmt, new_name in format_specs:
        name = key if new_name is None else new_name
        if key not in metrics:
            continue
        try:
            parts.append(f"{name}: {metrics[key]:>{fmt}}")
        except (KeyError, ValueError) as e:
            logger.debug(f"Format error for {key}: {e}")

    return " - ".join(parts)


def eval_log_msg(logs: dict[str, float | int]) -> str:
    """Format evaluation log message for console output."""
    parts = []
    format_specs = [
        ("step", "06", None),
        ("perplexity", ".3f", "eval_perplexity"),
        ("eval_loss", ".3f", None),
        ("train_loss", ".3f", None),
        ("text_eval_loss", ".3f", None),
        ("audio_eval_loss", ".3f", None),
    ]

    for key, fmt, new_name in format_specs:
        name = key if new_name is None else new_name
        if key in logs:
            parts.append(f"{name}: {logs[key]:>{fmt}}")

    return " - ".join(parts)


class MetricsLogger:
    """
    Unified metrics logger with TensorBoard (primary) and optional W&B support.

    Features:
    - TensorBoard logging with custom scalar layouts
    - Optional Weights & Biases integration
    - JSONL file logging for offline analysis
    - Graceful handling of import errors
    """

    def __init__(
        self,
        dst_dir: Path,
        tag: str,
        is_master: bool,
        wandb_args: WandbArgs,
        config: dict[str, Any] | None = None,
        enable_tensorboard: bool = True,
        enable_wandb: bool = False,  # Changed default to False
    ):
        self.dst_dir = dst_dir
        self.tag = tag
        self.is_master = is_master
        self.jsonl_path = dst_dir / f"metrics.{tag}.jsonl"
        self.tb_dir = dst_dir / "tensorboard"
        self.summary_writer = None
        self.is_wandb = False
        self.wandb_log = None
        self._loss_ema = None

        if not self.is_master:
            return

        # Initialize TensorBoard
        if enable_tensorboard:
            self._init_tensorboard()

        # Initialize W&B only if explicitly enabled and configured
        if enable_wandb and wandb_args.project is not None:
            self._init_wandb(wandb_args, config)

        # Setup custom scalar layouts for TensorBoard
        if self.summary_writer is not None:
            self._setup_custom_layouts()

    def _init_tensorboard(self):
        """Initialize TensorBoard with error handling."""
        SummaryWriter = _get_summary_writer()
        if SummaryWriter is None:
            logger.warning("TensorBoard not available - logging to JSONL only")
            return

        try:
            filename_suffix = f".{self.tag}"
            self.tb_dir.mkdir(parents=True, exist_ok=True)
            self.summary_writer = SummaryWriter(
                log_dir=str(self.tb_dir),
                max_queue=1000,
                filename_suffix=filename_suffix,
            )
            logger.info(f"TensorBoard initialized: {self.tb_dir}")
            logger.info(f"View with: tensorboard --logdir={self.tb_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.summary_writer = None

    def _init_wandb(self, wandb_args: WandbArgs, config: dict | None):
        """Initialize Weights & Biases with error handling."""
        try:
            import wandb

            if wandb_args.key is not None:
                wandb.login(key=wandb_args.key)
            if wandb_args.offline:
                os.environ["WANDB_MODE"] = "offline"

            if wandb.run is None:
                logger.info("Initializing W&B...")
                wandb.init(
                    config=config,
                    dir=self.dst_dir,
                    project=wandb_args.project,
                    job_type="training",
                    name=wandb_args.run_name or self.dst_dir.name,
                    resume=False,
                )

            self.wandb_log = wandb.log
            self.is_wandb = True
            logger.info("W&B initialized successfully")
        except ImportError:
            logger.info("W&B not installed - using TensorBoard only")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")

    def _setup_custom_layouts(self):
        """Setup custom scalar layouts for organized TensorBoard visualization."""
        if self.summary_writer is None:
            return

        try:
            layout = {
                "K-Moshi Training": {
                    "Loss Overview": ["Multiline", [
                        f"{self.tag}.loss",
                        f"{self.tag}.loss_ema",
                    ]],
                    "Component Losses": ["Multiline", [
                        f"{self.tag}.text_loss",
                        f"{self.tag}.audio_loss",
                    ]],
                },
                "Performance": {
                    "Memory (GB)": ["Multiline", [
                        f"{self.tag}.peak_allocated_mem",
                        f"{self.tag}.allocated_mem",
                    ]],
                    "Speed": ["Multiline", [
                        f"{self.tag}.wps",
                        f"{self.tag}.avg_wps",
                    ]],
                },
            }
            self.summary_writer.add_custom_scalars(layout)
        except Exception as e:
            logger.debug(f"Custom layouts not applied: {e}")

    def log(self, metrics: dict[str, float | int], step: int):
        """Log metrics to all enabled backends."""
        if not self.is_master:
            return

        metrics_to_ignore = {"step"}

        # TensorBoard logging
        if self.summary_writer is not None:
            for key, value in metrics.items():
                if key in metrics_to_ignore:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                try:
                    self.summary_writer.add_scalar(
                        tag=f"{self.tag}.{key}",
                        scalar_value=value,
                        global_step=step,
                    )
                except Exception as e:
                    logger.debug(f"TensorBoard log error for {key}: {e}")

            # Log smoothed loss
            if "loss" in metrics:
                if self._loss_ema is None:
                    self._loss_ema = metrics["loss"]
                else:
                    self._loss_ema = 0.1 * metrics["loss"] + 0.9 * self._loss_ema
                self.summary_writer.add_scalar(
                    f"{self.tag}.loss_ema", self._loss_ema, step
                )

        # W&B logging
        if self.is_wandb and self.wandb_log is not None:
            try:
                self.wandb_log(
                    {
                        f"{self.tag}/{key}": value
                        for key, value in metrics.items()
                        if key not in metrics_to_ignore
                        and isinstance(value, (int, float))
                    },
                    step=step,
                )
            except Exception as e:
                logger.debug(f"W&B log error: {e}")

        # JSONL logging (always enabled)
        self._log_jsonl(metrics, step)

    def _log_jsonl(self, metrics: dict, step: int):
        """Write metrics to JSONL file."""
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            record = dict(metrics)
            if "step" not in record:
                record["step"] = step
            record["at"] = datetime.utcnow().isoformat()

            with self.jsonl_path.open("a") as fp:
                fp.write(f"{json.dumps(record)}\n")
        except Exception as e:
            logger.debug(f"JSONL write error: {e}")

    def log_histogram(self, tag: str, values, step: int):
        """Log histogram to TensorBoard."""
        if not self.is_master or self.summary_writer is None:
            return
        try:
            self.summary_writer.add_histogram(tag, values, step)
        except Exception as e:
            logger.debug(f"Histogram log error: {e}")

    def log_text(self, tag: str, text: str, step: int):
        """Log text to TensorBoard."""
        if not self.is_master or self.summary_writer is None:
            return
        try:
            self.summary_writer.add_text(tag, text, step)
        except Exception as e:
            logger.debug(f"Text log error: {e}")

    def flush(self):
        """Flush all pending writes."""
        if self.summary_writer is not None:
            try:
                self.summary_writer.flush()
            except Exception:
                pass

    def close(self):
        """Close the logger and release resources."""
        if not self.is_master:
            return

        if self.summary_writer is not None:
            try:
                self.summary_writer.close()
            except Exception:
                pass
            self.summary_writer = None

        if self.is_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    def __del__(self):
        """Destructor - warn if not properly closed."""
        if self.summary_writer is not None:
            logger.warning(
                "MetricsLogger not closed properly! "
                "Call close() method before destroying."
            )
