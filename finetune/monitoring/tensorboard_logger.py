"""
Advanced TensorBoard Logging System for K-Moshi Training.

Features:
- Multi-level loss tracking (total, text, audio, per-codebook)
- Gradient statistics and health monitoring
- Learning rate scheduling visualization
- Memory usage tracking
- Training speed metrics
- Model parameter statistics
- Audio sample logging (optional)
- Custom scalars for grouped visualization
- Histogram logging for weight distributions
- Hyperparameter tracking
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger("tensorboard_logger")

# Lazy import to avoid TensorFlow dependency issues
_SummaryWriter = None


def get_summary_writer():
    """Lazy load SummaryWriter to avoid import issues."""
    global _SummaryWriter
    if _SummaryWriter is None:
        # Suppress TensorFlow warnings
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        try:
            from torch.utils.tensorboard import SummaryWriter
            _SummaryWriter = SummaryWriter
        except ImportError as e:
            logger.error(f"Failed to import SummaryWriter: {e}")
            logger.error("Install tensorboard: pip install tensorboard")
            raise
    return _SummaryWriter


GB = 1024**3
MB = 1024**2


@dataclass
class TensorBoardConfig:
    """Configuration for TensorBoard logging."""

    # Basic settings
    log_dir: str = "./runs"
    run_name: Optional[str] = None

    # Logging frequencies (in steps)
    scalar_freq: int = 1           # Loss, LR, speed metrics
    histogram_freq: int = 100      # Weight distributions
    gradient_freq: int = 50        # Gradient statistics
    memory_freq: int = 10          # Memory usage

    # Feature toggles
    log_histograms: bool = True
    log_gradients: bool = True
    log_memory: bool = True
    log_per_codebook: bool = True
    log_audio_samples: bool = False

    # Advanced features
    profile_batches: bool = False
    flush_secs: int = 30
    max_queue: int = 1000

    # Custom scalar layouts
    enable_custom_layouts: bool = True


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    # Basic metrics
    step: int = 0
    loss: float = 0.0
    text_loss: float = 0.0
    audio_loss: float = 0.0

    # Per-codebook losses (8 audio codebooks)
    codebook_losses: List[float] = field(default_factory=list)

    # Learning rate
    lr: float = 0.0

    # Speed metrics
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    step_time: float = 0.0

    # Memory metrics
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0
    gpu_memory_peak: float = 0.0

    # Gradient metrics
    grad_norm: float = 0.0
    grad_max: float = 0.0
    grad_min: float = 0.0

    # Progress
    percent_complete: float = 0.0
    eta_seconds: float = 0.0

    # Token statistics
    num_real_tokens: int = 0
    num_padding_tokens: int = 0
    token_efficiency: float = 0.0


class GradientMonitor:
    """Monitor gradient health during training."""

    def __init__(self):
        self.grad_norms: List[float] = []
        self.grad_max_history: List[float] = []
        self.nan_count = 0
        self.inf_count = 0
        self.zero_count = 0

    def analyze_gradients(self, model: nn.Module) -> Dict[str, float]:
        """Analyze gradients for the current step."""
        total_norm = 0.0
        grad_max = float('-inf')
        grad_min = float('inf')
        param_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                param_norm = grad.norm(2).item()
                total_norm += param_norm ** 2

                grad_max = max(grad_max, grad.max().item())
                grad_min = min(grad_min, grad.min().item())
                param_count += 1

                # Check for anomalies
                if torch.isnan(grad).any():
                    self.nan_count += 1
                if torch.isinf(grad).any():
                    self.inf_count += 1
                if (grad == 0).all():
                    self.zero_count += 1

        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        self.grad_max_history.append(grad_max if grad_max != float('-inf') else 0)

        return {
            "grad_norm": total_norm,
            "grad_max": grad_max if grad_max != float('-inf') else 0,
            "grad_min": grad_min if grad_min != float('inf') else 0,
            "grad_nan_count": self.nan_count,
            "grad_inf_count": self.inf_count,
            "grad_zero_count": self.zero_count,
            "param_count_with_grad": param_count,
        }


class AdvancedTensorBoardLogger:
    """
    Advanced TensorBoard logger for K-Moshi training.

    Provides comprehensive logging including:
    - Multi-level loss tracking
    - Gradient health monitoring
    - Memory usage tracking
    - Training speed metrics
    - Custom scalar layouts for organized visualization
    """

    def __init__(
        self,
        config: TensorBoardConfig,
        is_master: bool = True,
        hyperparams: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.is_master = is_master
        self.hyperparams = hyperparams or {}

        self.writer = None
        self.gradient_monitor = GradientMonitor()
        self.jsonl_path: Optional[Path] = None
        self.start_time = time.time()
        self.step_times: List[float] = []
        self._last_step_time = time.time()

        if not self.is_master:
            return

        # Create log directory
        run_name = config.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(config.log_dir) / run_name / "tensorboard"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create JSONL log file
        self.jsonl_path = Path(config.log_dir) / run_name / "metrics.jsonl"

        # Initialize SummaryWriter
        SummaryWriter = get_summary_writer()
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            max_queue=config.max_queue,
            flush_secs=config.flush_secs,
        )

        logger.info(f"TensorBoard logging to: {self.log_dir}")
        logger.info(f"View with: tensorboard --logdir={self.log_dir.parent}")

        # Log hyperparameters
        if hyperparams:
            self._log_hyperparams(hyperparams)

        # Setup custom scalar layouts
        if config.enable_custom_layouts:
            self._setup_custom_layouts()

    def _log_hyperparams(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters to TensorBoard."""
        if not self.is_master or self.writer is None:
            return

        # Flatten nested dict for TensorBoard
        flat_params = {}
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_params[f"{key}/{sub_key}"] = sub_value
            else:
                flat_params[key] = value

        # Filter out non-scalar types
        scalar_params = {
            k: v for k, v in flat_params.items()
            if isinstance(v, (int, float, str, bool)) and v is not None
        }

        self.writer.add_hparams(
            scalar_params,
            {"hparam/dummy": 0},  # Required metric
            run_name="hparams",
        )

    def _setup_custom_layouts(self):
        """Setup custom scalar layouts for organized visualization."""
        if not self.is_master or self.writer is None:
            return

        layout = {
            "Loss Overview": {
                "Total Loss": ["Multiline", ["loss/total", "loss/smoothed"]],
                "Component Losses": ["Multiline", ["loss/text", "loss/audio"]],
            },
            "Audio Codebooks": {
                "Semantic (CB0)": ["Multiline", ["codebook/0_semantic"]],
                "Acoustic (CB1-7)": ["Multiline", [
                    f"codebook/{i}_acoustic" for i in range(1, 8)
                ]],
            },
            "Training Speed": {
                "Throughput": ["Multiline", [
                    "speed/samples_per_sec",
                    "speed/tokens_per_sec",
                ]],
                "Step Time": ["Multiline", ["speed/step_time_ms"]],
            },
            "Memory Usage": {
                "GPU Memory": ["Multiline", [
                    "memory/allocated_gb",
                    "memory/reserved_gb",
                    "memory/peak_gb",
                ]],
            },
            "Gradient Health": {
                "Gradient Norm": ["Multiline", ["gradient/norm", "gradient/norm_ema"]],
                "Gradient Range": ["Multiline", ["gradient/max", "gradient/min"]],
            },
            "Learning Rate": {
                "LR Schedule": ["Multiline", ["optim/lr"]],
            },
        }

        self.writer.add_custom_scalars(layout)

    def log_metrics(self, metrics: TrainingMetrics):
        """Log all training metrics."""
        if not self.is_master or self.writer is None:
            return

        step = metrics.step

        # Calculate step time
        current_time = time.time()
        step_time = current_time - self._last_step_time
        self._last_step_time = current_time
        self.step_times.append(step_time)

        # Loss metrics
        self._log_loss_metrics(metrics, step)

        # Speed metrics
        self._log_speed_metrics(metrics, step, step_time)

        # Memory metrics (at configured frequency)
        if self.config.log_memory and step % self.config.memory_freq == 0:
            self._log_memory_metrics(metrics, step)

        # Learning rate
        self.writer.add_scalar("optim/lr", metrics.lr, step)

        # Progress metrics
        self.writer.add_scalar("progress/percent_complete", metrics.percent_complete, step)
        if metrics.eta_seconds > 0:
            self.writer.add_scalar("progress/eta_hours", metrics.eta_seconds / 3600, step)

        # Token efficiency
        if metrics.token_efficiency > 0:
            self.writer.add_scalar("data/token_efficiency", metrics.token_efficiency, step)

        # Write to JSONL
        self._write_jsonl(metrics, step_time)

    def _log_loss_metrics(self, metrics: TrainingMetrics, step: int):
        """Log loss-related metrics."""
        # Main losses
        self.writer.add_scalar("loss/total", metrics.loss, step)
        self.writer.add_scalar("loss/text", metrics.text_loss, step)
        self.writer.add_scalar("loss/audio", metrics.audio_loss, step)

        # Smoothed loss (EMA)
        if hasattr(self, '_loss_ema'):
            alpha = 0.1
            self._loss_ema = alpha * metrics.loss + (1 - alpha) * self._loss_ema
        else:
            self._loss_ema = metrics.loss
        self.writer.add_scalar("loss/smoothed", self._loss_ema, step)

        # Per-codebook losses
        if self.config.log_per_codebook and metrics.codebook_losses:
            for i, cb_loss in enumerate(metrics.codebook_losses):
                tag = f"codebook/{i}_semantic" if i == 0 else f"codebook/{i}_acoustic"
                self.writer.add_scalar(tag, cb_loss, step)

            # Mean acoustic loss (codebooks 1-7)
            if len(metrics.codebook_losses) > 1:
                mean_acoustic = sum(metrics.codebook_losses[1:]) / (len(metrics.codebook_losses) - 1)
                self.writer.add_scalar("codebook/mean_acoustic", mean_acoustic, step)

    def _log_speed_metrics(self, metrics: TrainingMetrics, step: int, step_time: float):
        """Log training speed metrics."""
        self.writer.add_scalar("speed/samples_per_sec", metrics.samples_per_second, step)
        self.writer.add_scalar("speed/tokens_per_sec", metrics.tokens_per_second, step)
        self.writer.add_scalar("speed/step_time_ms", step_time * 1000, step)

        # Average step time (moving window)
        if len(self.step_times) >= 10:
            avg_step_time = sum(self.step_times[-10:]) / 10
            self.writer.add_scalar("speed/avg_step_time_ms", avg_step_time * 1000, step)

    def _log_memory_metrics(self, metrics: TrainingMetrics, step: int):
        """Log memory usage metrics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / GB
            reserved = torch.cuda.memory_reserved() / GB
            peak = torch.cuda.max_memory_allocated() / GB

            self.writer.add_scalar("memory/allocated_gb", allocated, step)
            self.writer.add_scalar("memory/reserved_gb", reserved, step)
            self.writer.add_scalar("memory/peak_gb", peak, step)

            # Memory fragmentation
            if reserved > 0:
                fragmentation = 1.0 - (allocated / reserved)
                self.writer.add_scalar("memory/fragmentation", fragmentation, step)

    def log_gradients(self, model: nn.Module, step: int):
        """Log gradient statistics."""
        if not self.is_master or self.writer is None:
            return

        if not self.config.log_gradients:
            return

        if step % self.config.gradient_freq != 0:
            return

        grad_stats = self.gradient_monitor.analyze_gradients(model)

        self.writer.add_scalar("gradient/norm", grad_stats["grad_norm"], step)
        self.writer.add_scalar("gradient/max", grad_stats["grad_max"], step)
        self.writer.add_scalar("gradient/min", grad_stats["grad_min"], step)

        # Gradient norm EMA
        if hasattr(self, '_grad_norm_ema'):
            alpha = 0.1
            self._grad_norm_ema = alpha * grad_stats["grad_norm"] + (1 - alpha) * self._grad_norm_ema
        else:
            self._grad_norm_ema = grad_stats["grad_norm"]
        self.writer.add_scalar("gradient/norm_ema", self._grad_norm_ema, step)

        # Anomaly counts
        self.writer.add_scalar("gradient/nan_count", grad_stats["grad_nan_count"], step)
        self.writer.add_scalar("gradient/inf_count", grad_stats["grad_inf_count"], step)

    def log_histograms(self, model: nn.Module, step: int):
        """Log weight and gradient histograms."""
        if not self.is_master or self.writer is None:
            return

        if not self.config.log_histograms:
            return

        if step % self.config.histogram_freq != 0:
            return

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Weight histogram
                self.writer.add_histogram(f"weights/{name}", param.data, step)

                # Gradient histogram
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad.data, step)

    def log_model_stats(self, model: nn.Module, step: int):
        """Log model parameter statistics."""
        if not self.is_master or self.writer is None:
            return

        total_params = 0
        trainable_params = 0

        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        self.writer.add_scalar("model/total_params_millions", total_params / 1e6, step)
        self.writer.add_scalar("model/trainable_params_millions", trainable_params / 1e6, step)

    def log_audio_sample(
        self,
        tag: str,
        audio: torch.Tensor,
        step: int,
        sample_rate: int = 24000,
    ):
        """Log audio sample to TensorBoard."""
        if not self.is_master or self.writer is None:
            return

        if not self.config.log_audio_samples:
            return

        # Ensure audio is in correct format [C, T] or [T]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Normalize to [-1, 1]
        audio = audio.float()
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val

        self.writer.add_audio(tag, audio, step, sample_rate=sample_rate)

    def log_text(self, tag: str, text: str, step: int):
        """Log text to TensorBoard."""
        if not self.is_master or self.writer is None:
            return

        self.writer.add_text(tag, text, step)

    def log_eval_metrics(
        self,
        step: int,
        eval_loss: float,
        text_eval_loss: float,
        audio_eval_loss: float,
        perplexity: Optional[float] = None,
    ):
        """Log evaluation metrics."""
        if not self.is_master or self.writer is None:
            return

        self.writer.add_scalar("eval/loss", eval_loss, step)
        self.writer.add_scalar("eval/text_loss", text_eval_loss, step)
        self.writer.add_scalar("eval/audio_loss", audio_eval_loss, step)

        if perplexity is not None:
            self.writer.add_scalar("eval/perplexity", perplexity, step)

    def _write_jsonl(self, metrics: TrainingMetrics, step_time: float):
        """Write metrics to JSONL file for offline analysis."""
        if self.jsonl_path is None:
            return

        record = {
            "step": metrics.step,
            "timestamp": datetime.utcnow().isoformat(),
            "loss": metrics.loss,
            "text_loss": metrics.text_loss,
            "audio_loss": metrics.audio_loss,
            "lr": metrics.lr,
            "step_time_sec": step_time,
            "samples_per_sec": metrics.samples_per_second,
            "gpu_memory_gb": metrics.gpu_memory_allocated,
            "percent_complete": metrics.percent_complete,
        }

        if metrics.codebook_losses:
            record["codebook_losses"] = metrics.codebook_losses

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def flush(self):
        """Flush all pending writes."""
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        """Close the logger."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        elapsed = time.time() - self.start_time
        logger.info(f"TensorBoard logger closed. Total time: {timedelta(seconds=int(elapsed))}")


def create_tensorboard_logger(
    run_dir: Union[str, Path],
    hyperparams: Optional[Dict[str, Any]] = None,
    is_master: bool = True,
    **kwargs,
) -> AdvancedTensorBoardLogger:
    """
    Factory function to create TensorBoard logger.

    Args:
        run_dir: Directory for logs
        hyperparams: Training hyperparameters to log
        is_master: Whether this is the master process
        **kwargs: Additional config options

    Returns:
        AdvancedTensorBoardLogger instance
    """
    config = TensorBoardConfig(
        log_dir=str(run_dir),
        **kwargs,
    )

    return AdvancedTensorBoardLogger(
        config=config,
        is_master=is_master,
        hyperparams=hyperparams,
    )


# Utility functions for quick metrics creation
def create_training_metrics(
    step: int,
    loss: float,
    text_loss: float = 0.0,
    audio_loss: float = 0.0,
    lr: float = 0.0,
    codebook_losses: Optional[List[float]] = None,
    samples_per_second: float = 0.0,
    tokens_per_second: float = 0.0,
    percent_complete: float = 0.0,
    eta_seconds: float = 0.0,
    **kwargs,
) -> TrainingMetrics:
    """Create TrainingMetrics with common defaults."""
    return TrainingMetrics(
        step=step,
        loss=loss,
        text_loss=text_loss,
        audio_loss=audio_loss,
        lr=lr,
        codebook_losses=codebook_losses or [],
        samples_per_second=samples_per_second,
        tokens_per_second=tokens_per_second,
        percent_complete=percent_complete,
        eta_seconds=eta_seconds,
        **kwargs,
    )
