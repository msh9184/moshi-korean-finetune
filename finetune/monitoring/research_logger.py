"""
Research Logger for K-Moshi Training.

Provides comprehensive data logging for academic paper writing:
- Attention map visualization and storage
- Loss curves (training, validation, per-codebook)
- Gradient norm history
- Codebook usage statistics
- Training summary generation

All data is saved in formats suitable for paper figures:
- PNG/PDF for plots
- CSV for tabular data
- NPY for raw tensors (optional)
- JSON for metadata
"""

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger("research_logger")

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server usage
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Install with: pip install matplotlib")


@dataclass
class LossRecord:
    """Single loss record for tracking."""
    step: int
    loss: float
    text_loss: Optional[float] = None
    audio_loss: Optional[float] = None
    codebook_losses: Optional[List[float]] = None
    lr: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class GradientRecord:
    """Single gradient record for tracking."""
    step: int
    grad_norm: float
    has_nan: bool = False
    has_inf: bool = False
    per_layer_norms: Optional[Dict[str, float]] = None


class ResearchLogger:
    """
    Comprehensive research data logger for paper writing.

    Features:
    - Loss curve tracking with automatic plotting
    - Gradient norm history
    - Codebook usage statistics
    - Attention map visualization (when available)
    - Training summary generation
    """

    def __init__(
        self,
        run_dir: Path,
        num_codebooks: int = 8,
        save_attention_maps: bool = True,
        attention_freq: int = 2000,
        attention_samples: int = 2,
        save_raw_attention: bool = False,
        save_loss_curves: bool = True,
        save_codebook_stats: bool = True,
        save_gradient_norms: bool = True,
        generate_plots: bool = True,
        plot_freq: int = 1000,
        save_summary: bool = True,
    ):
        """
        Initialize research logger.

        Args:
            run_dir: Base directory for saving research data
            num_codebooks: Number of audio codebooks (for per-codebook tracking)
            save_attention_maps: Whether to save attention maps
            attention_freq: Steps between attention map saves
            attention_samples: Number of attention samples per save
            save_raw_attention: Whether to save raw attention tensors (.npy)
            save_loss_curves: Whether to save loss curves
            save_codebook_stats: Whether to save codebook statistics
            save_gradient_norms: Whether to save gradient norm history
            generate_plots: Whether to auto-generate plots
            plot_freq: Steps between plot generation
            save_summary: Whether to save final training summary
        """
        self.run_dir = Path(run_dir)
        self.num_codebooks = num_codebooks
        self.save_attention_maps = save_attention_maps
        self.attention_freq = attention_freq
        self.attention_samples = attention_samples
        self.save_raw_attention = save_raw_attention
        self.save_loss_curves = save_loss_curves
        self.save_codebook_stats = save_codebook_stats
        self.save_gradient_norms = save_gradient_norms
        self.generate_plots = generate_plots and MATPLOTLIB_AVAILABLE
        self.plot_freq = plot_freq
        self.save_summary = save_summary

        # Create directories
        self.research_dir = self.run_dir / "research"
        self.plots_dir = self.research_dir / "plots"
        self.attention_dir = self.research_dir / "attention"
        self.data_dir = self.research_dir / "data"

        for d in [self.research_dir, self.plots_dir, self.attention_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.train_losses: List[LossRecord] = []
        self.valid_losses: List[LossRecord] = []
        self.gradient_history: List[GradientRecord] = []
        self.codebook_usage: Dict[int, List[Dict[str, float]]] = defaultdict(list)

        # Training metadata
        self.start_time = datetime.utcnow()
        self.config: Dict[str, Any] = {}

        logger.info(f"ResearchLogger initialized: {self.research_dir}")

    def set_config(self, config: Dict[str, Any]):
        """Store training configuration for summary."""
        self.config = config

    def log_loss(
        self,
        step: int,
        loss: float,
        text_loss: Optional[float] = None,
        audio_loss: Optional[float] = None,
        codebook_losses: Optional[List[float]] = None,
        lr: Optional[float] = None,
        split: str = "train",
    ):
        """
        Log loss values.

        Args:
            step: Training step
            loss: Total loss
            text_loss: Text stream loss
            audio_loss: Audio stream loss
            codebook_losses: Per-codebook losses [8 values]
            lr: Current learning rate
            split: 'train' or 'valid'
        """
        record = LossRecord(
            step=step,
            loss=loss,
            text_loss=text_loss,
            audio_loss=audio_loss,
            codebook_losses=codebook_losses,
            lr=lr,
        )

        if split == "train":
            self.train_losses.append(record)
        else:
            self.valid_losses.append(record)

    def log_gradient(
        self,
        step: int,
        grad_norm: float,
        has_nan: bool = False,
        has_inf: bool = False,
        per_layer_norms: Optional[Dict[str, float]] = None,
    ):
        """Log gradient statistics."""
        record = GradientRecord(
            step=step,
            grad_norm=grad_norm,
            has_nan=has_nan,
            has_inf=has_inf,
            per_layer_norms=per_layer_norms,
        )
        self.gradient_history.append(record)

    def log_codebook_usage(
        self,
        step: int,
        codebook_idx: int,
        token_counts: Dict[int, int],
        entropy: float,
    ):
        """Log codebook usage statistics."""
        usage = {
            "step": step,
            "num_unique_tokens": len(token_counts),
            "total_tokens": sum(token_counts.values()),
            "entropy": entropy,
            "top_10_tokens": sorted(token_counts.items(), key=lambda x: -x[1])[:10],
        }
        self.codebook_usage[codebook_idx].append(usage)

    def save_attention_map(
        self,
        attention: torch.Tensor,
        step: int,
        layer_name: str,
        sample_idx: int = 0,
    ):
        """
        Save attention map visualization.

        Args:
            attention: Attention weights [H, T, T] or [B, H, T, T]
            step: Training step
            layer_name: Name of the attention layer
            sample_idx: Sample index in batch
        """
        if not self.save_attention_maps:
            return

        try:
            # Handle batch dimension
            if attention.dim() == 4:
                attention = attention[sample_idx]  # [H, T, T]

            # Convert to numpy
            attn_np = attention.detach().cpu().numpy()

            # Save raw tensor if requested
            if self.save_raw_attention:
                npy_path = self.attention_dir / f"step{step:06d}_{layer_name}_s{sample_idx}.npy"
                np.save(npy_path, attn_np)

            # Generate visualization if matplotlib available
            if self.generate_plots:
                self._plot_attention_map(attn_np, step, layer_name, sample_idx)

        except Exception as e:
            logger.debug(f"Attention save error: {e}")

    def _plot_attention_map(
        self,
        attention: np.ndarray,
        step: int,
        layer_name: str,
        sample_idx: int,
    ):
        """Generate attention map visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            # Average across heads for visualization
            if attention.ndim == 3:
                avg_attention = attention.mean(axis=0)  # [T, T]
            else:
                avg_attention = attention

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(avg_attention, cmap='viridis', aspect='auto')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            ax.set_title(f'Attention Map - {layer_name} (Step {step})')
            plt.colorbar(im)

            plot_path = self.attention_dir / f"step{step:06d}_{layer_name}_s{sample_idx}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            logger.debug(f"Attention plot error: {e}")

    def generate_loss_plots(self, step: int):
        """Generate loss curve plots."""
        if not self.generate_plots or not MATPLOTLIB_AVAILABLE:
            return

        if not self.train_losses:
            return

        try:
            # Extract data
            steps = [r.step for r in self.train_losses]
            losses = [r.loss for r in self.train_losses]

            # Main loss curve
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Training loss
            ax = axes[0, 0]
            ax.plot(steps, losses, 'b-', alpha=0.7, linewidth=0.5)
            # Add smoothed line
            if len(losses) > 10:
                window = min(50, len(losses) // 5)
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(steps[window-1:], smoothed, 'b-', linewidth=2, label='Smoothed')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True, alpha=0.3)

            # Learning rate
            ax = axes[0, 1]
            lrs = [r.lr for r in self.train_losses if r.lr is not None]
            if lrs:
                lr_steps = [r.step for r in self.train_losses if r.lr is not None]
                ax.plot(lr_steps, lrs, 'g-')
                ax.set_xlabel('Step')
                ax.set_ylabel('Learning Rate')
                ax.set_title('Learning Rate Schedule')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)

            # Text vs Audio loss
            ax = axes[1, 0]
            text_losses = [(r.step, r.text_loss) for r in self.train_losses if r.text_loss is not None]
            audio_losses = [(r.step, r.audio_loss) for r in self.train_losses if r.audio_loss is not None]
            if text_losses:
                ax.plot([s for s, _ in text_losses], [l for _, l in text_losses], 'r-', label='Text', alpha=0.7)
            if audio_losses:
                ax.plot([s for s, _ in audio_losses], [l for _, l in audio_losses], 'b-', label='Audio', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Text vs Audio Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Per-codebook loss (last available)
            ax = axes[1, 1]
            cb_records = [r for r in self.train_losses if r.codebook_losses is not None]
            if cb_records:
                last_record = cb_records[-1]
                ax.bar(range(len(last_record.codebook_losses)), last_record.codebook_losses)
                ax.set_xlabel('Codebook Index')
                ax.set_ylabel('Loss')
                ax.set_title(f'Per-Codebook Loss (Step {last_record.step})')
                ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plot_path = self.plots_dir / f"loss_curves_step{step:06d}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Generated loss plot: {plot_path}")

        except Exception as e:
            logger.debug(f"Loss plot error: {e}")

    def generate_gradient_plot(self, step: int):
        """Generate gradient norm history plot."""
        if not self.generate_plots or not MATPLOTLIB_AVAILABLE:
            return

        if not self.gradient_history:
            return

        try:
            steps = [r.step for r in self.gradient_history]
            norms = [r.grad_norm for r in self.gradient_history]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, norms, 'b-', alpha=0.7, linewidth=0.5)

            # Add smoothed line
            if len(norms) > 10:
                window = min(50, len(norms) // 5)
                smoothed = np.convolve(norms, np.ones(window)/window, mode='valid')
                ax.plot(steps[window-1:], smoothed, 'b-', linewidth=2, label='Smoothed')

            ax.set_xlabel('Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm History')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            plt.tight_layout()
            plot_path = self.plots_dir / f"gradient_norms_step{step:06d}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            logger.debug(f"Gradient plot error: {e}")

    def save_data_csv(self):
        """Save all tracked data to CSV files."""
        # Training losses
        if self.train_losses:
            csv_path = self.data_dir / "train_losses.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['step', 'loss', 'text_loss', 'audio_loss', 'lr', 'timestamp']
                for i in range(self.num_codebooks):
                    header.append(f'cb{i}_loss')
                writer.writerow(header)

                for r in self.train_losses:
                    row = [r.step, r.loss, r.text_loss, r.audio_loss, r.lr, r.timestamp]
                    if r.codebook_losses:
                        row.extend(r.codebook_losses)
                    else:
                        row.extend([None] * self.num_codebooks)
                    writer.writerow(row)

            logger.info(f"Saved training losses: {csv_path}")

        # Validation losses
        if self.valid_losses:
            csv_path = self.data_dir / "valid_losses.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'loss', 'text_loss', 'audio_loss', 'timestamp'])
                for r in self.valid_losses:
                    writer.writerow([r.step, r.loss, r.text_loss, r.audio_loss, r.timestamp])

        # Gradient history
        if self.gradient_history:
            csv_path = self.data_dir / "gradient_norms.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'grad_norm', 'has_nan', 'has_inf'])
                for r in self.gradient_history:
                    writer.writerow([r.step, r.grad_norm, r.has_nan, r.has_inf])

            logger.info(f"Saved gradient history: {csv_path}")

    def generate_summary(self, final_step: int):
        """Generate training summary JSON."""
        if not self.save_summary:
            return

        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()

        # Compute statistics
        summary = {
            "training_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "duration_hours": duration / 3600,
                "final_step": final_step,
            },
            "loss_statistics": {},
            "gradient_statistics": {},
            "config": self.config,
        }

        # Loss statistics
        if self.train_losses:
            losses = [r.loss for r in self.train_losses]
            summary["loss_statistics"]["train"] = {
                "initial": losses[0],
                "final": losses[-1],
                "min": min(losses),
                "max": max(losses),
                "improvement": (losses[0] - losses[-1]) / losses[0] * 100,
                "num_records": len(losses),
            }

        if self.valid_losses:
            losses = [r.loss for r in self.valid_losses]
            summary["loss_statistics"]["valid"] = {
                "initial": losses[0] if losses else None,
                "final": losses[-1] if losses else None,
                "min": min(losses) if losses else None,
                "best_step": self.valid_losses[losses.index(min(losses))].step if losses else None,
            }

        # Gradient statistics
        if self.gradient_history:
            norms = [r.grad_norm for r in self.gradient_history]
            nan_count = sum(1 for r in self.gradient_history if r.has_nan)
            inf_count = sum(1 for r in self.gradient_history if r.has_inf)

            summary["gradient_statistics"] = {
                "mean_norm": float(np.mean(norms)),
                "max_norm": max(norms),
                "min_norm": min(norms),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "num_records": len(norms),
            }

        # Save summary
        summary_path = self.research_dir / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Saved training summary: {summary_path}")

        return summary

    def periodic_save(self, step: int):
        """
        Periodic save and plot generation.

        Call this at regular intervals during training.
        """
        if step % self.plot_freq == 0:
            if self.generate_plots:
                self.generate_loss_plots(step)
                self.generate_gradient_plot(step)

            # Also save data periodically
            self.save_data_csv()

    def finalize(self, final_step: int):
        """
        Finalize research logging at end of training.

        Generates final plots, saves all data, and creates summary.
        """
        logger.info("Finalizing research logger...")

        # Save all CSV data
        self.save_data_csv()

        # Generate final plots
        if self.generate_plots:
            self.generate_loss_plots(final_step)
            self.generate_gradient_plot(final_step)

        # Generate summary
        summary = self.generate_summary(final_step)

        logger.info(f"Research data saved to: {self.research_dir}")

        return summary
