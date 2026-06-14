"""
Pretty Logger for Speech Model Training.

Provides enhanced visual logging for training monitoring:
- Clean header banner with project info
- Structured configuration display in table format
- Progress bar with real-time metrics
- Color-coded status indicators
- Training summary and checkpoint notifications

This module is designed for better terminal UX during long training runs.
All characters are ASCII-safe for maximum terminal compatibility.
"""

import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("pretty_logger")


# =============================================================================
# ANSI Color Codes for Terminal
# =============================================================================
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-terminal output)."""
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith('_'):
                setattr(cls, attr, "")


# Check if stdout is a terminal
if not sys.stdout.isatty():
    Colors.disable()


# =============================================================================
# Box Drawing Characters (ASCII-safe)
# =============================================================================
class Box:
    """ASCII-safe box drawing characters for maximum terminal compatibility."""
    # Heavy box (using ASCII equivalents)
    H_TOP_LEFT = "+"
    H_TOP_RIGHT = "+"
    H_BOTTOM_LEFT = "+"
    H_BOTTOM_RIGHT = "+"
    H_HORIZONTAL = "-"
    H_VERTICAL = "|"
    H_T_DOWN = "+"
    H_T_UP = "+"
    H_T_RIGHT = "+"
    H_T_LEFT = "+"
    H_CROSS = "+"

    # Light box (using ASCII equivalents)
    TOP_LEFT = "+"
    TOP_RIGHT = "+"
    BOTTOM_LEFT = "+"
    BOTTOM_RIGHT = "+"
    HORIZONTAL = "-"
    VERTICAL = "|"
    T_DOWN = "+"
    T_UP = "+"
    T_RIGHT = "+"
    T_LEFT = "+"
    CROSS = "+"

    # Double box (using ASCII equivalents)
    D_TOP_LEFT = "+"
    D_TOP_RIGHT = "+"
    D_BOTTOM_LEFT = "+"
    D_BOTTOM_RIGHT = "+"
    D_HORIZONTAL = "="
    D_VERTICAL = "|"


# =============================================================================
# Pretty Logger Class
# =============================================================================
class PrettyLogger:
    """
    Enhanced logger with beautiful terminal output.

    Features:
    - ASCII art banner with gradient colors
    - Structured configuration tables
    - Real-time progress indicators
    - Training summary statistics
    """

    # Generic ASCII Banner (ASCII-safe characters only)
    BANNER = None  # No ASCII art banner - use clean header instead

    MINI_BANNER = None  # Will be generated dynamically

    def __init__(self, rank: int = 0, width: int = 80):
        """
        Initialize PrettyLogger.

        Args:
            rank: Distributed training rank (only rank 0 prints)
            width: Terminal width for formatting
        """
        self.rank = rank
        self.width = width
        self.start_time = None
        self.last_step_time = None
        self.step_times: List[float] = []

    def _print(self, message: str = "", end: str = "\n"):
        """Print only on rank 0."""
        if self.rank == 0:
            print(message, end=end, flush=True)

    def _center(self, text: str, width: Optional[int] = None) -> str:
        """Center text within width."""
        w = width or self.width
        return text.center(w)

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text."""
        return f"{color}{text}{Colors.RESET}"

    # =========================================================================
    # Banner and Header Functions
    # =========================================================================

    def print_banner(self, version: str = "2.0", subtitle: str = "Speech Model Finetuning"):
        """Print a clean header banner (ASCII-safe)."""
        if self.rank != 0:
            return

        self._print()

        # Clean header design with double-line box
        border = "=" * (self.width - 4)
        inner_border = "-" * (self.width - 6)

        self._print(f"  {Colors.BRIGHT_CYAN}{border}{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_CYAN}|{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{subtitle.center(self.width - 8)}{Colors.RESET} {Colors.BRIGHT_CYAN}|{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_CYAN}|{Colors.RESET} {Colors.DIM}{inner_border}{Colors.RESET} {Colors.BRIGHT_CYAN}|{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_CYAN}|{Colors.RESET} {Colors.DIM}{'Version ' + version + ' | ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^{self.width - 8}}{Colors.RESET} {Colors.BRIGHT_CYAN}|{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_CYAN}{border}{Colors.RESET}")
        self._print()

    def print_section_header(self, title: str, icon: str = ">"):
        """Print a section header with decorative border (ASCII-safe)."""
        if self.rank != 0:
            return

        border = "-" * (self.width - 4)
        self._print()
        self._print(f"  {Colors.BRIGHT_CYAN}+{border}+{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_CYAN}|{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}[{icon}] {title.upper()}{Colors.RESET}".ljust(self.width + 15) + f"{Colors.BRIGHT_CYAN}|{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_CYAN}+{border}+{Colors.RESET}")

    def print_subsection(self, title: str, icon: str = "*"):
        """Print a subsection header (ASCII-safe)."""
        if self.rank != 0:
            return
        self._print(f"\n  {Colors.BRIGHT_YELLOW}[{icon}]{Colors.RESET} {Colors.BOLD}{title}{Colors.RESET}")
        self._print(f"  {Colors.DIM}{'-' * (len(title) + 4)}{Colors.RESET}")

    # =========================================================================
    # Configuration Display Functions
    # =========================================================================

    def print_config_table(self, title: str, config: Dict[str, Any], icon: str = "*"):
        """Print a configuration table with aligned columns (ASCII-safe)."""
        if self.rank != 0:
            return

        self.print_subsection(title, icon)

        # Find max key length for alignment
        max_key_len = max(len(str(k)) for k in config.keys()) if config else 10

        for key, value in config.items():
            # Format value based on type
            if isinstance(value, bool):
                val_str = f"{Colors.GREEN}[ON] Enabled{Colors.RESET}" if value else f"{Colors.DIM}[--] Disabled{Colors.RESET}"
            elif isinstance(value, float):
                val_str = f"{Colors.BRIGHT_CYAN}{value:.6g}{Colors.RESET}"
            elif isinstance(value, int):
                val_str = f"{Colors.BRIGHT_CYAN}{value:,}{Colors.RESET}"
            elif isinstance(value, Path) or (isinstance(value, str) and '/' in str(value)):
                val_str = f"{Colors.BRIGHT_BLUE}{value}{Colors.RESET}"
            elif value is None:
                val_str = f"{Colors.DIM}None{Colors.RESET}"
            else:
                val_str = f"{Colors.WHITE}{value}{Colors.RESET}"

            key_str = f"{Colors.BRIGHT_WHITE}{key}{Colors.RESET}"
            self._print(f"    {key_str:<{max_key_len + 15}} : {val_str}")

    def print_model_info(self, model_config: Dict[str, Any]):
        """Print model configuration in a special format."""
        if self.rank != 0:
            return

        self.print_section_header("Model Configuration", "M")

        # Model architecture
        arch_config = {
            "Model Type": model_config.get("model_type", "Moshiko"),
            "Audio Codebooks (n_q)": model_config.get("n_q", 16),
            "Depformer Codebooks (dep_q)": model_config.get("dep_q", 8),
            "Total Codebooks": model_config.get("num_codebooks", 17),
            "Audio Offset": model_config.get("audio_offset", 1),
            "Zero Token ID": model_config.get("zero_token_id", -1),
        }
        self.print_config_table("Architecture", arch_config, "A")

        # Training mode
        training_config = {
            "Full Finetuning": model_config.get("full_finetuning", True),
            "LoRA Enabled": model_config.get("lora_enabled", False),
            "LoRA Rank": model_config.get("lora_rank", "N/A"),
            "Gradient Checkpointing": model_config.get("gradient_checkpointing", True),
        }
        self.print_config_table("Training Mode", training_config, "T")

    def print_training_config(self, args: Dict[str, Any]):
        """Print comprehensive training configuration."""
        if self.rank != 0:
            return

        self.print_section_header("Training Configuration", "C")

        # Data configuration
        data_config = {
            "Train Data": args.get("train_data", "N/A"),
            "Eval Data": args.get("eval_data", "N/A") or "Disabled",
            "Duration (sec)": args.get("duration_sec", 120),
            "Shuffle": args.get("shuffle", True),
        }
        self.print_config_table("Data", data_config, "D")

        # Batch & optimization
        batch_config = {
            "Batch Size": args.get("batch_size", 24),
            "Microbatches": args.get("num_microbatches", 8),
            "Effective Batch": args.get("batch_size", 24) * args.get("num_microbatches", 8) * args.get("world_size", 1),
            "Max Steps": args.get("max_steps", 50000),
            "Max Gradient Norm": args.get("max_norm", 1.0),
        }
        self.print_config_table("Batch & Steps", batch_config, "B")

        # Optimizer
        optim_config = {
            "Learning Rate": args.get("lr", 3e-5),
            "Depformer LR": args.get("depformer_lr", "Same as LR"),
            "Weight Decay": args.get("weight_decay", 0.1),
            "Betas": f"({args.get('beta1', 0.9)}, {args.get('beta2', 0.95)})",
            "Epsilon": args.get("eps", 1e-5),
        }
        self.print_config_table("Optimizer (AdamW)", optim_config, "O")

        # Scheduler
        scheduler_config = {
            "Type": args.get("scheduler_type", "cosine_warmup"),
            "Warmup Steps": args.get("warmup_steps", 500),
            "Min LR": args.get("min_lr", 1e-7),
        }
        self.print_config_table("Scheduler", scheduler_config, "S")

        # Loss weights
        loss_config = {
            "First Codebook Weight": f"{args.get('first_codebook_weight', 100.0)}x",
            "Text Padding Weight": args.get("text_padding_weight", 0.5),
        }
        self.print_config_table("Loss Weights", loss_config, "L")

    def print_hardware_info(self, world_size: int, device_name: str = "NVIDIA A100"):
        """Print hardware configuration."""
        if self.rank != 0:
            return

        self.print_section_header("Hardware & Environment", "H")

        import torch

        hw_config = {
            "GPUs": f"{world_size}x {device_name}",
            "CUDA Version": torch.version.cuda or "N/A",
            "PyTorch Version": torch.__version__,
            "Distributed Backend": "FSDP" if world_size > 1 else "Single GPU",
            "Mixed Precision": "bfloat16",
        }
        self.print_config_table("Hardware", hw_config, "G")

        # Memory info per GPU
        if torch.cuda.is_available():
            for i in range(min(world_size, 4)):  # Show first 4 GPUs
                try:
                    props = torch.cuda.get_device_properties(i)
                    total_mem = props.total_memory / (1024**3)
                    self._print(f"    {Colors.DIM}GPU {i}: {props.name} ({total_mem:.1f} GB){Colors.RESET}")
                except:
                    pass

    def print_checkpoint_info(self, run_dir: str, ckpt_freq: int, num_keep: int):
        """Print checkpointing configuration."""
        if self.rank != 0:
            return

        self.print_section_header("Checkpointing & Logging", "K")

        ckpt_config = {
            "Run Directory": run_dir,
            "Checkpoint Frequency": f"Every {ckpt_freq} steps",
            "Checkpoints to Keep": num_keep,
            "TensorBoard": "Enabled",
        }
        self.print_config_table("Checkpointing", ckpt_config, "P")

    # =========================================================================
    # Progress and Status Functions
    # =========================================================================

    def print_training_start(self):
        """Print training start message with timestamp."""
        if self.rank != 0:
            return

        self.start_time = time.time()
        self.last_step_time = self.start_time

        self._print()
        border = "=" * (self.width - 4)
        self._print(f"  {Colors.BRIGHT_GREEN}{border}{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_GREEN}{Colors.BOLD}  [*] TRAINING STARTED @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_GREEN}{border}{Colors.RESET}")
        self._print()

    def print_step_progress(
        self,
        step: int,
        max_steps: int,
        loss: float,
        text_loss: float,
        audio_loss: float,
        lr: float,
        grad_norm: float,
        samples_per_sec: float,
        gpu_memory_gb: Optional[float] = None,
        moshi_loss: Optional[float] = None,
        user_loss: Optional[float] = None,
    ):
        """Print training step progress in a clean format.

        Args:
            step: Current training step
            max_steps: Total training steps
            loss: Total loss value
            text_loss: Text stream loss
            audio_loss: Audio stream loss (combined moshi + user if applicable)
            lr: Learning rate
            grad_norm: Gradient norm
            samples_per_sec: Training samples per second
            gpu_memory_gb: GPU memory usage in GB
            moshi_loss: Moshi speaker audio loss (for user stream mode)
            user_loss: User speaker audio loss (for user stream mode)
        """
        if self.rank != 0:
            return

        # Calculate progress
        progress = step / max_steps
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Estimate remaining time
        if step > 0 and elapsed > 0:
            steps_per_sec = step / elapsed
            remaining_steps = max_steps - step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "Calculating..."

        # Progress bar (ASCII-safe characters)
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "#" * filled + "." * (bar_width - filled)

        # Format metrics
        loss_color = Colors.GREEN if loss < 5.0 else (Colors.YELLOW if loss < 10.0 else Colors.RED)

        # Build output line (ASCII-safe)
        # Check if we have user stream metrics
        if moshi_loss is not None and user_loss is not None:
            # Full duplex mode with separate moshi/user losses
            line = (
                f"  {Colors.BRIGHT_WHITE}Step{Colors.RESET} {Colors.BOLD}{step:>6}{Colors.RESET}/{max_steps} "
                f"{Colors.BRIGHT_CYAN}[{bar}]{Colors.RESET} {progress*100:>5.1f}% "
                f"| {loss_color}loss:{loss:.4f}{Colors.RESET} "
                f"| {Colors.DIM}txt:{text_loss:.3f}{Colors.RESET} "
                f"| {Colors.BRIGHT_GREEN}moshi:{moshi_loss:.3f}{Colors.RESET} "
                f"| {Colors.BRIGHT_YELLOW}user:{user_loss:.3f}{Colors.RESET} "
                f"| {Colors.BRIGHT_BLUE}lr:{lr:.2e}{Colors.RESET} "
                f"| {Colors.DIM}ETA:{eta_str}{Colors.RESET}"
            )
        else:
            # Standard mono mode
            line = (
                f"  {Colors.BRIGHT_WHITE}Step{Colors.RESET} {Colors.BOLD}{step:>6}{Colors.RESET}/{max_steps} "
                f"{Colors.BRIGHT_CYAN}[{bar}]{Colors.RESET} {progress*100:>5.1f}% "
                f"| {loss_color}loss:{loss:.4f}{Colors.RESET} "
                f"| {Colors.DIM}txt:{text_loss:.3f} aud:{audio_loss:.3f}{Colors.RESET} "
                f"| {Colors.BRIGHT_BLUE}lr:{lr:.2e}{Colors.RESET} "
                f"| {Colors.DIM}ETA:{eta_str}{Colors.RESET}"
            )

        self._print(line)

    def print_eval_results(
        self,
        step: int,
        eval_loss: float,
        text_loss: float,
        audio_loss: float,
        perplexity: float,
        wer: Optional[float] = None,
        cer: Optional[float] = None,
    ):
        """Print evaluation results in a highlighted format."""
        if self.rank != 0:
            return

        self._print()
        self._print(f"  {Colors.BRIGHT_MAGENTA}{'-' * (self.width - 4)}{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_MAGENTA}{Colors.BOLD}[EVAL] EVALUATION @ Step {step}{Colors.RESET}")

        # Results table
        loss_color = Colors.GREEN if eval_loss < 5.0 else (Colors.YELLOW if eval_loss < 10.0 else Colors.RED)

        results = [
            f"    Loss:       {loss_color}{eval_loss:.4f}{Colors.RESET}",
            f"    Text Loss:  {Colors.CYAN}{text_loss:.4f}{Colors.RESET}",
            f"    Audio Loss: {Colors.CYAN}{audio_loss:.4f}{Colors.RESET}",
            f"    Perplexity: {Colors.BRIGHT_YELLOW}{perplexity:.2f}{Colors.RESET}",
        ]

        if wer is not None:
            wer_color = Colors.GREEN if wer < 0.3 else (Colors.YELLOW if wer < 0.5 else Colors.RED)
            results.append(f"    WER:        {wer_color}{wer*100:.1f}%{Colors.RESET}")

        if cer is not None:
            cer_color = Colors.GREEN if cer < 0.2 else (Colors.YELLOW if cer < 0.4 else Colors.RED)
            results.append(f"    CER:        {cer_color}{cer*100:.1f}%{Colors.RESET}")

        for r in results:
            self._print(r)

        self._print(f"  {Colors.BRIGHT_MAGENTA}{'-' * (self.width - 4)}{Colors.RESET}")
        self._print()

    def print_checkpoint_saved(self, step: int, path: str):
        """Print checkpoint saved notification."""
        if self.rank != 0:
            return

        self._print(f"\n  {Colors.BRIGHT_GREEN}[CKPT] Checkpoint saved @ step {step}{Colors.RESET}")
        self._print(f"  {Colors.DIM}   +-- {path}{Colors.RESET}\n")

    def print_sample_saved(self, step: int, split: str, num_samples: int, num_dialogues: int = 0):
        """Print sample saved notification."""
        if self.rank != 0:
            return

        dialogue_info = f" (+ {num_dialogues} stereo)" if num_dialogues > 0 else ""
        self._print(f"  {Colors.BRIGHT_BLUE}[AUDIO] {num_samples} samples saved ({split}){dialogue_info}{Colors.RESET}")

    def print_gradient_warning(self, step: int, issue: str, details: str = ""):
        """Print gradient health warning."""
        if self.rank != 0:
            return

        self._print(f"  {Colors.BRIGHT_YELLOW}[!] Gradient {issue} @ step {step}{Colors.RESET}")
        if details:
            self._print(f"  {Colors.DIM}   +-- {details}{Colors.RESET}")

    # =========================================================================
    # Training Summary Functions
    # =========================================================================

    def print_training_complete(
        self,
        total_steps: int,
        final_loss: float,
        best_loss: float,
        best_step: int,
        total_time_seconds: float,
    ):
        """Print training completion summary."""
        if self.rank != 0:
            return

        total_time = str(timedelta(seconds=int(total_time_seconds)))

        self._print()
        border = "=" * (self.width - 4)
        self._print(f"  {Colors.BRIGHT_GREEN}{border}{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_GREEN}{Colors.BOLD}  [*] TRAINING COMPLETED{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_GREEN}{border}{Colors.RESET}")
        self._print()

        summary = {
            "Total Steps": f"{total_steps:,}",
            "Final Loss": f"{final_loss:.4f}",
            "Best Loss": f"{best_loss:.4f} (step {best_step})",
            "Total Time": total_time,
            "Avg Step Time": f"{total_time_seconds / total_steps:.2f}s" if total_steps > 0 else "N/A",
        }

        for key, value in summary.items():
            self._print(f"    {Colors.BRIGHT_WHITE}{key:<16}{Colors.RESET} : {Colors.BRIGHT_CYAN}{value}{Colors.RESET}")

        self._print()
        self._print(f"  {Colors.DIM}Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        self._print()

    def print_error(self, message: str, details: str = ""):
        """Print error message."""
        if self.rank != 0:
            return

        self._print(f"\n  {Colors.BRIGHT_RED}[X] ERROR: {message}{Colors.RESET}")
        if details:
            self._print(f"  {Colors.DIM}   {details}{Colors.RESET}")
        self._print()

    def print_warning(self, message: str):
        """Print warning message."""
        if self.rank != 0:
            return

        self._print(f"  {Colors.BRIGHT_YELLOW}[!] {message}{Colors.RESET}")

    def print_success(self, message: str):
        """Print success message."""
        if self.rank != 0:
            return

        self._print(f"  {Colors.BRIGHT_GREEN}[OK] {message}{Colors.RESET}")

    def print_info(self, message: str):
        """Print info message."""
        if self.rank != 0:
            return

        self._print(f"  {Colors.BRIGHT_CYAN}[i] {message}{Colors.RESET}")

    # =========================================================================
    # Advanced Monitoring Display Functions
    # =========================================================================

    def print_text_predictions(
        self,
        step: int,
        samples: list,
        wer: float,
        cer: float,
        max_display: int = 3,
    ):
        """
        Print text prediction samples with WER/CER metrics in a structured format.

        Args:
            step: Current training step
            samples: List of dicts with 'reference', 'hypothesis', 'wer', 'cer'
            wer: Overall WER for the batch
            cer: Overall CER for the batch
            max_display: Maximum number of samples to display
        """
        if self.rank != 0:
            return

        # Header with metrics
        self._print()
        border = "=" * (self.width - 4)
        self._print(f"  {Colors.BRIGHT_MAGENTA}{border}{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_MAGENTA}|{Colors.RESET} {Colors.BOLD}[TEXT] Inner Monologue Predictions @ Step {step}{Colors.RESET}")

        # WER/CER summary with color coding
        wer_color = Colors.GREEN if wer < 0.3 else (Colors.YELLOW if wer < 0.5 else Colors.RED)
        cer_color = Colors.GREEN if cer < 0.2 else (Colors.YELLOW if cer < 0.4 else Colors.RED)

        metrics_line = (
            f"  {Colors.BRIGHT_MAGENTA}|{Colors.RESET} "
            f"WER: {wer_color}{wer*100:>5.1f}%{Colors.RESET}  "
            f"CER: {cer_color}{cer*100:>5.1f}%{Colors.RESET}  "
            f"{Colors.DIM}(lower is better){Colors.RESET}"
        )
        self._print(metrics_line)
        self._print(f"  {Colors.BRIGHT_MAGENTA}{'-' * (self.width - 4)}{Colors.RESET}")

        # Display samples
        for idx, sample in enumerate(samples[:max_display]):
            ref = sample.get('reference', '')
            hyp = sample.get('hypothesis', '')
            s_wer = sample.get('wer', 0)
            s_cer = sample.get('cer', 0)

            # Truncate long text for display (max 60 chars with proper Korean handling)
            max_len = 55
            ref_display = ref[:max_len] + "..." if len(ref) > max_len else ref
            hyp_display = hyp[:max_len] + "..." if len(hyp) > max_len else hyp

            # Sample header
            sample_wer_color = Colors.GREEN if s_wer < 0.3 else (Colors.YELLOW if s_wer < 0.5 else Colors.RED)
            self._print(f"  {Colors.BRIGHT_WHITE}[{idx+1}]{Colors.RESET} {sample_wer_color}WER:{s_wer*100:>4.0f}%{Colors.RESET} CER:{s_cer*100:>4.0f}%")

            # Reference (Ground Truth)
            self._print(f"      {Colors.BRIGHT_CYAN}REF:{Colors.RESET} {Colors.WHITE}{ref_display}{Colors.RESET}")

            # Hypothesis (Prediction)
            self._print(f"      {Colors.BRIGHT_YELLOW}HYP:{Colors.RESET} {Colors.WHITE}{hyp_display}{Colors.RESET}")

            if idx < min(len(samples), max_display) - 1:
                self._print(f"      {Colors.DIM}---{Colors.RESET}")

        self._print(f"  {Colors.BRIGHT_MAGENTA}{border}{Colors.RESET}")
        self._print()

    def print_codebook_analysis(
        self,
        step: int,
        losses: list,
        entropy: list = None,
        semantic_loss: float = None,
        acoustic_loss: float = None,
    ):
        """
        Print per-codebook loss analysis in a visual format.

        Args:
            step: Current training step
            losses: List of losses per codebook
            entropy: Optional list of entropy per codebook
            semantic_loss: Semantic (first codebook) weighted loss
            acoustic_loss: Combined acoustic (remaining codebooks) loss
        """
        if self.rank != 0:
            return

        self._print()
        self._print(f"  {Colors.BRIGHT_CYAN}+{'-' * (self.width - 6)}+{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_CYAN}|{Colors.RESET} {Colors.BOLD}[CODEBOOK] Per-Codebook Analysis @ Step {step}{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_CYAN}+{'-' * (self.width - 6)}+{Colors.RESET}")

        if semantic_loss is not None and acoustic_loss is not None:
            self._print(
                f"  {Colors.BRIGHT_CYAN}|{Colors.RESET} "
                f"Semantic (CB0): {Colors.BRIGHT_GREEN}{semantic_loss:.4f}{Colors.RESET}  "
                f"Acoustic (CB1-7): {Colors.BRIGHT_YELLOW}{acoustic_loss:.4f}{Colors.RESET}"
            )
            self._print(f"  {Colors.BRIGHT_CYAN}|{Colors.RESET}")

        # Visual bar chart for codebook losses
        if losses:
            max_loss = max(losses) if max(losses) > 0 else 1.0
            bar_width = 25

            for i, loss in enumerate(losses):
                # Determine color based on codebook type
                if i == 0:
                    label = "Semantic"
                    color = Colors.BRIGHT_GREEN
                else:
                    label = f"Acoustic"
                    color = Colors.BRIGHT_YELLOW if i < 4 else Colors.YELLOW

                # Create bar
                bar_len = int((loss / max_loss) * bar_width) if max_loss > 0 else 0
                bar = "#" * bar_len + "." * (bar_width - bar_len)

                # Entropy info if available
                ent_str = ""
                if entropy and i < len(entropy):
                    ent_str = f" H={entropy[i]:.2f}"

                self._print(
                    f"  {Colors.BRIGHT_CYAN}|{Colors.RESET} "
                    f"CB{i}: {color}[{bar}]{Colors.RESET} {loss:>6.3f}{Colors.DIM}{ent_str}{Colors.RESET}"
                )

        self._print(f"  {Colors.BRIGHT_CYAN}+{'-' * (self.width - 6)}+{Colors.RESET}")

    def print_gradient_health(
        self,
        step: int,
        grad_norm: float,
        has_nan: bool = False,
        has_inf: bool = False,
        is_exploding: bool = False,
        is_vanishing: bool = False,
    ):
        """
        Print gradient health status with visual indicators.

        Args:
            step: Current training step
            grad_norm: Gradient norm value
            has_nan: Whether NaN gradients were detected
            has_inf: Whether Inf gradients were detected
            is_exploding: Whether gradients are exploding
            is_vanishing: Whether gradients are vanishing
        """
        if self.rank != 0:
            return

        # Determine overall health status
        if has_nan or has_inf:
            status = "CRITICAL"
            status_color = Colors.BRIGHT_RED
            icon = "[X]"
        elif is_exploding:
            status = "WARNING"
            status_color = Colors.BRIGHT_YELLOW
            icon = "[!]"
        elif is_vanishing:
            status = "CAUTION"
            status_color = Colors.YELLOW
            icon = "[~]"
        else:
            status = "HEALTHY"
            status_color = Colors.BRIGHT_GREEN
            icon = "[OK]"

        # Build status line
        line = (
            f"  {status_color}{icon} [GRAD]{Colors.RESET} "
            f"norm={Colors.BRIGHT_CYAN}{grad_norm:.4f}{Colors.RESET}"
        )

        # Add warning indicators
        warnings = []
        if has_nan:
            warnings.append(f"{Colors.RED}NaN{Colors.RESET}")
        if has_inf:
            warnings.append(f"{Colors.RED}Inf{Colors.RESET}")
        if is_exploding:
            warnings.append(f"{Colors.YELLOW}Exploding{Colors.RESET}")
        if is_vanishing:
            warnings.append(f"{Colors.YELLOW}Vanishing{Colors.RESET}")

        if warnings:
            line += f" | {', '.join(warnings)}"
        else:
            line += f" | {status_color}{status}{Colors.RESET}"

        self._print(line)

    def print_training_summary_table(
        self,
        step: int,
        metrics: dict,
    ):
        """
        Print a compact training summary table.

        Args:
            step: Current training step
            metrics: Dictionary of metric name -> value pairs
        """
        if self.rank != 0 or not metrics:
            return

        self._print()
        header = f" Training Summary @ Step {step} "
        border_len = self.width - 4
        pad_left = (border_len - len(header)) // 2
        pad_right = border_len - len(header) - pad_left

        self._print(f"  {Colors.BRIGHT_CYAN}{'=' * pad_left}{header}{'=' * pad_right}{Colors.RESET}")

        # Split metrics into columns
        items = list(metrics.items())
        col_width = (self.width - 8) // 2

        for i in range(0, len(items), 2):
            left_key, left_val = items[i]
            left_str = f"{left_key}: {self._format_metric(left_val)}"

            if i + 1 < len(items):
                right_key, right_val = items[i + 1]
                right_str = f"{right_key}: {self._format_metric(right_val)}"
            else:
                right_str = ""

            self._print(f"  {Colors.WHITE}{left_str:<{col_width}}{right_str}{Colors.RESET}")

        self._print(f"  {Colors.BRIGHT_CYAN}{'=' * border_len}{Colors.RESET}")

    def _format_metric(self, value) -> str:
        """Format a metric value for display."""
        if isinstance(value, float):
            if value < 0.001:
                return f"{Colors.BRIGHT_CYAN}{value:.2e}{Colors.RESET}"
            elif value < 1:
                return f"{Colors.BRIGHT_CYAN}{value:.4f}{Colors.RESET}"
            else:
                return f"{Colors.BRIGHT_CYAN}{value:.3f}{Colors.RESET}"
        elif isinstance(value, int):
            return f"{Colors.BRIGHT_CYAN}{value:,}{Colors.RESET}"
        elif isinstance(value, bool):
            if value:
                return f"{Colors.GREEN}Yes{Colors.RESET}"
            else:
                return f"{Colors.DIM}No{Colors.RESET}"
        else:
            return f"{Colors.WHITE}{value}{Colors.RESET}"

    def print_user_stream_losses(
        self,
        step: int,
        moshi_semantic: float,
        moshi_acoustic: float,
        user_semantic: float,
        user_acoustic: float,
    ):
        """
        Print user stream (full duplex) loss breakdown.

        Args:
            step: Current training step
            moshi_semantic: Moshi semantic codebook loss
            moshi_acoustic: Moshi acoustic codebooks loss
            user_semantic: User semantic codebook loss
            user_acoustic: User acoustic codebooks loss
        """
        if self.rank != 0:
            return

        moshi_total = moshi_semantic + moshi_acoustic
        user_total = user_semantic + user_acoustic

        self._print()
        self._print(f"  {Colors.BRIGHT_GREEN}[DUPLEX]{Colors.RESET} Full Duplex Loss @ Step {step}")
        self._print(
            f"      Moshi: {Colors.BRIGHT_GREEN}sem={moshi_semantic:.3f}{Colors.RESET} "
            f"aco={moshi_acoustic:.3f} "
            f"total={Colors.BOLD}{moshi_total:.3f}{Colors.RESET}"
        )
        self._print(
            f"      User:  {Colors.BRIGHT_YELLOW}sem={user_semantic:.3f}{Colors.RESET} "
            f"aco={user_acoustic:.3f} "
            f"total={Colors.BOLD}{user_total:.3f}{Colors.RESET}"
        )

    def print_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        eval_loss: float = None,
        wer: float = None,
        cer: float = None,
        elapsed_time: float = None,
    ):
        """
        Print epoch summary with key metrics.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            eval_loss: Evaluation loss (optional)
            wer: Word Error Rate (optional)
            cer: Character Error Rate (optional)
            elapsed_time: Elapsed time in seconds (optional)
        """
        if self.rank != 0:
            return

        self._print()
        border = "~" * (self.width - 4)
        self._print(f"  {Colors.BRIGHT_BLUE}{border}{Colors.RESET}")
        self._print(f"  {Colors.BRIGHT_BLUE}{Colors.BOLD}[EPOCH {epoch}] Summary{Colors.RESET}")

        # Format metrics
        lines = [f"    Train Loss: {Colors.BRIGHT_CYAN}{train_loss:.4f}{Colors.RESET}"]

        if eval_loss is not None:
            eval_color = Colors.GREEN if eval_loss < train_loss else Colors.YELLOW
            lines.append(f"    Eval Loss:  {eval_color}{eval_loss:.4f}{Colors.RESET}")

        if wer is not None:
            wer_color = Colors.GREEN if wer < 0.3 else (Colors.YELLOW if wer < 0.5 else Colors.RED)
            lines.append(f"    WER:        {wer_color}{wer*100:.1f}%{Colors.RESET}")

        if cer is not None:
            cer_color = Colors.GREEN if cer < 0.2 else (Colors.YELLOW if cer < 0.4 else Colors.RED)
            lines.append(f"    CER:        {cer_color}{cer*100:.1f}%{Colors.RESET}")

        if elapsed_time is not None:
            time_str = str(timedelta(seconds=int(elapsed_time)))
            lines.append(f"    Time:       {Colors.DIM}{time_str}{Colors.RESET}")

        for line in lines:
            self._print(line)

        self._print(f"  {Colors.BRIGHT_BLUE}{border}{Colors.RESET}")


# =============================================================================
# Utility Functions
# =============================================================================

def format_number(n: Union[int, float], precision: int = 2) -> str:
    """Format large numbers with K, M, B suffixes."""
    if n >= 1e9:
        return f"{n/1e9:.{precision}f}B"
    elif n >= 1e6:
        return f"{n/1e6:.{precision}f}M"
    elif n >= 1e3:
        return f"{n/1e3:.{precision}f}K"
    else:
        return f"{n:.{precision}f}" if isinstance(n, float) else str(n)


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    return str(timedelta(seconds=int(seconds)))


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage information."""
    import torch

    if not torch.cuda.is_available():
        return {}

    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        info[f"gpu_{i}"] = {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "utilization": allocated / total * 100,
        }
    return info
