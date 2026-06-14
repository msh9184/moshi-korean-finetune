import logging
import os
from dataclasses import dataclass, field
from typing import Tuple

from simple_parsing.helpers import Serializable

from .data.args import DataArgs
from .backbone.config import UnifiedBackboneConfig


@dataclass
class LoraArgs(Serializable):
    enable: bool = False
    rank: int = 64
    scaling: float = 2.0
    ft_embed: bool = False

    def __post_init__(self) -> None:
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0


@dataclass
class OptimArgs(Serializable):
    """
    Optimizer configuration with support for two-rate learning (J-Moshi style).

    For Moshi architecture, separate learning rates for TempFormer and DepFormer
    can lead to better convergence. If depformer_lr is None, it uses the same lr.
    """
    # Primary learning rate (used for TempFormer / main transformer)
    lr: float = 3e-5

    # Separate learning rate for DepFormer (depth transformer)
    # If None, uses the same as lr
    depformer_lr: float | None = None

    # Separate learning rate for DimensionAdapter (backbone bridge)
    # If None, uses the same as lr
    adapter_lr: float | None = None

    # AdamW weight decay
    weight_decay: float = 0.1

    # AdamW betas - J-Moshi uses [0.9, 0.95] for faster convergence
    beta1: float = 0.9
    beta2: float = 0.95

    # AdamW epsilon for numerical stability
    eps: float = 1e-5

    # Legacy: warmup percentage for OneCycleLR (deprecated, use SchedulerArgs)
    pct_start: float = 0.05

    def __post_init__(self) -> None:
        if self.depformer_lr is None:
            self.depformer_lr = self.lr
        if self.adapter_lr is None:
            self.adapter_lr = self.lr


@dataclass
class SchedulerArgs(Serializable):
    """
    Learning rate scheduler configuration.

    Supports multiple scheduler types:
    - onecycle: PyTorch OneCycleLR (original moshi-finetune)
    - cosine_warmup: Cosine annealing with warmup (recommended)
    - warmup_linear: Linear warmup then constant (J-Moshi DeepSpeed style)
    - cosine_restarts: Cosine annealing with warm restarts
    """
    # Scheduler type
    type: str = "cosine_warmup"

    # Number of warmup steps (linear ramp from 0 to max_lr)
    warmup_steps: int = 500

    # Minimum learning rate (for cosine schedulers)
    min_lr: float = 1e-7

    # For cosine_restarts: number of iterations for first restart
    t_0: int = 1000

    # For cosine_restarts: factor to increase T_i after each restart
    t_mult: int = 2

    def __post_init__(self) -> None:
        valid_types = ("onecycle", "cosine_warmup", "warmup_linear", "cosine_restarts")
        if self.type not in valid_types:
            raise ValueError(f"scheduler.type must be one of {valid_types}, got '{self.type}'")


@dataclass
class WandbArgs(Serializable):
    """
    Weights & Biases logging configuration.

    W&B is optional - if not installed, logging will fall back to TensorBoard only.
    Set project to None to disable W&B completely.
    """
    project: str | None = None  # Set to enable W&B logging (optional)
    offline: bool = True  # Changed default to True for network-restricted environments
    key: str | None = None
    run_name: str | None = None
    enabled: bool = False  # Explicit enable flag (project != None will also enable)

    def __post_init__(self) -> None:
        if self.project is not None and len(self.project) > 0:
            try:
                import wandb  # noqa: F401
                self.enabled = True
            except ImportError:
                logging.warning(
                    "W&B project specified but `wandb` not installed. "
                    "Falling back to TensorBoard-only logging. "
                    "Install wandb with: pip install wandb"
                )
                self.project = None
                self.enabled = False

            if self.project is not None and len(self.project) == 0:
                logging.warning("`wandb.project` is empty string, disabling W&B.")
                self.project = None
                self.enabled = False


@dataclass
class TensorBoardArgs(Serializable):
    """
    TensorBoard logging configuration.

    TensorBoard is the primary logging backend, always enabled by default.
    """
    enabled: bool = True
    log_histograms: bool = True  # Weight distribution histograms
    histogram_freq: int = 500  # Steps between histogram logs (reduced frequency)
    log_gradients: bool = True  # Gradient statistics
    gradient_freq: int = 100  # Steps between gradient logs
    log_memory: bool = True  # GPU memory usage
    log_per_codebook: bool = True  # Per-codebook loss breakdown
    flush_secs: int = 30  # Seconds between flushes


# =============================================================================
# Advanced Monitoring Configuration (Phase 3)
# =============================================================================

@dataclass
class TextEvaluationArgs(Serializable):
    """
    Text stream (Inner Monologue) evaluation configuration.

    Computes WER between predicted text and ground truth alignments.
    Logs sample predictions for qualitative monitoring.
    """
    enabled: bool = True

    # Frequency of WER evaluation (steps)
    eval_freq: int = 500

    # Log prediction samples to console and TensorBoard
    log_predictions: bool = True

    # Number of prediction samples to log per evaluation
    max_prediction_samples: int = 5

    # Normalize text before WER calculation (lowercase, remove punctuation)
    normalize_text: bool = True


@dataclass
class CodebookAnalysisArgs(Serializable):
    """
    Per-codebook loss analysis configuration.

    Tracks individual losses for 8 audio codebooks + 1 text stream.
    Useful for understanding semantic vs acoustic learning dynamics.
    """
    enabled: bool = True

    # Frequency of per-codebook loss logging (steps)
    log_freq: int = 100

    # Log codebook usage statistics (token distribution)
    log_usage_stats: bool = True

    # Log entropy analysis for each codebook
    log_entropy: bool = True


@dataclass
class GradientMonitoringArgs(Serializable):
    """
    Gradient health monitoring configuration.

    Detects NaN/Inf gradients, exploding gradients, and vanishing gradients.
    Critical for debugging training instabilities.
    """
    enabled: bool = True

    # Frequency of gradient health checks (steps)
    log_freq: int = 50

    # Alert on NaN gradients (will also log warning)
    alert_on_nan: bool = True

    # Alert on Inf gradients
    alert_on_inf: bool = True

    # Threshold for exploding gradient detection
    exploding_threshold: float = 100.0

    # Threshold for vanishing gradient detection
    vanishing_threshold: float = 1e-7

    # Log detailed per-layer gradient norms
    log_per_layer: bool = False


@dataclass
class BackboneMonitoringArgs(Serializable):
    """
    Backbone-specific monitoring configuration.

    For modular backbone system (HFLM, custom LLM backbones).
    Monitors DimensionAdapter and backbone-specific metrics.
    """
    enabled: bool = True

    # Log DimensionAdapter input/output projection norms
    log_adapter_norms: bool = True

    # Log backbone hidden state activations (expensive)
    log_backbone_activations: bool = False

    # Compare backbone outputs with original Moshi (requires both loaded)
    compare_backbone_moshi: bool = False


@dataclass
class MonitoringArgs(Serializable):
    """
    Unified monitoring configuration.

    Combines text evaluation, codebook analysis, gradient monitoring,
    and backbone monitoring.
    """
    text_evaluation: TextEvaluationArgs = field(default_factory=TextEvaluationArgs)
    codebook_analysis: CodebookAnalysisArgs = field(default_factory=CodebookAnalysisArgs)
    gradient_monitoring: GradientMonitoringArgs = field(default_factory=GradientMonitoringArgs)
    backbone_monitoring: BackboneMonitoringArgs = field(default_factory=BackboneMonitoringArgs)


# =============================================================================
# Enhanced Evaluation Configuration
# =============================================================================

@dataclass
class SemanticEvaluationArgs(Serializable):
    """
    Semantic quality evaluation configuration (BLEU, Semantic Similarity).

    Evaluates the semantic quality of Inner Monologue predictions:
    - BLEU: N-gram based text similarity (requires sacrebleu)
    - Semantic Similarity: Embedding-based meaning similarity (optional)
    """
    enabled: bool = True

    # Compute BLEU score (requires: pip install sacrebleu)
    compute_bleu: bool = True

    # Compute semantic similarity using sentence embeddings
    # Requires: pip install sentence-transformers
    # Note: Computationally expensive, loads additional model
    compute_semantic: bool = False

    # Model for semantic similarity computation
    semantic_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Maximum samples to evaluate
    max_samples: int = 50


@dataclass
class AlignmentEvaluationArgs(Serializable):
    """
    Text-Audio alignment quality evaluation configuration.

    Evaluates how well Inner Monologue text is synchronized with audio:
    - Timing accuracy: Correct token placement in time
    - Boundary detection: Word boundary precision/recall
    - Sync score: Overall synchronization quality
    """
    enabled: bool = True

    # Tolerance in frames for "correct" timing (1 frame = 80ms at 12.5Hz)
    tolerance_frames: int = 2


@dataclass
class DialogueEvaluationArgs(Serializable):
    """
    Dialogue quality evaluation configuration (Full-Duplex mode only).

    Evaluates Moshi's real-time conversation abilities:
    - Turn-taking: Natural alternation between speakers
    - Response latency: Time between user and Moshi responses
    - Overlap handling: Detection and resolution of simultaneous speech

    Note: Only meaningful in Full-Duplex (V3) mode where user audio is available.
    """
    enabled: bool = True

    # Minimum frames to count as overlap (3 frames = 240ms)
    overlap_threshold_frames: int = 3

    # Minimum frames to count as silence gap (25 frames = 2 seconds)
    silence_threshold_frames: int = 25


@dataclass
class AudioQualityEvaluationArgs(Serializable):
    """
    Audio quality evaluation configuration (PESQ, STOI, MCD).

    Compares Mimi-decoded audio from GT vs predicted codes.

    WARNING: Computationally expensive! Consider:
    - Using max_samples to limit evaluation
    - Running less frequently than other monitors
    - Disabling by default (enabled: false)

    Dependencies:
    - PESQ: pip install pesq
    - STOI: pip install pystoi
    - MCD: pip install librosa
    """
    enabled: bool = False  # Disabled by default due to computational cost

    # Compute PESQ (Perceptual Evaluation of Speech Quality)
    # Requires 16kHz resampling
    compute_pesq: bool = True

    # Compute STOI (Short-Time Objective Intelligibility)
    compute_stoi: bool = True

    # Compute MCD (Mel Cepstral Distortion)
    compute_mcd: bool = True

    # Maximum samples to evaluate per batch
    max_samples: int = 10

    # Minimum audio duration to evaluate (seconds)
    min_duration_sec: float = 1.0


@dataclass
class EnhancedEvaluationArgs(Serializable):
    """
    Enhanced Evaluation System Configuration.

    Coordinates all advanced evaluation monitors beyond basic loss metrics:
    - Semantic: BLEU score, semantic similarity
    - Alignment: Text-audio timing accuracy
    - Dialogue: Turn-taking, response latency (Full-Duplex only)
    - Audio Quality: PESQ, STOI, MCD (expensive, disabled by default)

    All metrics are logged to TensorBoard under eval.{category}/{metric} tags.
    """
    # Semantic quality (BLEU, embedding similarity)
    semantic: SemanticEvaluationArgs = field(default_factory=SemanticEvaluationArgs)

    # Text-audio alignment quality
    alignment: AlignmentEvaluationArgs = field(default_factory=AlignmentEvaluationArgs)

    # Dialogue quality (Full-Duplex mode only)
    dialogue: DialogueEvaluationArgs = field(default_factory=DialogueEvaluationArgs)

    # Audio quality (expensive, disabled by default)
    audio_quality: AudioQualityEvaluationArgs = field(default_factory=AudioQualityEvaluationArgs)


# =============================================================================
# Sample Saving Configuration (Phase 4)
# =============================================================================

@dataclass
class SampleSavingArgs(Serializable):
    """
    Audio and text sample saving configuration.

    Saves decoded audio samples and text predictions during training
    for qualitative evaluation and debugging.

    Two types of sample savers:
    1. Segment samples (SampleSaver): 60-second training segments with GT vs Prediction
    2. Dialogue samples (DialogueSampleSaver): Complete dialogues with GT vs Prediction

    Output structure:
        samples/{split}/step_{step}/
        ├── sample_segment/    # Segment-level samples (60s)
        │   ├── sample_00_gt_dialogue.wav
        │   ├── sample_00_pred_dialogue.wav
        │   └── ...
        └── sample_dialogue/   # Full dialogue samples
            ├── sample_00_gt_dialogue.wav
            ├── sample_00_pred_dialogue.wav
            └── ...
    """
    enabled: bool = True

    # =================================================================
    # Sample Type Control
    # =================================================================
    # Save segment-level samples (60s training chunks) - uses SampleSaver
    # Good for: Quick GT vs Prediction comparison during training
    save_segment_samples: bool = True

    # Save complete dialogue samples - uses DialogueSampleSaver
    # Good for: Full conversation quality evaluation
    # Note: More expensive as it runs model inference on entire dialogue
    save_dialogue_samples: bool = True

    # =================================================================
    # Frequency and Limits
    # =================================================================
    # Frequency of sample saving (steps)
    save_freq: int = 1000

    # Maximum number of sample sets to keep per split (train/valid)
    # Older samples are deleted when limit is exceeded
    max_samples_per_split: int = 20

    # Number of random samples to save per save event (for segment samples)
    samples_per_save: int = 3

    # Maximum complete dialogues to save per split (for dialogue samples)
    max_dialogues_per_split: int = 5

    # =================================================================
    # Audio/Text Options
    # =================================================================
    # Save audio samples (requires mimi decoder)
    save_audio: bool = True

    # Save text predictions
    save_text: bool = True

    # Audio format: "wav" or "flac"
    audio_format: str = "wav"

    # Audio sample rate (should match mimi output: 24000)
    sample_rate: int = 24000

    # =================================================================
    # Debug Options
    # =================================================================
    # Enable verbose debug logging for audio consistency verification
    # Useful for debugging audio encoding/decoding issues
    # Warning: Can produce very verbose logs, use only for debugging
    debug_audio_consistency: bool = False

    def __post_init__(self) -> None:
        if self.audio_format not in ("wav", "flac"):
            raise ValueError(f"audio_format must be 'wav' or 'flac', got '{self.audio_format}'")


# =============================================================================
# Research Logging Configuration (Phase 5)
# =============================================================================

@dataclass
class ResearchLoggingArgs(Serializable):
    """
    Research and paper-writing data logging configuration.

    Saves detailed analysis data for academic paper writing:
    - Attention maps (Temporal/Depth Transformer)
    - Loss curves (train/valid, per-codebook)
    - Gradient norms over time
    - Codebook usage statistics
    - Model weight statistics
    """
    enabled: bool = True

    # Save attention map visualizations
    save_attention_maps: bool = True

    # Frequency of attention map saving (steps)
    attention_freq: int = 2000

    # Number of attention samples to save per event
    attention_samples: int = 2

    # Save raw attention tensors (.npy) in addition to plots
    save_raw_attention: bool = False

    # Save loss curves (CSV + PNG)
    save_loss_curves: bool = True

    # Save codebook usage statistics
    save_codebook_stats: bool = True

    # Save gradient norm history
    save_gradient_norms: bool = True

    # Automatically generate plots
    generate_plots: bool = True

    # Frequency of plot generation (steps)
    plot_freq: int = 1000

    # Save training summary JSON at end
    save_summary: bool = True


# =============================================================================
# Checkpoint Configuration (Resume Support)
# =============================================================================

@dataclass
class CheckpointArgs(Serializable):
    """
    Checkpoint saving and resume configuration.

    This module provides comprehensive checkpoint management with:
    - Metric-based checkpoint naming for easy model selection
    - Automatic best/last model tracking with symlinks
    - Full resume support (model + optimizer + scheduler + training state)
    - Configurable retention policy

    ═══════════════════════════════════════════════════════════════════════════
    CHECKPOINT FILE STRUCTURE
    ═══════════════════════════════════════════════════════════════════════════

    runs/{run_name}/checkpoints/
    ├── config.json                                           # Model config (shared)
    ├── checkpoint.eval_loss-5.430.step-000040.safetensors   # Model weights
    ├── checkpoint.eval_loss-4.472.step-000050.safetensors
    ├── checkpoint.eval_loss-3.570.step-000080.safetensors
    ├── checkpoint.eval_loss-3.570.step-000080.best.safetensors  # Symlink → best
    ├── checkpoint.eval_loss-3.570.step-000080.last.safetensors  # Symlink → latest
    ├── training_state.step-000080.last.pt                   # Training state (last)
    └── training_state.step-000080.best.pt                   # Training state (best)

    ═══════════════════════════════════════════════════════════════════════════
    NAMING FORMAT
    ═══════════════════════════════════════════════════════════════════════════

    Model weights: {name_prefix}.{metric_type}-{value:.3f}.step-{step:06d}.safetensors
    Training state: training_state.step-{step:06d}.{best|last}.pt

    ═══════════════════════════════════════════════════════════════════════════
    RESUME BEHAVIOR
    ═══════════════════════════════════════════════════════════════════════════

    1. resume_if_exist=True (default):
       - On startup, checks for *.last.safetensors
       - If found, loads model weights + training state
       - Training continues from the saved step

    2. resume_from="/path/to/checkpoint.safetensors":
       - Explicitly specifies checkpoint to resume from
       - Overrides resume_if_exist behavior

    ═══════════════════════════════════════════════════════════════════════════
    TRAINING STATE CONTENTS
    ═══════════════════════════════════════════════════════════════════════════

    training_state.step-{step}.{tag}.pt contains:
    {
        "step": int,                    # Current training step
        "optimizer_state_dict": dict,   # AdamW momentum/variance
        "scheduler_state_dict": dict,   # LR scheduler state
        "train_state": {
            "max_steps": int,
            "step": int,
            "elapsed_time": float,
            "n_seen_tokens": int,
            "best_metric": float,
            "best_step": int,
        },
        "rng_state": {
            "torch": tensor,
            "cuda": list[tensor],
        },
        "metadata": {
            "timestamp": str,
            "world_size": int,
            "metric_type": str,
            "metric_value": float,
            "model_file": str,
        }
    }
    """
    # =========================================================================
    # Enable/Disable
    # =========================================================================
    enabled: bool = True

    # =========================================================================
    # File Naming
    # =========================================================================
    # Prefix for checkpoint filenames
    # Example: "checkpoint" → checkpoint.eval_loss-3.570.step-000080.safetensors
    name_prefix: str = "checkpoint"

    # =========================================================================
    # Metric Tracking
    # =========================================================================
    # Metric to use for best model selection and filename
    # Options: "train_loss", "eval_loss", "eval_perplexity"
    metric_type: str = "eval_loss"

    # How to determine "best" metric
    # "min" = lower is better (for loss)
    # "max" = higher is better (for accuracy)
    metric_best: str = "min"

    # =========================================================================
    # Resume Options
    # =========================================================================
    # Automatically resume from last checkpoint if found
    # Checks for {name_prefix}.*.last.safetensors in checkpoint directory
    resume_if_exist: bool = True

    # Explicit path to checkpoint for resume (overrides resume_if_exist)
    # Can be absolute path or relative to run_dir/checkpoints/
    # Example: "/path/to/checkpoint.safetensors" or "checkpoint.eval_loss-3.570.step-000080.safetensors"
    resume_from: str | None = None

    # =========================================================================
    # State Saving Options
    # =========================================================================
    # Save optimizer state (required for perfect resume)
    # Note: For full finetuning, optimizer state can be ~2x model size
    # For LoRA, optimizer state is small (only trainable params)
    save_optimizer: bool = True

    # Save scheduler state (required for LR resume)
    save_scheduler: bool = True

    # Save RNG state for reproducibility
    # Enables bit-exact resume of training
    save_rng_state: bool = True

    # =========================================================================
    # Frequency and Retention
    # =========================================================================
    # Save checkpoint every N steps (0 = disabled, only save at end)
    save_freq: int = 1000

    # Maximum number of checkpoints to keep (excluding best/last symlinks)
    # Older checkpoints are deleted when limit is exceeded
    # None = keep all checkpoints
    max_keep: int | None = 5

    # =========================================================================
    # Save Options
    # =========================================================================
    # Save only LoRA adapters (smaller files, requires base model for inference)
    # If False, saves full merged model (larger but standalone)
    save_adapters_only: bool = True

    # Data type for saved weights
    # "bfloat16", "float16", "float32"
    save_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        # Validate metric_type
        valid_metrics = ("train_loss", "eval_loss", "eval_perplexity")
        if self.metric_type not in valid_metrics:
            raise ValueError(
                f"checkpoint.metric_type must be one of {valid_metrics}, "
                f"got '{self.metric_type}'"
            )

        # Validate metric_best
        if self.metric_best not in ("min", "max"):
            raise ValueError(
                f"checkpoint.metric_best must be 'min' or 'max', "
                f"got '{self.metric_best}'"
            )

        # Validate save_dtype
        valid_dtypes = ("bfloat16", "float16", "float32")
        if self.save_dtype not in valid_dtypes:
            raise ValueError(
                f"checkpoint.save_dtype must be one of {valid_dtypes}, "
                f"got '{self.save_dtype}'"
            )

        # Warning for full model saves
        if not self.save_adapters_only:
            logging.warning(
                "save_adapters_only=False: Full model will be saved at each checkpoint. "
                "This requires significant disk space (~14GB per checkpoint for Moshi 7B). "
                "For LoRA training, consider save_adapters_only=True for smaller checkpoints."
            )

        # Log resume configuration
        if self.resume_if_exist:
            logging.info(
                "Checkpoint resume enabled: Will automatically resume from last checkpoint if found."
            )
        if self.resume_from:
            logging.info(f"Explicit resume checkpoint specified: {self.resume_from}")


@dataclass
class PretrainedModelArgs(Serializable):
    """
    Stage-based pretrained model loading configuration.

    This configuration allows loading weights from a previous training stage
    while initializing new parameters (e.g., speaker_conditioner) from scratch.

    Key differences from Checkpoint Resume:
    ═══════════════════════════════════════════════════════════════════════════
    | Feature            | Checkpoint Resume    | Pretrained Loading         |
    ═══════════════════════════════════════════════════════════════════════════
    | Purpose            | Continue training    | Start new stage            |
    | Step               | Resume from saved    | Start from 0               |
    | Optimizer          | Restore              | Fresh initialization       |
    | Scheduler          | Restore              | Fresh initialization       |
    | New parameters     | Error (strict)       | Keep random init           |
    | Missing parameters | Error (strict)       | Warning only               |
    ═══════════════════════════════════════════════════════════════════════════

    Usage Example (YAML):
        pretrained:
          enabled: true
          path: "best"  # or absolute path to .safetensors
          checkpoint_dir: "./runs/korean_moshi_stage1_pretrain/checkpoints"
          strict: false
          expected_new_modules:
            - "speaker_conditioner"
          verbose: true

    Workflow:
        Stage 1: Train base model (no speaker conditioning)
                 → saves to runs/stage1/checkpoints/checkpoint.*.safetensors

        Stage 2: Load Stage 1 weights + add speaker conditioning
                 pretrained.path = "best"
                 pretrained.checkpoint_dir = "runs/stage1/checkpoints"
                 → speaker_conditioner initialized randomly
                 → starts from step 0
    """

    # =========================================================================
    # Enable/Disable
    # =========================================================================
    # Enable pretrained model loading from previous stage
    enabled: bool = False

    # =========================================================================
    # Path Configuration
    # =========================================================================
    # Path to safetensors file from previous stage
    # Supports:
    #   - Absolute path: "/path/to/checkpoint.safetensors"
    #   - Relative path (to checkpoint_dir): "checkpoint.eval_loss-2.434.step-000880.safetensors"
    #   - Keywords: "best" or "last" (auto-find in checkpoint_dir)
    path: str | None = None

    # Directory containing checkpoints (required for "best"/"last" keywords)
    # Can be absolute or relative to parent of run_dir
    checkpoint_dir: str | None = None

    # =========================================================================
    # Loading Behavior
    # =========================================================================
    # Strict loading mode:
    #   - True: Error on unexpected missing/extra keys
    #   - False: Warning only, continue with partial loading
    strict: bool = False

    # List of module prefixes expected to be newly initialized
    # These won't trigger warnings when missing in pretrained weights
    # Common values:
    #   - "speaker_conditioner": Speaker conditioning module
    #   - "dimension_adapter": Backbone dimension adapter (for HFLM)
    expected_new_modules: list = field(default_factory=lambda: [
        "speaker_conditioner",
        "dimension_adapter",
    ])

    # =========================================================================
    # Logging
    # =========================================================================
    # Log detailed loading information (loaded/skipped/new params)
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.enabled:
            if self.path is None:
                raise ValueError(
                    "pretrained.path must be specified when pretrained.enabled=True"
                )

            # Validate "best"/"last" keywords require checkpoint_dir
            if self.path.lower() in ("best", "last") and self.checkpoint_dir is None:
                raise ValueError(
                    f"pretrained.checkpoint_dir must be specified when path='{self.path}'"
                )

            logging.info(
                f"[PRETRAINED] Stage loading enabled: path={self.path}, "
                f"checkpoint_dir={self.checkpoint_dir}"
            )


@dataclass
class ModelPaths(Serializable):
    hf_repo_id: str | None = "kyutai/moshiko-pytorch-bf16"
    mimi_path: str | None = None
    moshi_path: str | None = None
    tokenizer_path: str | None = None
    config_path: str | None = None

    def __post_init__(self) -> None:
        if self.hf_repo_id is not None and self.config_path is None:
            print(
                "Warning: `hf_repo_id` is set but `config_path` is None. "
                "This will load default models."
            )


# =============================================================================
# Speaker Conditioning Configuration (Zero-Shot Speaker Adaptation)
# =============================================================================

@dataclass
class SpeakerEncoderArgs(Serializable):
    """
    Speaker encoder configuration for zero-shot speaker adaptation.

    The speaker encoder extracts speaker embeddings from reference audio,
    which are then used to condition the Temporal Transformer.

    Supported encoders:
        - ecapa_tdnn: ECAPA-TDNN from SpeechBrain (192-dim, VoxCeleb pre-trained)
        - w2v_bert2: W2v-BERT 2.0 SV (256-dim, 0.14% EER SOTA on VoxCeleb1-O)
                     Paper: https://arxiv.org/abs/2510.04213
                     Model: https://huggingface.co/zl389/w2v-bert-2.0_SV
        - dummy: Dummy encoder for testing (random embeddings)
        - custom: Custom speaker encoder (for team's future model)

    Reference: ZERO_SHOT_SPEAKER_CONDITIONING_SPECIFICATION.md
    """
    # Encoder type selection
    # - "ecapa_tdnn": SpeechBrain ECAPA-TDNN (192-dim)
    # - "w2v_bert2": W2v-BERT 2.0 SV (256-dim, SOTA 0.14% EER)
    # - "dummy": Testing without dependencies
    # - "custom": Team's future model
    encoder_type: str = "ecapa_tdnn"

    # Path or HuggingFace repo ID for pre-trained encoder
    # - ecapa_tdnn: "speechbrain/spkrec-ecapa-voxceleb"
    # - w2v_bert2: path to model_lmft_0.14.pth (or .safetensors)
    pretrained_path: str = "speechbrain/spkrec-ecapa-voxceleb"

    # Freeze encoder weights during training (recommended for pre-trained)
    freeze: bool = True

    # Output embedding dimension
    # - ecapa_tdnn: 192
    # - w2v_bert2: 256
    # - dummy: configurable
    output_dim: int = 192

    # Input sample rate (16000 for most speaker encoders)
    sample_rate: int = 16000

    # L2-normalize output embeddings
    normalize_embedding: bool = True

    # Custom encoder path (for future team model)
    custom_encoder_path: str | None = None

    # W2v-BERT 2.0 specific settings
    # Number of MFA layers to aggregate (-1 = all layers)
    w2v_bert2_n_mfa_layers: int = -1

    # W2v-BERT 2.0 pooling type: "ASP" (Attentive Statistics Pooling)
    w2v_bert2_pooling: str = "ASP"

    def __post_init__(self) -> None:
        valid_types = ("ecapa_tdnn", "w2v_bert2", "dummy", "custom")
        if self.encoder_type not in valid_types:
            raise ValueError(
                f"speaker.encoder.encoder_type must be one of {valid_types}, "
                f"got '{self.encoder_type}'"
            )

        # Auto-adjust output_dim for w2v_bert2 if using default
        if self.encoder_type == "w2v_bert2" and self.output_dim == 192:
            self.output_dim = 256


@dataclass
class SpeakerConditionerArgs(Serializable):
    """
    Speaker conditioner configuration (projection layer).

    Transforms speaker embeddings to the Temporal Transformer's hidden dimension
    via a learnable projection layer with optional LayerNorm and scaling.

    Shape flow:
        Speaker Embedding [B, D_spk] → Projection → [B, 1, D_model]

    Reference: ZERO_SHOT_SPEAKER_CONDITIONING_SPECIFICATION.md
    """
    # Output dimension matching Temporal TF hidden dim (4096 for Moshi)
    output_dim: int = 4096

    # Initial value for learnable scale parameter (small for stability)
    initial_scale: float = 0.1

    # Apply LayerNorm after projection
    use_layernorm: bool = True

    # Dropout probability after projection
    dropout: float = 0.0

    # Whether scale parameter is learnable
    learnable_scale: bool = True

    # Scale mode: "multiply" (scalar) or "gated" (element-wise sigmoid)
    scale_mode: str = "multiply"

    def __post_init__(self) -> None:
        if self.scale_mode not in ("multiply", "gated"):
            raise ValueError(
                f"speaker.conditioner.scale_mode must be 'multiply' or 'gated', "
                f"got '{self.scale_mode}'"
            )


@dataclass
class ReferenceSamplerArgs(Serializable):
    """
    Reference audio sampler configuration for training.

    During training, we sample a random segment from the MOSHI audio stream
    to use as speaker reference. This avoids using the same segment being predicted.

    Reference: ZERO_SHOT_SPEAKER_CONDITIONING_SPECIFICATION.md Section 5
    """
    # Minimum reference duration in seconds
    min_duration_sec: float = 3.0

    # Maximum reference duration in seconds
    max_duration_sec: float = 10.0

    # Source sample rate (Mimi's rate: 24000)
    sample_rate: int = 24000

    # Target sample rate (speaker encoder's rate: 16000)
    target_sample_rate: int = 16000


@dataclass
class AudioPromptArgs(Serializable):
    """
    PersonaPlex style audio/text prompting configuration.

    This enables prepending reference audio codes AND text tokens to the main
    training sequence, following the PersonaPlex approach. Unlike VALL-E which
    uses audio-only prompting, PersonaPlex always includes BOTH audio and text.

    Prompting Modes:
        - "speaker_embedding": Only global speaker embedding (no prompt prepending)
        - "audio_text": Prepend both audio codes AND text tokens (PersonaPlex style)

    Architecture:
        ┌─────────────────────────────┐  ┌────────────────────────────┐
        │ Reference Prompt (10-15s)   │→│ Main Sequence (Training)   │
        │ [text_tokens] [audio_codes] │→│ [text_tokens] [audio_codes]│
        └─────────────────────────────┘  └────────────────────────────┘
                ↑ prompt_mask=True             prompt_mask=False

    The prompt region is excluded from loss computation via prompt_mask.

    Reference: K-Moshi Zero-Shot Speaker Conditioning Specification
    """
    # Enable audio prompting (in addition to speaker embedding)
    enable: bool = False

    # Prompting mode
    # - "speaker_embedding": Only global speaker embedding (no audio prepend)
    # - "audio_text": Prepend both audio codes AND text tokens (PersonaPlex style)
    mode: str = "speaker_embedding"

    # Prompt duration (10-15 seconds recommended for good speaker adaptation)
    min_duration_sec: float = 10.0
    max_duration_sec: float = 15.0

    # Sampling strategy: "random", "start", "end", "voiced"
    # - "random": Random position and duration (TRAINING only)
    # - "start": Fixed position from start (RECOMMENDED for eval/inference)
    # - "end": Fixed position from end
    # - "voiced": Prefer voiced segments using VAD
    sample_strategy: str = "random"

    # Include special tokens (BOS/EOS) around prompt
    include_special_tokens: bool = True

    # Avoid sampling from the same segment being trained
    avoid_overlap: bool = True

    # =========================================================================
    # DETERMINISTIC MODE FOR EVALUATION/INFERENCE
    # =========================================================================
    # When deterministic=True:
    #   - NO random number generation
    #   - Fixed duration (fixed_duration_sec) instead of random [min, max]
    #   - Fixed position based on sample_strategy ("start" recommended)
    #   - Same input always produces same reference selection
    #
    # CRITICAL for reproducible evaluation and inference!
    # =========================================================================
    deterministic: bool = False

    # Fixed duration when deterministic=True (overrides min/max_duration_sec)
    fixed_duration_sec: float = 10.0

    # =========================================================================
    # WORD-COUNT BASED SELECTION (alternative to duration-based)
    # =========================================================================
    # When use_word_count=True:
    # - min_words/max_words define the valid range of non-padding text tokens
    # - Segment is selected to contain approximately min_words~max_words tokens
    # - This ensures meaningful speech content in reference audio
    # =========================================================================
    use_word_count: bool = False

    # Word count range for random sampling (when use_word_count=True)
    min_words: int = 5       # Minimum non-padding text tokens
    max_words: int = 30      # Maximum non-padding text tokens

    # Fixed word count for deterministic mode (evaluation/inference)
    fixed_word_count: int = 20

    # Text padding token IDs to exclude from word count
    # Default: 0=PAD, 3=EOS, 32000=END_OF_TEXT (for Moshi tokenizer)
    text_padding_token_ids: Tuple[int, ...] = (0, 3, 32000)

    def __post_init__(self) -> None:
        # PersonaPlex style: only two modes supported
        valid_modes = ("speaker_embedding", "audio_text")
        if self.mode not in valid_modes:
            raise ValueError(
                f"speaker.audio_prompt.mode must be one of {valid_modes}, "
                f"got '{self.mode}'. Note: 'audio_only' (VALL-E style) is not "
                "supported. Use 'audio_text' for PersonaPlex style prompting."
            )

        valid_strategies = ("random", "start", "end", "voiced")
        if self.sample_strategy not in valid_strategies:
            raise ValueError(
                f"speaker.audio_prompt.sample_strategy must be one of {valid_strategies}, "
                f"got '{self.sample_strategy}'"
            )

        # Warn if deterministic mode with random strategy
        if self.deterministic and self.sample_strategy == "random":
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "speaker.audio_prompt: deterministic=True but sample_strategy='random'. "
                "For truly deterministic behavior, use sample_strategy='start' or 'end'."
            )


@dataclass
class SpeakerConditioningArgs(Serializable):
    """
    Speaker conditioning configuration for zero-shot speaker adaptation.

    Enables the model to generate speech in a specific speaker's voice
    given a short reference audio sample.

    Architecture:
        Reference Audio → Speaker Encoder → Speaker Conditioner → sum_condition
                                                                    ↓
        Temporal TF Input: text_emb + audio_emb + speaker_condition

    Conditioning Methods:
        1. Speaker Encoder (Default):
           Extract speaker embedding from reference audio, add via sum_condition.
           Options: ecapa_tdnn (192-dim), w2v_bert2 (256-dim SOTA), dummy, custom

        2. Audio Prompt (VALL-E Style):
           Prepend reference audio/text codes to main sequence.
           Options: audio_only (codes only), audio_text (codes + text)

    These methods can be combined: speaker_encoder + audio_prompt for best results.

    Reference: ZERO_SHOT_SPEAKER_CONDITIONING_SPECIFICATION.md
    """
    # Enable speaker conditioning
    enabled: bool = False

    # Conditioning method: "encoder" (sum_condition), "prompt" (prepend), or "both"
    method: str = "encoder"

    # Speaker encoder configuration
    encoder: SpeakerEncoderArgs = field(default_factory=SpeakerEncoderArgs)

    # Speaker conditioner (projection layer) configuration
    conditioner: SpeakerConditionerArgs = field(default_factory=SpeakerConditionerArgs)

    # Reference sampler configuration (for training)
    reference_sampler: ReferenceSamplerArgs = field(default_factory=ReferenceSamplerArgs)

    # Audio prompting configuration (VALL-E style)
    audio_prompt: AudioPromptArgs = field(default_factory=AudioPromptArgs)

    # Reference audio path for inference (None during training)
    # During training, reference is sampled from MOSHI stream
    inference_reference_path: str | None = None

    # Reference text for inference (optional, used with audio prompt method)
    inference_reference_text: str | None = None

    def __post_init__(self) -> None:
        valid_methods = ("encoder", "prompt", "both")
        if self.method not in valid_methods:
            raise ValueError(
                f"speaker.method must be one of {valid_methods}, "
                f"got '{self.method}'"
            )

        # Auto-enable audio_prompt if method is "prompt" or "both"
        if self.method in ("prompt", "both") and not self.audio_prompt.enable:
            self.audio_prompt.enable = True
            if self.audio_prompt.mode == "speaker_embedding":
                self.audio_prompt.mode = "audio_text"  # PersonaPlex style (audio + text)


# =============================================================================
# UNIFIED SEGMENT FILTERING SYSTEM
# =============================================================================
# This is a complete redesign that merges:
#   - data_filtering (stream integrity)
#   - data_augmentation (filtering, smart_segmentation)
# into a single, clear, hierarchical structure.
#
# The old structure had overlapping and confusing options:
#   - stream_integrity.min_moshi_text_words vs filtering.min_moshi_words
#   - stream_integrity.allow_case2 vs smart_segmentation.require_both_speakers
#
# NEW STRUCTURE:
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                        segment_filtering (UNIFIED)                          │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │  Layer 1: case_control     - Structural validity (Case 1-5)                 │
# │  Layer 2: quality          - Quality requirements (min_words, duration)     │
# │  Layer 3: preferences      - Probabilistic preferences (optional)           │
# │  Layer 4: role_swapping    - Data augmentation (swap moshi↔user)            │
# │  Layer 5: logging          - Debug and statistics logging                   │
# └─────────────────────────────────────────────────────────────────────────────┘
# =============================================================================


@dataclass
class CaseControlArgs(Serializable):
    """
    Layer 1: Structural Validity Control (Case-based filtering).

    Determines which segment configurations are allowed based on
    the presence/absence of audio and text streams.

    +------+----------------------------------------+---------+---------------------+
    | Case | Configuration                          | Default | Description         |
    +------+----------------------------------------+---------+---------------------+
    | 1    | moshi_audio + moshi_text + user_audio  | [ALLOW] | Full dialogue       |
    | 2    | moshi_audio + moshi_text               | [ALLOW] | Moshi monologue     |
    | 3    | user_audio only                        | [SKIP]  | No Moshi content    |
    | 4    | moshi_text + user_audio                | [SKIP]  | Missing Moshi audio |
    | 5    | moshi_audio + user_audio               | [SKIP]  | Missing Moshi text  |
    +------+----------------------------------------+---------+---------------------+

    Use Cases:
    - DIALOGUE training: allow_case1=true, allow_case2=false (require both speakers)
    - MONOLOGUE training: allow_case1=true, allow_case2=true (allow Moshi-only)
    - Case 3-5: Always false (data corruption or invalid structure)
    """
    enabled: bool = True

    # Case 1: Full dialogue (moshi_audio + moshi_text + user_audio)
    allow_case1: bool = True

    # Case 2: Moshi monologue (moshi_audio + moshi_text, no user_audio)
    # Set to False to require both speakers (dialogue-only mode)
    allow_case2: bool = True

    # Case 3: User audio only (no Moshi content) - typically invalid data
    allow_case3: bool = False

    # Case 4: Missing Moshi audio (moshi_text + user_audio) - data corruption
    allow_case4: bool = False

    # Case 5: Missing Moshi text (moshi_audio + user_audio) - data corruption
    allow_case5: bool = False

    # -------------------------------------------------------------------------
    # Audio Detection Thresholds
    # -------------------------------------------------------------------------
    # Minimum RMS energy to consider audio channel as having content
    # Speech: 0.01-0.1, Silence: < 0.001
    min_audio_energy: float = 0.001

    # Minimum ratio of frames with audio above threshold
    # 0.05 = at least 5% of frames should have audio
    min_audio_presence_ratio: float = 0.05


@dataclass
class QualityRequirementsArgs(Serializable):
    """
    Layer 2: Quality Requirements (hard minimums).

    Enforces minimum quality standards for segments.
    These are NOT probabilistic - segments failing these are always filtered.

    [!] IMPORTANT for MONOLOGUE training:
       Set min_user_words=0 to allow segments without user speech!
    """
    enabled: bool = True

    # -------------------------------------------------------------------------
    # Moshi (SPEAKER_MAIN) Requirements
    # -------------------------------------------------------------------------
    # Minimum words from Moshi required (combined from all Moshi alignments)
    min_moshi_words: int = 3

    # Minimum Moshi speech duration in seconds
    min_moshi_duration_sec: float = 3.0

    # Minimum Moshi speech ratio: moshi_duration / total_speech_duration
    # Set to 0.0 to disable ratio check
    min_moshi_ratio: float = 0.1

    # -------------------------------------------------------------------------
    # User Requirements
    # -------------------------------------------------------------------------
    # Minimum words from User required
    # [!] Set to 0 for MONOLOGUE training!
    min_user_words: int = 0

    # -------------------------------------------------------------------------
    # Segment Duration Requirements
    # -------------------------------------------------------------------------
    # Minimum segment duration (shorter segments lack context)
    min_segment_duration_sec: float = 10.0

    # Maximum segment duration (None = no limit, uses duration_sec from config)
    max_segment_duration_sec: float | None = None


@dataclass
class PreferencesArgs(Serializable):
    """
    Layer 3: Probabilistic Preferences (soft filtering).

    These are OPTIONAL preferences that apply probabilistic filtering.
    Use for fine-tuning data distribution without hard requirements.

    Key Concept: Unlike Layer 1-2 (always enforce), Layer 3 uses
    probability to decide whether to skip non-preferred segments.
    """
    enabled: bool = False  # Disabled by default

    # -------------------------------------------------------------------------
    # First Speaker Preference
    # -------------------------------------------------------------------------
    # Prefer segments where Moshi speaks first
    prefer_moshi_start: bool = True

    # Probability of SKIPPING segments where User speaks first
    # 0.0 = accept all, 0.7 = skip 70% of user-first segments
    prefer_moshi_start_prob: float = 0.7

    # -------------------------------------------------------------------------
    # Both Speakers Preference (probabilistic, NOT hard requirement)
    # -------------------------------------------------------------------------
    # Note: For hard requirement, use case_control.allow_case2=false
    # This is a softer probabilistic preference
    prefer_both_speakers: bool = False

    # Probability of SKIPPING single-speaker segments
    # 0.0 = accept all single-speaker, 0.9 = skip 90% of single-speaker
    prefer_both_speakers_prob: float = 0.5


@dataclass
class RoleSwappingArgs(Serializable):
    """
    Layer 4: Role Swapping Data Augmentation (J-Moshi Style).

    Swaps Moshi and User roles to effectively double training data.
    The model learns from both speaker perspectives.

    Audio Swap: LEFT(Moshi) ↔ RIGHT(User)
    Text Swap:  SPEAKER_MAIN ↔ other speakers (in alignments)

    [!] IMPORTANT: When role_swapping is enabled:
       - Original sample: Moshi=LEFT, User=RIGHT
       - Swapped sample:  Moshi=RIGHT(was User), User=LEFT(was Moshi)
       - User text becomes Moshi text (and vice versa)
       - All filters are re-applied to swapped samples

    recheck_after_swap: When True, Case Control and Quality filters
    are re-applied after swapping. This prevents invalid swapped samples
    (e.g., original Case 1 → swapped Case 5 if new Moshi has no text).
    """
    enabled: bool = True

    # Probability of role swapping (0.0-1.0)
    probability: float = 1.0

    # Swap audio channels (LEFT ↔ RIGHT)
    swap_audio: bool = True

    # Swap text alignments (SPEAKER_MAIN ↔ others)
    swap_text: bool = True

    # Yield both original and swapped samples
    # True = 2x data (original + swapped)
    # False = probabilistic (either original or swapped, based on probability)
    yield_both: bool = True

    # Re-check Case Control + Quality after swap
    # CRITICAL: Prevents invalid swapped samples!
    recheck_after_swap: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be between 0.0 and 1.0, got {self.probability}")


@dataclass
class FilteringLoggerArgs(Serializable):
    """
    Layer 5: Logging and Debugging Configuration.

    Controls verbose output and statistics logging for debugging
    and monitoring the filtering pipeline.
    """
    # Master switch for all filtering logs
    enabled: bool = True

    # Verbosity level: 0=minimal, 1=summary, 2=detailed, 3=debug
    verbosity: int = 1

    # Log first filtered sample (useful for quick debugging)
    log_first_filter: bool = True

    # Log first passed sample
    log_first_pass: bool = True

    # Log detailed case detection for each sample
    log_case_detection: bool = False

    # Log quality check results for each sample
    log_quality_checks: bool = False

    # Log role swapping operations
    log_role_swapping: bool = False

    # Log statistics summary at epoch end
    log_epoch_summary: bool = True

    # Save detailed logs to file (in run_dir/logs/)
    save_to_file: bool = True

    # Log file name (relative to run_dir/logs/)
    log_filename: str = "segment_filtering.log"


@dataclass
class SegmentFilteringArgs(Serializable):
    """
    ═══════════════════════════════════════════════════════════════════════════
    UNIFIED SEGMENT FILTERING SYSTEM
    ═══════════════════════════════════════════════════════════════════════════

    This is the master configuration for all segment filtering and augmentation.
    It replaces the old separate data_filtering and data_augmentation configs.

    PROCESSING ORDER:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Phase 1: Load alignments & compute statistics                          │
    │     ↓                                                                   │
    │  Phase 2: CASE CONTROL (Layer 1) - Structural validity                  │
    │     ↓                                                                   │
    │  Phase 3: QUALITY (Layer 2) - Quality requirements                      │
    │     ↓                                                                   │
    │  Phase 4: PREFERENCES (Layer 3) - Probabilistic preferences             │
    │     ↓                                                                   │
    │  Phase 5: ROLE SWAPPING (Layer 4) - Data augmentation                   │
    │     ↓                                                                   │
    │  Phase 5.5: RE-CHECK after swap (if recheck_after_swap=true)            │
    │     ↓                                                                   │
    │  Phase 6-11: Encode, tokenize, yield samples                            │
    └─────────────────────────────────────────────────────────────────────────┘

    CONFIGURATION EXAMPLES:

    1. DIALOGUE Training (both speakers required):
       ```yaml
       segment_filtering:
         case_control:
           allow_case1: true
           allow_case2: false  # ← No monologue
         quality:
           min_moshi_words: 3
           min_user_words: 2   # ← Require user words
       ```

    2. MONOLOGUE Training (Moshi-only allowed):
       ```yaml
       segment_filtering:
         case_control:
           allow_case1: true
           allow_case2: true   # ← Allow monologue
         quality:
           min_moshi_words: 3
           min_user_words: 0   # ← No user requirement
         role_swapping:
           enabled: false      # ← No swapping for monologue
       ```

    3. Maximum Data Augmentation:
       ```yaml
       segment_filtering:
         case_control:
           allow_case1: true
           allow_case2: true
         quality:
           min_moshi_words: 3
           min_user_words: 0
         role_swapping:
           enabled: true
           yield_both: true    # ← 2x data
           recheck_after_swap: true
       ```
    """
    # Layer 1: Structural validity (Case 1-5)
    case_control: CaseControlArgs = field(default_factory=CaseControlArgs)

    # Layer 2: Quality requirements
    quality: QualityRequirementsArgs = field(default_factory=QualityRequirementsArgs)

    # Layer 3: Probabilistic preferences (optional)
    preferences: PreferencesArgs = field(default_factory=PreferencesArgs)

    # Layer 4: Role swapping augmentation
    role_swapping: RoleSwappingArgs = field(default_factory=RoleSwappingArgs)

    # Layer 5: Logging configuration
    logging: FilteringLoggerArgs = field(default_factory=FilteringLoggerArgs)

    def __post_init__(self) -> None:
        # Validation: If monologue not allowed (case2=false), role_swapping
        # should have recheck_after_swap=true to prevent swapped monologues
        if not self.case_control.allow_case2 and self.role_swapping.enabled:
            if not self.role_swapping.recheck_after_swap:
                logging.warning(
                    "case_control.allow_case2=false but role_swapping.recheck_after_swap=false. "
                    "This may allow swapped monologue samples. Consider enabling recheck_after_swap."
                )

        # Log configuration summary
        if self.logging.enabled and self.logging.verbosity >= 1:
            logging.info(
                f"[SEGMENT FILTERING] Initialized: "
                f"case_control={self.case_control.enabled}, "
                f"quality={self.quality.enabled}, "
                f"preferences={self.preferences.enabled}, "
                f"role_swapping={self.role_swapping.enabled}"
            )


@dataclass
class InterleaverArgs(Serializable):
    """
    Text-Audio interleaver configuration for precise alignment.

    Controls how text tokens are aligned with audio frames (12.5Hz).
    These settings are critical for Inner Monologue quality.

    Key concepts:
    - Audio frame rate: 12.5Hz = 80ms per frame
    - Text tokens must be placed within available audio frames
    - Overflow occurs when text tokens exceed available frames
    """
    # Use only SPEAKER_MAIN (Moshi) text, excluding user speech
    # When True, text stream only contains Moshi's Inner Monologue
    # When False, includes both speakers' text (not recommended for training)
    keep_main_only: bool = True

    # Token preservation mode when words overlap in time
    # When True: Queue-based approach, extends token queue (may delay tokens)
    # When False: Replace queue with new word tokens (original Moshi/J-Moshi behavior)
    # Recommendation: False (follows previous works, better for timing accuracy)
    keep_and_shift: bool = False

    # Adaptive token distribution when overflow is detected
    # When True: Distributes tokens evenly across available frames
    # When False: Drops overflow tokens at segment end
    # Recommendation: True (prevents token loss in fast speech)
    adaptive_distribute: bool = True

    # Log warnings when token overflow occurs
    # Useful for monitoring data quality and timing issues
    warn_on_overflow: bool = True

    # Character-level timestamp interpolation (J-Moshi style)
    # When True: Splits word timestamps to character-level for precise alignment
    # This provides more accurate token placement for languages like Korean/Japanese
    # where subword tokens span multiple characters within a word
    # Recommendation: True for Korean/Japanese, False for English
    character_level_interpolation: bool = True

    # Main speaker label in alignment data
    # Used to identify Moshi's speech in dialogue data
    main_speaker_label: str = "SPEAKER_MAIN"


@dataclass
class KoreanFinetuningArgs(Serializable):
    """
    Korean-specific finetuning configuration.

    These settings enable full finetuning with user stream support
    for Korean dialogue models.

    UNIFIED SEGMENT FILTERING:
        Use `segment_filtering` for all data filtering and augmentation.
        The old `data_filtering` and `data_augmentation` fields are deprecated.
    """
    # Enable user stream training (stereo → 17 codebooks)
    # When True, uses StereoInterleavedTokenizer and extended model (dep_q=16)
    enable_user_stream: bool = False

    # Enable full-duplex input (Original Moshi / J-Moshi default mode)
    # When True: Stereo data (17 codebooks) with dep_q=8 (user audio = context only)
    # When False: Mono data (9 codebooks) with dep_q=8
    # NOTE: Ignored when enable_user_stream=True
    full_duplex_input: bool = True

    # Path to pre-initialized model with user stream extension
    # Use tools/init_korean_moshi.py to create this model
    initialized_model_path: str | None = None

    # Path to Korean tokenizer directory or .model file
    # If None, uses the default Moshiko tokenizer
    korean_tokenizer_path: str | None = None

    # Tokenizer type: 'sentencepiece' (default) or 'klue'
    # - 'sentencepiece': Native SentencePiece .model file
    # - 'klue': KLUE BERT tokenizer (uses KoreanTokenizerWrapper)
    korean_tokenizer_type: str = "sentencepiece"

    # Token IDs to retain when reinitializing embeddings
    # Default: [0, 3, 32000] for PAD, EOS, and special tokens
    retain_token_ids: list = field(default_factory=lambda: [0, 3, 32000])

    # Interleaver configuration for text-audio alignment
    interleaver: InterleaverArgs = field(default_factory=InterleaverArgs)

    # =========================================================================
    # UNIFIED SEGMENT FILTERING (5-Layer System)
    # =========================================================================
    # Layer 1: Case Control - Structural validity (allow_case1..5)
    # Layer 2: Quality - Hard minimums (min_moshi_words, etc.)
    # Layer 3: Preferences - Probabilistic preferences (optional)
    # Layer 4: Role Swapping - Data augmentation (2x data)
    # Layer 5: Logging - Debug and statistics
    segment_filtering: SegmentFilteringArgs = field(default_factory=SegmentFilteringArgs)

    def __post_init__(self) -> None:
        # Log the training mode
        if self.enable_user_stream:
            logging.info(
                "[MODE] USER-STREAM: stereo input (17 codebooks), dep_q=16 output"
            )
            if self.initialized_model_path is None:
                logging.info(
                    "Dynamic model extension will be applied (no pre-init required)"
                )
        elif self.full_duplex_input:
            logging.info(
                "[MODE] FULL-DUPLEX: stereo input (17 codebooks), dep_q=8 output"
            )
            logging.info(
                "User audio used as context only (Original Moshi / J-Moshi default)"
            )
        else:
            logging.info(
                "[MODE] MONOLOGUE: mono input (9 codebooks), dep_q=8 output"
            )

        # Validate tokenizer type
        valid_types = ("sentencepiece", "klue", "hf_lm")
        if self.korean_tokenizer_type not in valid_types:
            raise ValueError(
                f"korean_tokenizer_type must be one of {valid_types}, "
                f"got '{self.korean_tokenizer_type}'"
            )

        if self.korean_tokenizer_type == "klue" and self.korean_tokenizer_path is None:
            logging.warning(
                "korean_tokenizer_type='klue' but korean_tokenizer_path is not set. "
                "Download KLUE BERT tokenizer with: python scripts/download_models.py --download-klue"
            )

        if self.korean_tokenizer_type == "hf_lm":
            logging.warning(
                "korean_tokenizer_type='hf_lm' requires text_card=105900. "
                "This is an experimental feature requiring model architecture changes. "
                "See tools/hf_lm_tokenizer_wrapper.py for details."
            )

        # Log interleaver configuration
        if self.interleaver.character_level_interpolation:
            logging.info(
                "Character-level interpolation enabled for text-audio alignment. "
                "This provides more precise timing for Korean text tokens."
            )


@dataclass
class TrainArgs(Serializable):
    """
    Main training configuration.

    This is the root configuration class that contains all training settings.
    """
    data: DataArgs

    # Path to the directory where everything will be saved. It needs to be empty.
    run_dir: str

    # Model paths configuration
    moshi_paths: ModelPaths = field(default_factory=ModelPaths)

    # Backbone configuration (modular LLM backend selection)
    # Supports: "moshi" (default), "hf_lm" (Phase 2)
    # See finetune/backbone/config.py for detailed options
    backbone: UnifiedBackboneConfig = field(default_factory=UnifiedBackboneConfig)

    # Loss function weights
    first_codebook_weight_multiplier: float = 100.0  # Semantic codebook weight (J-Moshi: 100)
    text_padding_weight: float = 0.5  # Text padding token weight

    # User stream loss weights (for Full Duplex mode with dep_q=16)
    # These weights control the relative importance of user audio vs moshi audio
    user_semantic_weight: float | None = None  # User semantic weight (default: same as first_codebook_weight)
    user_acoustic_weight: float | None = None  # User acoustic weight (default: 1.0)
    user_stream_loss_ratio: float = 1.0  # Ratio of user loss to moshi loss (0.5 = half weight)

    # Optimizer configuration (supports two-rate LR)
    optim: OptimArgs = field(default_factory=OptimArgs)

    # Scheduler configuration (supports multiple scheduler types)
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)

    # Random seed
    seed: int = 42

    # Number of steps to accumulate gradients before doing an optimizer step.
    num_microbatches: int = 4

    # Sequence duration in seconds
    duration_sec: float = 90  # Optimized for A100 80GB

    # Batch size per GPU
    batch_size: int = 6  # Optimized for A100 80GB with FSDP

    # Gradient clipping
    max_norm: float = 1.0

    # Total training steps
    max_steps: int = 50000

    # Logging frequency (steps)
    log_freq: int = 10

    # Checkpoint frequency (steps). If < 1, only the last checkpoint is saved.
    ckpt_freq: int = 1000
    save_adapters: bool = True
    do_ckpt: bool = True
    num_ckpt_keep: int | None = 5

    # Evaluation frequency and settings
    eval_freq: int = 500
    do_eval: bool = True
    eval_samples: int = 100  # Number of validation samples

    # Gradient checkpointing (saves memory at cost of compute)
    gradient_checkpointing: bool = True

    world_size: int | None = field(init=False, default=None)

    # Logging backends
    wandb: WandbArgs = field(default_factory=WandbArgs)
    tensorboard: TensorBoardArgs = field(default_factory=TensorBoardArgs)

    # Advanced monitoring (Phase 3)
    monitoring: MonitoringArgs = field(default_factory=MonitoringArgs)

    # Enhanced evaluation (BLEU, Alignment, Dialogue, Audio Quality)
    enhanced_evaluation: EnhancedEvaluationArgs = field(default_factory=EnhancedEvaluationArgs)

    # Sample saving (Phase 4)
    sample_saving: SampleSavingArgs = field(default_factory=SampleSavingArgs)

    # Research logging (Phase 5)
    research_logging: ResearchLoggingArgs = field(default_factory=ResearchLoggingArgs)

    # Checkpoint configuration (Phase 6 - Resume Support)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)

    # LoRA configuration
    lora: LoraArgs | None = field(default_factory=LoraArgs)
    full_finetuning: bool = False

    # Model precision
    param_dtype: str = "bfloat16"

    # Overwrite existing run directory
    overwrite_run_dir: bool = False

    # Distributed training backend: "fsdp" or "ddp"
    # - fsdp: FullyShardedDataParallel - shards model across GPUs, lower memory per GPU
    # - ddp: DistributedDataParallel - replicates full model, simpler but needs more memory
    # For LoRA finetuning on A100 80GB, both work. DDP is simpler and compatible with MPI.
    distributed_backend: str = "fsdp"

    # Korean finetuning configuration
    korean: KoreanFinetuningArgs = field(default_factory=KoreanFinetuningArgs)

    # Speaker conditioning configuration (Zero-Shot Speaker Adaptation)
    # Enables voice cloning from short reference audio
    speaker: SpeakerConditioningArgs = field(default_factory=SpeakerConditioningArgs)

    # Stage-based pretrained model loading
    # Loads weights from a previous training stage while starting from step 0
    # Useful for multi-stage training (e.g., Stage 1 base → Stage 2 with speaker conditioning)
    pretrained: PretrainedModelArgs = field(default_factory=PretrainedModelArgs)

    def __post_init__(self) -> None:
        assert getattr(self, "world_size", None) is None
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        if self.wandb.offline:
            command = f"cd {self.run_dir}; wandb sync --sync-all"
            logging.info(f"to sync wandb offline, run: {command}")

        assert self.num_microbatches >= 1

        assert self.num_ckpt_keep is None or self.num_ckpt_keep >= 1

        # Validate distributed backend
        assert self.distributed_backend in ("fsdp", "ddp"), (
            f"distributed_backend must be 'fsdp' or 'ddp', got '{self.distributed_backend}'"
        )

        if self.distributed_backend == "ddp" and self.full_finetuning:
            logging.warning(
                "Full finetuning with DDP requires ~42GB+ GPU memory for Moshi 7B. "
                "Consider using FSDP (distributed_backend='fsdp') for full finetuning."
            )

        # =================================================================
        # Backbone Configuration Validation
        # =================================================================
        # Log backbone type selection
        logging.info(f"[BACKBONE] Type: {self.backbone.type}")

        # Validate backbone type compatibility
        if self.backbone.type == "hf_lm":
            # HFLM requires dimension adapter when hidden_dim != 4096
            if not self.backbone.dimension_adapter.enable:
                if self.backbone.hf_lm.hidden_dim != 4096:
                    logging.warning(
                        f"[BACKBONE] HFLM hidden_dim ({self.backbone.hf_lm.hidden_dim}) "
                        "differs from Moshi's 4096. Consider enabling dimension_adapter."
                    )
            else:
                logging.info(
                    f"[BACKBONE] DimensionAdapter enabled: "
                    f"{self.backbone.dimension_adapter.moshi_dim} <-> "
                    f"{self.backbone.dimension_adapter.backbone_dim or self.backbone.hf_lm.hidden_dim}"
                )

            # HFLM is not yet fully implemented
            logging.warning(
                "[BACKBONE] HFLM backbone is experimental (Phase 2). "
                "For production, use backbone.type='moshi'."
            )

        if not self.save_adapters:
            logging.warning(
                "You have disabled `save_adapters` and are thus merging the "
                "trained LoRA checkpoint into the base model upon checkpointing. "
                "This might lead to OOM errors - make sure you have enough CPU "
                "and GPU memory."
            )

        # =================================================================
        # DEPRECATION WARNINGS for Legacy Checkpoint Options
        # =================================================================
        # When new CheckpointArgs.enabled=True, legacy options are ignored.
        # These warnings help users migrate to the new checkpoint system.
        if self.checkpoint.enabled:
            legacy_in_use = []
            if self.do_ckpt:
                legacy_in_use.append("do_ckpt")
            if self.ckpt_freq != 1000:  # non-default
                legacy_in_use.append("ckpt_freq")
            if self.num_ckpt_keep != 5:  # non-default
                legacy_in_use.append("num_ckpt_keep")

            if legacy_in_use:
                logging.info(
                    f"[DEPRECATION] Legacy checkpoint options ({', '.join(legacy_in_use)}) "
                    "are ignored when checkpoint.enabled=True. "
                    "Use checkpoint.save_freq and checkpoint.max_keep instead."
                )
        else:
            # Legacy mode: do_ckpt=True uses old Checkpointer
            if self.do_ckpt:
                logging.warning(
                    "[LEGACY MODE] Using legacy checkpoint system (no resume support). "
                    "Consider migrating to new checkpoint system: checkpoint.enabled=True"
                )
