"""
Enhanced Evaluation Orchestrator for K-Moshi Training.

Unified interface that coordinates all evaluation monitors:
- AdvancedTrainingMonitor (WER/CER, Codebook, Gradient)
- SemanticQualityMonitor (BLEU, Semantic Similarity)
- AlignmentQualityMonitor (Timing, Boundary)
- DialogueQualityMonitor (Turn-Taking, Latency)
- AudioQualityMonitor (PESQ, STOI, MCD)

This orchestrator manages the evaluation lifecycle and aggregates
all metrics for TensorBoard logging.
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


def _dataclass_to_dict(obj: Any) -> Any:
    """
    Recursively convert dataclass objects to dictionaries.

    This is needed because vars() only converts the top-level dataclass,
    leaving nested dataclasses as objects. This function handles nested
    dataclasses, lists, and dicts recursively.

    Args:
        obj: Any object (dataclass, dict, list, or primitive)

    Returns:
        Converted object with all dataclasses as dicts
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Convert dataclass to dict, recursively processing values
        return {k: _dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    elif isinstance(obj, dict):
        # Recursively process dict values
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively process list/tuple elements
        return type(obj)(_dataclass_to_dict(item) for item in obj)
    else:
        # Return primitive values as-is
        return obj

from finetune.monitoring.advanced_monitor import (
    AdvancedTrainingMonitor,
    TextEvaluationResult,
    CodebookLossResult,
    GradientHealthResult,
)
from finetune.monitoring.semantic_monitor import SemanticQualityMonitor, SemanticQualityResult
from finetune.monitoring.alignment_monitor import AlignmentQualityMonitor, AlignmentQualityResult
from finetune.monitoring.dialogue_monitor import DialogueQualityMonitor, DialogueQualityResult
from finetune.monitoring.audio_quality_monitor import AudioQualityMonitor, AudioQualityResult

logger = logging.getLogger("enhanced_evaluation")


@dataclass
class EnhancedEvaluationResult:
    """Combined result from all evaluation monitors."""
    # Text quality (WER/CER)
    text_result: Optional[TextEvaluationResult] = None

    # Semantic quality (BLEU)
    semantic_result: Optional[SemanticQualityResult] = None

    # Alignment quality
    alignment_result: Optional[AlignmentQualityResult] = None

    # Dialogue quality
    dialogue_result: Optional[DialogueQualityResult] = None

    # Audio quality
    audio_result: Optional[AudioQualityResult] = None

    # Codebook analysis
    codebook_result: Optional[CodebookLossResult] = None

    # Gradient health
    gradient_result: Optional[GradientHealthResult] = None


class EnhancedEvaluationOrchestrator:
    """
    Enhanced evaluation orchestrator for K-Moshi training.

    Coordinates all evaluation monitors and provides:
    1. Unified evaluation interface
    2. Metric aggregation for logging
    3. Configuration-based monitor activation
    4. TensorBoard-compatible metric formatting

    Usage:
        orchestrator = EnhancedEvaluationOrchestrator(
            args=train_args,
            tokenizer=spm,
            mimi_model=mimi,
            model_config={...},
        )

        # During evaluation
        result = orchestrator.evaluate_batch(batch, model_output)

        # Get metrics for logging
        metrics = orchestrator.get_all_metrics()

        # Reset for next epoch
        orchestrator.reset_all()
    """

    def __init__(
        self,
        args,
        tokenizer,
        mimi_model=None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize enhanced evaluation orchestrator.

        Args:
            args: TrainArgs with configuration
            tokenizer: SentencePiece tokenizer
            mimi_model: Mimi codec model (optional, for audio quality)
            model_config: Model configuration dict with:
                - text_padding_token_id
                - end_of_text_padding_id
                - zero_token_id
                - dep_q (number of codebooks)
        """
        self.args = args
        self.tokenizer = tokenizer
        self.mimi_model = mimi_model
        model_config = model_config or {}

        # Extract model config
        self.text_padding_token_id = model_config.get("text_padding_token_id", 0)
        self.end_of_text_padding_id = model_config.get("end_of_text_padding_id", 0)
        self.zero_token_id = model_config.get("zero_token_id", -1)
        self.dep_q = model_config.get("dep_q", 8)
        self.audio_offset = model_config.get("audio_offset", 1)

        # Get monitoring configuration (convert dataclass to dict recursively)
        monitoring_config = getattr(args, "monitoring", {})
        monitoring_config = _dataclass_to_dict(monitoring_config)
        if not isinstance(monitoring_config, dict):
            monitoring_config = {}

        # Get enhanced evaluation configuration (convert dataclass to dict recursively)
        enhanced_config = getattr(args, "enhanced_evaluation", {})
        enhanced_config = _dataclass_to_dict(enhanced_config)
        if not isinstance(enhanced_config, dict):
            enhanced_config = {}

        # Training mode flags
        korean_config = getattr(args, "korean", None)
        self.enable_user_stream = getattr(korean_config, "enable_user_stream", False) if korean_config else False
        self.full_duplex_input = getattr(korean_config, "full_duplex_input", True) if korean_config else False
        self.is_full_duplex = self.full_duplex_input and not self.enable_user_stream

        # Initialize monitors
        self._init_monitors(monitoring_config, enhanced_config)

        logger.info(f"EnhancedEvaluationOrchestrator initialized")
        logger.info(f"  Full-Duplex mode: {self.is_full_duplex}")
        logger.info(f"  Monitors: {self._get_enabled_monitors()}")

    def _init_monitors(
        self,
        monitoring_config: Dict[str, Any],
        enhanced_config: Dict[str, Any],
    ):
        """Initialize all evaluation monitors based on configuration."""

        # 1. Advanced Training Monitor (WER/CER, Codebook, Gradient)
        self.advanced_monitor = AdvancedTrainingMonitor(
            tokenizer=self.tokenizer,
            text_padding_token_id=self.text_padding_token_id,
            end_of_text_padding_id=self.end_of_text_padding_id,
            num_codebooks=self.dep_q,
            first_codebook_weight=getattr(self.args, "first_codebook_weight_multiplier", 100.0),
            config=monitoring_config,
        )

        # 2. Semantic Quality Monitor (BLEU)
        # Note: enhanced_config is already recursively converted to dict
        semantic_config = enhanced_config.get("semantic", {})

        self.semantic_monitor = None
        if semantic_config.get("enabled", True):
            self.semantic_monitor = SemanticQualityMonitor(
                tokenizer=self.tokenizer,
                enabled=True,
                compute_bleu=semantic_config.get("compute_bleu", True),
                compute_semantic=semantic_config.get("compute_semantic", False),
                max_samples=semantic_config.get("max_samples", 50),
            )

        # 3. Alignment Quality Monitor
        alignment_config = enhanced_config.get("alignment", {})

        self.alignment_monitor = None
        if alignment_config.get("enabled", True):
            self.alignment_monitor = AlignmentQualityMonitor(
                frame_rate=12.5,
                tolerance_frames=alignment_config.get("tolerance_frames", 2),
                enabled=True,
            )

        # 4. Dialogue Quality Monitor (Full-Duplex only)
        dialogue_config = enhanced_config.get("dialogue", {})

        self.dialogue_monitor = None
        if self.is_full_duplex and dialogue_config.get("enabled", True):
            self.dialogue_monitor = DialogueQualityMonitor(
                frame_rate=12.5,
                enabled=True,
                overlap_threshold_frames=dialogue_config.get("overlap_threshold_frames", 3),
                silence_threshold_frames=dialogue_config.get("silence_threshold_frames", 25),
            )

        # 5. Audio Quality Monitor
        audio_config = enhanced_config.get("audio_quality", {})

        self.audio_quality_monitor = None
        if audio_config.get("enabled", False):
            self.audio_quality_monitor = AudioQualityMonitor(
                mimi_model=self.mimi_model,
                sample_rate=24000,
                enabled=True,
                compute_pesq=audio_config.get("compute_pesq", True),
                compute_stoi=audio_config.get("compute_stoi", True),
                compute_mcd=audio_config.get("compute_mcd", True),
                max_samples=audio_config.get("max_samples", 10),
            )

    def _get_enabled_monitors(self) -> List[str]:
        """Get list of enabled monitor names."""
        monitors = ["AdvancedTraining"]
        if self.semantic_monitor:
            monitors.append("Semantic")
        if self.alignment_monitor:
            monitors.append("Alignment")
        if self.dialogue_monitor:
            monitors.append("Dialogue")
        if self.audio_quality_monitor:
            monitors.append("AudioQuality")
        return monitors

    def reset_all(self):
        """Reset all monitors for new evaluation epoch."""
        self.advanced_monitor.reset_all()

        if self.semantic_monitor:
            self.semantic_monitor.reset()
        if self.alignment_monitor:
            self.alignment_monitor.reset()
        if self.dialogue_monitor:
            self.dialogue_monitor.reset()
        if self.audio_quality_monitor:
            self.audio_quality_monitor.reset()

    def evaluate_batch(
        self,
        batch,
        model_output,
        model=None,
    ) -> EnhancedEvaluationResult:
        """
        Evaluate a batch using all enabled monitors.

        Args:
            batch: Batch object with:
                - codes: [B, K, T] tensor
                - user_text_alignments: Optional alignment data
                - moshi_text_raw_list: Optional raw text list
            model_output: Model output with:
                - text_logits: [B, T, V]
                - logits: [B, K, T, V]
                - text_mask: [B, 1, T]
                - mask: [B, K, T]
            model: Optional model for gradient checking

        Returns:
            EnhancedEvaluationResult with all monitor results
        """
        result = EnhancedEvaluationResult()
        codes = batch.codes

        # 1. Text Evaluation (WER/CER)
        text_codes = codes[:, :self.audio_offset]  # [B, 1, T]
        result.text_result = self.advanced_monitor.evaluate_text(
            model_output.text_logits,
            text_codes,
            model_output.text_mask,
        )

        # 2. Codebook Analysis
        audio_codes = codes[:, self.audio_offset:self.audio_offset + self.dep_q]
        result.codebook_result = self.advanced_monitor.analyze_codebooks(
            model_output.logits,
            audio_codes,
            model_output.mask,
        )

        # 3. Gradient Health (if model provided)
        if model is not None:
            result.gradient_result = self.advanced_monitor.check_gradients(model)

        # 4. Semantic Quality (BLEU)
        if self.semantic_monitor and result.text_result:
            # Get decoded texts from text evaluation
            references = []
            hypotheses = []

            for sample in result.text_result.samples:
                if sample.get("reference") and sample.get("hypothesis"):
                    references.append(sample["reference"])
                    hypotheses.append(sample["hypothesis"])

            if references and hypotheses:
                result.semantic_result = self.semantic_monitor.evaluate_batch(
                    references, hypotheses
                )

        # 5. Alignment Quality
        if self.alignment_monitor:
            alignments = getattr(batch, "user_text_alignments", None)
            if alignments:
                result.alignment_result = self.alignment_monitor.evaluate_alignment(
                    text_codes,
                    self.text_padding_token_id,
                    self.end_of_text_padding_id,
                    alignments,
                )

        # 6. Dialogue Quality (Full-Duplex mode only)
        if self.dialogue_monitor and self.is_full_duplex:
            # Extract Moshi and User audio codes from stereo data
            # In Full-Duplex: codes[:, 1:9] = Moshi audio, codes[:, 9:17] = User audio
            if codes.shape[1] >= 17:
                moshi_audio = codes[:, 1:9]  # [B, 8, T]
                user_audio = codes[:, 9:17]  # [B, 8, T]

                result.dialogue_result = self.dialogue_monitor.evaluate_dialogue(
                    moshi_audio,
                    user_audio,
                    self.zero_token_id,
                    text_codes,
                )

        # 7. Audio Quality (expensive, disabled by default)
        if self.audio_quality_monitor:
            # Get predicted audio codes from logits
            pred_audio_codes = model_output.logits.argmax(dim=-1)  # [B, K, T]
            gt_audio_codes = audio_codes

            result.audio_result = self.audio_quality_monitor.evaluate_batch(
                gt_audio_codes,
                pred_audio_codes,
            )

        return result

    def get_all_metrics(self) -> Dict[str, float]:
        """
        Get all metrics as a flat dictionary for TensorBoard logging.

        Returns:
            Dictionary with metric names as keys and values as floats.
            Metric names are prefixed by monitor category.
        """
        metrics = {}

        # Advanced monitor metrics (WER/CER, Codebook, Gradient)
        advanced_metrics = self.advanced_monitor.get_metrics_dict()
        metrics.update(advanced_metrics)

        # Semantic metrics
        if self.semantic_monitor:
            semantic_summary = self.semantic_monitor.get_summary()
            for key, value in semantic_summary.items():
                if isinstance(value, (int, float)):
                    metrics[f"semantic/{key}"] = value

        # Alignment metrics
        if self.alignment_monitor:
            alignment_summary = self.alignment_monitor.get_summary()
            for key, value in alignment_summary.items():
                if isinstance(value, (int, float)):
                    metrics[f"alignment/{key}"] = value

        # Dialogue metrics
        if self.dialogue_monitor:
            dialogue_summary = self.dialogue_monitor.get_summary()
            for key, value in dialogue_summary.items():
                if isinstance(value, (int, float)):
                    metrics[f"dialogue/{key}"] = value

        # Audio quality metrics
        if self.audio_quality_monitor:
            audio_summary = self.audio_quality_monitor.get_summary()
            for key, value in audio_summary.items():
                if isinstance(value, (int, float)):
                    metrics[f"audio_quality/{key}"] = value

        return metrics

    def format_summary_message(self, step: int) -> str:
        """
        Format a comprehensive summary log message.

        Args:
            step: Current training step

        Returns:
            Formatted multi-line summary string
        """
        lines = [f"[ENHANCED EVAL step={step}]"]

        # Advanced monitor summary
        lines.append(self.advanced_monitor.format_log_message(step))

        # Semantic
        if self.semantic_monitor:
            lines.append(self.semantic_monitor.format_log_message())

        # Alignment
        if self.alignment_monitor:
            lines.append(self.alignment_monitor.format_log_message())

        # Dialogue
        if self.dialogue_monitor:
            lines.append(self.dialogue_monitor.format_log_message())

        # Audio quality
        if self.audio_quality_monitor:
            lines.append(self.audio_quality_monitor.format_log_message())

        return " | ".join(lines)

    def get_tensorboard_layout(self) -> Dict[str, Any]:
        """
        Get TensorBoard custom layout configuration.

        Returns:
            Layout dictionary for add_custom_scalars()
        """
        layout = {
            "K-Moshi Enhanced Evaluation": {
                "Text Quality": ["Multiline", [
                    "eval.text_eval/wer",
                    "eval.text_eval/cer",
                    "eval.semantic/corpus_bleu",
                ]],
                "Audio Quality": ["Multiline", [
                    "eval.audio_quality/pesq_mean",
                    "eval.audio_quality/stoi_mean",
                    "eval.audio_quality/mcd_mean",
                ]],
                "Alignment": ["Multiline", [
                    "eval.alignment/timing_accuracy",
                    "eval.alignment/boundary_f1",
                    "eval.alignment/sync_score",
                ]],
                "Dialogue": ["Multiline", [
                    "eval.dialogue/turn_taking_score",
                    "eval.dialogue/overlap_ratio",
                    "eval.dialogue/avg_response_latency_ms",
                ]],
            },
            "Codebook Analysis": {
                "Per-Codebook Loss": ["Multiline", [
                    f"eval.codebook/codebook_{i}_avg_loss" for i in range(self.dep_q)
                ]],
                "Entropy": ["Multiline", [
                    f"eval.codebook/codebook_{i}_avg_entropy" for i in range(self.dep_q)
                ]],
            },
            "Training Health": {
                "Gradients": ["Multiline", [
                    "eval.gradient/avg_grad_norm",
                    "eval.gradient/max_grad_norm",
                ]],
            },
        }

        return layout

    def get_prediction_samples(self) -> List[Dict[str, str]]:
        """Get prediction samples from text evaluation for logging."""
        return self.advanced_monitor.get_prediction_samples()

    def set_mimi_model(self, mimi_model):
        """Update Mimi model for audio quality evaluation."""
        self.mimi_model = mimi_model
        if self.audio_quality_monitor:
            self.audio_quality_monitor.set_mimi_model(mimi_model)
