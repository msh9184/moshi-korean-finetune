"""
K-Moshi Training Monitoring Module.

Provides comprehensive training monitoring including:
- Metrics logging (TensorBoard, W&B, JSONL)
- Advanced monitoring (WER, codebook analysis, gradient health)
- Enhanced evaluation (BLEU, alignment, dialogue, audio quality)
- Sample saving (audio and text predictions)
- Research logging utilities
"""

from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.advanced_monitor import (
    AdvancedTrainingMonitor,
    TextEvaluationMonitor,
    CodebookLossMonitor,
    GradientHealthMonitor,
    TextEvaluationResult,
    CodebookLossResult,
    GradientHealthResult,
    calculate_wer,
    calculate_cer,
    normalize_text,
)
from finetune.monitoring.sample_saver import (
    SampleSaver,
    SampleSaveResult,
)
from finetune.monitoring.research_logger import (
    ResearchLogger,
    LossRecord,
    GradientRecord,
)

# Enhanced Evaluation Monitors (Phase 1-3)
from finetune.monitoring.semantic_monitor import (
    SemanticQualityMonitor,
    SemanticQualityResult,
)
from finetune.monitoring.alignment_monitor import (
    AlignmentQualityMonitor,
    AlignmentQualityResult,
)
from finetune.monitoring.dialogue_monitor import (
    DialogueQualityMonitor,
    DialogueQualityResult,
)
from finetune.monitoring.audio_quality_monitor import (
    AudioQualityMonitor,
    AudioQualityResult,
)
from finetune.monitoring.enhanced_evaluation import (
    EnhancedEvaluationOrchestrator,
    EnhancedEvaluationResult,
)

__all__ = [
    # Metrics Logger
    "MetricsLogger",
    "eval_log_msg",
    "get_eval_logs",
    "get_train_logs",
    "train_log_msg",
    # Advanced Monitor
    "AdvancedTrainingMonitor",
    "TextEvaluationMonitor",
    "CodebookLossMonitor",
    "GradientHealthMonitor",
    "TextEvaluationResult",
    "CodebookLossResult",
    "GradientHealthResult",
    "calculate_wer",
    "calculate_cer",
    "normalize_text",
    # Sample Saver
    "SampleSaver",
    "SampleSaveResult",
    # Research Logger
    "ResearchLogger",
    "LossRecord",
    "GradientRecord",
    # Enhanced Evaluation - Semantic
    "SemanticQualityMonitor",
    "SemanticQualityResult",
    # Enhanced Evaluation - Alignment
    "AlignmentQualityMonitor",
    "AlignmentQualityResult",
    # Enhanced Evaluation - Dialogue
    "DialogueQualityMonitor",
    "DialogueQualityResult",
    # Enhanced Evaluation - Audio Quality
    "AudioQualityMonitor",
    "AudioQualityResult",
    # Enhanced Evaluation - Orchestrator
    "EnhancedEvaluationOrchestrator",
    "EnhancedEvaluationResult",
]
