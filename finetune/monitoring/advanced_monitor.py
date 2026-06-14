"""
Advanced Monitoring for K-Moshi Training.

Provides comprehensive training analysis including:
- Text stream (Inner Monologue) WER evaluation
- Per-codebook loss analysis
- Gradient health monitoring
- Sample prediction logging

This module is designed for research-quality training with
detailed metrics for paper writing and debugging.
"""

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger("advanced_monitor")


# =============================================================================
# WER Calculation Utilities
# =============================================================================

def normalize_text(text: str, remove_punctuation: bool = True) -> str:
    """
    Normalize text for WER calculation.

    - Convert to lowercase
    - Remove punctuation (optional)
    - Normalize unicode
    - Collapse multiple spaces
    """
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Lowercase
    text = text.lower()

    # Remove punctuation if requested
    if remove_punctuation:
        # Keep Korean characters, alphanumeric, and spaces
        text = re.sub(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _postprocess_korean_text(text: str, mode: str = "display") -> str:
    """
    Post-process Korean text for display or metric calculation.

    SentencePiece with byte-level fallback may produce extra spaces between
    Korean syllables. This function offers different processing modes.

    Args:
        text: Input text with potential extra spaces
        mode: Processing mode
            - "display": Collapse multiple spaces to single space (default)
            - "compact": Remove ALL spaces between Korean syllables (for compact display)
            - "raw": No processing, return as-is

    Returns:
        Processed text based on mode
    """
    if not text or mode == "raw":
        return text

    if mode == "compact":
        # Remove ALL spaces between Korean characters (original behavior)
        # Pattern: Korean char + whitespace(s) + Korean char
        pattern = re.compile(r"([가-힣])\s+([가-힣])")
        prev_text = None
        while prev_text != text:
            prev_text = text
            text = pattern.sub(r"\1\2", text)
        return text

    # Default "display" mode: Collapse multiple spaces to single space
    # This preserves word boundaries while cleaning up tokenization artifacts
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_wer(reference: str, hypothesis: str, normalize: bool = True) -> Tuple[float, dict]:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.

    WER = (S + D + I) / N
    Where:
        S = substitutions
        D = deletions
        I = insertions
        N = number of words in reference

    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize: Whether to normalize text before comparison

    Returns:
        Tuple of (wer, details) where details contains S, D, I counts
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        # No reference words - can't calculate WER
        return 0.0 if m == 0 else 1.0, {"substitutions": 0, "deletions": 0, "insertions": m}

    # Dynamic programming for edit distance
    d = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize first column (deletions)
    for i in range(n + 1):
        d[i][0] = i

    # Initialize first row (insertions)
    for j in range(m + 1):
        d[0][j] = j

    # Fill the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # Deletion
                    d[i][j - 1] + 1,      # Insertion
                    d[i - 1][j - 1] + 1,  # Substitution
                )

    # Backtrack to count S, D, I
    substitutions = 0
    deletions = 0
    insertions = 0

    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    wer = (substitutions + deletions + insertions) / n

    return wer, {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "ref_words": n,
        "hyp_words": m,
    }


def calculate_cer(reference: str, hypothesis: str, normalize: bool = True) -> Tuple[float, dict]:
    """
    Calculate Character Error Rate (CER) - useful for Korean.

    Same algorithm as WER but at character level.
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    # Use character-level comparison (with spaces removed)
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    n = len(ref_chars)
    m = len(hyp_chars)

    if n == 0:
        return 0.0 if m == 0 else 1.0, {"substitutions": 0, "deletions": 0, "insertions": m}

    # Dynamic programming
    d = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1,
                )

    cer = d[n][m] / n

    return cer, {"ref_chars": n, "hyp_chars": m, "edit_distance": d[n][m]}


# =============================================================================
# Text Evaluation Monitor
# =============================================================================

@dataclass
class TextEvaluationResult:
    """Result of text stream evaluation."""
    wer: float
    cer: float
    samples: List[Dict[str, str]] = field(default_factory=list)
    wer_details: Dict[str, Any] = field(default_factory=dict)


class TextEvaluationMonitor:
    """
    Monitor for evaluating text stream (Inner Monologue) predictions.

    Computes WER/CER and logs sample predictions for qualitative analysis.
    """

    def __init__(
        self,
        tokenizer,
        text_padding_token_id: int,
        end_of_text_padding_id: int,
        max_samples: int = 5,
        normalize_text: bool = True,
    ):
        """
        Initialize text evaluation monitor.

        Args:
            tokenizer: SentencePiece tokenizer for decoding
            text_padding_token_id: Padding token ID
            end_of_text_padding_id: End-of-text padding token ID
            max_samples: Max prediction samples to log
            normalize_text: Whether to normalize text for WER
        """
        self.tokenizer = tokenizer
        self.text_padding_token_id = text_padding_token_id
        self.end_of_text_padding_id = end_of_text_padding_id
        self.max_samples = max_samples
        self.normalize_text = normalize_text

        # Accumulated statistics
        self.total_wer = 0.0
        self.total_cer = 0.0
        self.num_samples = 0
        self.samples_buffer = []

    def reset(self):
        """Reset accumulated statistics."""
        self.total_wer = 0.0
        self.total_cer = 0.0
        self.num_samples = 0
        self.samples_buffer = []

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text with UTF-8 safe handling and Korean post-processing.

        SentencePiece uses byte-level fallback for OOV characters (including Korean),
        which can produce invalid UTF-8 sequences and extra spaces between Korean syllables.
        This method handles both cases gracefully.

        Args:
            token_ids: Tensor of token IDs [T] or [1, T]

        Returns:
            Decoded text string (UTF-8 safe, Korean spaces cleaned)
        """
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)

        # Filter padding tokens
        valid_tokens = []
        for tid in token_ids.tolist():
            if tid not in (self.text_padding_token_id, self.end_of_text_padding_id):
                if tid >= 0:  # Skip negative tokens
                    valid_tokens.append(tid)

        if not valid_tokens:
            return ""

        try:
            decoded = self.tokenizer.decode(valid_tokens)
            # Ensure UTF-8 validity by encoding and decoding with error handling
            if isinstance(decoded, bytes):
                decoded = decoded.decode("utf-8", errors="replace")
            else:
                # Re-encode and decode to clean up any invalid sequences
                decoded = decoded.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            # Remove replacement characters for cleaner output
            decoded = decoded.replace("\ufffd", "")
            decoded = decoded.strip()

            # Apply Korean text post-processing to remove spaces between Hangul syllables
            decoded = _postprocess_korean_text(decoded)

            return decoded
        except Exception as e:
            logger.debug(f"Decode error: {e}")
            return ""

    def evaluate_batch(
        self,
        text_logits: torch.Tensor,
        text_targets: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> TextEvaluationResult:
        """
        Evaluate text predictions for a batch.

        Args:
            text_logits: Model output logits [B, T, V]
            text_targets: Ground truth token IDs [B, 1, T]
            text_mask: Valid position mask [B, 1, T]

        Returns:
            TextEvaluationResult with WER, CER, and samples
        """
        batch_size = text_logits.size(0)

        # Get predictions
        predictions = text_logits.argmax(dim=-1)  # [B, T]

        batch_wer = 0.0
        batch_cer = 0.0
        batch_samples = []

        for i in range(batch_size):
            # Get valid positions for this sample
            if text_targets.dim() == 3:
                target = text_targets[i, 0]  # [T]
                mask = text_mask[i, 0] if text_mask.dim() == 3 else text_mask[i]
            else:
                target = text_targets[i]
                mask = text_mask[i]

            pred = predictions[i]

            # Decode to text
            ref_text = self.decode_tokens(target)
            hyp_text = self.decode_tokens(pred)

            if ref_text:  # Only compute if reference exists
                wer, wer_details = calculate_wer(ref_text, hyp_text, self.normalize_text)
                cer, _ = calculate_cer(ref_text, hyp_text, self.normalize_text)

                batch_wer += wer
                batch_cer += cer
                self.num_samples += 1

                # Store sample for logging
                if len(batch_samples) < self.max_samples:
                    batch_samples.append({
                        "reference": ref_text,
                        "hypothesis": hyp_text,
                        "wer": wer,
                        "cer": cer,
                    })

        # Compute batch averages
        if batch_size > 0:
            batch_wer /= batch_size
            batch_cer /= batch_size

        self.total_wer += batch_wer
        self.total_cer += batch_cer
        self.samples_buffer.extend(batch_samples[:self.max_samples - len(self.samples_buffer)])

        return TextEvaluationResult(
            wer=batch_wer,
            cer=batch_cer,
            samples=batch_samples,
        )

    def get_summary(self) -> Dict[str, float]:
        """Get accumulated evaluation summary."""
        if self.num_samples == 0:
            return {"wer": 0.0, "cer": 0.0, "num_samples": 0}

        return {
            "wer": self.total_wer / self.num_samples,
            "cer": self.total_cer / self.num_samples,
            "num_samples": self.num_samples,
        }


# =============================================================================
# Per-Codebook Loss Monitor
# =============================================================================

@dataclass
class CodebookLossResult:
    """Result of per-codebook loss analysis."""
    losses: List[float]  # Loss for each codebook
    total_loss: float
    semantic_loss: float  # First codebook (weighted)
    acoustic_loss: float  # Remaining codebooks
    entropy: Optional[List[float]] = None  # Per-codebook entropy


class CodebookLossMonitor:
    """
    Monitor for per-codebook loss analysis.

    Tracks individual losses for 8 audio codebooks to understand
    semantic vs acoustic learning dynamics.
    """

    def __init__(
        self,
        num_codebooks: int = 8,
        first_codebook_weight: float = 100.0,
        log_entropy: bool = True,
    ):
        """
        Initialize codebook loss monitor.

        Args:
            num_codebooks: Number of audio codebooks (default 8)
            first_codebook_weight: Weight multiplier for semantic codebook
            log_entropy: Whether to compute entropy for each codebook
        """
        self.num_codebooks = num_codebooks
        self.first_codebook_weight = first_codebook_weight
        self.log_entropy = log_entropy

        # Accumulated statistics
        self.loss_history = defaultdict(list)
        self.entropy_history = defaultdict(list)

    def reset(self):
        """Reset accumulated statistics."""
        self.loss_history = defaultdict(list)
        self.entropy_history = defaultdict(list)

    def compute_per_codebook_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> CodebookLossResult:
        """
        Compute loss for each codebook separately.

        Args:
            logits: Model output logits [B, K, T, V]
            targets: Ground truth token IDs [B, K, T]
            mask: Valid position mask [B, K, T]

        Returns:
            CodebookLossResult with per-codebook losses
        """
        B, K, T, V = logits.shape

        # Ensure we only process the expected number of codebooks
        K = min(K, self.num_codebooks)

        losses = []
        entropies = [] if self.log_entropy else None

        for k in range(K):
            cb_logits = logits[:, k]  # [B, T, V]
            cb_targets = targets[:, k]  # [B, T]
            cb_mask = mask[:, k]  # [B, T]

            # Flatten for loss computation
            flat_logits = cb_logits.reshape(-1, V)
            flat_targets = cb_targets.reshape(-1)
            flat_mask = cb_mask.reshape(-1)

            if flat_mask.sum() > 0:
                # Compute cross entropy for valid positions
                valid_logits = flat_logits[flat_mask]
                valid_targets = flat_targets[flat_mask]

                loss = F.cross_entropy(
                    valid_logits, valid_targets, reduction="mean"
                ).item()
                losses.append(loss)

                # Compute entropy if requested
                if self.log_entropy:
                    probs = F.softmax(valid_logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                    entropies.append(entropy)

                # Store in history
                self.loss_history[k].append(loss)
                if self.log_entropy:
                    self.entropy_history[k].append(entropy)
            else:
                losses.append(0.0)
                if self.log_entropy:
                    entropies.append(0.0)

        # Compute summary metrics
        total_loss = sum(losses)
        semantic_loss = losses[0] * self.first_codebook_weight if losses else 0.0
        acoustic_loss = sum(losses[1:]) if len(losses) > 1 else 0.0

        return CodebookLossResult(
            losses=losses,
            total_loss=total_loss,
            semantic_loss=semantic_loss,
            acoustic_loss=acoustic_loss,
            entropy=entropies,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all codebooks."""
        summary = {}

        for k in range(self.num_codebooks):
            if self.loss_history[k]:
                summary[f"codebook_{k}_avg_loss"] = sum(self.loss_history[k]) / len(self.loss_history[k])
                summary[f"codebook_{k}_min_loss"] = min(self.loss_history[k])
                summary[f"codebook_{k}_max_loss"] = max(self.loss_history[k])

            if self.entropy_history[k]:
                summary[f"codebook_{k}_avg_entropy"] = sum(self.entropy_history[k]) / len(self.entropy_history[k])

        return summary


# =============================================================================
# Gradient Health Monitor
# =============================================================================

@dataclass
class GradientHealthResult:
    """Result of gradient health check."""
    has_nan: bool
    has_inf: bool
    is_exploding: bool
    is_vanishing: bool
    grad_norm: float
    per_layer_norms: Optional[Dict[str, float]] = None
    alerts: List[str] = field(default_factory=list)


class GradientHealthMonitor:
    """
    Monitor for gradient health during training.

    Detects NaN/Inf gradients, exploding/vanishing gradients,
    and provides detailed per-layer analysis.
    """

    def __init__(
        self,
        exploding_threshold: float = 100.0,
        vanishing_threshold: float = 1e-7,
        alert_on_nan: bool = True,
        alert_on_inf: bool = True,
        log_per_layer: bool = False,
    ):
        """
        Initialize gradient health monitor.

        Args:
            exploding_threshold: Threshold for exploding gradient alert
            vanishing_threshold: Threshold for vanishing gradient alert
            alert_on_nan: Whether to log alerts on NaN gradients
            alert_on_inf: Whether to log alerts on Inf gradients
            log_per_layer: Whether to compute per-layer gradient norms
        """
        self.exploding_threshold = exploding_threshold
        self.vanishing_threshold = vanishing_threshold
        self.alert_on_nan = alert_on_nan
        self.alert_on_inf = alert_on_inf
        self.log_per_layer = log_per_layer

        # History
        self.grad_norm_history = []
        self.nan_count = 0
        self.inf_count = 0

    def reset(self):
        """Reset accumulated statistics."""
        self.grad_norm_history = []
        self.nan_count = 0
        self.inf_count = 0

    def check_gradients(self, model: torch.nn.Module) -> GradientHealthResult:
        """
        Check gradient health for all model parameters.

        Args:
            model: PyTorch model to check

        Returns:
            GradientHealthResult with health status and alerts
        """
        alerts = []
        has_nan = False
        has_inf = False
        total_norm = 0.0
        param_count = 0
        per_layer_norms = {} if self.log_per_layer else None

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data

            # Check for NaN
            if torch.isnan(grad).any():
                has_nan = True
                self.nan_count += 1
                if self.alert_on_nan:
                    alerts.append(f"NaN gradient in {name}")

            # Check for Inf
            if torch.isinf(grad).any():
                has_inf = True
                self.inf_count += 1
                if self.alert_on_inf:
                    alerts.append(f"Inf gradient in {name}")

            # Compute norm
            grad_norm = grad.norm().item()

            if not (has_nan or has_inf):
                total_norm += grad_norm ** 2
                param_count += 1

            if self.log_per_layer:
                per_layer_norms[name] = grad_norm

        # Compute total gradient norm
        grad_norm = total_norm ** 0.5 if param_count > 0 else 0.0
        self.grad_norm_history.append(grad_norm)

        # Check for exploding/vanishing
        is_exploding = grad_norm > self.exploding_threshold
        is_vanishing = grad_norm < self.vanishing_threshold and grad_norm > 0

        if is_exploding:
            alerts.append(f"Exploding gradient: norm={grad_norm:.4f} > {self.exploding_threshold}")

        if is_vanishing:
            alerts.append(f"Vanishing gradient: norm={grad_norm:.2e} < {self.vanishing_threshold}")

        # Log alerts
        for alert in alerts:
            logger.warning(f"[GRADIENT ALERT] {alert}")

        return GradientHealthResult(
            has_nan=has_nan,
            has_inf=has_inf,
            is_exploding=is_exploding,
            is_vanishing=is_vanishing,
            grad_norm=grad_norm,
            per_layer_norms=per_layer_norms,
            alerts=alerts,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get gradient health summary."""
        if not self.grad_norm_history:
            return {}

        return {
            "avg_grad_norm": sum(self.grad_norm_history) / len(self.grad_norm_history),
            "max_grad_norm": max(self.grad_norm_history),
            "min_grad_norm": min(self.grad_norm_history),
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


# =============================================================================
# Unified Advanced Monitor
# =============================================================================

class AdvancedTrainingMonitor:
    """
    Unified advanced training monitor.

    Combines text evaluation, codebook analysis, and gradient monitoring
    into a single interface.
    """

    def __init__(
        self,
        tokenizer,
        text_padding_token_id: int,
        end_of_text_padding_id: int,
        num_codebooks: int = 8,
        first_codebook_weight: float = 100.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize advanced training monitor.

        Args:
            tokenizer: SentencePiece tokenizer
            text_padding_token_id: Padding token ID
            end_of_text_padding_id: End-of-text padding ID
            num_codebooks: Number of audio codebooks
            first_codebook_weight: Weight for semantic codebook
            config: Optional configuration dict from args
        """
        config = config or {}

        # Text evaluation config
        text_config = config.get("text_evaluation", {})
        self.text_monitor = TextEvaluationMonitor(
            tokenizer=tokenizer,
            text_padding_token_id=text_padding_token_id,
            end_of_text_padding_id=end_of_text_padding_id,
            max_samples=text_config.get("max_prediction_samples", 5),
            normalize_text=text_config.get("normalize_text", True),
        )

        # Codebook analysis config
        codebook_config = config.get("codebook_analysis", {})
        self.codebook_monitor = CodebookLossMonitor(
            num_codebooks=num_codebooks,
            first_codebook_weight=first_codebook_weight,
            log_entropy=codebook_config.get("log_entropy", True),
        )

        # Gradient monitoring config
        gradient_config = config.get("gradient_monitoring", {})
        self.gradient_monitor = GradientHealthMonitor(
            exploding_threshold=gradient_config.get("exploding_threshold", 100.0),
            vanishing_threshold=gradient_config.get("vanishing_threshold", 1e-7),
            alert_on_nan=gradient_config.get("alert_on_nan", True),
            alert_on_inf=gradient_config.get("alert_on_inf", True),
            log_per_layer=gradient_config.get("log_per_layer", False),
        )

        # Feature flags
        self.enable_text_eval = text_config.get("enabled", True)
        self.enable_codebook_analysis = codebook_config.get("enabled", True)
        self.enable_gradient_monitoring = gradient_config.get("enabled", True)

    def reset_all(self):
        """Reset all monitors."""
        self.text_monitor.reset()
        self.codebook_monitor.reset()
        self.gradient_monitor.reset()

    def evaluate_text(
        self,
        text_logits: torch.Tensor,
        text_targets: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Optional[TextEvaluationResult]:
        """Evaluate text predictions if enabled."""
        if not self.enable_text_eval:
            return None
        return self.text_monitor.evaluate_batch(text_logits, text_targets, text_mask)

    def analyze_codebooks(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[CodebookLossResult]:
        """Analyze per-codebook losses if enabled."""
        if not self.enable_codebook_analysis:
            return None
        return self.codebook_monitor.compute_per_codebook_loss(logits, targets, mask)

    def check_gradients(self, model: torch.nn.Module) -> Optional[GradientHealthResult]:
        """Check gradient health if enabled."""
        if not self.enable_gradient_monitoring:
            return None
        return self.gradient_monitor.check_gradients(model)

    def get_metrics_dict(self) -> Dict[str, float]:
        """
        Get all monitoring metrics as a flat dictionary for logging.

        Returns:
            Dictionary with all metrics prefixed by monitor type
        """
        metrics = {}

        # Text evaluation metrics
        if self.enable_text_eval:
            text_summary = self.text_monitor.get_summary()
            for k, v in text_summary.items():
                metrics[f"text_eval/{k}"] = v

        # Codebook analysis metrics
        if self.enable_codebook_analysis:
            codebook_summary = self.codebook_monitor.get_summary()
            for k, v in codebook_summary.items():
                metrics[f"codebook/{k}"] = v

        # Gradient health metrics
        if self.enable_gradient_monitoring:
            gradient_summary = self.gradient_monitor.get_summary()
            for k, v in gradient_summary.items():
                metrics[f"gradient/{k}"] = v

        return metrics

    def format_log_message(self, step: int) -> str:
        """
        Format a log message with key monitoring metrics.

        Args:
            step: Current training step

        Returns:
            Formatted log message string
        """
        parts = [f"[MONITOR step={step}]"]

        if self.enable_text_eval:
            summary = self.text_monitor.get_summary()
            if summary.get("num_samples", 0) > 0:
                parts.append(f"WER={summary['wer']:.2%} CER={summary['cer']:.2%}")

        if self.enable_gradient_monitoring:
            summary = self.gradient_monitor.get_summary()
            if summary:
                parts.append(f"grad_norm={summary.get('avg_grad_norm', 0):.4f}")
                if summary.get("nan_count", 0) > 0:
                    parts.append(f"NaN={summary['nan_count']}")

        return " | ".join(parts)

    def get_prediction_samples(self) -> List[Dict[str, str]]:
        """Get accumulated prediction samples for logging."""
        return self.text_monitor.samples_buffer
