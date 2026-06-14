"""
Semantic Quality Monitor for K-Moshi Training.

Provides semantic quality evaluation including:
- BLEU score (n-gram based text similarity)
- Semantic similarity (embedding based, optional)
- Text consistency metrics

This module evaluates the semantic quality of Inner Monologue predictions,
providing a different perspective from WER/CER:
- WER/CER: Token-level accuracy (order-sensitive)
- BLEU: N-gram overlap ratio (order-flexible)
- Semantic: Meaning preservation (expression-flexible)
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("semantic_monitor")

# Try to import sacrebleu (optional dependency)
_SACREBLEU_AVAILABLE = False
try:
    import sacrebleu
    _SACREBLEU_AVAILABLE = True
except ImportError:
    logger.info("sacrebleu not installed. BLEU metrics will be disabled. "
                "Install with: pip install sacrebleu")

# Try to import sentence-transformers (optional dependency)
_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.debug("sentence-transformers not installed. Semantic similarity will be disabled.")


def normalize_for_bleu(text: str) -> str:
    """
    Normalize text for BLEU calculation.

    - Unicode normalization
    - Lowercase conversion
    - Remove extra whitespace
    - Keep Korean characters and basic punctuation
    """
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Lowercase
    text = text.lower()

    # Keep Korean, alphanumeric, and basic punctuation
    # Remove other special characters
    text = re.sub(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


@dataclass
class SemanticQualityResult:
    """Result of semantic quality evaluation."""
    # BLEU scores
    bleu_score: float = 0.0           # Overall BLEU (0-100)
    bleu_1: float = 0.0               # 1-gram precision
    bleu_2: float = 0.0               # 2-gram precision
    bleu_3: float = 0.0               # 3-gram precision
    bleu_4: float = 0.0               # 4-gram precision
    brevity_penalty: float = 1.0      # BLEU brevity penalty

    # Semantic similarity (optional)
    semantic_similarity: Optional[float] = None  # Cosine similarity (0-1)

    # Statistics
    sample_count: int = 0
    avg_ref_length: float = 0.0
    avg_hyp_length: float = 0.0
    length_ratio: float = 0.0

    # Samples for logging
    samples: List[Dict[str, Any]] = field(default_factory=list)


class SemanticQualityMonitor:
    """
    Semantic quality evaluation monitor for Inner Monologue predictions.

    Computes:
    - BLEU: N-gram based text similarity score
    - Semantic Similarity: Embedding-based meaning similarity (optional)
    - Text statistics: Length ratios, word counts

    Key difference from WER/CER:
    - WER/CER measures exact token-level accuracy
    - BLEU measures n-gram overlap (more tolerant of paraphrasing)
    - Semantic similarity measures meaning preservation

    Usage:
        monitor = SemanticQualityMonitor(tokenizer, enabled=True)
        result = monitor.evaluate_batch(references, hypotheses)
        metrics = monitor.get_summary()
    """

    def __init__(
        self,
        tokenizer=None,
        enabled: bool = True,
        compute_bleu: bool = True,
        compute_semantic: bool = False,
        semantic_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_samples: int = 5,
        normalize_text: bool = True,
    ):
        """
        Initialize semantic quality monitor.

        Args:
            tokenizer: SentencePiece tokenizer (optional, for decoding)
            enabled: Whether to enable this monitor
            compute_bleu: Whether to compute BLEU scores
            compute_semantic: Whether to compute semantic similarity
                (requires sentence-transformers, computationally expensive)
            semantic_model_name: HuggingFace model for semantic similarity
            max_samples: Maximum samples to store for logging
            normalize_text: Whether to normalize text before evaluation
        """
        self.tokenizer = tokenizer
        self.enabled = enabled
        self.compute_bleu = compute_bleu and _SACREBLEU_AVAILABLE
        self.compute_semantic = compute_semantic and _SENTENCE_TRANSFORMERS_AVAILABLE
        self.semantic_model_name = semantic_model_name
        self.max_samples = max_samples
        self.normalize_text = normalize_text

        # Lazy-loaded semantic model
        self._semantic_model = None

        # Accumulated statistics
        self.total_bleu = 0.0
        self.total_semantic = 0.0
        self.num_batches = 0
        self.all_references: List[str] = []
        self.all_hypotheses: List[str] = []
        self.samples_buffer: List[Dict[str, Any]] = []

        # Warnings
        if compute_bleu and not _SACREBLEU_AVAILABLE:
            logger.warning("BLEU computation requested but sacrebleu not available")
        if compute_semantic and not _SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Semantic similarity requested but sentence-transformers not available")

    def reset(self):
        """Reset accumulated statistics."""
        self.total_bleu = 0.0
        self.total_semantic = 0.0
        self.num_batches = 0
        self.all_references = []
        self.all_hypotheses = []
        self.samples_buffer = []

    def _get_semantic_model(self):
        """Lazy load semantic similarity model."""
        if self._semantic_model is None and self.compute_semantic:
            try:
                logger.info(f"Loading semantic model: {self.semantic_model_name}")
                self._semantic_model = SentenceTransformer(self.semantic_model_name)
                logger.info("Semantic model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
                self.compute_semantic = False
        return self._semantic_model

    def evaluate_batch(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> SemanticQualityResult:
        """
        Evaluate semantic quality for a batch of text pairs.

        Args:
            references: List of ground truth texts
            hypotheses: List of predicted texts

        Returns:
            SemanticQualityResult with BLEU and semantic scores
        """
        if not self.enabled:
            return SemanticQualityResult()

        if len(references) != len(hypotheses):
            logger.warning(f"Reference/hypothesis count mismatch: {len(references)} vs {len(hypotheses)}")
            min_len = min(len(references), len(hypotheses))
            references = references[:min_len]
            hypotheses = hypotheses[:min_len]

        if len(references) == 0:
            return SemanticQualityResult()

        # Filter empty pairs
        valid_pairs = [
            (ref, hyp) for ref, hyp in zip(references, hypotheses)
            if ref.strip() and hyp.strip()
        ]

        if not valid_pairs:
            return SemanticQualityResult(sample_count=len(references))

        refs, hyps = zip(*valid_pairs)
        refs = list(refs)
        hyps = list(hyps)

        # Normalize if requested
        if self.normalize_text:
            refs_normalized = [normalize_for_bleu(r) for r in refs]
            hyps_normalized = [normalize_for_bleu(h) for h in hyps]
        else:
            refs_normalized = refs
            hyps_normalized = hyps

        # Compute BLEU
        bleu_result = self._compute_bleu(refs_normalized, hyps_normalized)

        # Compute semantic similarity (optional)
        semantic_sim = None
        if self.compute_semantic:
            semantic_sim = self._compute_semantic_similarity(refs, hyps)

        # Compute statistics
        avg_ref_len = np.mean([len(r.split()) for r in refs])
        avg_hyp_len = np.mean([len(h.split()) for h in hyps])
        length_ratio = avg_hyp_len / max(avg_ref_len, 1e-6)

        # Store for corpus-level calculation
        self.all_references.extend(refs_normalized)
        self.all_hypotheses.extend(hyps_normalized)
        self.num_batches += 1

        # Store samples for logging
        batch_samples = []
        for i, (ref, hyp) in enumerate(zip(refs[:self.max_samples], hyps[:self.max_samples])):
            if len(self.samples_buffer) >= self.max_samples:
                break
            sample = {
                "reference": ref,
                "hypothesis": hyp,
                "ref_normalized": refs_normalized[i] if i < len(refs_normalized) else "",
                "hyp_normalized": hyps_normalized[i] if i < len(hyps_normalized) else "",
            }
            batch_samples.append(sample)
            self.samples_buffer.append(sample)

        result = SemanticQualityResult(
            bleu_score=bleu_result.get("bleu", 0.0),
            bleu_1=bleu_result.get("bleu_1", 0.0),
            bleu_2=bleu_result.get("bleu_2", 0.0),
            bleu_3=bleu_result.get("bleu_3", 0.0),
            bleu_4=bleu_result.get("bleu_4", 0.0),
            brevity_penalty=bleu_result.get("bp", 1.0),
            semantic_similarity=semantic_sim,
            sample_count=len(valid_pairs),
            avg_ref_length=avg_ref_len,
            avg_hyp_length=avg_hyp_len,
            length_ratio=length_ratio,
            samples=batch_samples,
        )

        # Accumulate for summary
        self.total_bleu += result.bleu_score
        if semantic_sim is not None:
            self.total_semantic += semantic_sim

        return result

    def _compute_bleu(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """
        Compute BLEU score using sacrebleu.

        Args:
            references: Normalized reference texts
            hypotheses: Normalized hypothesis texts

        Returns:
            Dictionary with BLEU scores
        """
        if not self.compute_bleu or not references:
            return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0, "bp": 1.0}

        try:
            # sacrebleu expects references as list of list (multiple references per hypothesis)
            refs_wrapped = [[r] for r in references]

            # Compute sentence-level BLEU for batch average
            bleu_scores = []
            for ref, hyp in zip(references, hypotheses):
                try:
                    score = sacrebleu.sentence_bleu(hyp, [ref])
                    bleu_scores.append(score.score)
                except Exception:
                    bleu_scores.append(0.0)

            avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0

            # Also compute corpus-level BLEU for more stable metrics
            try:
                corpus_bleu = sacrebleu.corpus_bleu(hypotheses, [references])
                return {
                    "bleu": corpus_bleu.score,
                    "bleu_1": corpus_bleu.precisions[0] if len(corpus_bleu.precisions) > 0 else 0.0,
                    "bleu_2": corpus_bleu.precisions[1] if len(corpus_bleu.precisions) > 1 else 0.0,
                    "bleu_3": corpus_bleu.precisions[2] if len(corpus_bleu.precisions) > 2 else 0.0,
                    "bleu_4": corpus_bleu.precisions[3] if len(corpus_bleu.precisions) > 3 else 0.0,
                    "bp": corpus_bleu.bp,
                    "avg_sentence_bleu": avg_bleu,
                }
            except Exception as e:
                logger.debug(f"Corpus BLEU failed: {e}")
                return {"bleu": avg_bleu, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0, "bp": 1.0}

        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0, "bp": 1.0}

    def _compute_semantic_similarity(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> Optional[float]:
        """
        Compute embedding-based semantic similarity.

        Args:
            references: Reference texts (original, not normalized)
            hypotheses: Hypothesis texts

        Returns:
            Average cosine similarity (0-1), or None if unavailable
        """
        if not self.compute_semantic:
            return None

        model = self._get_semantic_model()
        if model is None:
            return None

        try:
            # Encode texts
            ref_embeddings = model.encode(references, convert_to_numpy=True)
            hyp_embeddings = model.encode(hypotheses, convert_to_numpy=True)

            # Compute cosine similarity for each pair
            similarities = []
            for ref_emb, hyp_emb in zip(ref_embeddings, hyp_embeddings):
                # Cosine similarity
                norm_ref = np.linalg.norm(ref_emb)
                norm_hyp = np.linalg.norm(hyp_emb)
                if norm_ref > 0 and norm_hyp > 0:
                    sim = np.dot(ref_emb, hyp_emb) / (norm_ref * norm_hyp)
                    similarities.append(float(sim))

            return np.mean(similarities) if similarities else None

        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return None

    def get_corpus_bleu(self) -> Dict[str, float]:
        """
        Compute corpus-level BLEU over all accumulated samples.

        Returns:
            Dictionary with corpus-level BLEU scores
        """
        if not self.all_references or not self.all_hypotheses:
            return {"corpus_bleu": 0.0}

        return self._compute_bleu(self.all_references, self.all_hypotheses)

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics for logging.

        Returns:
            Dictionary with all semantic quality metrics
        """
        summary = {
            "num_samples": len(self.all_references),
            "num_batches": self.num_batches,
        }

        if self.num_batches > 0:
            summary["avg_batch_bleu"] = self.total_bleu / self.num_batches

            if self.compute_semantic and self.total_semantic > 0:
                summary["avg_semantic_sim"] = self.total_semantic / self.num_batches

        # Corpus-level BLEU
        corpus_bleu = self.get_corpus_bleu()
        summary["corpus_bleu"] = corpus_bleu.get("bleu", 0.0)
        summary["corpus_bleu_1"] = corpus_bleu.get("bleu_1", 0.0)
        summary["corpus_bleu_2"] = corpus_bleu.get("bleu_2", 0.0)
        summary["corpus_bleu_3"] = corpus_bleu.get("bleu_3", 0.0)
        summary["corpus_bleu_4"] = corpus_bleu.get("bleu_4", 0.0)
        summary["brevity_penalty"] = corpus_bleu.get("bp", 1.0)

        return summary

    def get_samples(self) -> List[Dict[str, Any]]:
        """Get stored samples for logging."""
        return self.samples_buffer

    def format_log_message(self) -> str:
        """Format a summary log message."""
        summary = self.get_summary()
        parts = ["[SEMANTIC]"]

        if "corpus_bleu" in summary:
            parts.append(f"BLEU={summary['corpus_bleu']:.2f}")

        if "avg_semantic_sim" in summary:
            parts.append(f"SemSim={summary['avg_semantic_sim']:.3f}")

        parts.append(f"samples={summary.get('num_samples', 0)}")

        return " ".join(parts)
