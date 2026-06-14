"""
Audio Quality Monitor for K-Moshi Training.

Provides objective audio quality evaluation including:
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- MCD (Mel Cepstral Distortion)
- SNR (Signal-to-Noise Ratio)

This module evaluates the audio reconstruction quality by comparing
Mimi-decoded audio from ground truth codes vs predicted codes.

Note: PESQ and STOI require additional dependencies:
    pip install pesq pystoi
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger("audio_quality_monitor")

# Try to import optional dependencies
_PESQ_AVAILABLE = False
_STOI_AVAILABLE = False
_LIBROSA_AVAILABLE = False

try:
    from pesq import pesq as compute_pesq
    _PESQ_AVAILABLE = True
except ImportError:
    logger.debug("pesq not installed. PESQ metrics will be disabled. "
                 "Install with: pip install pesq")

try:
    from pystoi import stoi as compute_stoi
    _STOI_AVAILABLE = True
except ImportError:
    logger.debug("pystoi not installed. STOI metrics will be disabled. "
                 "Install with: pip install pystoi")

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    logger.debug("librosa not installed. MCD metrics will be disabled. "
                 "Install with: pip install librosa")


@dataclass
class AudioQualityResult:
    """Result of audio quality evaluation."""
    # PESQ (Perceptual quality)
    pesq_score: Optional[float] = None      # -0.5 to 4.5 (higher is better)
    pesq_samples: int = 0

    # STOI (Intelligibility)
    stoi_score: Optional[float] = None      # 0 to 1 (higher is better)
    stoi_samples: int = 0

    # MCD (Spectral distortion)
    mcd_score: Optional[float] = None       # dB (lower is better)
    mcd_samples: int = 0

    # SNR (Signal-to-noise ratio)
    snr_db: Optional[float] = None          # dB (higher is better)
    snr_samples: int = 0

    # Statistics
    sample_count: int = 0
    total_duration_sec: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)


class AudioQualityMonitor:
    """
    Audio quality evaluation monitor.

    Compares Mimi-decoded audio from ground truth vs predicted codes
    to assess audio reconstruction quality.

    Metrics:
    - PESQ: Perceptual quality score (-0.5 to 4.5)
    - STOI: Speech intelligibility (0 to 1)
    - MCD: Mel cepstral distortion (lower is better)
    - SNR: Signal-to-noise ratio in dB

    Requirements:
    - Mimi model for decoding audio codes to waveform
    - Optional: pesq, pystoi, librosa packages

    Note: This monitor is computationally expensive. Consider:
    - Using max_samples to limit evaluation
    - Running less frequently than other monitors

    Usage:
        monitor = AudioQualityMonitor(mimi_model, sample_rate=24000)
        result = monitor.evaluate_batch(gt_codes, pred_codes, mask)
        metrics = monitor.get_summary()
    """

    def __init__(
        self,
        mimi_model=None,
        sample_rate: int = 24000,
        enabled: bool = True,
        compute_pesq: bool = True,
        compute_stoi: bool = True,
        compute_mcd: bool = True,
        compute_snr: bool = True,
        max_samples: int = 10,
        min_duration_sec: float = 1.0,
    ):
        """
        Initialize audio quality monitor.

        Args:
            mimi_model: Mimi codec model for decoding (can be set later)
            sample_rate: Audio sample rate (Moshi default: 24000)
            enabled: Whether to enable this monitor
            compute_pesq: Whether to compute PESQ (requires pesq package)
            compute_stoi: Whether to compute STOI (requires pystoi package)
            compute_mcd: Whether to compute MCD (requires librosa package)
            compute_snr: Whether to compute SNR
            max_samples: Maximum samples to evaluate per batch
            min_duration_sec: Minimum audio duration to evaluate
        """
        self.mimi = mimi_model
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.max_samples = max_samples
        self.min_duration_sec = min_duration_sec

        # Feature flags based on available dependencies
        self.compute_pesq = compute_pesq and _PESQ_AVAILABLE
        self.compute_stoi = compute_stoi and _STOI_AVAILABLE
        self.compute_mcd = compute_mcd and _LIBROSA_AVAILABLE
        self.compute_snr = compute_snr

        # Accumulated statistics
        self.pesq_scores: List[float] = []
        self.stoi_scores: List[float] = []
        self.mcd_scores: List[float] = []
        self.snr_scores: List[float] = []
        self.num_samples = 0
        self.total_duration = 0.0

        # Log availability
        if enabled:
            self._log_availability()

    def _log_availability(self):
        """Log which metrics are available."""
        available = []
        unavailable = []

        if self.compute_pesq:
            available.append("PESQ")
        elif not _PESQ_AVAILABLE:
            unavailable.append("PESQ (pip install pesq)")

        if self.compute_stoi:
            available.append("STOI")
        elif not _STOI_AVAILABLE:
            unavailable.append("STOI (pip install pystoi)")

        if self.compute_mcd:
            available.append("MCD")
        elif not _LIBROSA_AVAILABLE:
            unavailable.append("MCD (pip install librosa)")

        if self.compute_snr:
            available.append("SNR")

        if available:
            logger.info(f"AudioQualityMonitor enabled: {', '.join(available)}")
        if unavailable:
            logger.info(f"AudioQualityMonitor disabled: {', '.join(unavailable)}")

    def set_mimi_model(self, mimi_model):
        """Set or update Mimi model for audio decoding."""
        self.mimi = mimi_model

    def reset(self):
        """Reset accumulated statistics."""
        self.pesq_scores = []
        self.stoi_scores = []
        self.mcd_scores = []
        self.snr_scores = []
        self.num_samples = 0
        self.total_duration = 0.0

    def evaluate_batch(
        self,
        audio_codes_gt: torch.Tensor,
        audio_codes_pred: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> AudioQualityResult:
        """
        Evaluate audio quality for a batch.

        Args:
            audio_codes_gt: Ground truth audio codes [B, K, T] (K=8 codebooks)
            audio_codes_pred: Predicted audio codes [B, K, T]
            audio_mask: Optional mask for valid positions [B, K, T] or [B, T]

        Returns:
            AudioQualityResult with quality metrics
        """
        if not self.enabled:
            return AudioQualityResult()

        if self.mimi is None:
            return AudioQualityResult(errors=["Mimi model not set"])

        result = AudioQualityResult()
        batch_size = min(audio_codes_gt.shape[0], self.max_samples)

        for b in range(batch_size):
            try:
                # Decode audio
                gt_codes = audio_codes_gt[b:b+1]  # [1, K, T]
                pred_codes = audio_codes_pred[b:b+1]

                gt_audio = self._decode_audio(gt_codes)
                pred_audio = self._decode_audio(pred_codes)

                if gt_audio is None or pred_audio is None:
                    result.errors.append(f"Failed to decode sample {b}")
                    continue

                # Check duration
                duration_sec = len(gt_audio) / self.sample_rate
                if duration_sec < self.min_duration_sec:
                    continue

                result.sample_count += 1
                result.total_duration_sec += duration_sec
                self.total_duration += duration_sec
                self.num_samples += 1

                # Compute metrics
                if self.compute_pesq:
                    pesq_val = self._compute_pesq(gt_audio, pred_audio)
                    if pesq_val is not None:
                        self.pesq_scores.append(pesq_val)
                        result.pesq_samples += 1

                if self.compute_stoi:
                    stoi_val = self._compute_stoi(gt_audio, pred_audio)
                    if stoi_val is not None:
                        self.stoi_scores.append(stoi_val)
                        result.stoi_samples += 1

                if self.compute_mcd:
                    mcd_val = self._compute_mcd(gt_audio, pred_audio)
                    if mcd_val is not None:
                        self.mcd_scores.append(mcd_val)
                        result.mcd_samples += 1

                if self.compute_snr:
                    snr_val = self._compute_snr(gt_audio, pred_audio)
                    if snr_val is not None:
                        self.snr_scores.append(snr_val)
                        result.snr_samples += 1

            except Exception as e:
                result.errors.append(f"Error processing sample {b}: {str(e)}")
                logger.debug(f"Audio quality evaluation error: {e}")

        # Compute batch averages
        if self.pesq_scores:
            result.pesq_score = np.mean(self.pesq_scores[-batch_size:])
        if self.stoi_scores:
            result.stoi_score = np.mean(self.stoi_scores[-batch_size:])
        if self.mcd_scores:
            result.mcd_score = np.mean(self.mcd_scores[-batch_size:])
        if self.snr_scores:
            result.snr_db = np.mean(self.snr_scores[-batch_size:])

        return result

    def _decode_audio(self, codes: torch.Tensor) -> Optional[np.ndarray]:
        """
        Decode audio codes to waveform using Mimi.

        Args:
            codes: Audio codes [1, K, T]

        Returns:
            Audio waveform as numpy array, or None on failure
        """
        try:
            with torch.no_grad():
                # Mimi decode expects codes in specific format
                device = next(self.mimi.parameters()).device
                codes = codes.to(device)

                # Decode to waveform
                audio = self.mimi.decode(codes)

                # Convert to numpy
                if isinstance(audio, torch.Tensor):
                    audio = audio.squeeze().cpu().numpy()

                return audio

        except Exception as e:
            logger.debug(f"Audio decode failed: {e}")
            return None

    def _compute_pesq(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
    ) -> Optional[float]:
        """
        Compute PESQ score.

        PESQ requires 8kHz or 16kHz audio, so we resample.

        Args:
            reference: Reference audio
            degraded: Degraded (predicted) audio

        Returns:
            PESQ score (-0.5 to 4.5) or None on failure
        """
        if not self.compute_pesq:
            return None

        try:
            # PESQ requires 16kHz for wideband
            target_sr = 16000

            # Resample if needed
            if self.sample_rate != target_sr:
                if _LIBROSA_AVAILABLE:
                    reference = librosa.resample(
                        reference, orig_sr=self.sample_rate, target_sr=target_sr
                    )
                    degraded = librosa.resample(
                        degraded, orig_sr=self.sample_rate, target_sr=target_sr
                    )
                else:
                    # Simple resampling fallback
                    ratio = target_sr / self.sample_rate
                    new_len = int(len(reference) * ratio)
                    reference = np.interp(
                        np.linspace(0, len(reference) - 1, new_len),
                        np.arange(len(reference)),
                        reference
                    )
                    degraded = np.interp(
                        np.linspace(0, len(degraded) - 1, new_len),
                        np.arange(len(degraded)),
                        degraded
                    )

            # Ensure same length
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]

            # Compute PESQ (wideband mode)
            score = compute_pesq(target_sr, reference, degraded, 'wb')
            return float(score)

        except Exception as e:
            logger.debug(f"PESQ computation failed: {e}")
            return None

    def _compute_stoi(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
    ) -> Optional[float]:
        """
        Compute STOI score.

        Args:
            reference: Reference audio
            degraded: Degraded (predicted) audio

        Returns:
            STOI score (0 to 1) or None on failure
        """
        if not self.compute_stoi:
            return None

        try:
            # Ensure same length
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]

            # Compute STOI
            score = compute_stoi(reference, degraded, self.sample_rate, extended=False)
            return float(score)

        except Exception as e:
            logger.debug(f"STOI computation failed: {e}")
            return None

    def _compute_mcd(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
        n_mfcc: int = 13,
    ) -> Optional[float]:
        """
        Compute Mel Cepstral Distortion (MCD).

        MCD measures the distance between MFCC features.
        Lower is better (0 = identical).

        Args:
            reference: Reference audio
            degraded: Degraded (predicted) audio
            n_mfcc: Number of MFCC coefficients

        Returns:
            MCD in dB or None on failure
        """
        if not self.compute_mcd or not _LIBROSA_AVAILABLE:
            return None

        try:
            # Extract MFCCs
            mfcc_ref = librosa.feature.mfcc(
                y=reference, sr=self.sample_rate, n_mfcc=n_mfcc
            )
            mfcc_deg = librosa.feature.mfcc(
                y=degraded, sr=self.sample_rate, n_mfcc=n_mfcc
            )

            # Align lengths
            min_frames = min(mfcc_ref.shape[1], mfcc_deg.shape[1])
            mfcc_ref = mfcc_ref[:, :min_frames]
            mfcc_deg = mfcc_deg[:, :min_frames]

            # Compute MCD (excluding 0th coefficient - energy)
            diff = mfcc_ref[1:] - mfcc_deg[1:]
            mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0)))

            # Convert to dB scale
            mcd_db = (10.0 / np.log(10)) * mcd

            return float(mcd_db)

        except Exception as e:
            logger.debug(f"MCD computation failed: {e}")
            return None

    def _compute_snr(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
    ) -> Optional[float]:
        """
        Compute Signal-to-Noise Ratio.

        Treats reference as signal and (reference - degraded) as noise.

        Args:
            reference: Reference audio
            degraded: Degraded (predicted) audio

        Returns:
            SNR in dB or None on failure
        """
        try:
            # Ensure same length
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]

            # Compute noise (difference)
            noise = reference - degraded

            # Signal power
            signal_power = np.mean(reference ** 2)
            noise_power = np.mean(noise ** 2)

            if noise_power < 1e-10:
                return 100.0  # Very high SNR (essentially identical)
            if signal_power < 1e-10:
                return 0.0  # No signal

            snr_db = 10 * np.log10(signal_power / noise_power)
            return float(snr_db)

        except Exception as e:
            logger.debug(f"SNR computation failed: {e}")
            return None

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for logging."""
        summary = {
            "num_samples": self.num_samples,
            "total_duration_sec": self.total_duration,
        }

        if self.pesq_scores:
            summary["pesq_mean"] = np.mean(self.pesq_scores)
            summary["pesq_std"] = np.std(self.pesq_scores)
            summary["pesq_min"] = np.min(self.pesq_scores)
            summary["pesq_max"] = np.max(self.pesq_scores)

        if self.stoi_scores:
            summary["stoi_mean"] = np.mean(self.stoi_scores)
            summary["stoi_std"] = np.std(self.stoi_scores)

        if self.mcd_scores:
            summary["mcd_mean"] = np.mean(self.mcd_scores)
            summary["mcd_std"] = np.std(self.mcd_scores)

        if self.snr_scores:
            summary["snr_mean"] = np.mean(self.snr_scores)
            summary["snr_std"] = np.std(self.snr_scores)

        return summary

    def format_log_message(self) -> str:
        """Format a summary log message."""
        summary = self.get_summary()
        parts = ["[AUDIO_QUALITY]"]

        if "pesq_mean" in summary:
            parts.append(f"PESQ={summary['pesq_mean']:.2f}")
        if "stoi_mean" in summary:
            parts.append(f"STOI={summary['stoi_mean']:.3f}")
        if "mcd_mean" in summary:
            parts.append(f"MCD={summary['mcd_mean']:.2f}dB")
        if "snr_mean" in summary:
            parts.append(f"SNR={summary['snr_mean']:.1f}dB")

        parts.append(f"samples={summary.get('num_samples', 0)}")

        return " ".join(parts)

    @staticmethod
    def is_available() -> Dict[str, bool]:
        """Check which audio quality metrics are available."""
        return {
            "pesq": _PESQ_AVAILABLE,
            "stoi": _STOI_AVAILABLE,
            "mcd": _LIBROSA_AVAILABLE,
            "snr": True,  # Always available
        }
