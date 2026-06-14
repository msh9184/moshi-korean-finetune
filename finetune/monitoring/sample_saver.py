"""
Sample Saver for K-Moshi Training.

Provides periodic saving of:
- Ground Truth vs Prediction audio/text comparison (60-second segments)
- Decoded audio samples (moshi only) using Mimi codec
- Text predictions with ground truth for qualitative evaluation
- Speaker conditioning metadata and reference audio (for reproducibility)
- Organized by step with automatic cleanup of old samples

This module enables qualitative evaluation during training by saving:
  - sample_XX_gt_moshi.wav: Ground truth Moshi audio (what the model should generate)
  - sample_XX_pred_moshi.wav: Predicted Moshi audio (what the model actually generates)
  - sample_XX_gt_dialogue.wav: Ground truth stereo dialogue (L=Moshi, R=User)
  - sample_XX_pred_dialogue.wav: Predicted stereo dialogue (L=Pred Moshi, R=GT User)
  - sample_XX_text.json: Ground truth vs predicted text comparison
  - sample_XX_reference.wav: Reference audio for speaker conditioning (if available)
  - sample_XX_speaker_metadata.json: Speaker conditioning metadata (if available)

Output Structure:
    samples/{split}/step_{step:06d}/sample_segment/
    ├── sample_00_gt_dialogue.wav
    ├── sample_00_gt_moshi.wav
    ├── sample_00_pred_dialogue.wav
    ├── sample_00_pred_moshi.wav
    ├── sample_00_text.json
    ├── sample_00_reference.wav (if speaker conditioning enabled)
    ├── sample_00_speaker_metadata.json (if speaker conditioning enabled)
    └── ...
"""

import json
import logging
import random
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


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

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except (ImportError, OSError) as e:
    SOUNDFILE_AVAILABLE = False
    sf = None  # type: ignore

# torchaudio is optional - we prefer soundfile for stability
TORCHAUDIO_AVAILABLE = False
torchaudio = None  # type: ignore

logger = logging.getLogger("sample_saver")

# Mimi codec always uses 8 codebooks per audio stream
# This is fixed regardless of dep_q setting
MIMI_CODEBOOKS_PER_STREAM = 8

# Debug mode for tracking audio consistency issues
# Can be overridden via YAML config: sample_saving.debug_audio_consistency
_DEBUG_AUDIO_CONSISTENCY = False  # Disabled by default for cleaner logs


def _compute_audio_fingerprint(audio: torch.Tensor) -> str:
    """
    Compute a fingerprint for audio tensor to verify consistency.

    Returns a string with: sum, mean, first 3 values, last 3 values.
    """
    if audio is None:
        return "None"
    try:
        audio_flat = audio.flatten()
        total = float(audio_flat.sum().item())
        mean = float(audio_flat.mean().item())
        first3 = [float(x.item()) for x in audio_flat[:3]]
        last3 = [float(x.item()) for x in audio_flat[-3:]]
        return f"sum={total:.4f}, mean={mean:.6f}, first3={first3}, last3={last3}"
    except Exception as e:
        return f"error: {e}"


@dataclass
class SampleSaveResult:
    """Result of sample save operation."""
    step: int
    num_samples: int
    gt_audio_paths: List[Path] = field(default_factory=list)
    pred_audio_paths: List[Path] = field(default_factory=list)
    dialogue_paths: List[Path] = field(default_factory=list)  # Stereo dialogue files
    text_paths: List[Path] = field(default_factory=list)
    # Speaker conditioning outputs (for reproducibility analysis)
    reference_audio_paths: List[Path] = field(default_factory=list)
    speaker_metadata_paths: List[Path] = field(default_factory=list)
    success: bool = True
    errors: List[str] = field(default_factory=list)


class SampleSaver:
    """
    Saves audio and text samples during training for qualitative evaluation.

    Key Features:
    - Saves BOTH Ground Truth and Model Predictions for comparison
    - Decodes audio tokens using Mimi codec
    - Supports moshi (AI) and user audio streams
    - Text predictions with ground truth comparison in JSON
    - Automatic cleanup of old samples to save disk space

    File Naming Convention:
        sample_00_gt_moshi.wav    - Ground truth Moshi audio
        sample_00_pred_moshi.wav  - Predicted Moshi audio (model output)
        sample_00_gt_dialogue.wav - GT stereo (L=Moshi, R=User)
        sample_00_pred_dialogue.wav - Pred stereo (L=Pred Moshi, R=GT User)
        sample_00_text.json       - Text comparison (GT vs Prediction)

    Note: gt_user.wav is NOT saved separately (available in gt_dialogue right channel).
    """

    def __init__(
        self,
        mimi,
        tokenizer,
        run_dir: Path,
        text_padding_token_id: int,
        end_of_text_padding_id: int,
        audio_offset: int = 1,
        dep_q: int = 8,
        has_user_audio: bool = False,
        sample_rate: int = 24000,
        audio_format: str = "wav",
        max_samples_per_split: int = 20,
        samples_per_save: int = 3,
        save_audio: bool = True,
        save_text: bool = True,
        debug_audio_consistency: bool = False,
    ):
        """
        Initialize sample saver.

        Args:
            mimi: Mimi codec model (for audio decoding)
            tokenizer: Text tokenizer (for decoding text tokens)
            run_dir: Base directory for saving samples
            text_padding_token_id: Padding token ID
            end_of_text_padding_id: End-of-text padding ID
            audio_offset: Index where audio codebooks start (default 1)
            dep_q: Number of audio codebooks (default 8)
            has_user_audio: Whether input data contains user audio (stereo modes)
            sample_rate: Audio sample rate (default 24000)
            audio_format: 'wav' or 'flac'
            max_samples_per_split: Max sample directories to keep per split
            samples_per_save: Number of samples to save per event
            save_audio: Whether to save audio samples
            save_text: Whether to save text predictions
            debug_audio_consistency: Enable verbose debug logging for audio consistency
        """
        self.mimi = mimi
        self.tokenizer = tokenizer
        self.run_dir = Path(run_dir)
        self.text_padding_token_id = text_padding_token_id
        self.end_of_text_padding_id = end_of_text_padding_id
        self.audio_offset = audio_offset
        self.dep_q = dep_q
        self.has_user_audio = has_user_audio
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.max_samples_per_split = max_samples_per_split
        self.samples_per_save = samples_per_save
        self.save_audio = save_audio
        self.save_text = save_text
        self.debug_audio_consistency = debug_audio_consistency

        # Mimi codebook vocabulary size
        self.codebook_size = 2048

        # Create sample directories
        self.samples_dir = self.run_dir / "samples"
        self.train_samples_dir = self.samples_dir / "train"
        self.valid_samples_dir = self.samples_dir / "valid"

        self.train_samples_dir.mkdir(parents=True, exist_ok=True)
        self.valid_samples_dir.mkdir(parents=True, exist_ok=True)

        # Track saved sample directories for cleanup
        self.train_sample_dirs: List[Path] = []
        self.valid_sample_dirs: List[Path] = []

        # Validate audio saving capability
        if self.save_audio and not SOUNDFILE_AVAILABLE and not TORCHAUDIO_AVAILABLE:
            logger.warning(
                "Neither soundfile nor torchaudio available. "
                "Install with: pip install soundfile torchaudio"
            )
            self.save_audio = False

        logger.info(f"SampleSaver initialized: audio={self.save_audio}, text={self.save_text}")
        logger.info(f"  Output: samples/{{split}}/step_{{step}}/sample_segment/")
        logger.info(f"  File naming: sample_XX_{{gt|pred}}_{{dialogue|moshi}}.wav")

    def decode_text(self, token_ids: torch.Tensor) -> str:
        """
        Decode text tokens to string with UTF-8 safe handling and Korean post-processing.

        SentencePiece uses byte-level fallback for OOV characters (including Korean),
        which can produce invalid UTF-8 sequences and extra spaces between Korean syllables.
        This method handles both cases gracefully.

        Args:
            token_ids: Token IDs [T] or [1, T]

        Returns:
            Decoded text string (UTF-8 safe, Korean spaces cleaned)
        """
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)

        # Filter padding tokens
        valid_tokens = []
        for tid in token_ids.tolist():
            if tid not in (self.text_padding_token_id, self.end_of_text_padding_id):
                if tid >= 0:
                    valid_tokens.append(tid)

        if not valid_tokens:
            return ""

        try:
            decoded = self.tokenizer.decode(valid_tokens)
            # Ensure UTF-8 validity
            if isinstance(decoded, bytes):
                decoded = decoded.decode("utf-8", errors="replace")
            else:
                decoded = decoded.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            decoded = decoded.replace("\ufffd", "")
            decoded = decoded.strip()

            # Apply Korean text post-processing to remove spaces between Hangul syllables
            decoded = _postprocess_korean_text(decoded)

            return decoded
        except Exception as e:
            logger.debug(f"Text decode error: {e}")
            return ""

    def decode_audio(self, audio_codes: torch.Tensor, source: str = "unknown") -> Optional[torch.Tensor]:
        """
        Decode audio codes using Mimi codec.

        Args:
            audio_codes: Audio token IDs [K, T] where K is num codebooks
            source: Description of the source for logging (e.g., "gt_moshi", "pred_moshi")

        Returns:
            Audio waveform [1, num_samples] or None if decoding fails
        """
        try:
            with torch.no_grad():
                # Mimi expects [B, K, T] format
                if audio_codes.dim() == 2:
                    audio_codes = audio_codes.unsqueeze(0)

                # Move to mimi's device
                device = next(self.mimi.parameters()).device
                audio_codes = audio_codes.to(device)

                # Validate audio codes range
                max_val = audio_codes.max().item()
                min_val = audio_codes.min().item()

                if max_val >= self.codebook_size or min_val < 0:
                    logger.debug(
                        f"[{source}] Audio codes out of range: min={min_val}, max={max_val}. Clamping."
                    )
                    audio_codes = audio_codes.clamp(0, self.codebook_size - 1)

                # Reset Mimi streaming state if active to ensure consistent decoding
                # Mimi is a streaming neural codec that can accumulate state across decode calls
                if hasattr(self.mimi, '_streaming_state') and self.mimi._streaming_state is not None:
                    self.mimi.reset_streaming()

                # Decode using mimi
                audio = self.mimi.decode(audio_codes)

                return audio.squeeze(0)  # [1, num_samples]

        except Exception as e:
            logger.warning(f"[{source}] Audio decode error: {e}")
            return None

    def decode_audio_batch(
        self,
        audio_codes_list: List[torch.Tensor],
        sources: List[str],
    ) -> List[Optional[torch.Tensor]]:
        """
        Decode multiple audio codes in a single batch for consistency.

        This method ensures all audio for a sample is decoded with the same
        Mimi codec state, preventing state contamination between decode calls.

        Args:
            audio_codes_list: List of audio token tensors [K, T]
            sources: List of source descriptions for logging

        Returns:
            List of audio waveforms [1, num_samples] or None for each input
        """
        if not audio_codes_list:
            return []

        results: List[Optional[torch.Tensor]] = [None] * len(audio_codes_list)

        try:
            with torch.no_grad():
                device = next(self.mimi.parameters()).device

                # Prepare all codes for batch processing
                valid_indices = []
                batch_codes = []
                max_time = 0

                for i, codes in enumerate(audio_codes_list):
                    if codes is None:
                        continue

                    # Validate
                    if codes.dim() != 2:
                        logger.warning(f"[{sources[i]}] Invalid codes dim: {codes.dim()}")
                        continue

                    max_val = codes.max().item()
                    min_val = codes.min().item()

                    if max_val >= self.codebook_size or min_val < 0:
                        codes = codes.clamp(0, self.codebook_size - 1)

                    valid_indices.append(i)
                    batch_codes.append(codes)
                    max_time = max(max_time, codes.size(-1))

                if not batch_codes:
                    return results

                # Pad all codes to same length
                padded_codes = []
                for codes in batch_codes:
                    if codes.size(-1) < max_time:
                        # Pad with zeros (will produce silence for the padding)
                        pad_size = max_time - codes.size(-1)
                        codes = torch.nn.functional.pad(codes, (0, pad_size), value=0)
                    padded_codes.append(codes)

                # Stack into batch [B, K, T]
                stacked_codes = torch.stack(padded_codes, dim=0).to(device)

                # Reset Mimi streaming state if active
                if hasattr(self.mimi, '_streaming_state') and self.mimi._streaming_state is not None:
                    self.mimi.reset_streaming()

                # Decode all at once
                batch_audio = self.mimi.decode(stacked_codes)  # [B, 1, num_samples]

                # Extract individual results
                for batch_idx, orig_idx in enumerate(valid_indices):
                    results[orig_idx] = batch_audio[batch_idx]  # [1, num_samples]

                # Note: debug logging controlled by calling context (SampleSaver.debug_audio_consistency)
                # This method is called from SampleSaver which handles debug flag

        except Exception as e:
            logger.warning(f"[BATCH_DECODE] Error: {e}")
            # Fall back to individual decoding
            for i, codes in enumerate(audio_codes_list):
                if codes is not None:
                    results[i] = self.decode_audio(codes, sources[i])

        return results

    def save_audio_file(
        self,
        audio: torch.Tensor,
        filepath: Path,
        sample_rate: Optional[int] = None
    ) -> bool:
        """
        Save audio tensor to file.

        Args:
            audio: Audio waveform [C, T] or [T]
            filepath: Output file path
            sample_rate: Sample rate for saving (default: self.sample_rate=24000)
                        IMPORTANT: Reference audio may be at 16kHz, so this must be
                        explicitly specified to avoid pitch/speed issues.

        Returns:
            True if successful
        """
        try:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            audio_np = audio.cpu().numpy()

            # Use provided sample_rate or default to self.sample_rate
            sr = sample_rate if sample_rate is not None else self.sample_rate

            if SOUNDFILE_AVAILABLE:
                sf.write(str(filepath), audio_np.T, sr)
            elif TORCHAUDIO_AVAILABLE and torchaudio is not None:
                torchaudio.save(str(filepath), torch.from_numpy(audio_np), sr)
            else:
                return False

            return True

        except Exception as e:
            logger.debug(f"Audio save error: {e}")
            return False

    def create_stereo_dialogue(
        self,
        moshi_audio: torch.Tensor,
        user_audio: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Create stereo dialogue audio by combining Moshi and User audio.

        Full-duplex dialogue format:
          - Left channel (0): Moshi audio (AI response)
          - Right channel (1): User audio (human input)

        This matches the Moshi training convention where the model learns to:
          - Generate Left channel (Moshi's response)
          - Listen to Right channel (User's input)

        Args:
            moshi_audio: Moshi audio waveform [1, T] or [T]
            user_audio: User audio waveform [1, T] or [T]

        Returns:
            Stereo audio tensor [2, T] or None if creation fails
        """
        try:
            # Ensure both are 2D tensors [1, T]
            if moshi_audio.dim() == 1:
                moshi_audio = moshi_audio.unsqueeze(0)
            if user_audio.dim() == 1:
                user_audio = user_audio.unsqueeze(0)

            # Get lengths
            moshi_len = moshi_audio.size(-1)
            user_len = user_audio.size(-1)

            # Align lengths by padding the shorter one with zeros
            if moshi_len != user_len:
                max_len = max(moshi_len, user_len)
                if moshi_len < max_len:
                    pad_size = max_len - moshi_len
                    moshi_audio = F.pad(moshi_audio, (0, pad_size), value=0.0)
                if user_len < max_len:
                    pad_size = max_len - user_len
                    user_audio = F.pad(user_audio, (0, pad_size), value=0.0)

            # Stack to create stereo: [2, T]
            # Channel 0 (Left) = Moshi, Channel 1 (Right) = User
            stereo_audio = torch.cat([moshi_audio, user_audio], dim=0)

            return stereo_audio

        except Exception as e:
            logger.debug(f"Stereo dialogue creation error: {e}")
            return None

    def save_samples(
        self,
        codes: torch.Tensor,
        text_logits: torch.Tensor,
        audio_logits: Optional[torch.Tensor],
        step: int,
        split: str = "train",
        user_text_alignments: Optional[list[list]] = None,
        moshi_text_raw_list: Optional[list[str]] = None,
        speaker_conditioning_info: Optional[Any] = None,
        prompt_length: int = 0,
    ) -> SampleSaveResult:
        """
        Save samples from a batch with both Ground Truth and Predictions.

        Args:
            codes: Batch of input codes [B, K, T] (K = text + audio codebooks)
                   This contains GROUND TRUTH data.
            text_logits: Model text predictions [B, T, V]
            audio_logits: Model audio predictions [B, dep_q, T, V] or None
                         If provided, predicted audio will be decoded and saved.
            step: Current training step
            split: 'train' or 'valid'
            user_text_alignments: Per-sample User text alignments for reference
            moshi_text_raw_list: Per-sample original Moshi text (no truncation)
            speaker_conditioning_info: Optional speaker conditioning metadata
                                      (EvalSpeakerConditioningInfo from eval.py)
            prompt_length: Number of prompt frames prepended to predictions.
                          Used for shape validation. When audio prompting is enabled,
                          predictions have shape [B, K, T_main + T_prompt] while codes
                          have shape [B, K, T_main]. The caller should pre-slice logits,
                          but if not, this value enables auto-slicing as fallback.

        Returns:
            SampleSaveResult with save details

        Saved Files per Sample:
            - sample_XX_gt_moshi.wav: Ground truth Moshi audio
            - sample_XX_pred_moshi.wav: Predicted Moshi audio (if audio_logits provided)
            - sample_XX_gt_user.wav: Ground truth User audio (if available)
            - sample_XX_text.json: Text comparison with GT and prediction
            - sample_XX_reference.wav: Reference audio used for speaker conditioning (if available)
            - sample_XX_speaker_metadata.json: Speaker conditioning metadata (if available)
        """
        result = SampleSaveResult(step=step, num_samples=0)

        if not self.save_audio and not self.save_text:
            return result

        # Select sample directory: samples/{split}/step_{step}/sample_segment/
        base_dir = self.train_samples_dir if split == "train" else self.valid_samples_dir
        step_parent_dir = base_dir / f"step_{step:06d}"
        step_dir = step_parent_dir / "sample_segment"

        # FIX: Clear existing step directory to prevent file mismatch from previous runs
        # This is critical because random.sample() selects different batch indices each run,
        # and leftover files from previous runs will have different content!
        if step_dir.exists():
            try:
                shutil.rmtree(step_dir)
                logger.debug(f"Cleared existing step directory: {step_dir}")
            except Exception as e:
                logger.warning(f"Failed to clear step directory {step_dir}: {e}")

        step_dir.mkdir(parents=True, exist_ok=True)

        # Track for cleanup
        if split == "train":
            self.train_sample_dirs.append(step_dir)
        else:
            self.valid_sample_dirs.append(step_dir)

        # Select random samples from batch
        batch_size = codes.size(0)
        num_samples = min(self.samples_per_save, batch_size)
        sample_indices = random.sample(range(batch_size), num_samples)

        # =================================================================
        # CRITICAL FIX: Shape validation with auto-slicing fallback
        # =================================================================
        # When audio prompting is enabled, caller should slice logits before
        # passing to this function. If not, we detect shape mismatch and
        # auto-slice as a safety fallback.
        # =================================================================
        if audio_logits is not None:
            gt_time = codes.size(2)
            pred_time = audio_logits.size(2)

            if pred_time != gt_time:
                time_diff = pred_time - gt_time
                if time_diff > 0:
                    # Prediction is longer - likely includes prompt region
                    if prompt_length == 0:
                        # prompt_length not provided, infer from shape difference
                        prompt_length = time_diff
                    logger.warning(
                        f"[SAMPLE SAVER] Auto-slicing: GT_time={gt_time}, Pred_time={pred_time}, "
                        f"removing {prompt_length} prompt frames from predictions"
                    )
                    text_logits = text_logits[:, :, prompt_length:]
                    audio_logits = audio_logits[:, :, prompt_length:]
                else:
                    # GT is longer than Pred - unexpected, log error
                    logger.error(
                        f"[SAMPLE SAVER] Shape mismatch: GT_time={gt_time}, Pred_time={pred_time}. "
                        f"Expected Pred >= GT. Audio comparison may be incorrect."
                    )

        for idx, batch_idx in enumerate(sample_indices):
            sample_codes = codes[batch_idx]  # [K, T] - Ground Truth codes
            sample_data = {
                "step": step,
                "split": split,
                "sample_idx": idx,
                "batch_idx": batch_idx,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # =================================================================
            # DATA CONSISTENCY VERIFICATION (Optional Debug Mode)
            # =================================================================
            # Compute unique fingerprints for audio codes to verify consistency
            # This helps debug any indexing mismatch between codes and text
            # Enable via YAML: sample_saving.debug_audio_consistency: true
            verification_data = {}
            if self.debug_audio_consistency:
                # Audio codes fingerprint (first 100 values sum)
                moshi_audio_codes = sample_codes[
                    self.audio_offset : self.audio_offset + MIMI_CODEBOOKS_PER_STREAM
                ]
                codes_fp = int(moshi_audio_codes[:, :100].sum().item()) if moshi_audio_codes.numel() > 0 else 0

                # User audio codes fingerprint
                user_audio_start = self.audio_offset + MIMI_CODEBOOKS_PER_STREAM
                user_audio_codes = sample_codes[user_audio_start : user_audio_start + MIMI_CODEBOOKS_PER_STREAM]
                user_codes_fp = int(user_audio_codes[:, :100].sum().item()) if user_audio_codes.numel() > 0 else 0

                # Text from codes
                text_codes = sample_codes[:self.audio_offset]
                text_from_codes = self.decode_text(text_codes.squeeze(0)) if text_codes.numel() > 0 else ""
                text_preview = text_from_codes[:50] if text_from_codes else "(empty)"

                # Text from moshi_text_raw_list
                raw_text = ""
                if moshi_text_raw_list and batch_idx < len(moshi_text_raw_list):
                    raw_text = moshi_text_raw_list[batch_idx] or ""
                raw_preview = raw_text[:50] if raw_text else "(empty)"

                # Text match check
                text_match = text_from_codes[:30] == raw_text[:30] if raw_text else None

                # Store verification data for JSON output
                verification_data = {
                    "batch_idx": batch_idx,
                    "moshi_codes_fingerprint": codes_fp,
                    "user_codes_fingerprint": user_codes_fp,
                    "text_from_codes_preview": text_preview,
                    "text_from_raw_preview": raw_preview,
                    "text_match": text_match,
                }

                # Log verification info (debug level)
                logger.debug(
                    f"[DATA VERIFY] sample_idx={idx}, batch_idx={batch_idx} | "
                    f"moshi_codes_fp={codes_fp}, user_codes_fp={user_codes_fp}, "
                    f"text_match={text_match}"
                )

            # Add verification data to sample metadata
            if verification_data:
                sample_data["_debug_verification"] = verification_data

            # =================================================================
            # TEXT: Save Ground Truth and Prediction
            # =================================================================
            if self.save_text:
                try:
                    # Ground truth text - prefer original text (no truncation) if available
                    gt_text_codes = sample_codes[:self.audio_offset]  # [1, T]
                    gt_text_from_codes = self.decode_text(gt_text_codes.squeeze(0))

                    # Use original Moshi text if available (no truncation)
                    gt_text_raw = None
                    if moshi_text_raw_list and batch_idx < len(moshi_text_raw_list):
                        gt_text_raw = moshi_text_raw_list[batch_idx]

                    # Predicted text (from model output)
                    pred_text_codes = text_logits[batch_idx].argmax(dim=-1)  # [T]
                    pred_text = self.decode_text(pred_text_codes)

                    sample_data["text"] = {
                        # Primary ground truth: original text (no truncation)
                        "ground_truth": gt_text_raw if gt_text_raw else gt_text_from_codes,
                        # Also keep the tokenized version for comparison
                        "ground_truth_tokenized": gt_text_from_codes,
                        "prediction": pred_text,
                        "gt_token_count": int((gt_text_codes.squeeze(0) != self.text_padding_token_id).sum().item()),
                        "pred_token_count": int((pred_text_codes != self.text_padding_token_id).sum().item()),
                        # Flag to indicate if original text was used
                        "used_original_text": gt_text_raw is not None,
                    }

                    # Add User text for reference (from alignments if available)
                    if user_text_alignments and batch_idx < len(user_text_alignments):
                        user_aligns = user_text_alignments[batch_idx]
                        if user_aligns:
                            user_text = " ".join([a[0] for a in user_aligns])
                            sample_data["text_user"] = {
                                "ground_truth": user_text,
                                "word_count": len(user_aligns),
                            }

                except Exception as e:
                    result.errors.append(f"Sample {idx} text error: {e}")

            # =================================================================
            # AUDIO: Save Ground Truth and Prediction
            # =================================================================
            # FIX: Use batch decoding to ensure consistent Mimi codec state
            # across all audio streams for this sample. This prevents state
            # contamination that can cause audio mismatch issues.
            # =================================================================
            if self.save_audio:
                # Track decoded audio tensors for stereo dialogue creation
                gt_moshi_audio = None
                pred_moshi_audio = None
                gt_user_audio = None

                try:
                    # ---------------------------------------------------------
                    # Prepare all audio codes for batch decoding
                    # ---------------------------------------------------------
                    audio_codes_list = []
                    audio_sources = []

                    # 1. Ground Truth Moshi codes
                    gt_moshi_codes = sample_codes[
                        self.audio_offset : self.audio_offset + MIMI_CODEBOOKS_PER_STREAM
                    ]  # [8, T]
                    audio_codes_list.append(gt_moshi_codes)
                    audio_sources.append(f"sample_{idx}_gt_moshi")

                    # 2. Predicted Moshi codes (if available)
                    pred_moshi_codes = None
                    if audio_logits is not None:
                        pred_all_codes = audio_logits[batch_idx].argmax(dim=-1)  # [dep_q, T]
                        pred_moshi_codes = pred_all_codes[:MIMI_CODEBOOKS_PER_STREAM]  # [8, T]
                        audio_codes_list.append(pred_moshi_codes)
                        audio_sources.append(f"sample_{idx}_pred_moshi")
                    else:
                        audio_codes_list.append(None)
                        audio_sources.append(f"sample_{idx}_pred_moshi_none")

                    # 3. Ground Truth User codes (if stereo mode: USER-STREAM or FULL-DUPLEX)
                    gt_user_codes = None
                    user_start = self.audio_offset + MIMI_CODEBOOKS_PER_STREAM  # = 9
                    user_end = user_start + MIMI_CODEBOOKS_PER_STREAM  # = 17

                    if self.has_user_audio and sample_codes.size(0) >= user_end:
                        gt_user_codes = sample_codes[user_start:user_end]  # [8, T]
                        # Check if user audio exists (not all padding)
                        if (gt_user_codes == -1).all() or (gt_user_codes < 0).all():
                            gt_user_codes = None

                    if gt_user_codes is not None:
                        audio_codes_list.append(gt_user_codes)
                        audio_sources.append(f"sample_{idx}_gt_user")
                    else:
                        audio_codes_list.append(None)
                        audio_sources.append(f"sample_{idx}_gt_user_none")

                    # ---------------------------------------------------------
                    # Batch decode all audio at once for consistent state
                    # ---------------------------------------------------------
                    decoded_audio = self.decode_audio_batch(audio_codes_list, audio_sources)

                    gt_moshi_audio = decoded_audio[0]
                    pred_moshi_audio = decoded_audio[1] if len(decoded_audio) > 1 else None
                    gt_user_audio = decoded_audio[2] if len(decoded_audio) > 2 else None

                    # Debug logging for consistency verification
                    if self.debug_audio_consistency:
                        # Compute and store audio fingerprints
                        gt_moshi_fp = _compute_audio_fingerprint(gt_moshi_audio)
                        pred_moshi_fp = _compute_audio_fingerprint(pred_moshi_audio)
                        gt_user_fp = _compute_audio_fingerprint(gt_user_audio)

                        # Add to verification data for JSON
                        if verification_data:
                            verification_data["gt_moshi_audio_fp"] = gt_moshi_fp
                            verification_data["pred_moshi_audio_fp"] = pred_moshi_fp
                            verification_data["gt_user_audio_fp"] = gt_user_fp

                        logger.debug(
                            f"[SAMPLE {idx}] Audio fingerprints: gt_moshi={gt_moshi_fp[:30]}..., "
                            f"pred_moshi={pred_moshi_fp[:30] if pred_moshi_fp else 'None'}..."
                        )

                    # ---------------------------------------------------------
                    # Save individual audio files
                    # ---------------------------------------------------------
                    if gt_moshi_audio is not None:
                        gt_path = step_dir / f"sample_{idx:02d}_gt_moshi.{self.audio_format}"
                        if self.save_audio_file(gt_moshi_audio, gt_path):
                            result.gt_audio_paths.append(gt_path)
                            sample_data.setdefault("audio", {})["gt_moshi_path"] = str(gt_path.name)

                    if pred_moshi_audio is not None:
                        pred_path = step_dir / f"sample_{idx:02d}_pred_moshi.{self.audio_format}"
                        if self.save_audio_file(pred_moshi_audio, pred_path):
                            result.pred_audio_paths.append(pred_path)
                            sample_data.setdefault("audio", {})["pred_moshi_path"] = str(pred_path.name)

                    # NOTE: gt_user.wav is NOT saved separately.
                    # User audio is available in gt_dialogue right channel.

                except Exception as e:
                    result.errors.append(f"Sample {idx} audio decode error: {e}")

                # ---------------------------------------------------------
                # STEREO DIALOGUE: Full-duplex conversation audio
                # Left channel = Moshi, Right channel = User
                # ---------------------------------------------------------
                if gt_user_audio is not None:
                    # GT Dialogue: GT Moshi (Left) + GT User (Right)
                    if gt_moshi_audio is not None:
                        try:
                            gt_dialogue = self.create_stereo_dialogue(gt_moshi_audio, gt_user_audio)
                            if gt_dialogue is not None:
                                # Debug: Verify dialogue matches components
                                if self.debug_audio_consistency:
                                    # Compute fingerprints
                                    orig_moshi_fp = _compute_audio_fingerprint(gt_moshi_audio)
                                    dialogue_left_fp = _compute_audio_fingerprint(gt_dialogue[0:1])
                                    orig_user_fp = _compute_audio_fingerprint(gt_user_audio)
                                    dialogue_right_fp = _compute_audio_fingerprint(gt_dialogue[1:2])

                                    # CRITICAL: Verify left channel matches gt_moshi_audio
                                    orig_slice = gt_moshi_audio.flatten()[:1000]
                                    dial_slice = gt_dialogue[0].flatten()[:1000]
                                    is_match = torch.allclose(orig_slice, dial_slice, atol=1e-6)

                                    if not is_match:
                                        logger.error(
                                            f"[SAMPLE {idx}] CRITICAL MISMATCH! "
                                            f"gt_moshi_audio != gt_dialogue[0]"
                                        )
                                    else:
                                        logger.debug(f"[SAMPLE {idx}] Stereo verification passed")

                                    # Add dialogue fingerprints to verification data
                                    if verification_data:
                                        verification_data["gt_dialogue_left_fp"] = dialogue_left_fp
                                        verification_data["gt_dialogue_right_fp"] = dialogue_right_fp
                                        verification_data["stereo_moshi_match"] = is_match
                                gt_dialogue_path = step_dir / f"sample_{idx:02d}_gt_dialogue.{self.audio_format}"
                                if self.save_audio_file(gt_dialogue, gt_dialogue_path):
                                    result.dialogue_paths.append(gt_dialogue_path)
                                    sample_data.setdefault("audio", {})["gt_dialogue_path"] = str(gt_dialogue_path.name)
                        except Exception as e:
                            result.errors.append(f"Sample {idx} gt_dialogue error: {e}")

                    # Pred Dialogue: Pred Moshi (Left) + GT User (Right)
                    if pred_moshi_audio is not None:
                        try:
                            pred_dialogue = self.create_stereo_dialogue(pred_moshi_audio, gt_user_audio)
                            if pred_dialogue is not None:
                                pred_dialogue_path = step_dir / f"sample_{idx:02d}_pred_dialogue.{self.audio_format}"
                                if self.save_audio_file(pred_dialogue, pred_dialogue_path):
                                    result.dialogue_paths.append(pred_dialogue_path)
                                    sample_data.setdefault("audio", {})["pred_dialogue_path"] = str(pred_dialogue_path.name)
                        except Exception as e:
                            result.errors.append(f"Sample {idx} pred_dialogue error: {e}")

            # =================================================================
            # SPEAKER CONDITIONING: Save Reference Audio & Metadata
            # =================================================================
            # For reproducibility analysis, save:
            # 1. Reference audio used for speaker embedding extraction
            # 2. Comprehensive speaker conditioning metadata
            #
            # CRITICAL: Sample Rate Handling
            # - Reference audio from interleaver (train): 16kHz (for speaker encoder)
            # - Reference audio from mimi.decode (eval): 24kHz (Mimi's native rate)
            # - We MUST use the correct sample rate when saving to avoid pitch issues
            # =================================================================
            if speaker_conditioning_info is not None and getattr(
                speaker_conditioning_info, 'enabled', False
            ):
                try:
                    # Get reference audio sample rate (CRITICAL for correct playback)
                    ref_audio_sample_rate = getattr(
                        speaker_conditioning_info, 'reference_audio_sample_rate', 24000
                    )
                    ref_duration_sec = getattr(
                        speaker_conditioning_info, 'reference_duration_sec', 0.0
                    )
                    ref_num_frames = getattr(
                        speaker_conditioning_info, 'reference_num_frames', 0
                    )
                    source_file = getattr(
                        speaker_conditioning_info, 'source_file', None
                    )

                    # =============================================================
                    # CRITICAL FIX: Use batch-level fields for per-sample metadata
                    # =============================================================
                    # If batch-level lists are available, use batch_idx to get
                    # the correct per-sample information. Otherwise, fall back
                    # to legacy single-value fields for backward compatibility.
                    # =============================================================

                    # Get per-sample timing (prefer batch-level lists)
                    ref_start_secs = getattr(speaker_conditioning_info, 'reference_start_secs', None)
                    ref_end_secs = getattr(speaker_conditioning_info, 'reference_end_secs', None)
                    if ref_start_secs is not None and batch_idx < len(ref_start_secs):
                        sample_start_sec = ref_start_secs[batch_idx]
                        sample_end_sec = ref_end_secs[batch_idx] if ref_end_secs else 0.0
                    else:
                        # Fallback to legacy single fields
                        sample_start_sec = getattr(speaker_conditioning_info, 'reference_start_sec', 0.0)
                        sample_end_sec = getattr(speaker_conditioning_info, 'reference_end_sec', 0.0)

                    # Get per-sample source file (prefer batch-level lists)
                    source_files = getattr(speaker_conditioning_info, 'source_files', None)
                    if source_files is not None and batch_idx < len(source_files):
                        sample_source_file = source_files[batch_idx]
                    else:
                        sample_source_file = source_file  # Legacy single field

                    # Get per-sample reference text (prefer batch-level lists)
                    ref_texts = getattr(speaker_conditioning_info, 'reference_texts', None)
                    if ref_texts is not None and batch_idx < len(ref_texts):
                        sample_ref_text = ref_texts[batch_idx]
                    else:
                        sample_ref_text = getattr(speaker_conditioning_info, 'reference_text', None)

                    # Build comprehensive speaker metadata with PER-SAMPLE values
                    speaker_metadata = {
                        "enabled": True,
                        "method": getattr(speaker_conditioning_info, 'method', 'none'),
                        "deterministic": getattr(speaker_conditioning_info, 'deterministic', True),
                        "sampling_strategy": getattr(speaker_conditioning_info, 'sampling_strategy', 'start'),
                        "fixed_duration_sec": getattr(speaker_conditioning_info, 'fixed_duration_sec', 10.0),
                        # Reference segment timing (PER-SAMPLE from batch_idx)
                        "reference_segment": {
                            "start_sec": sample_start_sec,
                            "end_sec": sample_end_sec,
                        },
                        # Reference audio properties
                        "reference_audio_info": {
                            "sample_rate": ref_audio_sample_rate,
                            "duration_sec": ref_duration_sec,
                            "num_samples": ref_num_frames,
                        },
                        # Batch index for debugging
                        "batch_idx": batch_idx,
                    }

                    # Add source file if available (PER-SAMPLE)
                    if sample_source_file:
                        speaker_metadata["source_file"] = sample_source_file

                    # Add reference text if available (PER-SAMPLE)
                    if sample_ref_text:
                        speaker_metadata["reference_text"] = sample_ref_text
                        speaker_metadata["reference_text_length"] = len(sample_ref_text)

                    # Add speaker embedding statistics if available
                    spk_emb = getattr(speaker_conditioning_info, 'speaker_embedding', None)
                    if spk_emb is not None and isinstance(spk_emb, torch.Tensor):
                        # Store embedding statistics (not the full embedding to save space)
                        # For batch, use the corresponding batch index
                        if spk_emb.dim() == 2 and batch_idx < spk_emb.size(0):
                            emb_sample = spk_emb[batch_idx]
                        elif spk_emb.dim() == 1:
                            emb_sample = spk_emb
                        else:
                            emb_sample = spk_emb.flatten()[:256]  # Fallback

                        speaker_metadata["embedding_stats"] = {
                            "shape": list(emb_sample.shape),
                            "dim": int(emb_sample.numel()),
                            "mean": float(emb_sample.mean().item()),
                            "std": float(emb_sample.std().item()),
                            "min": float(emb_sample.min().item()),
                            "max": float(emb_sample.max().item()),
                            "norm": float(emb_sample.norm().item()),
                        }

                    # Save speaker metadata JSON
                    speaker_meta_path = step_dir / f"sample_{idx:02d}_speaker_metadata.json"
                    with open(speaker_meta_path, "w", encoding="utf-8") as f:
                        json.dump(speaker_metadata, f, ensure_ascii=False, indent=2)
                    result.speaker_metadata_paths.append(speaker_meta_path)

                    # Add to sample_data for combined metadata
                    sample_data["speaker_conditioning"] = speaker_metadata

                    # Save reference audio if available
                    ref_audio = getattr(speaker_conditioning_info, 'reference_audio', None)
                    if ref_audio is not None and isinstance(ref_audio, torch.Tensor):
                        # Reference audio shape: [B, T] or [T]
                        if ref_audio.dim() == 2 and batch_idx < ref_audio.size(0):
                            ref_audio_sample = ref_audio[batch_idx]  # [T]
                        elif ref_audio.dim() == 1:
                            ref_audio_sample = ref_audio
                        else:
                            ref_audio_sample = None

                        if ref_audio_sample is not None and ref_audio_sample.numel() > 0:
                            # Ensure 2D for save_audio_file: [1, T]
                            if ref_audio_sample.dim() == 1:
                                ref_audio_sample = ref_audio_sample.unsqueeze(0)

                            ref_audio_path = step_dir / f"sample_{idx:02d}_reference.{self.audio_format}"

                            # CRITICAL: Use correct sample rate for reference audio
                            # - Train split: 16kHz (from interleaver resampling)
                            # - Eval split: 24kHz (from mimi.decode)
                            actual_duration = ref_audio_sample.size(-1) / ref_audio_sample_rate

                            if self.save_audio_file(
                                ref_audio_sample,
                                ref_audio_path,
                                sample_rate=ref_audio_sample_rate  # Use correct sample rate!
                            ):
                                result.reference_audio_paths.append(ref_audio_path)
                                sample_data["speaker_conditioning"]["reference_audio_path"] = str(ref_audio_path.name)
                                # Update audio info with actual saved values
                                sample_data["speaker_conditioning"]["reference_audio_info"]["saved_sample_rate"] = ref_audio_sample_rate
                                sample_data["speaker_conditioning"]["reference_audio_info"]["saved_num_samples"] = ref_audio_sample.size(-1)
                                sample_data["speaker_conditioning"]["reference_audio_info"]["saved_duration_sec"] = actual_duration
                                logger.debug(
                                    f"[SAMPLE {idx}] Saved reference audio: "
                                    f"sample_rate={ref_audio_sample_rate}Hz, "
                                    f"duration={actual_duration:.2f}s"
                                )

                except Exception as e:
                    result.errors.append(f"Sample {idx} speaker conditioning save error: {e}")
                    logger.warning(f"[SAMPLE {idx}] Speaker conditioning save error: {e}")

            # =================================================================
            # Save metadata JSON with all info
            # =================================================================
            try:
                text_path = step_dir / f"sample_{idx:02d}_text.json"
                with open(text_path, "w", encoding="utf-8") as f:
                    json.dump(sample_data, f, ensure_ascii=False, indent=2)
                result.text_paths.append(text_path)
            except Exception as e:
                result.errors.append(f"Sample {idx} metadata save error: {e}")

            result.num_samples += 1

        # Cleanup old samples
        self._cleanup_old_samples(split)

        # Verify all expected files were created (detect partial save failures)
        # Files per sample: text.json + gt_moshi + pred_moshi + (gt_dialogue + pred_dialogue if stereo)
        # + speaker_metadata.json + reference.wav (if speaker conditioning enabled)
        expected_files_per_sample = 1  # text.json always
        if self.save_audio:
            expected_files_per_sample += 2  # gt_moshi + pred_moshi
            if self.has_user_audio:
                expected_files_per_sample += 2  # gt_dialogue + pred_dialogue (no gt_user)

        # NOTE: Speaker conditioning files (speaker_metadata.json, reference.wav) are optional
        # and depend on whether speaker_conditioning_info contains valid data.
        # We don't add them to expected_files_per_sample to avoid false warnings.
        # Instead, we just log actual file count for debugging.

        actual_files = list(step_dir.glob("*"))
        min_expected = num_samples * expected_files_per_sample
        max_expected = num_samples * (expected_files_per_sample + 2)  # +2 for speaker files

        if len(actual_files) < min_expected:
            logger.warning(
                f"[SAMPLE SAVER] File count mismatch! Expected at least {min_expected}, "
                f"got {len(actual_files)}. Some files may be missing."
            )
        elif len(actual_files) > max_expected:
            logger.debug(
                f"[SAMPLE SAVER] Extra files detected: expected {min_expected}-{max_expected}, "
                f"got {len(actual_files)}."
            )

        # Log summary
        has_speaker_cond = (
            speaker_conditioning_info is not None and
            getattr(speaker_conditioning_info, 'enabled', False)
        )
        speaker_info = ""
        if has_speaker_cond:
            method = getattr(speaker_conditioning_info, 'method', 'none')
            speaker_info = f" | speaker_cond={method}"

        logger.info(
            f"[SAMPLE SAVER] step={step} split={split} | "
            f"samples={result.num_samples} | "
            f"gt_audio={len(result.gt_audio_paths)} | "
            f"pred_audio={len(result.pred_audio_paths)} | "
            f"dialogue={len(result.dialogue_paths)} | "
            f"text={len(result.text_paths)}{speaker_info}"
        )

        if result.errors:
            result.success = False
            for error in result.errors[:3]:  # Limit error logging
                logger.warning(f"  Error: {error}")
            if len(result.errors) > 3:
                logger.warning(f"  ... and {len(result.errors) - 3} more errors")

        return result

    def _cleanup_old_samples(self, split: str):
        """Remove old sample directories to save disk space."""
        if split == "train":
            sample_dirs = self.train_sample_dirs
            max_keep = self.max_samples_per_split
        else:
            sample_dirs = self.valid_sample_dirs
            max_keep = self.max_samples_per_split

        while len(sample_dirs) > max_keep:
            old_dir = sample_dirs.pop(0)
            try:
                if old_dir.exists():
                    shutil.rmtree(old_dir)
                    logger.debug(f"Cleaned up old sample dir: {old_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up {old_dir}: {e}")

    def save_metadata(self, step: int, metrics: Dict[str, Any], split: str = "train"):
        """
        Save training metadata alongside samples.

        Args:
            step: Training step
            metrics: Metrics to save
            split: 'train' or 'valid'
        """
        base_dir = self.train_samples_dir if split == "train" else self.valid_samples_dir
        step_dir = base_dir / f"step_{step:06d}"

        if not step_dir.exists():
            return

        metadata_path = step_dir / "metadata.json"
        metadata = {
            "step": step,
            "split": split,
            "timestamp": datetime.utcnow().isoformat(),
            **metrics,
        }

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save metadata: {e}")

    def get_latest_samples(self, split: str = "train", n: int = 5) -> List[Path]:
        """
        Get paths to the most recent sample directories.

        Args:
            split: 'train' or 'valid'
            n: Number of directories to return

        Returns:
            List of sample directory paths
        """
        if split == "train":
            sample_dirs = self.train_sample_dirs
        else:
            sample_dirs = self.valid_sample_dirs

        return sample_dirs[-n:] if sample_dirs else []
