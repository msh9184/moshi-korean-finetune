"""
Dialogue Sample Saver for K-Moshi Training.

This module saves COMPLETE dialogues with BOTH Ground Truth AND Predictions.
It processes entire dialogues in streaming chunks for memory efficiency.

PURPOSE:
    Save complete dialogue samples with model predictions for qualitative evaluation.
    Unlike SampleSaver (60s segments), this saves the ENTIRE dialogue.

Key Features:
- Streaming chunk processing for memory efficiency
- Ground Truth AND Prediction generation (not GT only!)
- Moshi and User audio/text separation
- Model inference on complete dialogue via forward pass
- Unified output format matching SampleSaver

Output Structure:
    samples/{split}/step_{step:06d}/sample_dialogue/
    ├── sample_00_gt_dialogue.wav      # GT Stereo (L=Moshi, R=User)
    ├── sample_00_gt_moshi.wav         # GT Moshi audio
    ├── sample_00_pred_dialogue.wav    # Pred Stereo (L=Pred Moshi, R=GT User)
    ├── sample_00_pred_moshi.wav       # Pred Moshi audio
    ├── sample_00_text.json            # Text comparison (GT vs Pred)
    └── ...
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import sphn
    SPHN_AVAILABLE = True
except ImportError:
    SPHN_AVAILABLE = False
    sphn = None

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None

logger = logging.getLogger("dialogue_sample_saver")

# Mimi codec constants
MIMI_CODEBOOKS = 8
MIMI_FRAME_RATE = 12.5  # 12.5 Hz = 80ms per frame
SAMPLE_RATE = 24000


@dataclass
class DialogueSaveResult:
    """Result of dialogue save operation."""
    path: str
    success: bool = True
    duration_sec: float = 0.0
    moshi_word_count: int = 0
    user_word_count: int = 0
    output_dir: Optional[Path] = None
    errors: List[str] = field(default_factory=list)


class DialogueSampleSaver:
    """
    Saves complete dialogues with Ground Truth AND Predictions.

    This class processes entire dialogues in streaming chunks and generates
    model predictions using the forward pass (teacher forcing).

    Architecture:
    1. Read original stereo WAV file and alignment JSON
    2. Encode audio with Mimi codec in chunks
    3. Run model forward pass to get predictions
    4. Decode both GT and predicted codes
    5. Save unified format: gt_dialogue, gt_moshi, pred_dialogue, pred_moshi, text.json
    """

    def __init__(
        self,
        model,
        mimi,
        tokenizer,
        run_dir: Path,
        text_padding_token_id: int,
        end_of_text_padding_id: int,
        audio_offset: int = 1,
        dep_q: int = 8,
        sample_rate: int = SAMPLE_RATE,
        chunk_duration_sec: float = 60.0,
        max_dialogues_per_split: int = 5,
    ):
        """
        Initialize dialogue sample saver.

        Args:
            model: The LM model (FSDP wrapped) for inference
            mimi: Mimi codec model for audio encoding/decoding
            tokenizer: Text tokenizer
            run_dir: Base directory for saving samples
            text_padding_token_id: Padding token ID for text
            end_of_text_padding_id: End-of-text padding token ID
            audio_offset: Index where audio codebooks start (default 1)
            dep_q: Number of audio codebooks for prediction (default 8)
            sample_rate: Audio sample rate (default 24000)
            chunk_duration_sec: Duration of each processing chunk
            max_dialogues_per_split: Maximum dialogues to save per split (train/valid)
        """
        self.model = model
        self.mimi = mimi
        self.tokenizer = tokenizer
        self.run_dir = Path(run_dir)
        self.text_padding_token_id = text_padding_token_id
        self.end_of_text_padding_id = end_of_text_padding_id
        self.audio_offset = audio_offset
        self.dep_q = dep_q
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.max_dialogues_per_split = max_dialogues_per_split

        # Mimi codebook vocabulary size
        self.codebook_size = 2048

        # Create base samples directory (unified with SampleSaver)
        self.samples_dir = self.run_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        # Track saved dialogues per split
        self.saved_dialogues: Dict[str, List[Path]] = {"train": [], "valid": []}
        self.dialogue_count: Dict[str, int] = {"train": 0, "valid": 0}

        # Track already saved paths to avoid duplicates
        self._saved_paths: set = set()

        logger.info(f"DialogueSampleSaver initialized: {self.samples_dir}")
        logger.info(f"  chunk_duration={chunk_duration_sec}s, max_per_split={max_dialogues_per_split}")
        logger.info(f"  dep_q={dep_q}, audio_offset={audio_offset}")
        logger.info(f"  Outputs to: samples/{{split}}/step_{{step}}/sample_dialogue/")

    def _read_audio(self, path: str) -> Optional[np.ndarray]:
        """
        Read stereo audio file.

        Returns:
            Audio array [2, samples] or None if failed
        """
        try:
            if SPHN_AVAILABLE:
                audio, sr = sphn.read(path)
                if sr != self.sample_rate:
                    logger.warning(f"Sample rate mismatch: {sr} vs {self.sample_rate}")
                return audio
            elif SOUNDFILE_AVAILABLE:
                audio, sr = sf.read(path)
                if audio.ndim == 1:
                    logger.warning(f"Mono audio in {path}, expected stereo")
                    return None
                return audio.T
            else:
                logger.error("No audio library available (sphn or soundfile)")
                return None
        except Exception as e:
            logger.error(f"Failed to read audio {path}: {e}")
            return None

    def _read_alignments(self, audio_path: str) -> Optional[List]:
        """Read alignment JSON for audio file."""
        try:
            audio_path = Path(audio_path)
            json_paths = [
                audio_path.with_suffix(".json"),
                audio_path.parent / f"{audio_path.stem}.json",
                audio_path.parent / "alignments" / f"{audio_path.stem}.json",
            ]

            for json_path in json_paths:
                if json_path.exists():
                    with open(json_path, encoding="utf-8") as f:
                        data = json.load(f)
                        return data.get("alignments", [])

            logger.warning(f"No alignment JSON found for {audio_path}")
            return None

        except Exception as e:
            logger.error(f"Failed to read alignments for {audio_path}: {e}")
            return None

    def _extract_speaker_text(
        self,
        alignments: List,
        speaker_label: str,
    ) -> Tuple[str, List]:
        """Extract text and alignments for a specific speaker."""
        speaker_aligns = [a for a in alignments if a[2] == speaker_label]
        speaker_aligns = sorted(speaker_aligns, key=lambda x: x[1][0])
        full_text = " ".join([a[0] for a in speaker_aligns])
        return full_text, speaker_aligns

    def _encode_audio(self, audio: np.ndarray, source: str = "unknown") -> Optional[torch.Tensor]:
        """
        Encode audio to Mimi codes.

        Args:
            audio: Audio waveform [samples] (mono)
            source: Description for logging

        Returns:
            Codes tensor [1, 8, T] or None
        """
        try:
            with torch.no_grad():
                device = next(self.mimi.parameters()).device

                # Convert to tensor [1, 1, samples]
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
                audio_tensor = audio_tensor.to(device)

                # Encode
                codes = self.mimi.encode(audio_tensor)  # [1, 8, T]
                return codes

        except Exception as e:
            logger.error(f"[{source}] Mimi encode failed: {e}")
            return None

    def _decode_audio(self, codes: torch.Tensor, source: str = "unknown") -> Optional[torch.Tensor]:
        """
        Decode Mimi codes to audio.

        Args:
            codes: Audio codes [1, 8, T] or [8, T]
            source: Description for logging

        Returns:
            Audio waveform [1, samples] or None
        """
        try:
            with torch.no_grad():
                device = next(self.mimi.parameters()).device

                if codes.dim() == 2:
                    codes = codes.unsqueeze(0)

                codes = codes.to(device)

                # Clamp to valid range
                codes = codes.clamp(0, self.codebook_size - 1)

                # Decode
                audio = self.mimi.decode(codes)  # [1, 1, samples]
                return audio.squeeze(0)  # [1, samples]

        except Exception as e:
            logger.error(f"[{source}] Mimi decode failed: {e}")
            return None

    def _decode_text(self, token_ids: torch.Tensor) -> str:
        """Decode text tokens to string."""
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)

        valid_tokens = []
        for tid in token_ids.tolist():
            if tid not in (self.text_padding_token_id, self.end_of_text_padding_id):
                if tid >= 0:
                    valid_tokens.append(tid)

        if not valid_tokens:
            return ""

        try:
            decoded = self.tokenizer.decode(valid_tokens)
            if isinstance(decoded, bytes):
                decoded = decoded.decode("utf-8", errors="replace")
            else:
                decoded = decoded.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            decoded = decoded.replace("\ufffd", "").strip()
            return decoded
        except Exception:
            return ""

    def _create_stereo_dialogue(
        self,
        moshi_audio: torch.Tensor,
        user_audio: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Create stereo dialogue: Left=Moshi, Right=User.

        Args:
            moshi_audio: Moshi waveform [1, T] or [T]
            user_audio: User waveform [1, T] or [T]

        Returns:
            Stereo tensor [2, T] or None
        """
        try:
            if moshi_audio.dim() == 1:
                moshi_audio = moshi_audio.unsqueeze(0)
            if user_audio.dim() == 1:
                user_audio = user_audio.unsqueeze(0)

            moshi_len = moshi_audio.size(-1)
            user_len = user_audio.size(-1)

            if moshi_len != user_len:
                max_len = max(moshi_len, user_len)
                if moshi_len < max_len:
                    moshi_audio = F.pad(moshi_audio, (0, max_len - moshi_len), value=0.0)
                if user_len < max_len:
                    user_audio = F.pad(user_audio, (0, max_len - user_len), value=0.0)

            return torch.cat([moshi_audio, user_audio], dim=0)

        except Exception as e:
            logger.debug(f"Stereo creation error: {e}")
            return None

    def _save_audio_file(self, audio: torch.Tensor, filepath: Path) -> bool:
        """Save audio tensor to file."""
        try:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            audio_np = audio.cpu().numpy()

            if SOUNDFILE_AVAILABLE:
                sf.write(str(filepath), audio_np.T, self.sample_rate)
                return True
            else:
                return False

        except Exception as e:
            logger.debug(f"Audio save error: {e}")
            return False

    def _process_dialogue_with_inference(
        self,
        audio: np.ndarray,
        alignments: List,
        output_dir: Path,
        sample_idx: int,
        main_speaker_label: str = "SPEAKER_MAIN",
    ) -> bool:
        """
        Process dialogue with model inference to generate predictions.

        This method:
        1. Encodes both channels with Mimi
        2. Creates input codes for model (17 codebooks for user stream)
        3. Runs model forward pass to get predictions
        4. Decodes both GT and predicted audio
        5. Saves all outputs in unified format

        Args:
            audio: Stereo audio [2, samples]
            alignments: List of alignments
            output_dir: Directory to save outputs
            sample_idx: Index for file naming
            main_speaker_label: Label for main speaker (Moshi)

        Returns:
            True if successful
        """
        try:
            device = next(self.mimi.parameters()).device
            total_samples = audio.shape[1]
            chunk_samples = int(self.chunk_duration_sec * self.sample_rate)

            # Separate channels: Moshi=Left, User=Right
            moshi_audio_np = audio[0]
            user_audio_np = audio[1]

            # Process in chunks and collect codes
            all_moshi_codes = []
            all_user_codes = []

            for start_sample in range(0, total_samples, chunk_samples):
                end_sample = min(start_sample + chunk_samples, total_samples)

                moshi_chunk = moshi_audio_np[start_sample:end_sample]
                user_chunk = user_audio_np[start_sample:end_sample]

                # Encode each chunk
                moshi_codes = self._encode_audio(moshi_chunk, f"moshi_chunk_{start_sample}")
                user_codes = self._encode_audio(user_chunk, f"user_chunk_{start_sample}")

                if moshi_codes is not None:
                    all_moshi_codes.append(moshi_codes)
                if user_codes is not None:
                    all_user_codes.append(user_codes)

            if not all_moshi_codes or not all_user_codes:
                logger.error("Failed to encode audio chunks")
                return False

            # Concatenate codes along time dimension
            gt_moshi_codes = torch.cat(all_moshi_codes, dim=-1)  # [1, 8, T]
            gt_user_codes = torch.cat(all_user_codes, dim=-1)    # [1, 8, T]

            # Align time dimensions
            T = min(gt_moshi_codes.size(-1), gt_user_codes.size(-1))
            gt_moshi_codes = gt_moshi_codes[:, :, :T]
            gt_user_codes = gt_user_codes[:, :, :T]

            # =================================================================
            # Create full input codes for model (17 codebooks for user stream)
            # Format: [1, 17, T] = [text(1), moshi_audio(8), user_audio(8)]
            # =================================================================
            B = 1

            # Create text placeholder (padding tokens)
            text_codes = torch.full((B, 1, T), self.text_padding_token_id, device=device, dtype=torch.long)

            # Concatenate: text + moshi_audio + user_audio
            full_codes = torch.cat([
                text_codes,                    # [1, 1, T]
                gt_moshi_codes.squeeze(0).unsqueeze(0),  # [1, 8, T]
                gt_user_codes.squeeze(0).unsqueeze(0),   # [1, 8, T]
            ], dim=1)  # [1, 17, T]

            # =================================================================
            # Run model forward pass to get predictions
            # =================================================================
            # Save current training state to restore after inference
            was_training = self.model.training

            with torch.no_grad():
                self.model.eval()
                try:
                    output = self.model(codes=full_codes, condition_tensors=None)

                    # Get predicted tokens via argmax
                    pred_text_codes = output.text_logits.argmax(dim=-1)  # [B, 1, T]
                    pred_audio_codes = output.logits.argmax(dim=-1)       # [B, dep_q, T]

                    # Extract Moshi predictions (first 8 codebooks of audio)
                    pred_moshi_codes = pred_audio_codes[:, :MIMI_CODEBOOKS, :]  # [B, 8, T]
                finally:
                    # CRITICAL: Restore model to training mode if it was training
                    if was_training:
                        self.model.train()

            # =================================================================
            # Decode audio
            # =================================================================
            # GT Moshi audio
            gt_moshi_audio = self._decode_audio(gt_moshi_codes.squeeze(0), "gt_moshi")

            # GT User audio
            gt_user_audio = self._decode_audio(gt_user_codes.squeeze(0), "gt_user")

            # Predicted Moshi audio
            pred_moshi_audio = self._decode_audio(pred_moshi_codes.squeeze(0), "pred_moshi")

            if gt_moshi_audio is None or gt_user_audio is None:
                logger.error("Failed to decode GT audio")
                return False

            # =================================================================
            # Save audio files
            # =================================================================
            prefix = f"sample_{sample_idx:02d}"

            # GT Moshi
            if gt_moshi_audio is not None:
                self._save_audio_file(gt_moshi_audio, output_dir / f"{prefix}_gt_moshi.wav")

            # Pred Moshi
            if pred_moshi_audio is not None:
                self._save_audio_file(pred_moshi_audio, output_dir / f"{prefix}_pred_moshi.wav")

            # GT Dialogue (stereo)
            if gt_moshi_audio is not None and gt_user_audio is not None:
                gt_dialogue = self._create_stereo_dialogue(gt_moshi_audio, gt_user_audio)
                if gt_dialogue is not None:
                    self._save_audio_file(gt_dialogue, output_dir / f"{prefix}_gt_dialogue.wav")

            # Pred Dialogue (stereo: Pred Moshi + GT User)
            if pred_moshi_audio is not None and gt_user_audio is not None:
                pred_dialogue = self._create_stereo_dialogue(pred_moshi_audio, gt_user_audio)
                if pred_dialogue is not None:
                    self._save_audio_file(pred_dialogue, output_dir / f"{prefix}_pred_dialogue.wav")

            # =================================================================
            # Decode and save text
            # =================================================================
            # GT text from alignments
            moshi_gt_text, moshi_aligns = self._extract_speaker_text(alignments, main_speaker_label)

            user_labels = ["SPEAKER_USER", "SPEAKER_OTHER", "SPEAKER_1", "USER"]
            user_gt_text = ""
            user_aligns = []
            for label in user_labels:
                user_gt_text, user_aligns = self._extract_speaker_text(alignments, label)
                if user_aligns:
                    break

            # Predicted text
            pred_moshi_text = self._decode_text(pred_text_codes.squeeze(0).squeeze(0))

            # Create text.json
            text_data = {
                "step": None,  # Will be set in save_dialogue
                "sample_idx": sample_idx,
                "source_path": str(output_dir.parent.parent.name),  # step_XXXXXX
                "timestamp": datetime.utcnow().isoformat(),
                "moshi": {
                    "ground_truth": moshi_gt_text,
                    "prediction": pred_moshi_text,
                    "gt_word_count": len(moshi_aligns),
                },
                "user": {
                    "ground_truth": user_gt_text,
                    "gt_word_count": len(user_aligns),
                },
                "full_dialogue": {
                    "ground_truth": f"[Moshi]: {moshi_gt_text}\n[User]: {user_gt_text}",
                },
                "duration_sec": total_samples / self.sample_rate,
            }

            with open(output_dir / f"{prefix}_text.json", "w", encoding="utf-8") as f:
                json.dump(text_data, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to process dialogue with inference: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_dialogue(
        self,
        audio_path: str,
        step: int,
        main_speaker_label: str = "SPEAKER_MAIN",
        split: str = "train",
    ) -> DialogueSaveResult:
        """
        Save a complete dialogue with GT and predictions.

        Args:
            audio_path: Path to stereo audio file
            step: Current training step
            main_speaker_label: Label for main speaker (Moshi)
            split: "train" or "valid"

        Returns:
            DialogueSaveResult with save details
        """
        result = DialogueSaveResult(path=audio_path)

        # Validate split
        if split not in ("train", "valid"):
            split = "train"

        # Check max dialogues
        if self.dialogue_count[split] >= self.max_dialogues_per_split:
            result.success = False
            result.errors.append(f"Max dialogues for {split} ({self.max_dialogues_per_split}) reached")
            return result

        # Skip duplicates
        if audio_path in self._saved_paths:
            result.success = False
            result.errors.append(f"Already saved: {audio_path}")
            return result

        try:
            # Read audio
            audio = self._read_audio(audio_path)
            if audio is None:
                result.success = False
                result.errors.append("Failed to read audio")
                return result

            if audio.ndim != 2 or audio.shape[0] != 2:
                result.success = False
                result.errors.append(f"Invalid audio shape: {audio.shape}")
                return result

            result.duration_sec = audio.shape[1] / self.sample_rate

            # Read alignments
            alignments = self._read_alignments(audio_path)
            if alignments is None:
                result.success = False
                result.errors.append("Failed to read alignments")
                return result

            # Count words
            moshi_aligns = [a for a in alignments if a[2] == main_speaker_label]
            user_aligns = [a for a in alignments if a[2] != main_speaker_label]
            result.moshi_word_count = len(moshi_aligns)
            result.user_word_count = len(user_aligns)

            # Create output directory: samples/{split}/step_{step}/sample_dialogue/
            step_dir = self.samples_dir / split / f"step_{step:06d}"
            output_dir = step_dir / "sample_dialogue"

            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            result.output_dir = output_dir

            # Get sample index for this dialogue
            sample_idx = self.dialogue_count[split]

            # Process dialogue with model inference
            success = self._process_dialogue_with_inference(
                audio=audio,
                alignments=alignments,
                output_dir=output_dir,
                sample_idx=sample_idx,
                main_speaker_label=main_speaker_label,
            )

            if not success:
                result.success = False
                result.errors.append("Failed to process dialogue")
                return result

            # Update step in text.json files
            for text_file in output_dir.glob("*_text.json"):
                try:
                    with open(text_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["step"] = step
                    with open(text_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            # Track saved dialogue
            self.saved_dialogues[split].append(output_dir)
            self.dialogue_count[split] += 1
            self._saved_paths.add(audio_path)

            logger.info(
                f"[DIALOGUE SAVER] [{split.upper()}] Saved dialogue {self.dialogue_count[split]}/{self.max_dialogues_per_split}: "
                f"step={step}, {result.duration_sec:.1f}s, "
                f"moshi={result.moshi_word_count}w, user={result.user_word_count}w"
            )

            return result

        except Exception as e:
            result.success = False
            result.errors.append(f"Exception: {e}")
            logger.error(f"Failed to save dialogue {audio_path}: {e}")
            return result

    def save_dialogue_from_batch(
        self,
        batch_paths: List[str],
        step: int,
        main_speaker_label: str = "SPEAKER_MAIN",
        max_per_batch: int = 1,
        split: str = "train",
    ) -> List[DialogueSaveResult]:
        """
        Save dialogues from a batch of audio paths.

        Args:
            batch_paths: List of audio file paths
            step: Current training step
            main_speaker_label: Label for main speaker
            max_per_batch: Maximum dialogues to save per batch
            split: "train" or "valid"

        Returns:
            List of DialogueSaveResult
        """
        results = []

        if split not in ("train", "valid"):
            split = "train"

        # Get unique paths
        unique_paths = list(set(batch_paths))

        for i, path in enumerate(unique_paths[:max_per_batch]):
            if self.dialogue_count[split] >= self.max_dialogues_per_split:
                break

            result = self.save_dialogue(
                audio_path=path,
                step=step,
                main_speaker_label=main_speaker_label,
                split=split,
            )
            results.append(result)

        return results

    def get_saved_dialogues(self, split: Optional[str] = None) -> Dict[str, List[Path]]:
        """Get list of saved dialogue directories."""
        if split is not None and split in self.saved_dialogues:
            return {split: self.saved_dialogues[split].copy()}
        return {k: v.copy() for k, v in self.saved_dialogues.items()}

    def cleanup_old_dialogues(self, keep_last: int = 3, split: Optional[str] = None):
        """Remove old dialogue directories to save disk space."""
        splits_to_clean = [split] if split else ["train", "valid"]

        for s in splits_to_clean:
            if s not in self.saved_dialogues:
                continue
            while len(self.saved_dialogues[s]) > keep_last:
                old_dir = self.saved_dialogues[s].pop(0)
                try:
                    if old_dir.exists():
                        shutil.rmtree(old_dir)
                        logger.debug(f"Cleaned up old dialogue: {old_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {old_dir}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about saved dialogues."""
        return {
            "train_count": self.dialogue_count["train"],
            "valid_count": self.dialogue_count["valid"],
            "total_count": self.dialogue_count["train"] + self.dialogue_count["valid"],
            "max_per_split": self.max_dialogues_per_split,
            "saved_paths_count": len(self._saved_paths),
        }
