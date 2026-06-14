# Copyright (c) 2026 Sunghwan Mun. Licensed under the Apache License, Version 2.0.
"""Supertonic-2 ONNX Inference Wrapper for K-Moshi.

Supertonic-2 is a Korean TTS model from Supertone (Naver).
It supports 6 voice styles and runs on ONNX runtime.

Installation:
    pip install onnxruntime-gpu numpy soundfile

Model Download:
    git lfs install
    git clone https://huggingface.co/Supertone/supertonic-2 /models/supertonic-2

Usage:
    tts = SupertonicTTS(model_dir="/models/supertonic-2", voice_style="F4")
    audio = tts.synthesize("안녕하세요, 케이모시입니다.")
"""

from pathlib import Path
from typing import Optional, Literal, Union
from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger(__name__)


# Voice style type
VoiceStyle = Literal["M3", "M4", "M5", "F3", "F4", "F5"]


@dataclass
class VoiceInfo:
    """Voice style information."""
    style: VoiceStyle
    gender: str
    age: str
    description: str


# Available voice styles
VOICE_STYLES: dict[VoiceStyle, VoiceInfo] = {
    "M3": VoiceInfo("M3", "male", "middle", "중년 남성, 안정적"),
    "M4": VoiceInfo("M4", "male", "young", "청년 남성, 활기"),
    "M5": VoiceInfo("M5", "male", "young", "청년 남성, 부드러움"),
    "F3": VoiceInfo("F3", "female", "middle", "중년 여성, 따뜻함"),
    "F4": VoiceInfo("F4", "female", "young", "청년 여성, 밝음"),
    "F5": VoiceInfo("F5", "female", "young", "청년 여성, 차분함"),
}


class SupertonicTTS:
    """Supertonic-2 ONNX-based TTS Engine.

    This class provides a Python interface to the Supertonic-2 TTS model,
    which runs on ONNX runtime for efficient inference.

    Attributes:
        model_dir: Path to the Supertonic-2 model directory.
        voice_style: Current voice style (M3-F5).
        sample_rate: Output sample rate (24000 Hz).
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str = "cuda",
        voice_style: VoiceStyle = "F4",
    ):
        """Initialize Supertonic-2 TTS.

        Args:
            model_dir: Path to supertonic-2 model directory.
            device: "cuda" or "cpu".
            voice_style: One of M3, M4, M5, F3, F4, F5.

        Raises:
            FileNotFoundError: If model files are not found.
            RuntimeError: If ONNX runtime initialization fails.
        """
        self.model_dir = Path(model_dir)
        self.voice_style = voice_style
        self.sample_rate = 24000
        self.device = device

        # Validate model directory
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # Initialize ONNX sessions
        self._init_onnx_sessions()

        # Load speaker embedding
        self._load_speaker_embedding()

        logger.info(
            f"SupertonicTTS initialized: device={device}, voice={voice_style}"
        )

    def _init_onnx_sessions(self):
        """Initialize ONNX inference sessions."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime-gpu"
            )

        # Select execution providers
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Model paths (structure may vary - adjust based on actual model)
        models_dir = self.model_dir / "assets" / "models"
        if not models_dir.exists():
            # Try alternative structure
            models_dir = self.model_dir / "models"
        if not models_dir.exists():
            models_dir = self.model_dir

        # Find ONNX files
        onnx_files = list(models_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX files found in {models_dir}")

        logger.info(f"Found ONNX files: {[f.name for f in onnx_files]}")

        # Load models based on available files
        self._sessions = {}
        for onnx_file in onnx_files:
            name = onnx_file.stem.lower()
            try:
                self._sessions[name] = ort.InferenceSession(
                    str(onnx_file),
                    sess_options,
                    providers=providers,
                )
                logger.debug(f"Loaded ONNX model: {name}")
            except Exception as e:
                logger.warning(f"Failed to load {onnx_file}: {e}")

        if not self._sessions:
            raise RuntimeError("No ONNX models could be loaded")

    def _load_speaker_embedding(self):
        """Load speaker embedding for selected voice style."""
        # Try multiple possible paths for speaker embeddings
        possible_paths = [
            self.model_dir / "assets" / "speakers" / f"{self.voice_style}.npy",
            self.model_dir / "speakers" / f"{self.voice_style}.npy",
            self.model_dir / f"speaker_{self.voice_style}.npy",
        ]

        for speaker_path in possible_paths:
            if speaker_path.exists():
                self.speaker_embedding = np.load(speaker_path)
                logger.info(f"Loaded speaker embedding: {speaker_path}")
                return

        # If no pre-computed embedding found, use default
        logger.warning(
            f"Speaker embedding not found for {self.voice_style}, using default"
        )
        self.speaker_embedding = None

    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speed: float = 1.0,
    ) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Korean text to synthesize.
            output_path: Optional path to save WAV file.
            speed: Speaking speed (0.5-2.0).

        Returns:
            Audio waveform as numpy array (24kHz, mono, float32).

        Raises:
            ValueError: If text is empty or speed is out of range.
            RuntimeError: If synthesis fails.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if not 0.5 <= speed <= 2.0:
            raise ValueError(f"Speed must be between 0.5 and 2.0, got {speed}")

        # Encode text
        text_encoded = self._encode_text(text)

        # Run inference pipeline
        try:
            audio = self._run_inference(text_encoded, speed)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise RuntimeError(f"TTS synthesis failed: {e}")

        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        # Save if output path provided
        if output_path:
            self._save_audio(audio, output_path)

        return audio.astype(np.float32)

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode Korean text to model input format.

        This is a placeholder implementation. The actual encoding
        depends on the specific model's text frontend (grapheme,
        phoneme, or character-based).
        """
        # Simple character-level encoding (placeholder)
        # Actual implementation would use the model's specific tokenizer
        chars = list(text)
        # Pad to minimum length
        while len(chars) < 10:
            chars.append(" ")

        # Convert to indices (placeholder - actual vocab depends on model)
        encoded = np.array(
            [ord(c) % 256 for c in chars], dtype=np.int64
        ).reshape(1, -1)

        return encoded

    def _run_inference(
        self,
        text_encoded: np.ndarray,
        speed: float,
    ) -> np.ndarray:
        """Run the TTS inference pipeline.

        The actual pipeline structure depends on the model architecture.
        Common patterns: Encoder → Decoder → Vocoder
        """
        # This is a simplified placeholder
        # Actual implementation depends on model structure

        # Check what sessions are available
        if "encoder" in self._sessions and "decoder" in self._sessions:
            return self._run_encoder_decoder_pipeline(text_encoded, speed)
        elif "tts" in self._sessions or "model" in self._sessions:
            return self._run_single_model_pipeline(text_encoded, speed)
        else:
            # Use first available session
            session_name = list(self._sessions.keys())[0]
            logger.warning(f"Using fallback session: {session_name}")
            return self._run_fallback_pipeline(text_encoded, speed, session_name)

    def _run_encoder_decoder_pipeline(
        self,
        text_encoded: np.ndarray,
        speed: float,
    ) -> np.ndarray:
        """Run encoder-decoder-vocoder pipeline."""
        # Encoder
        encoder_inputs = {"text": text_encoded}
        if self.speaker_embedding is not None:
            encoder_inputs["speaker"] = self.speaker_embedding

        encoder_output = self._sessions["encoder"].run(
            None, encoder_inputs
        )[0]

        # Decoder
        decoder_inputs = {
            "encoder_output": encoder_output,
            "speed": np.array([speed], dtype=np.float32),
        }
        mel_output = self._sessions["decoder"].run(
            None, decoder_inputs
        )[0]

        # Vocoder
        if "vocoder" in self._sessions:
            audio = self._sessions["vocoder"].run(
                None, {"mel": mel_output}
            )[0].squeeze()
        else:
            # No vocoder - return mel (shouldn't happen in practice)
            audio = mel_output.squeeze()

        return audio

    def _run_single_model_pipeline(
        self,
        text_encoded: np.ndarray,
        speed: float,
    ) -> np.ndarray:
        """Run single model end-to-end pipeline."""
        session_name = "tts" if "tts" in self._sessions else "model"
        session = self._sessions[session_name]

        # Get input names
        input_names = [inp.name for inp in session.get_inputs()]
        logger.debug(f"Model input names: {input_names}")

        # Build inputs
        inputs = {}
        for name in input_names:
            if "text" in name.lower():
                inputs[name] = text_encoded
            elif "speaker" in name.lower() and self.speaker_embedding is not None:
                inputs[name] = self.speaker_embedding
            elif "speed" in name.lower():
                inputs[name] = np.array([speed], dtype=np.float32)

        audio = session.run(None, inputs)[0].squeeze()
        return audio

    def _run_fallback_pipeline(
        self,
        text_encoded: np.ndarray,
        speed: float,
        session_name: str,
    ) -> np.ndarray:
        """Fallback pipeline for unknown model structure."""
        session = self._sessions[session_name]

        # Get input info
        inputs = {}
        for inp in session.get_inputs():
            if inp.type == "tensor(int64)":
                inputs[inp.name] = text_encoded
            elif inp.type == "tensor(float)":
                shape = inp.shape
                if shape and len(shape) == 1:
                    inputs[inp.name] = np.array([speed], dtype=np.float32)
                else:
                    inputs[inp.name] = np.zeros(
                        [1] + [s if isinstance(s, int) else 1 for s in shape[1:]],
                        dtype=np.float32,
                    )

        audio = session.run(None, inputs)[0].squeeze()
        return audio

    def _save_audio(self, audio: np.ndarray, output_path: Union[str, Path]):
        """Save audio to WAV file."""
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("soundfile not installed. Run: pip install soundfile")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, self.sample_rate)
        logger.info(f"Saved audio: {output_path}")

    def change_voice(self, voice_style: VoiceStyle):
        """Change voice style.

        Args:
            voice_style: New voice style (M3-F5).
        """
        if voice_style not in VOICE_STYLES:
            raise ValueError(
                f"Invalid voice style: {voice_style}. "
                f"Choose from: {list(VOICE_STYLES.keys())}"
            )

        self.voice_style = voice_style
        self._load_speaker_embedding()
        logger.info(f"Changed voice to: {voice_style}")

    @classmethod
    def list_voices(cls) -> dict[VoiceStyle, VoiceInfo]:
        """List available voice styles.

        Returns:
            Dictionary of voice styles and their information.
        """
        return VOICE_STYLES.copy()

    def get_voice_info(self) -> VoiceInfo:
        """Get information about current voice.

        Returns:
            VoiceInfo for the current voice style.
        """
        return VOICE_STYLES[self.voice_style]


# Convenience function
def synthesize_korean(
    text: str,
    model_dir: Union[str, Path],
    voice_style: VoiceStyle = "F4",
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Convenience function for one-off synthesis.

    Args:
        text: Korean text to synthesize.
        model_dir: Path to Supertonic-2 model.
        voice_style: Voice style (default: F4).
        output_path: Optional output path for WAV file.

    Returns:
        Audio waveform as numpy array.
    """
    tts = SupertonicTTS(model_dir, voice_style=voice_style)
    return tts.synthesize(text, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Supertonic-2 TTS")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument(
        "--model-dir",
        default="/models/supertonic-2",
        help="Model directory",
    )
    parser.add_argument(
        "--voice",
        choices=list(VOICE_STYLES.keys()),
        default="F4",
        help="Voice style",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output WAV file",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speaking speed (0.5-2.0)",
    )

    args = parser.parse_args()

    # List voices if requested
    if args.text.lower() == "list":
        print("Available voices:")
        for style, info in VOICE_STYLES.items():
            print(f"  {style}: {info.description}")
    else:
        # Synthesize
        tts = SupertonicTTS(
            model_dir=args.model_dir,
            voice_style=args.voice,
        )
        audio = tts.synthesize(
            args.text,
            output_path=args.output,
            speed=args.speed,
        )
        print(f"Generated {len(audio)/24000:.2f}s of audio")
        if args.output:
            print(f"Saved to: {args.output}")
