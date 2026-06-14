"""
HFLM Tokenizer Wrapper for Moshi

Provides a SentencePiece-compatible wrapper for the HF causal LM tokenizer.
This enables future integration of the HF LM's vocabulary with Moshi's architecture.

HFLM Tokenizer Specifications:
    - Vocab Size: 105,900 tokens
    - Format: BPE (HuggingFace tokenizers)
    - Special Tokens: 53 (IDs 0-52)
    - BOS: <|begin_of_text|> (ID 1)
    - EOS: <|turn_end|> (ID 37), <|end_of_text|> (ID 0)
    - PAD: <|end_of_text|> (ID 0)

Compatibility Notes:
    - Moshi's text_card is 32,000 by default
    - Using HFLM vocab requires text_card=105,900
    - This means resizing text_emb, depformer_text_emb, and linears[0]

Usage:
    from tools.hf_lm_tokenizer_wrapper import HFLMTokenizerWrapper

    # Load from local HFLM model directory
    wrapper = HFLMTokenizerWrapper.from_local(
        '/path/to/model'
    )

    # Use like SentencePiece
    tokens = wrapper.encode("안녕하세요")
    bos = wrapper.bos_id()
    eos = wrapper.eos_id()
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


# HFLM special token mappings
HFLM_SPECIAL_TOKENS = {
    "end_of_text": {"id": 0, "content": "<|end_of_text|>"},
    "begin_of_text": {"id": 1, "content": "<|begin_of_text|>"},
    "fim_prefix": {"id": 2, "content": "<|fim_prefix|>"},
    "fim_middle": {"id": 3, "content": "<|fim_middle|>"},
    "fim_suffix": {"id": 4, "content": "<|fim_suffix|>"},
    "fim_pad": {"id": 5, "content": "<|fim_pad|>"},
    "turn_start": {"id": 36, "content": "<|turn_start|>"},
    "turn_end": {"id": 37, "content": "<|turn_end|>"},
}

# Moshi compatibility mapping
MOSHI_TOKEN_MAPPING = {
    "bos": 1,   # <|begin_of_text|> maps to Moshi's BOS concept
    "eos": 37,  # <|turn_end|> maps to Moshi's EOS concept
    "pad": 0,   # <|end_of_text|> maps to Moshi's PAD concept
    "unk": 0,   # Use end_of_text as UNK (same as the HF LM's config)
}


class HFLMTokenizerWrapper:
    """
    SentencePiece-compatible wrapper for a Hugging Face causal LM tokenizer.

    This wrapper provides the same interface that Moshi's interleaver.py
    expects from SentencePiece, enabling future HFLM LLM integration.

    Attributes:
        tokenizer: Underlying HuggingFace tokenizer
        vocab_size: 105,900 tokens
        _bos_id: BOS token ID (1)
        _eos_id: EOS token ID (37)
        _pad_id: PAD token ID (0)
    """

    VOCAB_SIZE = 105900

    def __init__(
        self,
        tokenizer,
        use_turn_tokens: bool = True,
    ):
        """
        Initialize wrapper with a HuggingFace tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer instance (loaded from HFLM)
            use_turn_tokens: If True, use <|turn_end|> as EOS (ID 37)
                           If False, use <|end_of_text|> as EOS (ID 0)
        """
        self.tokenizer = tokenizer
        self.use_turn_tokens = use_turn_tokens

        # Set up token IDs based on the HF LM's convention
        self._bos_id = MOSHI_TOKEN_MAPPING["bos"]  # 1
        self._eos_id = MOSHI_TOKEN_MAPPING["eos"] if use_turn_tokens else 0
        self._pad_id = MOSHI_TOKEN_MAPPING["pad"]  # 0
        self._unk_id = MOSHI_TOKEN_MAPPING["unk"]  # 0

        # Validate vocab size
        actual_vocab_size = len(tokenizer)
        if actual_vocab_size != self.VOCAB_SIZE:
            logger.warning(
                f"Vocab size mismatch: expected {self.VOCAB_SIZE}, got {actual_vocab_size}"
            )
        self._vocab_size = actual_vocab_size

        logger.info(f"HFLMTokenizerWrapper initialized:")
        logger.info(f"  Vocab size: {self._vocab_size}")
        logger.info(f"  BOS ID: {self._bos_id} (<|begin_of_text|>)")
        logger.info(f"  EOS ID: {self._eos_id} (<|turn_end|> or <|end_of_text|>)")
        logger.info(f"  PAD ID: {self._pad_id} (<|end_of_text|>)")

    @classmethod
    def from_local(
        cls,
        model_dir: str,
        use_turn_tokens: bool = True,
    ) -> "HFLMTokenizerWrapper":
        """
        Load wrapper from local HFLM model directory.

        Args:
            model_dir: Directory containing tokenizer.json and tokenizer_config.json
            use_turn_tokens: If True, use <|turn_end|> as EOS

        Returns:
            Initialized HFLMTokenizerWrapper
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Check for required files
        tokenizer_json = model_path / "tokenizer.json"
        tokenizer_config = model_path / "tokenizer_config.json"

        if not tokenizer_json.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

        logger.info(f"Loading HFLM tokenizer from: {model_dir}")

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            use_fast=True,
            trust_remote_code=True,  # For custom HFLM tokenizer
        )

        return cls(tokenizer, use_turn_tokens=use_turn_tokens)

    # =========================================================================
    # SentencePiece-compatible methods
    # =========================================================================

    def encode(
        self,
        text: Union[str, List[str]],
        enable_sampling: bool = False,
        alpha: float = 0.0,
        nbest_size: int = -1,
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token IDs (SentencePiece-compatible).

        Args:
            text: String or list of strings to encode
            enable_sampling: Ignored (for SentencePiece compatibility)
            alpha: Ignored (for SentencePiece compatibility)
            nbest_size: Ignored (for SentencePiece compatibility)

        Returns:
            List of token IDs, or list of lists if input is a list
        """
        if isinstance(text, str):
            # Single string encoding without special tokens
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # List of strings - encode each separately
            return [
                self.tokenizer.encode(t, add_special_tokens=False)
                for t in text
            ]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def bos_id(self) -> int:
        """Return BOS (Beginning of Sentence) token ID."""
        return self._bos_id

    def eos_id(self) -> int:
        """Return EOS (End of Sentence) token ID."""
        return self._eos_id

    def pad_id(self) -> int:
        """Return PAD token ID."""
        return self._pad_id

    def unk_id(self) -> int:
        """Return UNK (Unknown) token ID."""
        return self._unk_id

    def GetPieceSize(self) -> int:
        """Return vocabulary size (SentencePiece method name)."""
        return self._vocab_size

    def get_piece_size(self) -> int:
        """Return vocabulary size (alternative naming)."""
        return self.GetPieceSize()

    def IdToPiece(self, token_id: int) -> str:
        """Convert token ID to token string."""
        return self.tokenizer.convert_ids_to_tokens(token_id)

    def PieceToId(self, piece: str) -> int:
        """Convert token string to token ID."""
        return self.tokenizer.convert_tokens_to_ids(piece)

    def EncodeAsIds(self, text: str) -> List[int]:
        """Encode text as token IDs (alias for encode)."""
        return self.encode(text)

    def EncodeAsPieces(self, text: str) -> List[str]:
        """Encode text as token pieces/strings."""
        token_ids = self.encode(text)
        return [self.IdToPiece(tid) for tid in token_ids]

    def DecodeIds(self, token_ids: List[int]) -> str:
        """Decode token IDs to text (alias for decode)."""
        return self.decode(token_ids)

    # =========================================================================
    # HFLM-specific methods
    # =========================================================================

    def get_special_token_id(self, token_name: str) -> int:
        """
        Get ID for a HFLM special token.

        Args:
            token_name: One of: end_of_text, begin_of_text, fim_prefix,
                       fim_middle, fim_suffix, fim_pad, turn_start, turn_end

        Returns:
            Token ID
        """
        if token_name not in HFLM_SPECIAL_TOKENS:
            raise ValueError(f"Unknown special token: {token_name}")
        return HFLM_SPECIAL_TOKENS[token_name]["id"]

    def get_turn_tokens(self) -> dict:
        """
        Get turn-based conversation tokens (HFLM-specific).

        Returns:
            Dictionary with turn_start and turn_end token info
        """
        return {
            "turn_start": HFLM_SPECIAL_TOKENS["turn_start"],
            "turn_end": HFLM_SPECIAL_TOKENS["turn_end"],
        }

    # =========================================================================
    # Moshi compatibility methods
    # =========================================================================

    def validate_for_moshi(self, expected_text_card: int = 105900) -> bool:
        """
        Validate that this tokenizer is compatible with modified Moshi.

        Note: the HF LM's vocab (105,900) is larger than Moshi's default (32,000).
        Using HFLM requires modifying Moshi's text_card parameter.

        Args:
            expected_text_card: Expected text_card value (should be 105,900)

        Returns:
            True if compatible, False otherwise
        """
        vocab_size = self.GetPieceSize()

        if vocab_size != expected_text_card:
            logger.warning(
                f"Vocab size ({vocab_size}) doesn't match expected text_card ({expected_text_card}). "
                f"You need to set text_card={vocab_size} in Moshi config."
            )
            return False

        if self._bos_id < 0 or self._eos_id < 0:
            logger.error("BOS or EOS token ID is invalid")
            return False

        logger.info(f"HFLM tokenizer validation passed (vocab_size={vocab_size})")
        logger.info(f"  ⚠️ Remember to set text_card={vocab_size} in Moshi config!")
        return True

    def get_moshi_config_overrides(self) -> dict:
        """
        Get Moshi configuration overrides required for HFLM compatibility.

        Returns:
            Dictionary of config values to override
        """
        return {
            "text_card": self._vocab_size,
            "existing_text_padding_id": self._pad_id,
            # Note: These would require model architecture changes
            # "text_emb": f"Embedding({self._vocab_size + 1}, 4096)",
            # "depformer_text_emb": f"Embedding({self._vocab_size + 1}, 1024)",
        }

    def test_korean(self) -> None:
        """Test Korean tokenization quality."""
        test_sentences = [
            "안녕하세요",
            "오늘 날씨가 좋네요",
            "인공지능 기술이 발전하고 있습니다",
            "한국어 대화 모델을 학습합니다",
            "오픈소스 모델입니다",
        ]

        print("=" * 60)
        print("HFLM Korean Tokenization Test")
        print("=" * 60)

        for sentence in test_sentences:
            tokens = self.encode(sentence)
            pieces = self.EncodeAsPieces(sentence)
            decoded = self.decode(tokens)

            print(f"\nOriginal: {sentence}")
            print(f"Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            print(f"Pieces: {pieces[:10]}{'...' if len(pieces) > 10 else ''}")
            print(f"Decoded: {decoded}")

            if decoded.replace(" ", "") != sentence.replace(" ", ""):
                print("⚠️  Warning: Round-trip mismatch!")

    def __repr__(self) -> str:
        return (
            f"HFLMTokenizerWrapper("
            f"vocab_size={self._vocab_size}, "
            f"bos_id={self._bos_id}, "
            f"eos_id={self._eos_id})"
        )


def get_hf_lm_moshi_compatibility_report(hf_lm_dir: str) -> str:
    """
    Generate a compatibility report for using HFLM with Moshi.

    Args:
        hf_lm_dir: Path to HFLM model directory

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append("HFLM ↔ Moshi Compatibility Report")
    report.append("=" * 70)

    try:
        wrapper = HFLMTokenizerWrapper.from_local(hf_lm_dir)

        report.append(f"\n✅ HFLM tokenizer loaded successfully")
        report.append(f"   Vocab size: {wrapper.GetPieceSize()}")
        report.append(f"   BOS ID: {wrapper.bos_id()}")
        report.append(f"   EOS ID: {wrapper.eos_id()}")
        report.append(f"   PAD ID: {wrapper.pad_id()}")

        report.append("\n⚠️  Required Moshi Configuration Changes:")
        report.append("   1. text_card: 32000 → 105900")
        report.append("   2. Resize text_emb: Embedding(32001, 4096) → Embedding(105901, 4096)")
        report.append("   3. Resize depformer_text_emb: Embedding(32001, 1024) → Embedding(105901, 1024)")
        report.append("   4. Resize linears[0]: Linear(1024, 32001) → Linear(1024, 105901)")

        report.append("\n📊 Memory Impact:")
        report.append("   Additional memory: ~455 MB (bf16)")
        report.append("   - text_emb: +303 MB")
        report.append("   - depformer_text_emb: +76 MB")
        report.append("   - linears[0]: +76 MB")

        report.append("\n🔧 Implementation Steps:")
        report.append("   1. Create HFLMTokenizerWrapper (this file)")
        report.append("   2. Modify init_korean_moshi.py to resize embeddings")
        report.append("   3. Update TrainArgs to support text_card override")
        report.append("   4. Reinitialize text embeddings for new vocab")

    except Exception as e:
        report.append(f"\n❌ Error loading HFLM tokenizer: {e}")

    report.append("\n" + "=" * 70)
    return "\n".join(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test HFLM Tokenizer Wrapper")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/path/to/model",
        help="Path to HFLM model directory",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run Korean tokenization test",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate compatibility report",
    )

    args = parser.parse_args()

    if args.report:
        print(get_hf_lm_moshi_compatibility_report(args.model_dir))
    else:
        print(f"Loading tokenizer: {args.model_dir}")
        wrapper = HFLMTokenizerWrapper.from_local(args.model_dir)
        print(wrapper)

        if args.test:
            wrapper.test_korean()

        wrapper.validate_for_moshi()
