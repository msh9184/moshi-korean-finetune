"""
Korean Tokenizer Wrapper for Moshi

Provides a SentencePiece-compatible wrapper for HuggingFace tokenizers.
This enables using KLUE BERT (32K WordPiece) with Moshi's core code.

Moshi's interleaver.py expects these methods:
    - encode(text) -> List[int]
    - bos_id() -> int
    - eos_id() -> int

Usage:
    from tools.korean_tokenizer_wrapper import KoreanTokenizerWrapper

    # Load KLUE BERT as SentencePiece-compatible tokenizer
    wrapper = KoreanTokenizerWrapper.from_pretrained("klue/bert-base")

    # Use like SentencePiece
    tokens = wrapper.encode("안녕하세요")
    bos = wrapper.bos_id()
    eos = wrapper.eos_id()
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class KoreanTokenizerWrapper:
    """
    SentencePiece-compatible wrapper for HuggingFace tokenizers.

    Wraps KLUE BERT (or similar) to provide the same interface
    that Moshi's interleaver.py expects from SentencePiece.

    Attributes:
        tokenizer: Underlying HuggingFace tokenizer
        _bos_id: BOS token ID (mapped from [CLS])
        _eos_id: EOS token ID (mapped from [SEP])
        _pad_id: PAD token ID
        _unk_id: UNK token ID
    """

    # Default token mappings for BERT-style tokenizers
    DEFAULT_SPECIAL_TOKENS = {
        "bos": "[CLS]",  # BERT uses [CLS] as sequence start
        "eos": "[SEP]",  # BERT uses [SEP] as sequence end
        "pad": "[PAD]",
        "unk": "[UNK]",
    }

    def __init__(
        self,
        tokenizer,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        unk_token: Optional[str] = None,
    ):
        """
        Initialize wrapper with a HuggingFace tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer instance
            bos_token: Override BOS token (default: [CLS])
            eos_token: Override EOS token (default: [SEP])
            pad_token: Override PAD token (default: [PAD])
            unk_token: Override UNK token (default: [UNK])
        """
        self.tokenizer = tokenizer

        # Map special tokens
        bos = bos_token or self.DEFAULT_SPECIAL_TOKENS["bos"]
        eos = eos_token or self.DEFAULT_SPECIAL_TOKENS["eos"]
        pad = pad_token or self.DEFAULT_SPECIAL_TOKENS["pad"]
        unk = unk_token or self.DEFAULT_SPECIAL_TOKENS["unk"]

        # Get token IDs
        self._bos_id = self._get_token_id(bos, "bos")
        self._eos_id = self._get_token_id(eos, "eos")
        self._pad_id = self._get_token_id(pad, "pad")
        self._unk_id = self._get_token_id(unk, "unk")

        logger.info(f"KoreanTokenizerWrapper initialized:")
        logger.info(f"  Vocab size: {self.GetPieceSize()}")
        logger.info(f"  BOS ID ({bos}): {self._bos_id}")
        logger.info(f"  EOS ID ({eos}): {self._eos_id}")
        logger.info(f"  PAD ID ({pad}): {self._pad_id}")
        logger.info(f"  UNK ID ({unk}): {self._unk_id}")

    def _get_token_id(self, token: str, name: str) -> int:
        """Get token ID with error handling."""
        try:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id and token != "[UNK]":
                logger.warning(f"{name} token '{token}' not found, mapped to UNK")
            return token_id
        except Exception as e:
            logger.error(f"Failed to get {name} token ID for '{token}': {e}")
            return 0

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "KoreanTokenizerWrapper":
        """
        Load wrapper from a pretrained HuggingFace model.

        Args:
            model_name_or_path: HuggingFace model ID or local path
                e.g., "klue/bert-base"
            cache_dir: Directory to cache downloaded files
            **kwargs: Additional arguments for wrapper initialization

        Returns:
            Initialized KoreanTokenizerWrapper
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        logger.info(f"Loading tokenizer from: {model_name_or_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            use_fast=True,  # Use fast tokenizer for better performance
        )

        return cls(tokenizer, **kwargs)

    @classmethod
    def from_local(
        cls,
        tokenizer_dir: str,
        **kwargs,
    ) -> "KoreanTokenizerWrapper":
        """
        Load wrapper from local tokenizer files.

        Args:
            tokenizer_dir: Directory containing tokenizer files
                (vocab.txt, tokenizer.json, tokenizer_config.json)
            **kwargs: Additional arguments for wrapper initialization

        Returns:
            Initialized KoreanTokenizerWrapper
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        tokenizer_path = Path(tokenizer_dir)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

        logger.info(f"Loading tokenizer from local path: {tokenizer_dir}")

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
            use_fast=True,
        )

        return cls(tokenizer, **kwargs)

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
        if enable_sampling:
            logger.debug("enable_sampling is ignored in HuggingFace tokenizer")

        if isinstance(text, str):
            # Single string encoding
            # Note: add_special_tokens=False to match SentencePiece behavior
            # SentencePiece.encode() doesn't add BOS/EOS by default
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
        return len(self.tokenizer)

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
    # Additional utility methods
    # =========================================================================

    def validate_for_moshi(self, required_vocab_size: int = 32000) -> bool:
        """
        Validate that this tokenizer is compatible with Moshi.

        Args:
            required_vocab_size: Expected vocabulary size

        Returns:
            True if compatible, False otherwise
        """
        vocab_size = self.GetPieceSize()

        if vocab_size != required_vocab_size:
            logger.warning(
                f"Vocab size mismatch: got {vocab_size}, expected {required_vocab_size}"
            )
            return False

        if self._bos_id < 0 or self._eos_id < 0:
            logger.error("BOS or EOS token ID is invalid")
            return False

        logger.info(f"Tokenizer validation passed (vocab_size={vocab_size})")
        return True

    def test_korean(self) -> None:
        """Test Korean tokenization quality."""
        test_sentences = [
            "안녕하세요",
            "오늘 날씨가 좋네요",
            "인공지능 기술이 발전하고 있습니다",
            "한국어 대화 모델을 학습합니다",
        ]

        print("=" * 60)
        print("Korean Tokenization Test")
        print("=" * 60)

        for sentence in test_sentences:
            tokens = self.encode(sentence)
            pieces = self.EncodeAsPieces(sentence)
            decoded = self.decode(tokens)

            print(f"\nOriginal: {sentence}")
            print(f"Tokens ({len(tokens)}): {tokens}")
            print(f"Pieces: {pieces}")
            print(f"Decoded: {decoded}")

            if decoded.replace(" ", "") != sentence.replace(" ", ""):
                print("⚠️  Warning: Round-trip mismatch!")

    def __repr__(self) -> str:
        return (
            f"KoreanTokenizerWrapper("
            f"vocab_size={self.GetPieceSize()}, "
            f"bos_id={self._bos_id}, "
            f"eos_id={self._eos_id})"
        )


def load_tokenizer(
    tokenizer_type: str = "default",
    tokenizer_path: Optional[str] = None,
    korean_model: str = "klue/bert-base",
) -> Union["KoreanTokenizerWrapper", "sentencepiece.SentencePieceProcessor"]:
    """
    Load appropriate tokenizer based on type.

    Args:
        tokenizer_type: Tokenizer type to load
            - "default" or "sentencepiece": Native SentencePiece .model file
            - "klue": KLUE BERT tokenizer (32K vocab, uses KoreanTokenizerWrapper)
            - "hf_lm": a Hugging Face causal LM tokenizer (105K vocab, uses HFLMTokenizerWrapper)
        tokenizer_path: Path to local tokenizer file/directory
        korean_model: HuggingFace model ID for Korean tokenizer

    Returns:
        Tokenizer instance (SentencePiece, KoreanTokenizerWrapper, or HFLMTokenizerWrapper)

    Note:
        HFLM tokenizer requires text_card=105900 in Moshi config.
        This is a significant architecture change from the default text_card=32000.
    """
    if tokenizer_type in ("default", "sentencepiece"):
        import sentencepiece as spm
        if tokenizer_path is None:
            raise ValueError("tokenizer_path required for default tokenizer")
        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_path)
        return sp

    elif tokenizer_type == "klue":
        if tokenizer_path and Path(tokenizer_path).exists():
            return KoreanTokenizerWrapper.from_local(tokenizer_path)
        else:
            return KoreanTokenizerWrapper.from_pretrained(korean_model)

    elif tokenizer_type == "hf_lm":
        from tools.hf_lm_tokenizer_wrapper import HFLMTokenizerWrapper
        if tokenizer_path is None:
            raise ValueError("tokenizer_path required for hf_lm tokenizer")
        return HFLMTokenizerWrapper.from_local(tokenizer_path)

    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Korean Tokenizer Wrapper")
    parser.add_argument(
        "--model",
        type=str,
        default="klue/bert-base",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run Korean tokenization test",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate for Moshi compatibility",
    )

    args = parser.parse_args()

    # Load wrapper
    print(f"Loading tokenizer: {args.model}")
    wrapper = KoreanTokenizerWrapper.from_pretrained(args.model)
    print(wrapper)

    if args.test:
        wrapper.test_korean()

    if args.validate:
        is_valid = wrapper.validate_for_moshi()
        print(f"\nMoshi compatibility: {'✅ Valid' if is_valid else '❌ Invalid'}")
