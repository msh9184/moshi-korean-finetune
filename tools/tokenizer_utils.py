"""
K-Moshi Tokenizer Utilities

Utilities for Korean tokenizer setup and validation:
- SPM tokenizer validation for Moshi compatibility
- Tokenizer comparison and analysis
- Special token ID extraction

Moshi Tokenizer Requirements:
    - Format: SentencePiece (.model file)
    - Vocab Size: 32,000 (32K) - must match embedding layer
    - Special Tokens: Must have BOS, EOS, PAD tokens

Korean Tokenizer Options:
    1. KLUE/RoBERTa-base: 32K vocab, Korean-trained
       - HuggingFace: klue/roberta-base (need conversion to SPM)
    2. KoBERT: 8K vocab (too small, not recommended)
    3. Train custom SPM on Korean dialogue data

Usage:
    # Validate a tokenizer for Moshi compatibility
    python -m tools.tokenizer_utils validate /path/to/tokenizer.model

    # Compare original and Korean tokenizer
    python -m tools.tokenizer_utils compare \
        /path/to/original.model /path/to/korean.model
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import sentencepiece as spm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Moshi's expected tokenizer configuration
MOSHI_VOCAB_SIZE = 32000
MOSHI_SPECIAL_TOKENS = {
    "pad_id": 0,      # Typically PAD
    "unk_id": 1,      # Unknown token
    "bos_id": 2,      # Beginning of sentence (or similar)
    "eos_id": 3,      # End of sentence
    # Note: Token 32000 may be text_padding or end_of_text_padding
}


def load_tokenizer(tokenizer_path: str) -> spm.SentencePieceProcessor:
    """
    Load a SentencePiece tokenizer from file.

    Args:
        tokenizer_path: Path to .model file

    Returns:
        Loaded SentencePieceProcessor
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    return sp


def get_tokenizer_info(sp: spm.SentencePieceProcessor) -> Dict[str, Any]:
    """
    Extract detailed information from a SentencePiece tokenizer.

    Args:
        sp: SentencePieceProcessor instance

    Returns:
        Dictionary with tokenizer information
    """
    info = {
        "vocab_size": sp.GetPieceSize(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "pad_id": sp.pad_id(),
        "unk_id": sp.unk_id(),
    }

    # Get some sample tokens for analysis
    sample_tokens = {}
    for i in range(min(10, sp.GetPieceSize())):
        sample_tokens[i] = sp.IdToPiece(i)

    # Get last few tokens
    for i in range(max(0, sp.GetPieceSize() - 5), sp.GetPieceSize()):
        sample_tokens[i] = sp.IdToPiece(i)

    info["sample_tokens"] = sample_tokens

    return info


def validate_tokenizer_for_moshi(
    sp: spm.SentencePieceProcessor,
    required_vocab_size: int = MOSHI_VOCAB_SIZE,
    strict: bool = True,
) -> tuple[bool, List[str]]:
    """
    Validate that a tokenizer is compatible with Moshi.

    Args:
        sp: SentencePieceProcessor to validate
        required_vocab_size: Expected vocabulary size (default: 32000)
        strict: If True, fail on vocab size mismatch

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    warnings = []

    # Check vocab size
    actual_size = sp.GetPieceSize()
    if actual_size != required_vocab_size:
        msg = f"Vocab size mismatch: got {actual_size}, expected {required_vocab_size}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    # Check special tokens exist
    if sp.bos_id() < 0:
        errors.append("BOS token not defined")
    if sp.eos_id() < 0:
        errors.append("EOS token not defined")
    if sp.unk_id() < 0:
        warnings.append("UNK token not defined (may be intentional)")

    # Check PAD token (Moshi uses model.text_padding_token_id)
    if sp.pad_id() < 0:
        warnings.append("PAD token not explicitly defined (will use model's text_padding)")

    # Log results
    is_valid = len(errors) == 0

    if warnings:
        for w in warnings:
            logger.warning(f"Warning: {w}")

    if errors:
        for e in errors:
            logger.error(f"Error: {e}")

    return is_valid, errors


def compare_tokenizers(
    sp1: spm.SentencePieceProcessor,
    sp2: spm.SentencePieceProcessor,
    test_texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare two tokenizers to understand differences.

    Args:
        sp1: First tokenizer (e.g., original Moshiko)
        sp2: Second tokenizer (e.g., Korean)
        test_texts: Optional list of test texts

    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "tokenizer1": get_tokenizer_info(sp1),
        "tokenizer2": get_tokenizer_info(sp2),
        "vocab_size_diff": sp1.GetPieceSize() - sp2.GetPieceSize(),
    }

    # Test with sample texts if provided
    if test_texts:
        tokenization_comparison = []
        for text in test_texts:
            tokens1 = sp1.EncodeAsIds(text)
            tokens2 = sp2.EncodeAsIds(text)
            pieces1 = sp1.EncodeAsPieces(text)
            pieces2 = sp2.EncodeAsPieces(text)

            tokenization_comparison.append({
                "text": text,
                "tokenizer1_tokens": tokens1,
                "tokenizer1_pieces": pieces1,
                "tokenizer2_tokens": tokens2,
                "tokenizer2_pieces": pieces2,
                "length_diff": len(tokens1) - len(tokens2),
            })
        comparison["tokenization_comparison"] = tokenization_comparison

    return comparison


def get_special_token_ids_to_retain(
    sp: spm.SentencePieceProcessor,
    additional_ids: Optional[List[int]] = None,
) -> List[int]:
    """
    Get list of special token IDs that should be retained during embedding reinitialization.

    These tokens typically have specific meanings that should be preserved
    even when switching tokenizers.

    Args:
        sp: SentencePieceProcessor
        additional_ids: Additional token IDs to retain

    Returns:
        List of token IDs to retain
    """
    retain_ids = []

    # Add defined special tokens
    if sp.bos_id() >= 0:
        retain_ids.append(sp.bos_id())
    if sp.eos_id() >= 0:
        retain_ids.append(sp.eos_id())
    if sp.pad_id() >= 0:
        retain_ids.append(sp.pad_id())
    if sp.unk_id() >= 0:
        retain_ids.append(sp.unk_id())

    # Add additional specified IDs
    if additional_ids:
        retain_ids.extend(additional_ids)

    # Remove duplicates while preserving order
    seen = set()
    unique_ids = []
    for id in retain_ids:
        if id not in seen:
            seen.add(id)
            unique_ids.append(id)

    return unique_ids


def test_korean_tokenization(sp: spm.SentencePieceProcessor) -> None:
    """
    Test Korean text tokenization quality.

    Args:
        sp: SentencePieceProcessor to test
    """
    test_sentences = [
        "안녕하세요",
        "오늘 날씨가 좋네요",
        "인공지능 기술이 발전하고 있습니다",
        "한국어 대화 모델을 학습합니다",
        "네, 알겠습니다. 감사합니다.",
    ]

    logger.info("=" * 60)
    logger.info("Korean Tokenization Test")
    logger.info("=" * 60)

    for sentence in test_sentences:
        tokens = sp.EncodeAsIds(sentence)
        pieces = sp.EncodeAsPieces(sentence)
        decoded = sp.DecodeIds(tokens)

        logger.info(f"\nOriginal: {sentence}")
        logger.info(f"Tokens ({len(tokens)}): {tokens}")
        logger.info(f"Pieces: {pieces}")
        logger.info(f"Decoded: {decoded}")

        if decoded != sentence:
            logger.warning(f"Round-trip mismatch! Original != Decoded")


def main():
    parser = argparse.ArgumentParser(
        description="K-Moshi Tokenizer Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate tokenizer for Moshi compatibility"
    )
    validate_parser.add_argument(
        "tokenizer_path", type=str, help="Path to tokenizer .model file"
    )
    validate_parser.add_argument(
        "--strict", action="store_true", help="Fail on any mismatch"
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two tokenizers"
    )
    compare_parser.add_argument(
        "tokenizer1", type=str, help="First tokenizer path"
    )
    compare_parser.add_argument(
        "tokenizer2", type=str, help="Second tokenizer path"
    )
    compare_parser.add_argument(
        "--test-korean", action="store_true", help="Include Korean test texts"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show tokenizer information"
    )
    info_parser.add_argument(
        "tokenizer_path", type=str, help="Path to tokenizer .model file"
    )

    # Test Korean command
    test_parser = subparsers.add_parser(
        "test-korean", help="Test Korean tokenization"
    )
    test_parser.add_argument(
        "tokenizer_path", type=str, help="Path to tokenizer .model file"
    )

    args = parser.parse_args()

    if args.command == "validate":
        logger.info(f"Validating tokenizer: {args.tokenizer_path}")
        sp = load_tokenizer(args.tokenizer_path)
        is_valid, errors = validate_tokenizer_for_moshi(sp, strict=args.strict)

        info = get_tokenizer_info(sp)
        logger.info(f"Tokenizer info: {json.dumps(info, indent=2, default=str)}")

        if is_valid:
            logger.info("✅ Tokenizer is valid for Moshi!")
        else:
            logger.error("❌ Tokenizer validation failed!")
            exit(1)

    elif args.command == "compare":
        logger.info(f"Comparing tokenizers:")
        logger.info(f"  1: {args.tokenizer1}")
        logger.info(f"  2: {args.tokenizer2}")

        sp1 = load_tokenizer(args.tokenizer1)
        sp2 = load_tokenizer(args.tokenizer2)

        test_texts = None
        if args.test_korean:
            test_texts = [
                "안녕하세요",
                "오늘 날씨가 좋네요",
                "Hello, world!",
            ]

        comparison = compare_tokenizers(sp1, sp2, test_texts)
        logger.info(f"Comparison results:\n{json.dumps(comparison, indent=2, default=str)}")

    elif args.command == "info":
        logger.info(f"Tokenizer info: {args.tokenizer_path}")
        sp = load_tokenizer(args.tokenizer_path)
        info = get_tokenizer_info(sp)
        logger.info(json.dumps(info, indent=2, default=str))

        # Also show special tokens to retain
        retain_ids = get_special_token_ids_to_retain(sp)
        logger.info(f"\nSpecial token IDs to retain: {retain_ids}")

    elif args.command == "test-korean":
        logger.info(f"Testing Korean tokenization: {args.tokenizer_path}")
        sp = load_tokenizer(args.tokenizer_path)
        test_korean_tokenization(sp)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
