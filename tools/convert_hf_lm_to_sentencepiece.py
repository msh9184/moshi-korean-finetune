#!/usr/bin/env python3
"""
HFLM BBPE (Byte-level BPE) Tokenizer -> SentencePiece .model Converter

This script converts the HF causal LM's HuggingFace tokenizer.json (BBPE format)
to SentencePiece .model (protobuf format) for compatibility with Moshi's Rust server.

ARCHITECTURE OVERVIEW:
======================

HFLM BBPE (tokenizer.json)           SentencePiece BPE (.model)
================================       ================================
{                                      ModelProto {
  "model": {                             pieces: [
    "vocab": {                             SentencePiece {
      "<|end_of_text|>": 0,                  piece: "<|end_of_text|>",
      "<|begin_of_text|>": 1,                score: 0.0,
      ...                                    type: CONTROL
      "<0x00>": 53,    <-- BYTE TOKENS       },
      "<0x01>": 54,        COMPATIBLE!       SentencePiece {
      ...                                      piece: "<0x00>",
      "hello": 1000,                           score: 0.0,
    },                                         type: BYTE
    "merges": [                              },
      "h e",           <-- MERGE ORDER       ...
      "he l",              TO SCORES       ]
      "hel lo",                            trainer_spec {
      ...                                    model_type: BPE,
    ]                                        byte_fallback: true,
  },                                         ...
  "added_tokens": [...]                    }
}                                        }

KEY CONVERSION DECISIONS:
=========================

1. BYTE TOKENS: the HF LM's <0xHH> format matches SentencePiece's BYTE piece format
   - This is a fortunate compatibility; no byte mapping conversion needed

2. SCORE CALCULATION: BPE merge order -> SentencePiece scores
   - Earlier merge = higher priority = more negative score
   - score = -(total_merges - merge_index)
   - Base vocabulary tokens get score = 0.0

3. NORMALIZATION: Disabled (identity)
   - SentencePiece default is NFKC which would change tokenization
   - We set add_dummy_prefix=False, remove_extra_whitespaces=False

4. SPECIAL TOKENS: Mapped to CONTROL type
   - HFLM has 53 special tokens (IDs 0-52)
   - BOS=1, EOS=37, PAD=0 for Moshi compatibility

PERFORMANCE CONSIDERATIONS:
===========================

Expected differences after conversion:
- Identical tokenization for most inputs
- Slight differences possible due to:
  * Score quantization (float precision)
  * Tie-breaking in merge selection
  * Any normalization edge cases

References:
    - SentencePiece proto: https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
    - HuggingFace tokenizers: https://github.com/huggingface/tokenizers
    - HFLM model: any Hugging Face causal LM

Usage:
    python convert_hf_lm_to_sentencepiece.py \\
        --input /path/to/HF-causal-LM/tokenizer.json \\
        --output /path/to/hf_lm.model \\
        --validate

Author: K-Moshi Project
License: Apache 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SentencePiece Protocol Buffer Constants
# =============================================================================

class ModelType:
    """SentencePiece TrainerSpec.ModelType enum values."""
    UNIGRAM = 1
    BPE = 2
    WORD = 3
    CHAR = 4


class PieceType:
    """SentencePiece ModelProto.SentencePiece.Type enum values."""
    NORMAL = 1       # Normal vocabulary token
    UNKNOWN = 2      # Unknown token (<unk>)
    CONTROL = 3      # Control tokens (BOS, EOS, PAD, etc.)
    USER_DEFINED = 4 # User-defined placeholder tokens
    UNUSED = 5       # Unused vocabulary slot
    BYTE = 6         # Byte tokens for byte_fallback mode (<0xHH>)


# =============================================================================
# HFLM Special Token Configuration
# =============================================================================

# HFLM special tokens (IDs 0-52)
# These map to CONTROL type in SentencePiece
HFLM_CONTROL_TOKENS = {
    0: "<|end_of_text|>",    # PAD token
    1: "<|begin_of_text|>",  # BOS token
    2: "<|fim_prefix|>",
    3: "<|fim_middle|>",
    4: "<|fim_suffix|>",
    5: "<|fim_pad|>",
    36: "<|turn_start|>",
    37: "<|turn_end|>",      # EOS token for Moshi
}

# Moshi-compatible token ID mapping
# Note: HFLM uses <|end_of_text|> (ID 0) as both PAD and UNK token
# This is valid because byte_fallback ensures no actual unknown tokens occur,
# but SentencePiece still requires unk_id to be defined
MOSHI_TOKEN_IDS = {
    "bos": 1,   # <|begin_of_text|>
    "eos": 37,  # <|turn_end|>
    "pad": 0,   # <|end_of_text|>
    "unk": 0,   # <|end_of_text|> - same as PAD, per HFLM tokenizer_config.json
}

# Byte token pattern regex for SentencePiece style (<0xHH>)
BYTE_TOKEN_PATTERN = re.compile(r'^<0x([0-9A-Fa-f]{2})>$')


def bytes_to_unicode() -> dict[int, str]:
    """
    Returns GPT-2 style byte-to-unicode mapping.

    This maps bytes to unicode strings, avoiding whitespace/control characters.
    Used by GPT-2, RoBERTa, and many modern LLMs including HFLM.
    """
    # Printable ASCII and extended Latin characters that map to themselves
    bs = (
        list(range(ord("!"), ord("~") + 1)) +  # 33-126
        list(range(ord("¡"), ord("¬") + 1)) +  # 161-172
        list(range(ord("®"), ord("ÿ") + 1))    # 174-255
    )
    cs = bs[:]
    n = 0
    # Map remaining bytes (0-32, 127-160, 173) to chars starting at 256
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def unicode_to_bytes() -> dict[str, int]:
    """Inverse of bytes_to_unicode()."""
    return {v: k for k, v in bytes_to_unicode().items()}


# Pre-compute GPT-2 byte mappings
GPT2_BYTE_TO_UNICODE = bytes_to_unicode()
GPT2_UNICODE_TO_BYTE = unicode_to_bytes()
GPT2_BYTE_CHARS = set(GPT2_BYTE_TO_UNICODE.values())


# =============================================================================
# Data Classes for Tokenizer Components
# =============================================================================

@dataclass
class TokenizerConfig:
    """Configuration extracted from HFLM tokenizer.json."""
    vocab: dict[str, int] = field(default_factory=dict)
    merges: list[str] = field(default_factory=list)
    added_tokens: list[dict[str, Any]] = field(default_factory=list)
    vocab_size: int = 0

    def validate(self) -> bool:
        """Validate the extracted configuration."""
        if not self.vocab:
            logger.error("Empty vocabulary")
            return False
        if self.vocab_size != len(self.vocab):
            logger.warning(
                f"Vocab size mismatch: declared={self.vocab_size}, actual={len(self.vocab)}"
            )
        return True


@dataclass
class ConversionResult:
    """Result of the conversion process."""
    success: bool
    output_path: str | None = None
    vocab_size: int = 0
    num_merges: int = 0
    num_special_tokens: int = 0
    num_byte_tokens: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Tokenizer Loading and Parsing
# =============================================================================

def load_tokenizer_json(path: str | Path) -> TokenizerConfig:
    """
    Load and parse HFLM tokenizer.json file.

    Args:
        path: Path to tokenizer.json

    Returns:
        TokenizerConfig with extracted data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        KeyError: If required fields are missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")

    logger.info(f"Loading tokenizer from: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract model section
    model = data.get("model", {})
    if not model:
        raise KeyError("Missing 'model' section in tokenizer.json")

    vocab = model.get("vocab", {})
    merges = model.get("merges", [])
    added_tokens = data.get("added_tokens", [])

    config = TokenizerConfig(
        vocab=vocab,
        merges=merges,
        added_tokens=added_tokens,
        vocab_size=len(vocab)
    )

    logger.info(f"Loaded tokenizer config:")
    logger.info(f"  - Vocabulary size: {len(vocab)}")
    logger.info(f"  - Number of merges: {len(merges)}")
    logger.info(f"  - Added tokens: {len(added_tokens)}")

    # Log merge format for debugging
    if merges:
        sample_merge = merges[0]
        if isinstance(sample_merge, list):
            logger.info(f"  - Merge format: list (e.g., {sample_merge})")
        elif isinstance(sample_merge, str):
            logger.info(f"  - Merge format: string (e.g., '{sample_merge}')")
        else:
            logger.warning(f"  - Merge format: unknown ({type(sample_merge)})")

    # Log some sample tokens for debugging
    sample_tokens = list(vocab.keys())[:10]
    logger.info(f"  - Sample tokens: {sample_tokens}")

    return config


def is_gpt2_byte_token(token: str) -> bool:
    """
    Check if a token is a single GPT-2 style byte character.

    GPT-2 style BBPE uses single Unicode characters to represent bytes.
    These are the 256 characters from the bytes_to_unicode() mapping.
    """
    if len(token) != 1:
        return False
    return token in GPT2_BYTE_CHARS


def gpt2_token_to_sp_byte(token: str) -> str | None:
    """
    Convert a GPT-2 style byte token to SentencePiece <0xHH> format.

    Args:
        token: Single character GPT-2 byte token

    Returns:
        SentencePiece byte token format, or None if not a byte token
    """
    if len(token) != 1 or token not in GPT2_UNICODE_TO_BYTE:
        return None
    byte_val = GPT2_UNICODE_TO_BYTE[token]
    return f"<0x{byte_val:02X}>"


def classify_token(
    token: str,
    token_id: int,
    special_ids: set[int],
    unk_id: int = 0
) -> int:
    """
    Classify a token into SentencePiece piece type.

    Args:
        token: Token string
        token_id: Token ID in vocabulary
        special_ids: Set of special token IDs
        unk_id: UNK token ID (default: 0, the <|end_of_text|> token)

    Returns:
        PieceType enum value
    """
    # CRITICAL: UNK token must be classified as UNKNOWN type
    # SentencePiece requires exactly one piece with type=UNKNOWN at unk_id position
    if token_id == unk_id and unk_id >= 0:
        return PieceType.UNKNOWN

    # Check if it's a known control token (excluding UNK which is handled above)
    if token_id in HFLM_CONTROL_TOKENS:
        return PieceType.CONTROL

    # Check if it's a special/added token
    if token_id in special_ids:
        return PieceType.USER_DEFINED

    # Check if it's a byte token (<0xHH> format)
    if BYTE_TOKEN_PATTERN.match(token):
        return PieceType.BYTE

    # Check if it's a GPT-2 style single-byte token
    if is_gpt2_byte_token(token):
        return PieceType.BYTE

    # Default to normal token
    return PieceType.NORMAL


def compute_scores(
    vocab: dict[str, int],
    merges: list[str]
) -> dict[str, float]:
    """
    Compute SentencePiece scores from BPE merge order.

    In BPE, merges are applied in order. The first merge has highest priority.
    In SentencePiece, lower score = higher priority.

    Algorithm:
    1. Base vocabulary tokens (not from merges) get score = 0.0
    2. Merged tokens get score = -(total_merges - merge_index)
       - First merge: score = -N (highest priority)
       - Last merge: score = -1 (lowest priority)

    Args:
        vocab: Token -> ID mapping
        merges: List of merge rules ("token1 token2")

    Returns:
        Token -> score mapping
    """
    scores: dict[str, float] = {}
    total_merges = len(merges)

    # Initialize all tokens with base score
    for token in vocab:
        scores[token] = 0.0

    # Build merge mapping: result_token -> merge_index
    merge_results: dict[str, int] = {}

    for idx, merge in enumerate(merges):
        # Handle both formats:
        # - String format: "token1 token2"
        # - List format: ["token1", "token2"]
        if isinstance(merge, list):
            # List format (HFLM style)
            if len(merge) == 2:
                parts = merge
            else:
                continue
        elif isinstance(merge, str):
            # String format (standard BPE)
            parts = merge.split(" ", 1)
            if len(parts) != 2:
                continue
        else:
            continue

        # Merge parts[0] + parts[1] -> merged_token
        merged_token = parts[0] + parts[1]
        if merged_token in vocab:
            merge_results[merged_token] = idx

    # Assign scores based on merge order
    for token, merge_idx in merge_results.items():
        # Earlier merge (smaller idx) = higher priority = more negative score
        scores[token] = float(-(total_merges - merge_idx))

    logger.info(f"Computed scores for {len(merge_results)} merged tokens")
    logger.info(f"  - Total merges processed: {total_merges}")
    logger.info(f"  - Merges found in vocab: {len(merge_results)}")
    logger.info(f"  - Score range: [{-total_merges}, -1] for merges")
    logger.info(f"  - Base tokens (score=0.0): {len(vocab) - len(merge_results)}")

    return scores


# =============================================================================
# SentencePiece Model Building
# =============================================================================

def build_sentencepiece_model_protobuf(
    config: TokenizerConfig,
    scores: dict[str, float],
    bos_id: int = 1,
    eos_id: int = 37,
    pad_id: int = 0,
    unk_id: int = 0
) -> bytes:
    """
    Build SentencePiece model using protobuf.

    This requires the sentencepiece package with protobuf support.

    Args:
        config: Tokenizer configuration
        scores: Token -> score mapping
        bos_id: BOS token ID
        eos_id: EOS token ID
        pad_id: PAD token ID
        unk_id: UNK token ID (default: 0, <|end_of_text|>)

    Returns:
        Serialized protobuf bytes
    """
    try:
        from sentencepiece import sentencepiece_model_pb2 as sp_pb2
    except ImportError:
        raise ImportError(
            "SentencePiece protobuf module not found.\n"
            "Install with: pip install sentencepiece protobuf"
        )

    model = sp_pb2.ModelProto()

    # Configure TrainerSpec
    trainer = model.trainer_spec
    trainer.model_type = ModelType.BPE
    trainer.vocab_size = len(config.vocab)
    trainer.bos_id = bos_id
    trainer.eos_id = eos_id
    trainer.pad_id = pad_id
    trainer.unk_id = unk_id
    trainer.byte_fallback = True  # Enable byte fallback for OOV handling

    # Configure NormalizerSpec (disable normalization for exact tokenization)
    normalizer = model.normalizer_spec
    normalizer.name = "identity"  # No normalization
    normalizer.add_dummy_prefix = False
    normalizer.remove_extra_whitespaces = False
    normalizer.escape_whitespaces = False

    # Collect special token IDs
    special_ids = {
        at["id"] for at in config.added_tokens
        if at.get("special", False)
    }

    # Sort vocabulary by ID and create pieces
    sorted_vocab = sorted(config.vocab.items(), key=lambda x: x[1])

    stats = {
        "unknown": 0,
        "control": 0,
        "user_defined": 0,
        "byte": 0,
        "byte_added": 0,  # Track added missing byte tokens
        "normal": 0
    }

    # Track which byte values we've seen (0-255)
    seen_byte_values: set[int] = set()

    for token, token_id in sorted_vocab:
        piece = sp_pb2.ModelProto.SentencePiece()
        piece.score = scores.get(token, 0.0)
        piece.type = classify_token(token, token_id, special_ids, unk_id=unk_id)

        # Convert GPT-2 style byte tokens to SentencePiece format
        if piece.type == PieceType.BYTE and is_gpt2_byte_token(token):
            sp_byte_token = gpt2_token_to_sp_byte(token)
            if sp_byte_token:
                piece.piece = sp_byte_token
                # Track the byte value we've seen
                byte_val = GPT2_UNICODE_TO_BYTE.get(token)
                if byte_val is not None:
                    seen_byte_values.add(byte_val)
            else:
                piece.piece = token
        else:
            piece.piece = token

        # Update statistics
        if piece.type == PieceType.UNKNOWN:
            stats["unknown"] += 1
        elif piece.type == PieceType.CONTROL:
            stats["control"] += 1
        elif piece.type == PieceType.USER_DEFINED:
            stats["user_defined"] += 1
        elif piece.type == PieceType.BYTE:
            stats["byte"] += 1
        else:
            stats["normal"] += 1

        model.pieces.append(piece)

    # CRITICAL: SentencePiece requires exactly 256 byte pieces for byte_fallback=True
    # Add any missing byte tokens (GPT-2 BBPE may not have all 256 bytes in vocab)
    missing_bytes = set(range(256)) - seen_byte_values
    if missing_bytes:
        logger.info(f"  - Adding {len(missing_bytes)} missing byte tokens for byte_fallback compatibility")
        for byte_val in sorted(missing_bytes):
            piece = sp_pb2.ModelProto.SentencePiece()
            piece.piece = f"<0x{byte_val:02X}>"
            piece.score = 0.0  # Base score for byte tokens
            piece.type = PieceType.BYTE
            model.pieces.append(piece)
            stats["byte_added"] += 1

    logger.info(f"Built SentencePiece model:")
    logger.info(f"  - UNKNOWN token: {stats['unknown']} (ID={unk_id})")
    logger.info(f"  - CONTROL tokens: {stats['control']}")
    logger.info(f"  - USER_DEFINED tokens: {stats['user_defined']}")
    logger.info(f"  - BYTE tokens (from vocab): {stats['byte']}")
    logger.info(f"  - BYTE tokens (added for 256 completeness): {stats['byte_added']}")
    logger.info(f"  - BYTE tokens (total): {stats['byte'] + stats['byte_added']}")
    logger.info(f"  - NORMAL tokens: {stats['normal']}")

    return model.SerializeToString()


# =============================================================================
# Validation and Testing
# =============================================================================

def validate_model(
    model_path: str | Path,
    test_texts: list[str] | None = None
) -> tuple[bool, list[str]]:
    """
    Validate the converted SentencePiece model.

    Args:
        model_path: Path to .model file
        test_texts: List of test strings (optional)

    Returns:
        Tuple of (success, list of issues)
    """
    issues: list[str] = []

    try:
        import sentencepiece as spm
    except ImportError:
        issues.append("SentencePiece package not installed")
        return False, issues

    try:
        sp = spm.SentencePieceProcessor()
        sp.Load(str(model_path))
    except Exception as e:
        issues.append(f"Failed to load model: {e}")
        return False, issues

    # Basic validation
    vocab_size = sp.GetPieceSize()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()
    unk_id = sp.unk_id()

    logger.info("Model validation:")
    logger.info(f"  - Vocab size: {vocab_size}")
    logger.info(f"  - BOS ID: {bos_id}")
    logger.info(f"  - EOS ID: {eos_id}")
    logger.info(f"  - PAD ID: {pad_id}")
    logger.info(f"  - UNK ID: {unk_id}")

    # Validate special token IDs
    if bos_id < 0:
        issues.append(f"Invalid BOS ID: {bos_id}")
    if eos_id < 0:
        issues.append(f"Invalid EOS ID: {eos_id}")
    if unk_id < 0:
        issues.append(f"Invalid UNK ID: {unk_id} (SentencePiece requires a valid UNK token)")

    # Test encoding/decoding
    if test_texts is None:
        test_texts = [
            "Hello, world!",
            "안녕하세요",
            "The quick brown fox",
            "한국어 대화 모델 테스트",
            "Mixed 한영 text 테스트",
            "Numbers: 12345",
            "Special chars: @#$%^&*()",
            "Emoji: 😀🎉",
        ]

    logger.info("\nTest encoding/decoding:")
    for text in test_texts:
        try:
            # Encode
            encoded = sp.Encode(text)
            # Decode
            decoded = sp.Decode(encoded)

            # Check round-trip
            # Note: Some differences are expected due to tokenization
            normalized_original = text.replace(" ", "")
            normalized_decoded = decoded.replace(" ", "").replace("▁", "")

            if normalized_original != normalized_decoded:
                # This may be a warning, not error (whitespace handling can differ)
                logger.warning(f"  Round-trip mismatch:")
                logger.warning(f"    Original: '{text}'")
                logger.warning(f"    Decoded:  '{decoded}'")
            else:
                logger.info(f"  OK: '{text[:30]}...' -> {len(encoded)} tokens")

        except Exception as e:
            issues.append(f"Encoding failed for '{text}': {e}")

    return len(issues) == 0, issues


def compare_tokenizations(
    original_tokenizer_path: str | Path,
    converted_model_path: str | Path,
    test_texts: list[str] | None = None
) -> dict[str, Any]:
    """
    Compare tokenization between original and converted tokenizer.

    Args:
        original_tokenizer_path: Path to original HFLM tokenizer directory
        converted_model_path: Path to converted .model file
        test_texts: Test strings (optional)

    Returns:
        Comparison report dictionary
    """
    try:
        from transformers import AutoTokenizer
        import sentencepiece as spm
    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}

    # Load tokenizers
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(
            str(original_tokenizer_path),
            local_files_only=True,
            trust_remote_code=True
        )
    except Exception as e:
        return {"error": f"Failed to load HuggingFace tokenizer: {e}"}

    try:
        sp = spm.SentencePieceProcessor()
        sp.Load(str(converted_model_path))
    except Exception as e:
        return {"error": f"Failed to load SentencePiece model: {e}"}

    if test_texts is None:
        test_texts = [
            "안녕하세요",
            "Hello, world!",
            "한국어 대화 모델 테스트",
            "The quick brown fox jumps over the lazy dog.",
            "Mixed 한영 text with numbers 12345",
        ]

    results = {
        "total_tests": len(test_texts),
        "exact_matches": 0,
        "token_count_matches": 0,
        "details": []
    }

    for text in test_texts:
        # Original tokenization (HuggingFace)
        hf_ids = hf_tokenizer.encode(text, add_special_tokens=False)

        # Converted tokenization (SentencePiece)
        sp_ids = sp.Encode(text)

        detail = {
            "text": text,
            "hf_ids": hf_ids,
            "sp_ids": sp_ids,
            "hf_count": len(hf_ids),
            "sp_count": len(sp_ids),
            "exact_match": hf_ids == sp_ids,
            "count_match": len(hf_ids) == len(sp_ids)
        }

        if detail["exact_match"]:
            results["exact_matches"] += 1
        if detail["count_match"]:
            results["token_count_matches"] += 1

        results["details"].append(detail)

    results["exact_match_rate"] = results["exact_matches"] / results["total_tests"]
    results["count_match_rate"] = results["token_count_matches"] / results["total_tests"]

    return results


# =============================================================================
# Main Conversion Function
# =============================================================================

def convert_tokenizer(
    input_path: str | Path,
    output_path: str | Path,
    bos_id: int = 1,
    eos_id: int = 37,
    pad_id: int = 0,
    unk_id: int = 0,
    validate: bool = True
) -> ConversionResult:
    """
    Convert HFLM tokenizer.json to SentencePiece .model format.

    Args:
        input_path: Path to tokenizer.json
        output_path: Output path for .model file
        bos_id: BOS token ID (default: 1)
        eos_id: EOS token ID (default: 37)
        pad_id: PAD token ID (default: 0)
        unk_id: UNK token ID (default: 0, <|end_of_text|>)
        validate: Whether to validate the output

    Returns:
        ConversionResult with status and statistics
    """
    result = ConversionResult(success=False)

    try:
        # Step 1: Load tokenizer configuration
        logger.info("=" * 60)
        logger.info("STEP 1: Loading tokenizer configuration")
        logger.info("=" * 60)

        config = load_tokenizer_json(input_path)
        if not config.validate():
            result.errors.append("Invalid tokenizer configuration")
            return result

        result.vocab_size = len(config.vocab)
        result.num_merges = len(config.merges)

        # Count special and byte tokens
        special_ids = {
            at["id"] for at in config.added_tokens
            if at.get("special", False)
        }
        result.num_special_tokens = len(special_ids)

        # Count byte tokens (both <0xHH> and GPT-2 style)
        byte_tokens_sp = [t for t in config.vocab if BYTE_TOKEN_PATTERN.match(t)]
        byte_tokens_gpt2 = [t for t in config.vocab if is_gpt2_byte_token(t)]
        result.num_byte_tokens = len(byte_tokens_sp) + len(byte_tokens_gpt2)

        logger.info(f"  - Special tokens: {result.num_special_tokens}")
        logger.info(f"  - Byte tokens (SP format): {len(byte_tokens_sp)}")
        logger.info(f"  - Byte tokens (GPT-2 style): {len(byte_tokens_gpt2)}")
        logger.info(f"  - Total byte tokens: {result.num_byte_tokens}")

        # Step 2: Compute scores from merges
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: Computing scores from merge order")
        logger.info("=" * 60)

        scores = compute_scores(config.vocab, config.merges)

        # Step 3: Build SentencePiece model
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 3: Building SentencePiece model")
        logger.info("=" * 60)

        model_bytes = build_sentencepiece_model_protobuf(
            config=config,
            scores=scores,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            unk_id=unk_id
        )

        # Step 4: Save model
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 4: Saving model")
        logger.info("=" * 60)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(model_bytes)

        result.output_path = str(output_path)
        logger.info(f"Model saved to: {output_path}")
        logger.info(f"  - File size: {output_path.stat().st_size:,} bytes")

        # Step 5: Validate if requested
        if validate:
            logger.info("")
            logger.info("=" * 60)
            logger.info("STEP 5: Validating model")
            logger.info("=" * 60)

            valid, issues = validate_model(output_path)
            if not valid:
                result.warnings.extend(issues)
                logger.warning(f"Validation issues: {issues}")

        result.success = True

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        result.errors.append(str(e))
        import traceback
        traceback.print_exc()

    return result


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Convert HFLM BBPE tokenizer to SentencePiece .model format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python convert_hf_lm_to_sentencepiece.py \\
        --input /path/to/HFLM/tokenizer.json \\
        --output /path/to/hf_lm.model

    # With validation and comparison
    python convert_hf_lm_to_sentencepiece.py \\
        --input /path/to/HFLM/tokenizer.json \\
        --output /path/to/hf_lm.model \\
        --validate \\
        --compare /path/to/HFLM/

    # Custom special token IDs
    python convert_hf_lm_to_sentencepiece.py \\
        --input /path/to/tokenizer.json \\
        --output /path/to/output.model \\
        --bos-id 1 --eos-id 37 --pad-id 0
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to HFLM tokenizer.json"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for SentencePiece .model file"
    )
    parser.add_argument(
        "--bos-id",
        type=int,
        default=1,
        help="BOS (Beginning of Sentence) token ID (default: 1)"
    )
    parser.add_argument(
        "--eos-id",
        type=int,
        default=37,
        help="EOS (End of Sentence) token ID (default: 37, <|turn_end|>)"
    )
    parser.add_argument(
        "--pad-id",
        type=int,
        default=0,
        help="PAD token ID (default: 0, <|end_of_text|>)"
    )
    parser.add_argument(
        "--unk-id",
        type=int,
        default=0,
        help="UNK token ID (default: 0, <|end_of_text|>)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the converted model after creation"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to original HFLM model directory for tokenization comparison"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Run conversion
    logger.info("HFLM BBPE -> SentencePiece Converter")
    logger.info("=" * 60)

    result = convert_tokenizer(
        input_path=args.input,
        output_path=args.output,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        pad_id=args.pad_id,
        unk_id=args.unk_id,
        validate=args.validate
    )

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 60)

    if result.success:
        logger.info("Status: SUCCESS")
        logger.info(f"Output: {result.output_path}")
        logger.info(f"Vocabulary size: {result.vocab_size:,}")
        logger.info(f"Merge rules: {result.num_merges:,}")
        logger.info(f"Special tokens: {result.num_special_tokens}")
        logger.info(f"Byte tokens: {result.num_byte_tokens}")

        if result.warnings:
            logger.warning(f"Warnings: {len(result.warnings)}")
            for w in result.warnings:
                logger.warning(f"  - {w}")
    else:
        logger.error("Status: FAILED")
        for e in result.errors:
            logger.error(f"  - {e}")
        sys.exit(1)

    # Optional comparison
    if args.compare and result.success:
        logger.info("")
        logger.info("=" * 60)
        logger.info("TOKENIZATION COMPARISON")
        logger.info("=" * 60)

        comparison = compare_tokenizations(
            original_tokenizer_path=args.compare,
            converted_model_path=args.output
        )

        if "error" in comparison:
            logger.error(f"Comparison failed: {comparison['error']}")
        else:
            logger.info(f"Total tests: {comparison['total_tests']}")
            logger.info(f"Exact matches: {comparison['exact_matches']} ({comparison['exact_match_rate']:.1%})")
            logger.info(f"Token count matches: {comparison['token_count_matches']} ({comparison['count_match_rate']:.1%})")

            if comparison['exact_match_rate'] < 1.0:
                logger.info("\nMismatched tokenizations:")
                for detail in comparison['details']:
                    if not detail['exact_match']:
                        logger.info(f"  Text: '{detail['text']}'")
                        logger.info(f"    HF:  {detail['hf_ids'][:10]}... ({detail['hf_count']} tokens)")
                        logger.info(f"    SP:  {detail['sp_ids'][:10]}... ({detail['sp_count']} tokens)")

    logger.info("")
    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
