# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Word-level alignment modules.

Lazy imports to avoid dependency issues when aligners are not needed (Phase 1 only).

Available aligners:
- WhisperTimestampedAligner: ASR-based alignment using whisper-timestamped
- NFAAligner: CTC-based forced alignment using NeMo Forced Aligner (NFA)

NFA is recommended for Korean data when transcripts are available from Phase 1.
"""


def get_whisper_aligner():
    """Get WhisperTimestampedAligner (lazy import).

    Returns:
        WhisperTimestampedAligner class
    """
    from .whisper_timestamped import WhisperTimestampedAligner
    return WhisperTimestampedAligner


def get_nfa_aligner():
    """Get NFAAligner (lazy import).

    NFA (NeMo Forced Aligner) uses CTC-based acoustic models for
    forced alignment. Recommended for Korean data when transcripts
    are available from Phase 1 metadata.

    Returns:
        NFAAligner class
    """
    from .nfa_aligner import NFAAligner
    return NFAAligner


def get_aligner(aligner_type: str = "nfa"):
    """Get aligner class by type.

    Args:
        aligner_type: "nfa" or "whisper"

    Returns:
        Aligner class

    Raises:
        ValueError: If aligner_type is not supported
    """
    if aligner_type == "nfa":
        return get_nfa_aligner()
    elif aligner_type == "whisper":
        return get_whisper_aligner()
    else:
        raise ValueError(f"Unknown aligner type: {aligner_type}. Use 'nfa' or 'whisper'.")


def get_alignment_classes():
    """Get alignment data classes (lazy import).

    Returns:
        Tuple of (WordAlignment, AlignmentResult) classes
    """
    from .whisper_timestamped import WordAlignment, AlignmentResult
    return WordAlignment, AlignmentResult


__all__ = [
    "get_whisper_aligner",
    "get_nfa_aligner",
    "get_aligner",
    "get_alignment_classes",
]
