"""
K-Moshi Tools Package

Tools for Korean Moshi full finetuning:
- Model initialization with Korean tokenizer
- User stream extension (dep_q=8 -> 16)
- Text embedding reinitialization
- Model validation and conversion

Usage:
    # Initialize model for Korean finetuning
    python -m tools.init_korean_moshi \
        --save_dir ./models/k-moshi-init \
        --extend_modules_for_user_stream \
        --init_text_embeddings
"""

from tools.model_utils import (
    extend_moshi_modules_for_user_stream,
    remove_moshi_modules_for_user_stream,
    init_embedding_module,
    validate_extended_model,
    validate_original_model,
    get_model_architecture_info,
)

from tools.tokenizer_utils import (
    load_tokenizer,
    get_tokenizer_info,
    validate_tokenizer_for_moshi,
    compare_tokenizers,
    get_special_token_ids_to_retain,
    test_korean_tokenization,
)

__all__ = [
    # User stream extension/removal
    "extend_moshi_modules_for_user_stream",
    "remove_moshi_modules_for_user_stream",
    # Embedding reinitialization
    "init_embedding_module",
    # Validation utilities
    "validate_extended_model",
    "validate_original_model",
    "get_model_architecture_info",
    # Tokenizer utilities
    "load_tokenizer",
    "get_tokenizer_info",
    "validate_tokenizer_for_moshi",
    "compare_tokenizers",
    "get_special_token_ids_to_retain",
    "test_korean_tokenization",
]
