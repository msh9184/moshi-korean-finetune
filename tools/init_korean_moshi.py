#!/usr/bin/env python3
"""
K-Moshi Model Initialization Script

Initializes a Korean Moshi model for full finetuning by:
1. Loading the original Moshiko-7B model from HuggingFace
2. Extending depth transformer modules for user stream (dep_q=8 → 16)
3. Reinitializing text embeddings for Korean tokenizer (optional)
4. Saving the initialized model for training

Usage:
    # Basic initialization with user stream extension
    python tools/init_korean_moshi.py \
        --save_dir ./models/k-moshi-init \
        --extend_modules_for_user_stream

    # Full initialization with Korean tokenizer embedding reinitialization
    python tools/init_korean_moshi.py \
        --save_dir ./models/k-moshi-init \
        --extend_modules_for_user_stream \
        --init_text_embeddings \
        --retain_text_token_ids 0 3 32000

    # Local model file instead of HuggingFace
    python tools/init_korean_moshi.py \
        --save_dir ./models/k-moshi-init \
        --moshi_lm_path ./models/moshiko-pytorch-bf16/model.safetensors \
        --extend_modules_for_user_stream
"""

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

# Add project root to path for proper imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from moshi.models.lm import LMModel

from tools.model_utils import (
    extend_moshi_modules_for_user_stream,
    init_embedding_module,
    get_model_architecture_info,
    validate_extended_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_moshiko_model(
    moshi_lm_path: Optional[str] = None,
    moshi_lm_repo: str = "kyutai/moshiko-pytorch-bf16",
    moshi_lm_name: str = "model.safetensors",
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> LMModel:
    """
    Load the original Moshiko LM model.

    Args:
        moshi_lm_path: Local path to model.safetensors (if None, download from HF)
        moshi_lm_repo: HuggingFace repository ID
        moshi_lm_name: Model filename in repository
        device: Device to load model on
        dtype: Data type for model parameters

    Returns:
        Loaded LMModel
    """
    if moshi_lm_path and Path(moshi_lm_path).exists():
        logger.info(f"Loading model from local path: {moshi_lm_path}")
        model_path = moshi_lm_path
    else:
        logger.info(f"Downloading model from HuggingFace: {moshi_lm_repo}/{moshi_lm_name}")
        model_path = hf_hub_download(moshi_lm_repo, moshi_lm_name)
        logger.info(f"Model downloaded to: {model_path}")

    # Load the model using moshi's loader
    moshi_lm = loaders.get_moshi_lm(model_path, device=device)
    moshi_lm = moshi_lm.to(dtype)

    logger.info(f"Model loaded successfully on {device} with dtype {dtype}")
    logger.info(f"Architecture info: {get_model_architecture_info(moshi_lm)}")

    return moshi_lm


def update_lm_kwargs_for_user_stream() -> dict:
    """
    Get updated LM kwargs for user stream (dep_q=16).

    Returns:
        Dictionary of kwargs to override for user stream support
    """
    # Copy default kwargs and update for user stream
    lm_kwargs = deepcopy(loaders._lm_kwargs)
    lm_kwargs.update({
        "dep_q": 16,  # 8 (moshi) + 8 (user)
        "depformer_context": 16,  # Match dep_q
    })
    return lm_kwargs


def save_model_for_finetuning(
    model: LMModel,
    save_dir: str,
    lm_kwargs: dict,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Save the initialized model for finetuning.

    Creates:
        - model.safetensors: Model weights
        - config.json: Model configuration (lm_kwargs)

    Args:
        model: Initialized LMModel
        save_dir: Directory to save model
        lm_kwargs: LM kwargs dictionary to save as config
        dtype: Data type for saved weights
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Convert model to target dtype
    model = model.to(dtype)

    # Save model weights
    model_path = save_path / "model.safetensors"
    logger.info(f"Saving model weights to: {model_path}")

    state_dict = model.state_dict()

    # Ensure all tensors are on CPU for saving
    state_dict = {k: v.cpu() for k, v in state_dict.items()}

    safetensors.torch.save_file(state_dict, model_path)
    logger.info(f"Model weights saved ({len(state_dict)} tensors)")

    # Save config
    config_path = save_path / "config.json"
    logger.info(f"Saving config to: {config_path}")

    config = {
        "lm_kwargs": lm_kwargs,
        "dtype": str(dtype).replace("torch.", ""),
        "architecture_info": get_model_architecture_info(model),
        "user_stream_extended": lm_kwargs.get("dep_q", 8) == 16,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Model initialization complete. Files saved to: {save_path}")


def main(args):
    """Main initialization routine."""
    logger.info("=" * 60)
    logger.info("K-Moshi Model Initialization")
    logger.info("=" * 60)

    # Determine dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.model_dtype]

    # Load original model
    logger.info("\n[Step 1] Loading original Moshiko model...")
    moshi_lm = load_moshiko_model(
        moshi_lm_path=args.moshi_lm_path,
        moshi_lm_repo=args.moshi_lm_repo,
        moshi_lm_name=args.moshi_lm_name,
        device="cpu",  # Load on CPU for manipulation
        dtype=dtype,
    )

    # Track lm_kwargs for config
    lm_kwargs = deepcopy(loaders._lm_kwargs)

    # Reinitialize text embeddings if requested
    if args.init_text_embeddings:
        logger.info("\n[Step 2] Reinitializing text embedding modules for Korean tokenizer...")
        logger.info(f"  Retaining embeddings for token IDs: {args.retain_text_token_ids}")

        # Reinitialize main text embedding
        logger.info("  - Reinitializing text_emb...")
        init_embedding_module(moshi_lm.text_emb, args.retain_text_token_ids)

        # Reinitialize depformer text embedding
        logger.info("  - Reinitializing depformer_text_emb...")
        init_embedding_module(moshi_lm.depformer_text_emb, args.retain_text_token_ids)

        logger.info("  Text embeddings reinitialized successfully!")
    else:
        logger.info("\n[Step 2] Skipping text embedding reinitialization (--init_text_embeddings not set)")

    # Extend modules for user stream if requested
    if args.extend_modules_for_user_stream:
        logger.info("\n[Step 3] Extending depth transformer modules for user stream...")
        logger.info("  Original architecture:")
        logger.info(f"    - depformer_in: {len(moshi_lm.depformer_in)}")
        logger.info(f"    - depformer_emb: {len(moshi_lm.depformer_emb)}")
        logger.info(f"    - linears: {len(moshi_lm.linears)}")

        moshi_lm = extend_moshi_modules_for_user_stream(moshi_lm)

        logger.info("  Extended architecture:")
        logger.info(f"    - depformer_in: {len(moshi_lm.depformer_in)}")
        logger.info(f"    - depformer_emb: {len(moshi_lm.depformer_emb)}")
        logger.info(f"    - linears: {len(moshi_lm.linears)}")

        # Validate extension
        validate_extended_model(moshi_lm)
        logger.info("  Model extension validated successfully!")

        # Update lm_kwargs for user stream
        lm_kwargs = update_lm_kwargs_for_user_stream()
    else:
        logger.info("\n[Step 3] Skipping user stream extension (--extend_modules_for_user_stream not set)")

    # Save the initialized model
    logger.info(f"\n[Step 4] Saving initialized model to: {args.save_dir}")
    save_model_for_finetuning(
        model=moshi_lm,
        save_dir=args.save_dir,
        lm_kwargs=lm_kwargs,
        dtype=dtype,
    )

    logger.info("\n" + "=" * 60)
    logger.info("K-Moshi Model Initialization Complete!")
    logger.info("=" * 60)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Prepare Korean training data (stereo WAV + transcriptions)")
    logger.info(f"  2. Update training config to use: {args.save_dir}/model.safetensors")
    logger.info(f"  3. Run training with: torchrun -m train example/korean_7B.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize K-Moshi model for Korean full finetuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Output configuration
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory path to save the initialized model",
    )

    # Model source configuration
    parser.add_argument(
        "--moshi_lm_path",
        type=str,
        default=None,
        help="Local path to model.safetensors (optional, downloads from HF if not set)",
    )
    parser.add_argument(
        "--moshi_lm_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="HuggingFace repository ID for Moshi model",
    )
    parser.add_argument(
        "--moshi_lm_name",
        type=str,
        default="model.safetensors",
        help="Model filename in the HuggingFace repository",
    )

    # Model configuration
    parser.add_argument(
        "--model_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Data type for model parameters",
    )

    # User stream extension
    parser.add_argument(
        "--extend_modules_for_user_stream",
        action="store_true",
        help="Extend depth transformer modules for user stream (dep_q=8→16)",
    )

    # Text embedding reinitialization
    parser.add_argument(
        "--init_text_embeddings",
        action="store_true",
        help="Reinitialize text embeddings for new tokenizer (Korean)",
    )
    parser.add_argument(
        "--retain_text_token_ids",
        nargs="+",
        type=int,
        default=[0, 3, 32000],
        help="Token IDs to preserve original embeddings (special tokens)",
    )

    args = parser.parse_args()
    main(args)
