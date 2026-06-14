#!/usr/bin/env python3
# Copyright (c) K-Moshi Project
"""
Convert K-Moshi full finetuning checkpoint to Rust/Candle compatible format.

This script transforms PyTorch state_dict tensor names to the format expected
by the Candle-based Rust backend (moshi-backend).

Usage:
    python scripts/convert_to_rust.py \
        --checkpoint ./runs/korean_v2/checkpoints/checkpoint_010000/consolidated/consolidated.safetensors \
        --output ./models/korean-moshi-full.safetensors \
        --base-model /path/to/moshiko-pytorch-bf16

Key differences between PyTorch and Rust tensor naming:
    PyTorch: depformer.layers.{layer}.self_attn.in_projs.{step}.weight
    Rust:    depformer.{step}.transformer.layers.{layer}.self_attn.in_proj_weight
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_model_structure(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Detect model structure from state dict."""

    # Count input embeddings (n_q)
    in_n_q = 0
    for idx in range(999):
        if f"emb.{idx}.weight" not in state_dict:
            in_n_q = idx
            break

    # Count output codebooks (dep_q)
    out_n_q = 0
    for idx in range(999):
        if f"linears.{idx}.weight" not in state_dict:
            out_n_q = idx
            break

    # Count depformer layers
    depformer_layers = 0
    for idx in range(999):
        # Check both possible naming conventions
        if f"depformer.layers.{idx}.self_attn.in_projs.0.weight" not in state_dict:
            if f"depformer.layers.{idx}.self_attn.in_proj.weight" not in state_dict:
                depformer_layers = idx
                break

    # Check if already in Rust format (depformer.0.transformer.layers.0...)
    is_rust_format = any(k.startswith("depformer.0.transformer") for k in state_dict.keys())

    return {
        "in_n_q": in_n_q,
        "out_n_q": out_n_q,
        "depformer_layers": depformer_layers,
        "is_rust_format": is_rust_format,
    }


def convert_to_rust_format(
    state_dict: Dict[str, torch.Tensor],
    max_out_n_q: int | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert PyTorch state dict to Rust/Candle compatible format.

    Args:
        state_dict: PyTorch model state dict
        max_out_n_q: Maximum number of depformer slices to export (None = all)

    Returns:
        Rust-compatible state dict
    """
    structure = detect_model_structure(state_dict)
    logger.info(f"Detected structure: {structure}")

    if structure["is_rust_format"]:
        logger.warning("Model appears to already be in Rust format!")
        return state_dict

    in_n_q = structure["in_n_q"]
    out_n_q = structure["out_n_q"]
    depformer_layers = structure["depformer_layers"]

    if max_out_n_q is not None:
        exported_out_n_q = min(max_out_n_q, out_n_q)
        logger.info(f"Exporting first {exported_out_n_q} of {out_n_q} depformer slices")
    else:
        exported_out_n_q = out_n_q
        logger.info(f"Exporting all {out_n_q} depformer slices")

    model = {}

    # 1. Copy text embedding and linear layers (unchanged names)
    for name in ["text_emb.weight", "text_linear.weight", "out_norm.alpha"]:
        if name in state_dict:
            model[name] = state_dict[name]
            logger.debug(f"Copied: {name}")

    # 2. Copy condition provider layers (unchanged names)
    for name in state_dict.keys():
        if name.startswith("condition_provider.conditioners"):
            model[name] = state_dict[name]

    # 3. Copy input embeddings (unchanged names)
    for idx in range(in_n_q):
        name = f"emb.{idx}.weight"
        if name in state_dict:
            model[name] = state_dict[name]

    # 4. Transform main transformer layers
    for k, v in state_dict.items():
        if k.startswith("transformer"):
            # Transform attention projection naming
            # PyTorch: in_projs.0.weight → Rust: in_proj_weight
            k_new = k.replace(".in_projs.0.weight", ".in_proj_weight")
            k_new = k_new.replace(".out_projs.0.weight", ".out_proj.weight")
            # Also handle singular form if present
            if ".self_attn.in_proj.weight" in k_new:
                k_new = k_new.replace(".self_attn.in_proj.weight", ".self_attn.in_proj_weight")
            model[k_new] = v
            if k != k_new:
                logger.debug(f"Transformed: {k} → {k_new}")

    # 5. Transform DepFormer layers (major restructuring)
    for slice_idx in range(exported_out_n_q):
        tch_idx = slice_idx
        base = f"depformer.{slice_idx}."

        # Linear in/out
        if f"depformer_in.{tch_idx}.weight" in state_dict:
            model[base + "linear_in.weight"] = state_dict[f"depformer_in.{tch_idx}.weight"].clone()
        if f"linears.{slice_idx}.weight" in state_dict:
            model[base + "linear_out.weight"] = state_dict[f"linears.{slice_idx}.weight"]

        # Embeddings
        if slice_idx == 0:
            if "depformer_text_emb.weight" in state_dict:
                model[base + "emb.weight"] = state_dict["depformer_text_emb.weight"]
            if "depformer_text_emb.low_rank.weight" in state_dict:
                model[base + "emb.low_rank.weight"] = state_dict["depformer_text_emb.low_rank.weight"].clone()
        else:
            emb_key = f"depformer_emb.{tch_idx-1}.weight"
            if emb_key in state_dict:
                model[base + "emb.weight"] = state_dict[emb_key].clone()
            lr_key = f"depformer_emb.{tch_idx-1}.low_rank.weight"
            if lr_key in state_dict:
                model[base + "emb.low_rank.weight"] = state_dict[lr_key].clone()

        # Transformer layers within depformer
        for layer_idx in range(depformer_layers):
            layer = base + f"transformer.layers.{layer_idx}."

            # Attention projections
            in_proj_key = f"depformer.layers.{layer_idx}.self_attn.in_projs.{tch_idx}.weight"
            if in_proj_key in state_dict:
                model[layer + "self_attn.in_proj_weight"] = state_dict[in_proj_key]

            out_proj_key = f"depformer.layers.{layer_idx}.self_attn.out_projs.{tch_idx}.weight"
            if out_proj_key in state_dict:
                model[layer + "self_attn.out_proj.weight"] = state_dict[out_proj_key]

            # Norms
            norm1_key = f"depformer.layers.{layer_idx}.norm1.alpha"
            if norm1_key in state_dict:
                model[layer + "norm1.alpha"] = state_dict[norm1_key].clone()

            norm2_key = f"depformer.layers.{layer_idx}.norm2.alpha"
            if norm2_key in state_dict:
                model[layer + "norm2.alpha"] = state_dict[norm2_key].clone()

            # Gating
            gating_in_key = f"depformer.layers.{layer_idx}.gating.{tch_idx}.linear_in.weight"
            if gating_in_key in state_dict:
                model[layer + "gating.linear_in.weight"] = state_dict[gating_in_key].clone()

            gating_out_key = f"depformer.layers.{layer_idx}.gating.{tch_idx}.linear_out.weight"
            if gating_out_key in state_dict:
                model[layer + "gating.linear_out.weight"] = state_dict[gating_out_key].clone()

    logger.info(f"Conversion complete: {len(state_dict)} → {len(model)} tensors")

    # Log sample of transformed keys
    sample_keys = [k for k in sorted(model.keys()) if "transformer.layers.0" in k][:5]
    logger.info(f"Sample output keys: {sample_keys}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Convert K-Moshi checkpoint to Rust/Candle format"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        required=True,
        help="Path to consolidated.safetensors from K-Moshi training"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output path for Rust-compatible safetensors"
    )
    parser.add_argument(
        "--max-out-n-q",
        type=int,
        default=None,
        help="Limit number of depformer slices to export (default: all)"
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Output data type (default: bf16)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    state_dict = load_file(str(args.checkpoint))
    logger.info(f"Loaded {len(state_dict)} tensors")

    # Convert
    rust_state_dict = convert_to_rust_format(state_dict, args.max_out_n_q)

    # Convert dtype if needed
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map[args.dtype]

    rust_state_dict = {
        k: v.to(dtype=target_dtype) for k, v in rust_state_dict.items()
    }
    logger.info(f"Converted to {args.dtype}")

    # Save
    logger.info(f"Saving to: {args.output}")
    save_file(rust_state_dict, str(args.output))

    # Print summary
    file_size_mb = args.output.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Saved {len(rust_state_dict)} tensors ({file_size_mb:.1f} MB)")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Update config-korean.json with: \"lm_model_file\": \"{args.output}\"")
    logger.info(f"  2. Start Rust server: cargo run --release -p moshi-backend -- --config config-korean.json")


if __name__ == "__main__":
    main()
