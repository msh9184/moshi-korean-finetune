#!/usr/bin/env python3
"""
Fix default_config.json by removing non-model fields that cause initialization errors.

The moshi LMModel only accepts specific _lm_kwargs parameters. Any other fields
in the config JSON (like model_type, moshi_name, _note, etc.) will be passed
to the model constructor and cause TypeError.

This script removes ALL non-lm_kwargs fields from the config.

Usage:
    python scripts/fix_config.py /path/to/model

Or to auto-detect:
    python scripts/fix_config.py --auto
"""

import json
import sys
from pathlib import Path


# Valid _lm_kwargs fields from moshi/models/loaders.py
# ONLY these fields should be in default_config.json
VALID_LM_KWARGS = {
    "dim",
    "text_card",
    "existing_text_padding_id",
    "n_q",
    "dep_q",
    "card",
    "num_heads",
    "num_layers",
    "hidden_scale",
    "causal",
    "layer_scale",
    "context",
    "max_period",
    "gating",
    "norm",
    "positional_embedding",
    "depformer_dim",
    "depformer_dim_feedforward",
    "depformer_num_heads",
    "depformer_num_layers",
    "depformer_layer_scale",
    "depformer_multi_linear",
    "depformer_context",
    "depformer_max_period",
    "depformer_gating",
    "depformer_pos_emb",
    "depformer_weights_per_step",
    "delays",
}


def fix_config(config_path: Path) -> bool:
    """Remove non-lm_kwargs fields from config file."""
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return False

    # Read original config
    with open(config_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
        config = json.loads(original_content)

    # Find all non-lm_kwargs fields to remove
    removed_keys = []
    keys_to_remove = [k for k in config.keys() if k not in VALID_LM_KWARGS]

    for key in keys_to_remove:
        del config[key]
        removed_keys.append(key)

    if not removed_keys:
        print(f"Config is already clean: {config_path}")
        print(f"Valid fields present: {list(config.keys())}")
        return True

    # Backup original (only if not already backed up)
    backup_path = config_path.with_suffix('.json.backup')
    if not backup_path.exists():
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"Backup saved: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

    # Write fixed config
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Fixed {config_path}")
    print(f"Removed {len(removed_keys)} non-lm_kwargs fields: {removed_keys}")
    print(f"Remaining valid fields: {list(config.keys())}")
    return True


def auto_detect_configs() -> list:
    """Auto-detect config files in common locations."""
    possible_paths = [
        Path("/path/to/model"),
        Path("/path/to/model"),
        Path("./models/kyutai/moshiko-pytorch-bf16/default_config.json"),
    ]
    return [p for p in possible_paths if p.exists()]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--auto":
        configs = auto_detect_configs()
        if not configs:
            print("No config files found in common locations.")
            print("Please specify the path manually:")
            print("  python scripts/fix_config.py /path/to/default_config.json")
            sys.exit(1)

        for config_path in configs:
            print(f"\nProcessing: {config_path}")
            fix_config(config_path)
    else:
        config_path = Path(sys.argv[1])
        if not fix_config(config_path):
            sys.exit(1)

    print("\nDone! You can now run training.")


if __name__ == "__main__":
    main()
