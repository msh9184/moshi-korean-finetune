#!/usr/bin/env python3
"""
=============================================================================
K-Moshi Model Download Script
=============================================================================

Downloads all required models for K-Moshi finetuning to a specified directory.

Models downloaded:
  - Moshi LM (7B parameters): ~14GB from Kyutai
  - Mimi Audio Codec: ~385MB from Kyutai
  - SentencePiece Tokenizer: ~540KB from Kyutai
  - Korean Tokenizer (optional): Various sources

Usage:
    # Download to default location
    python scripts/download_models.py --output-dir /path/to/model

    # Include Korean tokenizer
    python scripts/download_models.py --output-dir /path/to/model --korean-tokenizer

    # Skip verification
    python scripts/download_models.py --output-dir /path/to/model --skip-verify

Directory structure:
    /path/to/model
    ├── kyutai/                           # Provider: Kyutai Labs
    │   └── moshiko-pytorch-bf16/
    │       ├── model.safetensors         # ~14GB (bf16)
    │       ├── mimi.safetensors          # ~385MB (Mimi codec)
    │       ├── tokenizer_spm_32k_3.model # ~540KB (SentencePiece)
    │       └── default_config.json       # Generated from moshi defaults
    └── korean/                           # Korean-specific resources
        └── tokenizers/
            └── (optional Korean tokenizers)

Note:
    - config.json is NOT provided by the Kyutai HuggingFace repo
    - We generate default_config.json from moshi's hardcoded _lm_kwargs
    - This is the expected behavior for the legacy moshiko-pytorch-bf16 repo

=============================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# HuggingFace Repository Configuration
# =============================================================================

# Kyutai's Moshi repository
KYUTAI_REPO_ID = "kyutai/moshiko-pytorch-bf16"

# Files to download from Kyutai
KYUTAI_FILES = {
    "model.safetensors": {"description": "Moshi LM weights (7B)", "size_gb": 14.3, "required": True},
    "tokenizer-e351c8d8-checkpoint125.safetensors": {"description": "Mimi audio codec", "size_gb": 0.37, "required": True},
    "tokenizer_spm_32k_3.model": {"description": "SentencePiece tokenizer", "size_gb": 0.0005, "required": True},
}

# Default LM kwargs from moshi/models/loaders.py (for generating config)
# This is the configuration used when config.json is not present
DEFAULT_LM_KWARGS = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": 2048,  # from quantizer bins
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": 4224,  # int(4.125 * 1024)
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}

# Korean tokenizer sources (optional)
KOREAN_TOKENIZER_SOURCES = {
    "klue/bert-base": {
        "description": "KLUE BERT (32000 tokens, recommended)",
        "files": ["vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        "type": "wordpiece",
        "vocab_size": 32000,
    },
    "monologg/koelectra-base-v3-discriminator": {
        "description": "KoELECTRA (35000 tokens)",
        "files": ["vocab.txt", "tokenizer.json", "tokenizer_config.json"],
        "type": "wordpiece",
        "vocab_size": 35000,
    },
}


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_huggingface_hub():
    """Check if huggingface_hub is installed."""
    try:
        from huggingface_hub import hf_hub_download
        return True
    except ImportError:
        logger.error("huggingface_hub is not installed.")
        logger.info("Install with: pip install huggingface_hub")
        return False


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} bytes"


def download_file(
    repo_id: str,
    filename: str,
    local_dir: Path,
    description: str = "",
) -> bool:
    """Download a single file from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    target = local_dir / filename
    if target.exists():
        logger.info(f"  ✓ Already exists: {filename}")
        return True

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
        )
        logger.info(f"  ✓ Downloaded: {filename}")
        return True
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Entry Not Found" in error_msg:
            logger.warning(f"  ⚠ Not found (optional): {filename}")
            return False
        logger.error(f"  ✗ Failed to download {filename}: {e}")
        return False


def download_kyutai_models(output_dir: Path) -> bool:
    """Download models from Kyutai's HuggingFace repository."""
    print_banner("Downloading from Kyutai Labs")

    # Create directory structure with provider info
    kyutai_dir = output_dir / "kyutai" / "moshiko-pytorch-bf16"
    kyutai_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📦 Repository: {KYUTAI_REPO_ID}")
    logger.info(f"📁 Target directory: {kyutai_dir}")
    print()

    success = True
    for filename, info in KYUTAI_FILES.items():
        logger.info(f"📥 Downloading {info['description']}...")
        if not download_file(KYUTAI_REPO_ID, filename, kyutai_dir, info['description']):
            if info['required']:
                success = False

    # Generate default config (since config.json doesn't exist in the repo)
    # IMPORTANT: Only include valid _lm_kwargs fields!
    # Any other fields (model_type, moshi_name, _note, etc.) will be passed
    # to the model constructor and cause TypeError.
    config_path = kyutai_dir / "default_config.json"
    if not config_path.exists():
        logger.info("📝 Generating default_config.json from moshi defaults...")
        # Only include valid _lm_kwargs - NO metadata fields!
        config = DEFAULT_LM_KWARGS.copy()
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ Generated: default_config.json (pure _lm_kwargs only)")

    return success


def download_korean_tokenizers(output_dir: Path, download_klue: bool = False) -> bool:
    """Download Korean tokenizer resources (optional)."""
    print_banner("Korean Tokenizer Resources")

    korean_dir = output_dir / "korean" / "tokenizers"
    korean_dir.mkdir(parents=True, exist_ok=True)

    success = True

    if download_klue:
        # Download KLUE BERT tokenizer files
        logger.info("📥 Downloading KLUE BERT tokenizer (32K vocab, recommended)...")
        klue_dir = korean_dir / "klue-bert-base"
        klue_dir.mkdir(parents=True, exist_ok=True)

        klue_config = KOREAN_TOKENIZER_SOURCES["klue/bert-base"]
        for filename in klue_config["files"]:
            if not download_file("klue/bert-base", filename, klue_dir, f"KLUE BERT {filename}"):
                logger.warning(f"  ⚠ Failed to download: {filename}")

        logger.info(f"  ✓ KLUE BERT tokenizer saved to: {klue_dir}")
        logger.info("")
        logger.info("  ℹ️  To use KLUE BERT with Moshi, use the wrapper:")
        logger.info("      from tools.korean_tokenizer_wrapper import KoreanTokenizerWrapper")
        logger.info(f"      wrapper = KoreanTokenizerWrapper.from_local('{klue_dir}')")
        logger.info("")
    else:
        logger.info("ℹ️  Korean tokenizer options:")
        logger.info("")
        logger.info("   1. Default Moshiko tokenizer (32K vocab)")
        logger.info("      - Works for Korean, no additional setup needed")
        logger.info("")
        logger.info("   2. KLUE BERT tokenizer (32K vocab, recommended for Korean)")
        logger.info("      - Run with --download-klue to download")
        logger.info("      - Uses wrapper for SentencePiece compatibility")
        logger.info("")

    # Create a README for Korean tokenizers
    readme_path = korean_dir / "README.md"
    readme_content = """# Korean Tokenizer Options for K-Moshi

## Option 1: Default Moshiko Tokenizer (Recommended for Simplicity)

Use the original `tokenizer_spm_32k_3.model` without changes.
- Vocab size: 32,000 (matches model's text_card)
- Format: SentencePiece (native compatibility)
- Quality: Works for Korean, but English-optimized

## Option 2: KLUE BERT Tokenizer (Recommended for Best Korean)

KLUE BERT provides a Korean-optimized 32K vocabulary that matches Moshi's requirements.

### Setup

1. Download with: `python scripts/download_models.py --download-klue`

2. Use the wrapper in training config:
   ```yaml
   korean:
     korean_tokenizer_path: '/path/to/korean/tokenizers/klue-bert-base'
     korean_tokenizer_type: 'klue'  # Activates wrapper
   ```

3. The `KoreanTokenizerWrapper` provides SentencePiece-compatible API:
   ```python
   from tools.korean_tokenizer_wrapper import KoreanTokenizerWrapper

   wrapper = KoreanTokenizerWrapper.from_local('./korean/tokenizers/klue-bert-base')
   tokens = wrapper.encode("안녕하세요")  # Works like SentencePiece
   ```

## Token Mapping

| Moshi Requirement | Moshiko SPM | KLUE BERT |
|-------------------|-------------|-----------|
| BOS token | ID 2 | [CLS] (ID 2) |
| EOS token | ID 3 | [SEP] (ID 3) |
| PAD token | ID 0 | [PAD] (ID 0) |
| UNK token | ID 1 | [UNK] (ID 1) |
| Vocab size | 32,000 | 32,000 ✓ |

## Important Notes

- Moshi's `text_card` is 32000 - tokenizer vocab MUST match
- When switching tokenizers, reinitialize text embeddings using `tools/init_korean_moshi.py`
- The wrapper handles API differences automatically
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info(f"  ✓ Created/Updated: {readme_path}")

    return success


def verify_downloads(output_dir: Path) -> bool:
    """Verify all required files are present."""
    print_banner("Verifying Downloads")

    kyutai_dir = output_dir / "kyutai" / "moshiko-pytorch-bf16"

    required_files = {
        "model.safetensors": ("Moshi LM weights", 14 * 1024**3),
        "tokenizer-e351c8d8-checkpoint125.safetensors": ("Mimi codec", 350 * 1024**2),
        "tokenizer_spm_32k_3.model": ("SentencePiece tokenizer", 500 * 1024),
        "default_config.json": ("Default config (generated)", 0),
    }

    all_present = True
    total_size = 0

    for filename, (description, min_size) in required_files.items():
        full_path = kyutai_dir / filename
        exists = full_path.exists()

        if exists:
            file_size = full_path.stat().st_size
            total_size += file_size

            if min_size > 0 and file_size < min_size * 0.9:
                status = "⚠"
                logger.warning(f"  {status} {description}: {filename} (size mismatch)")
            else:
                status = "✓"
                print(f"  {status} {description}: {filename} ({format_size(file_size)})")
        else:
            print(f"  ✗ {description}: NOT FOUND")
            if "generated" not in description.lower():
                all_present = False

    print(f"\n  Total size: {format_size(total_size)}")
    return all_present


def print_config_example(output_dir: Path):
    """Print example YAML configuration."""
    print_banner("Configuration Example")

    kyutai_dir = output_dir / "kyutai" / "moshiko-pytorch-bf16"

    moshi_path = kyutai_dir / "model.safetensors"
    mimi_path = kyutai_dir / "tokenizer-e351c8d8-checkpoint125.safetensors"
    tokenizer_path = kyutai_dir / "tokenizer_spm_32k_3.model"
    config_path = kyutai_dir / "default_config.json"

    print(f"""
Add the following to your training YAML config:

# =============================================================================
# Option 1: Use local model files (recommended for GPU cluster)
# =============================================================================
moshi_paths:
  hf_repo_id: null  # Set to null to use local paths
  moshi_path: '{moshi_path}'
  mimi_path: '{mimi_path}'
  tokenizer_path: '{tokenizer_path}'
  config_path: '{config_path}'

# =============================================================================
# Option 2: Use HuggingFace Hub (requires internet, slower)
# =============================================================================
# moshi_paths:
#   hf_repo_id: 'kyutai/moshiko-pytorch-bf16'
#   moshi_path: null
#   mimi_path: null
#   tokenizer_path: null

# =============================================================================
# Korean finetuning with user stream
# =============================================================================
korean:
  enable_user_stream: true
  initialized_model_path: './models/k-moshi-init'  # After init_korean_moshi.py
  korean_tokenizer_path: null  # Or path to Korean tokenizer

# Checkpoints are saved to the working directory (not model storage)
run_dir: './runs/korean_v1'
""")


def print_directory_structure(output_dir: Path):
    """Print the directory structure."""
    print_banner("Directory Structure")

    print(f"""
{output_dir}/
├── kyutai/                              # Provider: Kyutai Labs
│   └── moshiko-pytorch-bf16/            # Model: Moshiko 7B
│       ├── model.safetensors            # LM weights (~14GB)
│       ├── tokenizer-*.safetensors      # Mimi codec (~385MB)
│       ├── tokenizer_spm_32k_3.model    # Text tokenizer (~540KB)
│       └── default_config.json          # Generated config
│
└── korean/                              # Korean-specific resources
    └── tokenizers/
        └── README.md                    # Korean tokenizer guide
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download K-Moshi models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/path/to/model",
        help="Directory to download models to (default: /path/to/model)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification step after download",
    )
    parser.add_argument(
        "--korean-tokenizer",
        action="store_true",
        help="Setup Korean tokenizer resources (README only)",
    )
    parser.add_argument(
        "--download-klue",
        action="store_true",
        help="Download KLUE BERT tokenizer (32K vocab, recommended for Korean)",
    )
    parser.add_argument(
        "--kyutai-only",
        action="store_true",
        help="Only download Kyutai models (skip Korean setup)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print_banner("K-Moshi Model Download")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Provider: Kyutai Labs ({KYUTAI_REPO_ID})")
    print()

    # Check huggingface_hub
    if not check_huggingface_hub():
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download Kyutai models
    success = download_kyutai_models(output_dir)

    # Setup Korean tokenizer resources
    if args.korean_tokenizer or args.download_klue or not args.kyutai_only:
        download_korean_tokenizers(output_dir, download_klue=args.download_klue)

    # Verify downloads
    if not args.skip_verify:
        all_verified = verify_downloads(output_dir)
        if not all_verified:
            success = False

    # Print directory structure
    print_directory_structure(output_dir)

    # Print config example
    print_config_example(output_dir)

    # Summary
    print_banner("Download Complete")
    if success:
        logger.info("✅ All required models downloaded successfully!")
        logger.info(f"   Location: {output_dir}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Update your YAML config with the paths shown above")
        logger.info("  2. For Korean finetuning, run: python -m tools.init_korean_moshi ...")
        logger.info("  3. Start training: torchrun -m train example/korean_fsdp.yaml")
    else:
        logger.error("❌ Some downloads failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
