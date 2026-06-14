#!/usr/bin/env python3
"""Create unified training manifest from multiple datasets.

This script combines manifests from multiple AIHub datasets into a single
training JSONL file for K-Moshi finetuning.

Dataset structure expected:
    /path/to/data
    ├── aihub-broadcast-key463-839g/
    │   ├── train/
    │   │   ├── audio/
    │   │   │   ├── CONV_ID.flac
    │   │   │   └── CONV_ID.json -> ../alignments/CONV_ID.json
    │   │   ├── alignments/
    │   │   │   └── CONV_ID.json
    │   │   └── manifest.jsonl
    │   └── valid/
    │       └── ...
    └── aihub-broadcast-key71314-559g/
        ├── train/
        └── valid/

Usage:
    # Create training manifest (train splits only)
    python scripts/create_unified_manifest.py \
        --output ./data/korean_v1_train.jsonl

    # Create validation manifest
    python scripts/create_unified_manifest.py \
        --output ./data/korean_v1_valid.jsonl \
        --split valid

    # Custom dataset selection
    python scripts/create_unified_manifest.py \
        --datasets key463-train key71314-train \
        --output ./data/korean_v1_train.jsonl

    # With validation
    python scripts/create_unified_manifest.py \
        --output ./data/korean_v1_train.jsonl \
        --validate
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default dataset base directory
DEFAULT_BASE_DIR = "/path/to/data"

# Available datasets
# Path format: {base_dir}/{dataset_name}/{split}/
DATASETS = {
    "key463-train": "aihub-broadcast-key463-839g/train",
    "key463-valid": "aihub-broadcast-key463-839g/valid",
    "key71314-train": "aihub-broadcast-key71314-559g/train",
    "key71314-valid": "aihub-broadcast-key71314-559g/valid",
}


def validate_entry(entry: dict, dataset_dir: Path) -> tuple[bool, str]:
    """Validate a single manifest entry.

    Checks:
    1. Audio file exists
    2. Alignment JSON exists (via symlink or alignments/ dir)
    3. Duration is valid

    Returns:
        (is_valid, error_message)
    """
    # Check required fields
    if "audio" not in entry:
        return False, "Missing 'audio' field"
    if "duration" not in entry:
        return False, "Missing 'duration' field"

    audio_rel = entry["audio"]
    audio_path = dataset_dir / audio_rel

    # Check audio file exists
    if not audio_path.exists():
        return False, f"Audio not found: {audio_path}"

    # Check alignment exists (try symlink first, then alignments/ dir)
    json_symlink = audio_path.with_suffix(".json")
    conv_id = audio_path.stem
    alignments_path = dataset_dir / "alignments" / f"{conv_id}.json"

    if not json_symlink.exists() and not alignments_path.exists():
        return False, f"Alignment not found for {conv_id}"

    # Check duration
    duration = entry.get("duration", 0)
    if duration <= 0:
        return False, f"Invalid duration: {duration}"

    return True, ""


def process_dataset(
    dataset_key: str,
    base_dir: Path,
    validate: bool = False,
    min_duration: float = 1.0,
    max_duration: float = 600.0,
) -> tuple[list[dict], dict]:
    """Process a single dataset and return entries.

    Args:
        dataset_key: Dataset identifier (e.g., "key463-train")
        base_dir: Base directory containing datasets
        validate: Whether to validate each entry
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds

    Returns:
        (entries, stats)
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASETS.keys())}")

    dataset_name = DATASETS[dataset_key]
    dataset_dir = base_dir / dataset_name

    # Priority order for manifest files:
    # 1. manifest_moshi.jsonl - Prepared specifically for moshi finetuning
    # 2. manifest.jsonl - Generic manifest (may have different format)
    manifest_candidates = [
        ("manifest_moshi.jsonl", "path"),   # moshi format: {"path": "audio/XXX.flac", "duration": ...}
        ("manifest.jsonl", "audio"),         # phase2 format: {"audio": "audio/XXX.flac", ...}
    ]

    manifest_path = None
    path_field = "path"

    for manifest_name, field_name in manifest_candidates:
        candidate = dataset_dir / manifest_name
        if candidate.exists():
            manifest_path = candidate
            path_field = field_name
            logger.info(f"Using manifest: {manifest_name} (path field: '{field_name}')")
            break

    if manifest_path is None:
        logger.warning(f"No manifest found in: {dataset_dir}")
        logger.warning(f"  Tried: {[c[0] for c in manifest_candidates]}")
        return [], {
            "total": 0,
            "included": 0,
            "skipped_duration": 0,
            "skipped_validation": 0,
            "total_duration_hours": 0,
        }

    logger.info(f"Processing: {dataset_name}")

    entries = []
    stats = {
        "total": 0,
        "included": 0,
        "skipped_duration": 0,
        "skipped_validation": 0,
        "skipped_no_path": 0,
        "total_duration_hours": 0,
    }

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stats["total"] += 1

            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                logger.warning(f"{dataset_name} line {line_num}: Invalid JSON")
                stats["skipped_validation"] += 1
                continue

            duration = entry.get("duration", 0)

            # Filter by duration
            if duration < min_duration or duration > max_duration:
                stats["skipped_duration"] += 1
                continue

            # Get audio path from the appropriate field
            audio_rel = entry.get(path_field, "")
            if not audio_rel:
                # Fallback: try other common field names
                for fallback_field in ["path", "audio", "audio_path"]:
                    audio_rel = entry.get(fallback_field, "")
                    if audio_rel:
                        break

            if not audio_rel:
                logger.debug(f"{dataset_name} line {line_num}: No audio path found")
                stats["skipped_no_path"] = stats.get("skipped_no_path", 0) + 1
                continue

            # Build absolute path
            audio_abs = str(dataset_dir / audio_rel)

            # Validate if requested
            if validate:
                if not Path(audio_abs).exists():
                    logger.debug(f"{dataset_name} line {line_num}: File not found: {audio_abs}")
                    stats["skipped_validation"] += 1
                    continue

            output_entry = {
                "path": audio_abs,
                "duration": duration,
            }

            entries.append(output_entry)
            stats["included"] += 1
            stats["total_duration_hours"] += duration / 3600

    logger.info(
        f"  {dataset_name}: {stats['included']:,}/{stats['total']:,} entries, "
        f"{stats['total_duration_hours']:.1f}h"
    )
    if stats.get("skipped_no_path", 0) > 0:
        logger.warning(f"  Skipped {stats['skipped_no_path']} entries with no audio path")

    return entries, stats


def create_unified_manifest(
    output_path: Path,
    dataset_keys: list[str],
    base_dir: Path,
    validate: bool = False,
    min_duration: float = 1.0,
    max_duration: float = 600.0,
    shuffle: bool = True,
) -> dict:
    """Create unified manifest from multiple datasets.

    Args:
        output_path: Output JSONL path
        dataset_keys: List of dataset identifiers
        base_dir: Base directory containing datasets
        validate: Whether to validate each entry
        min_duration: Minimum duration filter
        max_duration: Maximum duration filter
        shuffle: Whether to shuffle output entries

    Returns:
        Combined statistics
    """
    all_entries = []
    combined_stats = {
        "total_entries": 0,
        "total_hours": 0,
        "datasets": {},
    }

    for key in dataset_keys:
        entries, stats = process_dataset(
            key, base_dir, validate, min_duration, max_duration
        )
        all_entries.extend(entries)
        combined_stats["datasets"][key] = stats
        combined_stats["total_entries"] += stats["included"]
        combined_stats["total_hours"] += stats["total_duration_hours"]

    # Shuffle if requested
    if shuffle:
        import random
        random.shuffle(all_entries)
        logger.info(f"Shuffled {len(all_entries):,} entries")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Check if any entries were found
    if len(all_entries) == 0:
        logger.error("No entries found! Check dataset paths and manifest files.")
        logger.error("Expected path format: {base_dir}/{dataset_name}/{split}/manifest.jsonl")
        for key in dataset_keys:
            expected_path = base_dir / DATASETS[key] / "manifest.jsonl"
            logger.error(f"  Tried: {expected_path}")

    logger.info(f"Created: {output_path}")
    logger.info(f"  Total entries: {combined_stats['total_entries']:,}")
    logger.info(f"  Total duration: {combined_stats['total_hours']:.1f} hours")

    return combined_stats


def main():
    parser = argparse.ArgumentParser(
        description="Create unified training manifest from multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path (e.g., ./data/korean_v1_train.jsonl)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Dataset keys to include. Available: {list(DATASETS.keys())}",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "all", "combined"],
        help="Which split to include: train, valid, all (both separately), combined (merge train+valid for max data)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(DEFAULT_BASE_DIR),
        help=f"Base directory containing datasets (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each entry (check file existence)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=86400.0,  # 24 hours - effectively no limit
        help="Maximum duration in seconds (default: 86400.0 = 24h, effectively unlimited)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle output entries",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine datasets to process
    if args.datasets:
        dataset_keys = args.datasets
    else:
        # Auto-select based on split
        if args.split == "train":
            dataset_keys = ["key463-train", "key71314-train"]
        elif args.split == "valid":
            dataset_keys = ["key463-valid", "key71314-valid"]
        elif args.split == "combined":
            # Use ALL data (train + valid) for maximum training data
            dataset_keys = list(DATASETS.keys())
            logger.info("COMBINED mode: Using both train and valid splits for maximum data")
        else:  # all
            dataset_keys = list(DATASETS.keys())

    logger.info("=" * 60)
    logger.info("Creating Unified Manifest")
    logger.info("=" * 60)
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Datasets: {dataset_keys}")
    logger.info(f"  Base dir: {args.base_dir}")
    logger.info(f"  Validate: {args.validate}")
    logger.info(f"  Duration: {args.min_duration}s - {args.max_duration}s")
    logger.info("=" * 60)

    # Check base directory
    if not args.base_dir.exists():
        logger.error(f"Base directory not found: {args.base_dir}")
        sys.exit(1)

    # Create manifest
    stats = create_unified_manifest(
        output_path=args.output,
        dataset_keys=dataset_keys,
        base_dir=args.base_dir,
        validate=args.validate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        shuffle=not args.no_shuffle,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Output: {args.output}")
    print(f"  Total entries: {stats['total_entries']:,}")
    print(f"  Total duration: {stats['total_hours']:.1f} hours")
    print("\nPer-dataset breakdown:")
    for key, ds_stats in stats["datasets"].items():
        included = ds_stats.get('included', 0)
        hours = ds_stats.get('total_duration_hours', 0)
        print(f"  {key}: {included:,} entries, {hours:.1f}h")
    print("=" * 60)

    # Print next steps
    print("\nNext steps:")
    print(f"  1. Verify manifest: head -5 {args.output}")
    print(f"  2. Update config: data.train_data: '{args.output}'")
    print(f"  3. Run training: ./scripts/run_training_v1.sh")
    print("")
    print("For V2 training with train/valid split:")
    print("  # Create training manifest (train splits only)")
    print("  python scripts/create_unified_manifest.py \\")
    print("      --datasets key463-train key71314-train \\")
    print("      --output ./data/korean_v2_train.jsonl")
    print("")
    print("  # Create validation manifest (key71314-valid)")
    print("  python scripts/create_unified_manifest.py \\")
    print("      --datasets key71314-valid \\")
    print("      --output ./data/korean_v2_valid.jsonl")


if __name__ == "__main__":
    main()
