#!/usr/bin/env python3
"""Prepare training data for Moshi finetuning.

This script converts data from the data_preparation output format to the format
expected by Moshi's training code.

Data Preparation Output:
    {dataset}/train/
    ├── audio/
    │   └── conv_000001.wav
    ├── alignments/
    │   └── conv_000001.json (Moshi format)
    ├── alignment_speaker01/
    │   └── conv_000001.json
    ├── alignment_speaker02/
    │   └── conv_000001.json
    └── manifest.jsonl

Moshi Training Expected:
    - JSONL with {"path": "audio/conv_000001.wav", "duration": 45.32}
    - JSON alignment file accessible via audio path (same name, different extension)
      OR in the manifest with alignment path

This script:
1. Validates the input data format
2. Creates a Moshi-compatible training JSONL
3. Optionally creates symlinks for alignment files

Usage:
    # Validate dataset
    python scripts/prepare_training_data.py \\
        --input-dir /path/to/data \\
        --validate-only

    # Create training JSONL (uses alignments/ directory)
    python scripts/prepare_training_data.py \\
        --input-dir /path/to/data \\
        --output-jsonl ./data/korean_train.jsonl

    # Create training JSONL and symlinks for co-located alignments
    python scripts/prepare_training_data.py \\
        --input-dir /path/to/data \\
        --output-jsonl ./data/korean_train.jsonl \\
        --create-symlinks
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def validate_alignment_file(alignment_path: Path) -> tuple[bool, str, int]:
    """Validate alignment JSON file format.

    Expected format:
        {"alignments": [["word", [start, end], "SPEAKER_MAIN"], ...]}

    Returns:
        (is_valid, error_message, word_count)
    """
    try:
        with open(alignment_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "alignments" not in data:
            return False, "Missing 'alignments' key", 0

        alignments = data["alignments"]
        if not isinstance(alignments, list):
            return False, "'alignments' is not a list", 0

        if len(alignments) == 0:
            # Empty alignments are acceptable (silence segments)
            return True, "", 0

        # Check first alignment entry format
        first = alignments[0]
        if not isinstance(first, list) or len(first) != 3:
            return False, f"Invalid alignment entry format: {first}", 0

        word, times, speaker = first
        if not isinstance(word, str):
            return False, f"Word is not a string: {word}", 0
        if not isinstance(times, list) or len(times) != 2:
            return False, f"Times is not [start, end]: {times}", 0
        if not isinstance(speaker, str):
            return False, f"Speaker is not a string: {speaker}", 0

        return True, "", len(alignments)

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", 0
    except Exception as e:
        return False, f"Error reading file: {e}", 0


def validate_manifest_entry(entry: dict, input_dir: Path) -> tuple[bool, str]:
    """Validate a single manifest entry.

    Returns:
        (is_valid, error_message)
    """
    required_fields = ["audio", "duration"]
    for field in required_fields:
        if field not in entry:
            return False, f"Missing required field: {field}"

    # Check audio file exists
    audio_path = input_dir / entry["audio"]
    if not audio_path.exists():
        return False, f"Audio file not found: {audio_path}"

    # Check for alignment file (try multiple locations)
    alignment_found = False
    alignment_paths = []

    # 1. Try alignments/ directory (Moshi combined format)
    if "alignment" in entry:
        alignment_path = input_dir / entry["alignment"]
        alignment_paths.append(alignment_path)
        if alignment_path.exists():
            alignment_found = True

    # 2. Try co-located JSON (same name as audio)
    audio_json = audio_path.with_suffix(".json")
    alignment_paths.append(audio_json)
    if audio_json.exists():
        alignment_found = True

    # 3. Try alignment_speaker01/ directory
    if "alignment_speaker01" in entry:
        speaker01_path = input_dir / entry["alignment_speaker01"]
        alignment_paths.append(speaker01_path)
        if speaker01_path.exists():
            alignment_found = True

    if not alignment_found:
        paths_str = ", ".join(str(p) for p in alignment_paths)
        return False, f"No alignment file found. Tried: {paths_str}"

    return True, ""


def validate_dataset(input_dir: Path, limit: Optional[int] = None) -> dict:
    """Validate entire dataset.

    Args:
        input_dir: Path to dataset directory
        limit: Maximum number of entries to validate (None = all)

    Returns:
        Validation statistics
    """
    manifest_path = input_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "total_words": 0,
        "total_duration_hours": 0,
        "errors": [],
    }

    logger.info(f"Validating dataset: {input_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            stats["total"] += 1

            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError as e:
                stats["invalid"] += 1
                stats["errors"].append(f"Line {i+1}: Invalid JSON - {e}")
                continue

            # Validate entry
            is_valid, error = validate_manifest_entry(entry, input_dir)
            if not is_valid:
                stats["invalid"] += 1
                stats["errors"].append(f"Line {i+1}: {error}")
                continue

            # Validate alignment file if exists
            alignment_path = None
            if "alignment" in entry:
                alignment_path = input_dir / entry["alignment"]
            elif "alignment_speaker01" in entry:
                alignment_path = input_dir / entry["alignment_speaker01"]

            if alignment_path and alignment_path.exists():
                is_valid, error, word_count = validate_alignment_file(alignment_path)
                if not is_valid:
                    stats["invalid"] += 1
                    stats["errors"].append(f"Line {i+1}: Alignment error - {error}")
                    continue
                stats["total_words"] += word_count

            stats["valid"] += 1
            stats["total_duration_hours"] += entry.get("duration", 0) / 3600

    return stats


def create_training_jsonl(
    input_dir: Path,
    output_jsonl: Path,
    create_symlinks: bool = False,
    use_alignment_dir: str = "alignments",
    min_duration: float = 1.0,
    max_duration: float = 600.0,
) -> dict:
    """Create Moshi-compatible training JSONL.

    Args:
        input_dir: Path to dataset directory
        output_jsonl: Path to output JSONL file
        create_symlinks: If True, create symlinks for alignment files next to audio
        use_alignment_dir: Which alignment directory to use
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds

    Returns:
        Statistics about the created JSONL
    """
    manifest_path = input_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    stats = {
        "total": 0,
        "included": 0,
        "skipped_duration": 0,
        "skipped_no_alignment": 0,
        "total_duration_hours": 0,
    }

    # Ensure output directory exists
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating training JSONL: {output_jsonl}")
    logger.info(f"Using alignment directory: {use_alignment_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f_in, \
         open(output_jsonl, "w", encoding="utf-8") as f_out:

        for line in f_in:
            stats["total"] += 1

            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            duration = entry.get("duration", 0)

            # Filter by duration
            if duration < min_duration or duration > max_duration:
                stats["skipped_duration"] += 1
                continue

            # Get absolute audio path
            audio_rel_path = entry.get("audio", "")
            audio_abs_path = input_dir / audio_rel_path

            if not audio_abs_path.exists():
                logger.warning(f"Audio not found: {audio_abs_path}")
                continue

            # Find alignment file
            alignment_path = None

            # Try alignments/ directory first (Moshi combined format)
            if use_alignment_dir == "alignments" and "alignment" in entry:
                alignment_path = input_dir / entry["alignment"]
            elif use_alignment_dir == "alignment_speaker01" and "alignment_speaker01" in entry:
                alignment_path = input_dir / entry["alignment_speaker01"]
            else:
                # Default: try alignments/
                conv_id = audio_abs_path.stem
                alignment_path = input_dir / "alignments" / f"{conv_id}.json"

            if not alignment_path or not alignment_path.exists():
                stats["skipped_no_alignment"] += 1
                logger.debug(f"No alignment for: {audio_rel_path}")
                continue

            # Create symlink if requested
            if create_symlinks:
                symlink_path = audio_abs_path.with_suffix(".json")
                if not symlink_path.exists():
                    try:
                        # Create relative symlink
                        rel_alignment = os.path.relpath(alignment_path, symlink_path.parent)
                        symlink_path.symlink_to(rel_alignment)
                        logger.debug(f"Created symlink: {symlink_path} -> {rel_alignment}")
                    except OSError as e:
                        logger.warning(f"Failed to create symlink: {e}")

            # Write training entry
            # Use absolute path for audio
            training_entry = {
                "path": str(audio_abs_path),
                "duration": duration,
            }
            f_out.write(json.dumps(training_entry, ensure_ascii=False) + "\n")

            stats["included"] += 1
            stats["total_duration_hours"] += duration / 3600

    logger.info(f"Created training JSONL with {stats['included']} entries")
    logger.info(f"Total duration: {stats['total_duration_hours']:.1f} hours")

    return stats


def print_stats(stats: dict, title: str = "Statistics") -> None:
    """Print statistics in a formatted way."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for key, value in stats.items():
        if key == "errors":
            if value:
                print(f"\nErrors ({len(value)}):")
                for error in value[:10]:  # Show first 10 errors
                    print(f"  - {error}")
                if len(value) > 10:
                    print(f"  ... and {len(value) - 10} more errors")
        elif "duration" in key.lower() or "hours" in key.lower():
            print(f"  {key}: {value:.2f}h")
        else:
            print(f"  {key}: {value:,}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for Moshi finetuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to dataset directory (e.g., /path/to)",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Path to output training JSONL file",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the dataset, don't create output",
    )
    parser.add_argument(
        "--create-symlinks",
        action="store_true",
        help="Create symlinks for alignment files next to audio files",
    )
    parser.add_argument(
        "--alignment-dir",
        type=str,
        default="alignments",
        choices=["alignments", "alignment_speaker01"],
        help="Which alignment directory to use (default: alignments)",
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
        default=600.0,
        help="Maximum duration in seconds (default: 600.0 = 10 min)",
    )
    parser.add_argument(
        "--validate-limit",
        type=int,
        default=None,
        help="Limit validation to first N entries",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Validate dataset
    print("\n" + "=" * 60)
    print("STEP 1: Validating Dataset")
    print("=" * 60)

    try:
        stats = validate_dataset(args.input_dir, limit=args.validate_limit)
        print_stats(stats, "Validation Results")

        if stats["invalid"] > 0:
            logger.warning(f"Found {stats['invalid']} invalid entries")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.validate_only:
        if stats["valid"] > 0:
            print("✅ Dataset validation completed successfully")
        else:
            print("❌ No valid entries found")
            sys.exit(1)
        return

    # Create training JSONL
    if not args.output_jsonl:
        logger.error("--output-jsonl is required when not using --validate-only")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STEP 2: Creating Training JSONL")
    print("=" * 60)

    create_stats = create_training_jsonl(
        input_dir=args.input_dir,
        output_jsonl=args.output_jsonl,
        create_symlinks=args.create_symlinks,
        use_alignment_dir=args.alignment_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    print_stats(create_stats, "Training Data Statistics")

    print(f"✅ Training JSONL created: {args.output_jsonl}")
    print(f"   Total entries: {create_stats['included']:,}")
    print(f"   Total duration: {create_stats['total_duration_hours']:.1f} hours")

    # Print next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"""
1. Update your training config (example/korean_fsdp.yaml):

   data:
     train_data: '{args.output_jsonl}'
     shuffle: true

2. Run training:

   # Single GPU
   torchrun --nproc-per-node 1 -m train example/korean_fsdp.yaml

   # Multi-GPU with mpirun
   mpirun -np 4 python -m train example/korean_fsdp.yaml
""")


if __name__ == "__main__":
    main()
