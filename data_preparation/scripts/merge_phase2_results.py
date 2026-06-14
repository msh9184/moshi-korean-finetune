#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Merge Phase 2 results from distributed processing.

After running Phase 2 on multiple GPU machines, this script merges all
machine-specific manifests into a single final manifest and validates
the alignment outputs.

Example usage:
    # Merge results from 8 machines
    python merge_phase2_results.py --total-machines 8

    # Merge with validation
    python merge_phase2_results.py --total-machines 8 --validate

    # Dry run to check completeness
    python merge_phase2_results.py --total-machines 8 --dry-run

    # Merge specific dataset
    python merge_phase2_results.py --total-machines 8 --dataset aihub-broadcast-key463-839g-train
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_preparation.config import PipelineConfig, get_default_config, DatasetConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class MergeStats:
    """Statistics from merge operation."""
    total_entries: int = 0
    machines_found: int = 0
    machines_missing: list = None
    validation_passed: int = 0
    validation_failed: int = 0
    total_duration_hours: float = 0.0
    unique_conversations: int = 0
    duplicate_entries: int = 0

    def __post_init__(self):
        if self.machines_missing is None:
            self.machines_missing = []


def load_machine_manifest(
    dataset_config: DatasetConfig,
    machine_id: int,
    phase: int = 2,
) -> list[dict]:
    """Load manifest entries from a single machine.

    Args:
        dataset_config: Dataset configuration
        machine_id: Machine ID
        phase: Phase number (1 or 2)

    Returns:
        List of manifest entries
    """
    checkpoint_dir = dataset_config.checkpoint_dir

    if phase == 1:
        manifest_path = checkpoint_dir / f"manifest_phase1_{machine_id:03d}.jsonl"
    else:
        manifest_path = checkpoint_dir / f"manifest_phase2_{machine_id:03d}.jsonl"

    entries = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in {manifest_path}: {e}")

    return entries


def validate_entry(
    entry: dict,
    dataset_config: DatasetConfig,
    check_audio: bool = True,
    check_alignments: bool = True,
    check_extended: bool = False,
) -> tuple[bool, str]:
    """Validate a manifest entry.

    Args:
        entry: Manifest entry dict
        dataset_config: Dataset configuration
        check_audio: Whether to verify audio file exists
        check_alignments: Whether to verify alignment files exist
        check_extended: Whether to verify extended alignment files exist (optional)

    Returns:
        Tuple of (is_valid, error_message)
    """
    output_dir = dataset_config.output_path / dataset_config.split

    # Check required fields
    required_fields = ["conversation_id", "audio", "duration"]
    for field in required_fields:
        if field not in entry:
            return False, f"Missing field: {field}"

    # Check audio file
    if check_audio:
        audio_path = output_dir / entry["audio"]
        if not audio_path.exists():
            return False, f"Audio file missing: {audio_path}"

    # Check alignment files
    if check_alignments:
        # Check Moshi format alignment
        if "alignment" in entry:
            align_path = output_dir / entry["alignment"]
            if not align_path.exists():
                return False, f"Alignment missing: {align_path}"

            # Validate alignment content
            try:
                with open(align_path, "r", encoding="utf-8") as f:
                    align_data = json.load(f)
                if "alignments" not in align_data:
                    return False, f"Invalid alignment format: {align_path}"
                if len(align_data["alignments"]) == 0:
                    return False, f"Empty alignment: {align_path}"
            except Exception as e:
                return False, f"Cannot read alignment: {e}"

        # Check speaker-specific alignments
        if "alignment_speaker01" in entry:
            align01_path = output_dir / entry["alignment_speaker01"]
            if not align01_path.exists():
                return False, f"Speaker01 alignment missing: {align01_path}"

        if "alignment_speaker02" in entry:
            align02_path = output_dir / entry["alignment_speaker02"]
            if not align02_path.exists():
                return False, f"Speaker02 alignment missing: {align02_path}"

    # Check extended alignment (optional, for future extensibility)
    if check_extended and "alignment_extended" in entry:
        extended_path = output_dir / entry["alignment_extended"]
        if not extended_path.exists():
            return False, f"Extended alignment missing: {extended_path}"

        # Validate extended alignment content
        try:
            with open(extended_path, "r", encoding="utf-8") as f:
                extended_data = json.load(f)
            if "alignments" not in extended_data:
                return False, f"Invalid extended alignment format: {extended_path}"
            # Extended format should have format_version field
            if extended_data.get("format_version") != "2.0":
                logger.debug(f"Unexpected format version in extended alignment: {extended_path}")
        except Exception as e:
            return False, f"Cannot read extended alignment: {e}"

    return True, ""


def merge_manifests(
    dataset_config: DatasetConfig,
    total_machines: int,
    validate: bool = True,
    phase: int = 2,
) -> tuple[list[dict], MergeStats]:
    """Merge manifests from all machines.

    Args:
        dataset_config: Dataset configuration
        total_machines: Total number of machines
        validate: Whether to validate entries
        phase: Phase number

    Returns:
        Tuple of (merged_entries, stats)
    """
    stats = MergeStats()
    all_entries = []
    seen_conversations = set()

    for machine_id in range(total_machines):
        entries = load_machine_manifest(dataset_config, machine_id, phase)

        if entries:
            stats.machines_found += 1
            logger.info(f"Machine {machine_id}: {len(entries)} entries")

            for entry in entries:
                conv_id = entry.get("conversation_id", "")

                # Check for duplicates
                if conv_id in seen_conversations:
                    stats.duplicate_entries += 1
                    continue

                seen_conversations.add(conv_id)

                # Validate if requested
                if validate:
                    is_valid, error = validate_entry(
                        entry, dataset_config,
                        check_audio=True,
                        check_alignments=(phase == 2),
                    )
                    if is_valid:
                        stats.validation_passed += 1
                        all_entries.append(entry)
                    else:
                        stats.validation_failed += 1
                        logger.debug(f"Validation failed for {conv_id}: {error}")
                else:
                    all_entries.append(entry)

                stats.total_duration_hours += entry.get("duration", 0) / 3600
        else:
            stats.machines_missing.append(machine_id)
            logger.warning(f"Machine {machine_id}: No manifest found")

    stats.total_entries = len(all_entries)
    stats.unique_conversations = len(seen_conversations)

    # Sort by conversation ID for consistent ordering
    all_entries.sort(key=lambda x: x.get("conversation_id", ""))

    return all_entries, stats


def write_merged_manifest(
    entries: list[dict],
    output_path: Path,
    include_alignment_path: bool = True,
    include_extended_path: bool = True,
) -> None:
    """Write merged manifest file.

    Args:
        entries: List of manifest entries
        output_path: Output file path
        include_alignment_path: Whether to include alignment path for Moshi format
        include_extended_path: Whether to include extended alignment path with speaker metadata
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            # Create output entry with required fields
            out_entry = {
                "audio": entry["audio"],
                "duration": entry["duration"],
                "speakers": entry.get("speakers", 2),
            }

            # Add alignment paths
            if "alignment_speaker01" in entry:
                out_entry["alignment_speaker01"] = entry["alignment_speaker01"]
            if "alignment_speaker02" in entry:
                out_entry["alignment_speaker02"] = entry["alignment_speaker02"]

            # Add Moshi format alignment path
            if include_alignment_path and "alignment" in entry:
                out_entry["alignment"] = entry["alignment"]

            # Add extended alignment path with speaker metadata
            if include_extended_path and "alignment_extended" in entry:
                out_entry["alignment_extended"] = entry["alignment_extended"]

            f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")

    logger.info(f"Wrote merged manifest: {output_path} ({len(entries)} entries)")


def write_moshi_manifest(
    entries: list[dict],
    output_path: Path,
    dataset_config: DatasetConfig,
) -> None:
    """Write Moshi training format manifest.

    Moshi expects:
    - audio path (will look for alignment at same path with .json extension)
    - duration

    Args:
        entries: List of manifest entries
        output_path: Output file path
        dataset_config: Dataset configuration
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = dataset_config.output_path / dataset_config.split

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            # Moshi format: {"path": "relative/path/to/audio.flac", "duration": 123.45}
            audio_path = entry["audio"]

            # Convert .wav to .flac if needed (Moshi expects .flac for training)
            # But our pipeline uses .flac by default now
            out_entry = {
                "path": audio_path,
                "duration": round(entry["duration"], 3),
            }

            f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")

    logger.info(f"Wrote Moshi training manifest: {output_path} ({len(entries)} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Phase 2 results from distributed processing"
    )

    parser.add_argument(
        "--total-machines",
        type=int,
        required=True,
        help="Total number of machines used for processing",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Merge specific dataset by name (default: all)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate alignment files exist and are valid (default: True)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without writing files",
    )
    parser.add_argument(
        "--moshi-format",
        action="store_true",
        default=True,
        help="Also create Moshi training format manifest (default: True)",
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

    # Load configuration
    if args.config and args.config.exists():
        config = PipelineConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = get_default_config()
        logger.info("Using default configuration")

    # Filter datasets if specified
    if args.dataset:
        config.datasets = [d for d in config.datasets if d.name == args.dataset]
        if not config.datasets:
            logger.error(f"Dataset not found: {args.dataset}")
            sys.exit(1)

    validate = args.validate and not args.no_validate

    print("=" * 70)
    print("Phase 2 Merge Configuration")
    print("=" * 70)
    print(f"Total machines:   {args.total_machines}")
    print(f"Validation:       {'Enabled' if validate else 'Disabled'}")
    print(f"Moshi format:     {'Enabled' if args.moshi_format else 'Disabled'}")
    print(f"Datasets:         {len(config.datasets)}")
    for d in config.datasets:
        print(f"  - {d.name} ({d.split})")
    print("=" * 70)
    print("")

    # Process each dataset
    overall_stats = MergeStats()
    start_time = time.time()

    for dataset_config in config.datasets:
        print(f"\nProcessing: {dataset_config.name}")
        print("-" * 40)

        # Merge manifests
        entries, stats = merge_manifests(
            dataset_config,
            args.total_machines,
            validate=validate,
            phase=2,
        )

        # Update overall stats
        overall_stats.total_entries += stats.total_entries
        overall_stats.machines_found = max(overall_stats.machines_found, stats.machines_found)
        overall_stats.machines_missing.extend(
            [f"{dataset_config.name}:{m}" for m in stats.machines_missing]
        )
        overall_stats.validation_passed += stats.validation_passed
        overall_stats.validation_failed += stats.validation_failed
        overall_stats.total_duration_hours += stats.total_duration_hours
        overall_stats.unique_conversations += stats.unique_conversations
        overall_stats.duplicate_entries += stats.duplicate_entries

        # Print stats
        print(f"  Machines found:     {stats.machines_found}/{args.total_machines}")
        if stats.machines_missing:
            print(f"  Machines missing:   {stats.machines_missing}")
        print(f"  Total entries:      {stats.total_entries:,}")
        print(f"  Unique conversations: {stats.unique_conversations:,}")
        if stats.duplicate_entries > 0:
            print(f"  Duplicates skipped: {stats.duplicate_entries}")
        if validate:
            print(f"  Validation passed:  {stats.validation_passed:,}")
            print(f"  Validation failed:  {stats.validation_failed:,}")
        print(f"  Duration:           {stats.total_duration_hours:.1f} hours")

        if args.dry_run:
            print("  [DRY RUN - no files written]")
            continue

        if entries:
            output_dir = dataset_config.output_path / dataset_config.split

            # Write merged manifest
            manifest_path = output_dir / "manifest_phase2.jsonl"
            write_merged_manifest(entries, manifest_path)

            # Write Moshi training format
            if args.moshi_format:
                moshi_manifest_path = output_dir / "manifest_moshi.jsonl"
                write_moshi_manifest(entries, moshi_manifest_path, dataset_config)

            # Write final manifest.jsonl (main output)
            final_manifest_path = dataset_config.manifest_path
            write_merged_manifest(entries, final_manifest_path)
        else:
            print("  No entries to merge!")

    # Print overall summary
    elapsed = time.time() - start_time
    print("")
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total entries:        {overall_stats.total_entries:,}")
    print(f"Unique conversations: {overall_stats.unique_conversations:,}")
    print(f"Total duration:       {overall_stats.total_duration_hours:.1f} hours ({overall_stats.total_duration_hours/24:.1f} days)")
    if validate:
        print(f"Validation passed:    {overall_stats.validation_passed:,}")
        print(f"Validation failed:    {overall_stats.validation_failed:,}")
    if overall_stats.duplicate_entries > 0:
        print(f"Duplicates skipped:   {overall_stats.duplicate_entries}")
    if overall_stats.machines_missing:
        print(f"Missing machines:     {len(overall_stats.machines_missing)}")
    print(f"Processing time:      {elapsed:.1f}s")
    print("=" * 70)

    if not args.dry_run:
        print("")
        print("Next steps:")
        print("  The data is now ready for Moshi finetuning!")
        print("  Manifests created:")
        for d in config.datasets:
            output_dir = d.output_path / d.split
            print(f"    - {output_dir / 'manifest_moshi.jsonl'}")

    print("")
    logger.info("Phase 2 merge complete!")


if __name__ == "__main__":
    main()
