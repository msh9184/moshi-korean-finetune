#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Merge Phase 1 results from distributed processing.

This script merges manifests and statistics from all machines
after distributed Phase 1 processing is complete.

Example usage:
    # Merge results from 10 machines
    python merge_phase1_results.py --total-machines 10

    # Verify completion status only
    python merge_phase1_results.py --total-machines 10 --verify-only

    # Force merge even if some machines are incomplete
    python merge_phase1_results.py --total-machines 10 --force

    # Use custom config
    python merge_phase1_results.py --total-machines 10 --config /path/to/config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_preparation.config import PipelineConfig, get_default_config, DatasetConfig
from data_preparation.orchestrators.phase1_cpu import Phase1Orchestrator, Phase1Stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def verify_machine_completion(
    dataset_config: DatasetConfig,
    machine_id: int,
) -> dict:
    """Check completion status of a single machine.

    Returns:
        Dict with status info
    """
    checkpoint_path = dataset_config.get_machine_checkpoint_path(machine_id)
    manifest_path = dataset_config.get_machine_manifest_path(machine_id)

    result = {
        "machine_id": machine_id,
        "checkpoint_exists": checkpoint_path.exists(),
        "manifest_exists": manifest_path.exists(),
        "processed": 0,
        "status": "missing",
    }

    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            result["processed"] = data.get("stats", {}).get("processed_conversations", 0)
            result["timestamp"] = data.get("timestamp", "unknown")
        except Exception as e:
            result["error"] = str(e)

    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                result["manifest_entries"] = sum(1 for _ in f)
        except Exception as e:
            result["manifest_error"] = str(e)

    if result["checkpoint_exists"] and result["manifest_exists"]:
        result["status"] = "complete"
    elif result["checkpoint_exists"] or result["manifest_exists"]:
        result["status"] = "partial"

    return result


def verify_all_machines(
    dataset_config: DatasetConfig,
    total_machines: int,
) -> dict:
    """Verify completion status of all machines.

    Returns:
        Summary dict with all machine statuses
    """
    summary = {
        "total_machines": total_machines,
        "complete": [],
        "partial": [],
        "missing": [],
        "total_processed": 0,
        "machines": [],
    }

    for machine_id in range(total_machines):
        status = verify_machine_completion(dataset_config, machine_id)
        summary["machines"].append(status)

        if status["status"] == "complete":
            summary["complete"].append(machine_id)
            summary["total_processed"] += status.get("processed", 0)
        elif status["status"] == "partial":
            summary["partial"].append(machine_id)
            summary["total_processed"] += status.get("processed", 0)
        else:
            summary["missing"].append(machine_id)

    summary["all_complete"] = len(summary["complete"]) == total_machines
    return summary


def merge_dataset_results(
    dataset_config: DatasetConfig,
    total_machines: int,
    force: bool = False,
    validate: bool = True,
) -> Optional[Phase1Stats]:
    """Merge results for a single dataset.

    Args:
        dataset_config: Dataset configuration
        total_machines: Total number of machines
        force: Force merge even if incomplete
        validate: Run post-merge validation

    Returns:
        Merged stats or None if merge failed
    """
    # Verify completion
    summary = verify_all_machines(dataset_config, total_machines)

    logger.info(f"Dataset: {dataset_config.name}")
    logger.info(f"  Complete: {len(summary['complete'])}/{total_machines}")
    if summary["partial"]:
        logger.warning(f"  Partial: {summary['partial']}")
    if summary["missing"]:
        logger.warning(f"  Missing: {summary['missing']}")

    if not summary["all_complete"] and not force:
        logger.error("  Merge aborted - not all machines complete. Use --force to override.")
        return None

    # Merge manifests with duplicate detection
    logger.info("  Merging manifests (with duplicate detection)...")
    merged_manifest, merge_report = Phase1Orchestrator.merge_manifests_with_validation(
        dataset_config, total_machines, deduplicate=True
    )

    logger.info(f"  Created: {merged_manifest}")
    logger.info(f"  Total entries: {merge_report['total_entries']}")
    if merge_report['duplicate_count'] > 0:
        logger.warning(f"  Duplicates removed: {merge_report['duplicate_count']}")

    # Validate merged manifest if requested
    validation_report = None
    if validate:
        logger.info("  Validating merged manifest...")
        validation_report = Phase1Orchestrator.validate_merged_manifest(
            dataset_config, total_machines
        )
        if not validation_report["is_valid"]:
            logger.error(f"  Validation failed: {validation_report.get('error', 'see details')}")
            if validation_report.get("duplicates"):
                logger.error(f"    Duplicates found: {len(validation_report['duplicates'])}")
            if validation_report.get("format_errors"):
                logger.error(f"    Format errors: {len(validation_report['format_errors'])}")
        else:
            logger.info("  Validation passed!")

    # Merge stats
    logger.info("  Merging statistics...")
    merged_stats = Phase1Orchestrator.merge_stats(dataset_config, total_machines)

    # Save merged stats
    merged_stats_path = dataset_config.stats_path.parent / "merged_stats.json"
    with open(merged_stats_path, "w") as f:
        json.dump(merged_stats.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved merged stats: {merged_stats_path}")

    # Save merge report
    merge_report_path = dataset_config.stats_path.parent / "merge_report.json"
    with open(merge_report_path, "w") as f:
        json.dump({
            "merge": merge_report,
            "validation": validation_report if validate else None,
            "machines_summary": summary,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved merge report: {merge_report_path}")

    return merged_stats


def print_summary(all_stats: dict) -> None:
    """Print overall summary."""
    print("")
    print("=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)

    total_hours = 0
    total_conversations = 0
    total_audio_mb = 0

    for dataset_name, stats in all_stats.items():
        if stats is None:
            print(f"  {dataset_name}: FAILED")
            continue

        hours = stats.total_duration_hours
        convs = stats.processed_conversations
        audio_mb = stats.total_audio_bytes / (1024 * 1024)

        print(f"  {dataset_name}:")
        print(f"    Conversations: {convs:,}")
        print(f"    Duration: {hours:.1f} hours")
        print(f"    Audio size: {audio_mb:.1f} MB")

        total_hours += hours
        total_conversations += convs
        total_audio_mb += audio_mb

    print("")
    print("-" * 60)
    print(f"TOTAL:")
    print(f"  Conversations: {total_conversations:,}")
    print(f"  Duration: {total_hours:.1f} hours ({total_hours/24:.1f} days)")
    print(f"  Audio size: {total_audio_mb:.1f} MB ({total_audio_mb/1024:.1f} GB)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge Phase 1 results from distributed processing"
    )

    parser.add_argument(
        "--total-machines",
        type=int,
        required=True,
        help="Total number of machines used in distributed processing",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Merge specific dataset only",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify completion status without merging",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force merge even if some machines are incomplete",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        help="Save detailed report to JSON file",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-merge validation (faster but no integrity check)",
    )

    args = parser.parse_args()

    # Load config
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

    print("=" * 60)
    print("Phase 1 Results Merger")
    print("=" * 60)
    print(f"Total machines: {args.total_machines}")
    print(f"Datasets: {len(config.datasets)}")
    print("")

    report = {
        "total_machines": args.total_machines,
        "datasets": {},
    }
    all_stats = {}
    all_complete = True

    for dataset_config in config.datasets:
        # Verify
        summary = verify_all_machines(dataset_config, args.total_machines)
        report["datasets"][dataset_config.name] = summary

        print(f"\n{dataset_config.name}:")
        print("-" * 40)
        print(f"  Status: {len(summary['complete'])}/{args.total_machines} machines complete")
        print(f"  Total processed: {summary['total_processed']:,} conversations")

        if not summary["all_complete"]:
            all_complete = False
            if summary["partial"]:
                print(f"  Partial: {summary['partial']}")
            if summary["missing"]:
                print(f"  Missing: {summary['missing']}")

        if args.verify_only:
            all_stats[dataset_config.name] = None
            continue

        # Merge
        merged_stats = merge_dataset_results(
            dataset_config,
            args.total_machines,
            force=args.force,
            validate=not args.no_validate,
        )
        all_stats[dataset_config.name] = merged_stats

    # Print summary
    if not args.verify_only:
        print_summary(all_stats)
    else:
        print("")
        print("=" * 60)
        if all_complete:
            print("STATUS: All machines complete - ready to merge")
        else:
            print("STATUS: Some machines incomplete - check details above")
        print("=" * 60)

    # Save report if requested
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_report, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved report to {args.output_report}")


if __name__ == "__main__":
    main()
