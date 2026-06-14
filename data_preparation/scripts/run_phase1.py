#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Phase 1 processing CLI - Enhanced version.

CPU-based processing of Lhotse Shar data:
- Speaker selection (hybrid score: duration + turn count)
- Stereo conversion (L=SPEAKER_MAIN, R=SPEAKER_USER)
- FLAC compression (50-60% disk savings)
- Rich metadata for Phase 2

Example usage:
    # Process all datasets with default config
    python run_phase1.py --parallel

    # Process specific dataset
    python run_phase1.py --dataset aihub-broadcast-key463-839g-train

    # Distributed processing (machine 0 of 10)
    python run_phase1.py --machine-id 0 --total-machines 10 --parallel

    # Use WAV format instead of FLAC
    python run_phase1.py --audio-format wav --parallel

    # Use config file
    python run_phase1.py --config /path/to/config.yaml --parallel

    # Resume from checkpoint
    python run_phase1.py --resume --parallel

    # Dry run to see what would be processed
    python run_phase1.py --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_preparation.config import PipelineConfig, get_default_config
from data_preparation.orchestrators.phase1_cpu import Phase1Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_size(bytes_size: int) -> str:
    """Format size in human-readable format."""
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f}KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f}MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.1f}GB"


def print_progress(processed: int, total: int, start_time: float) -> None:
    """Print progress update."""
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0
    eta = (total - processed) / rate if rate > 0 else 0

    logger.info(
        f"Progress: {processed:,}/{total:,} ({100*processed/total:.1f}%) | "
        f"Rate: {rate:.1f}/s | ETA: {format_duration(eta)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: CPU-based preprocessing for Korean Moshi dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing with all defaults (FLAC, 16 workers)
  python run_phase1.py --parallel

  # Distributed processing on machine 0 of 10
  python run_phase1.py --machine-id 0 --total-machines 10 --parallel

  # Generate scripts for distributed processing
  python generate_distributed_scripts.py --total-machines 10

  # After all machines complete, merge results
  python merge_phase1_results.py --total-machines 10
        """,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Process specific dataset by name (default: all)",
    )

    # Distributed processing
    parser.add_argument(
        "--machine-id",
        type=int,
        default=0,
        help="Machine ID for distributed processing (0-indexed)",
    )
    parser.add_argument(
        "--total-machines",
        type=int,
        default=1,
        help="Total number of machines for distributed processing",
    )

    # Processing options
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of worker processes per machine (default: 16)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing within machine",
    )

    # Audio format
    parser.add_argument(
        "--audio-format",
        type=str,
        choices=["flac", "wav"],
        default="flac",
        help="Output audio format (default: flac for 50-60%% disk savings)",
    )

    # Output
    parser.add_argument(
        "--output-base",
        type=Path,
        help="Override output base directory",
    )

    # Resume/checkpoint
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N conversations (default: 50)",
    )

    # Debug/test
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    if args.config and args.config.exists():
        config = PipelineConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = get_default_config()
        logger.info("Using default configuration")

    # Override settings from arguments
    config.phase1.machine_id = args.machine_id
    config.phase1.total_machines = args.total_machines
    config.phase1.num_workers = args.num_workers
    config.phase1.checkpoint_interval = args.checkpoint_interval
    config.phase1.resume_from_checkpoint = args.resume and not args.no_resume

    # Audio format
    config.audio.format = args.audio_format

    if args.output_base:
        config.output_base = args.output_base

    # Filter datasets if specified
    if args.dataset:
        config.datasets = [d for d in config.datasets if d.name == args.dataset]
        if not config.datasets:
            logger.error(f"Dataset not found: {args.dataset}")
            logger.info("Available datasets:")
            for d in get_default_config().datasets:
                logger.info(f"  - {d.name}")
            sys.exit(1)

    # Print configuration
    print("=" * 70)
    print("Phase 1 Processing Configuration")
    print("=" * 70)
    print(f"Machine:          {config.phase1.machine_id + 1}/{config.phase1.total_machines}")
    print(f"Workers:          {config.phase1.num_workers}")
    print(f"Parallel:         {args.parallel}")
    print(f"Audio format:     {config.audio.format.upper()}")
    print(f"Sample rate:      {config.audio.sample_rate} Hz")
    print(f"Resume enabled:   {config.phase1.resume_from_checkpoint}")
    print(f"Checkpoint every: {config.phase1.checkpoint_interval}")
    print("")
    print("Datasets to process:")
    for d in config.datasets:
        print(f"  - {d.name} ({d.split})")
        print(f"    Source: {d.source_path}")
        print(f"    Output: {d.output_path / d.split}")
    print("=" * 70)
    print("")

    if args.dry_run:
        logger.info("Dry run - exiting without processing")
        print("")
        print("To actually process, remove --dry-run flag")
        sys.exit(0)

    # Process datasets
    orchestrator = Phase1Orchestrator(config)
    all_stats = []
    overall_start = time.time()

    for dataset_config in config.datasets:
        print("")
        print("=" * 70)
        print(f"Processing: {dataset_config.name}")
        print("=" * 70)

        dataset_start = time.time()

        # Define progress callback
        def progress_cb(processed: int, total: int):
            print_progress(processed, total, dataset_start)

        # Process
        if args.parallel:
            stats = orchestrator.process_dataset_parallel(
                dataset_config,
                num_workers=args.num_workers,
            )
        else:
            stats = orchestrator.process_dataset(
                dataset_config,
                progress_callback=progress_cb,
            )

        all_stats.append((dataset_config.name, stats))

        # Print dataset results
        print("")
        print("-" * 40)
        print(f"Dataset Results: {dataset_config.name}")
        print("-" * 40)
        stats_dict = stats.to_dict()
        print(f"  Processed:    {stats_dict['processed_conversations']:,}")
        print(f"  Skipped:      {stats_dict['skipped_conversations']:,}")
        print(f"  Failed:       {stats_dict['failed_conversations']:,}")
        print(f"  Duration:     {stats_dict['total_duration_hours']:.1f} hours")
        print(f"  MAIN speaker: {stats_dict['total_main_duration_hours']:.1f} hours")
        print(f"  USER speaker: {stats_dict['total_user_duration_hours']:.1f} hours")
        print(f"  Audio size:   {stats_dict['total_audio_mb']:.1f} MB")
        print(f"  Processing:   {format_duration(stats_dict['processing_time_sec'])}")
        print(f"  Speed ratio:  {stats_dict['speed_ratio']:.1f}x realtime")

    # Print overall summary
    overall_time = time.time() - overall_start
    print("")
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Machine:        {config.phase1.machine_id + 1}/{config.phase1.total_machines}")
    print(f"Total datasets: {len(all_stats)}")
    print("")

    total_processed = 0
    total_hours = 0
    total_audio_mb = 0

    for name, stats in all_stats:
        s = stats.to_dict()
        total_processed += s["processed_conversations"]
        total_hours += s["total_duration_hours"]
        total_audio_mb += s["total_audio_mb"]
        print(f"  {name}: {s['processed_conversations']:,} conversations, {s['total_duration_hours']:.1f}h")

    print("")
    print("-" * 40)
    print(f"Total processed:  {total_processed:,} conversations")
    print(f"Total duration:   {total_hours:.1f} hours ({total_hours/24:.1f} days)")
    print(f"Total audio:      {total_audio_mb:.1f} MB ({total_audio_mb/1024:.1f} GB)")
    print(f"Processing time:  {format_duration(overall_time)}")
    print("=" * 70)

    if config.phase1.total_machines > 1:
        print("")
        print("Next steps:")
        print(f"  1. Wait for all {config.phase1.total_machines} machines to complete")
        print(f"  2. Run: python merge_phase1_results.py --total-machines {config.phase1.total_machines}")

    print("")
    logger.info("Phase 1 processing complete!")


if __name__ == "__main__":
    main()
