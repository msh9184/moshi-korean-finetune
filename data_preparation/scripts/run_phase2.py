#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Phase 2 processing CLI - Distributed GPU-based word alignment.

Uses NeMo Forced Aligner (NFA) with Korean CTC model for word-level timestamps.
Supports 8-machine distributed processing with single GPU per machine.

Example usage:
    # Process all datasets on single GPU
    python run_phase2.py --single-gpu --gpu 0

    # Distributed processing (machine 0 of 8)
    python run_phase2.py --machine-id 0 --total-machines 8 --gpu 0

    # Process specific dataset
    python run_phase2.py --dataset aihub-broadcast-key463-839g-train --machine-id 0 --total-machines 8

    # Use Whisper-timestamped instead of NFA
    python run_phase2.py --aligner whisper --machine-id 0 --total-machines 8

    # Dry run to see what would be processed
    python run_phase2.py --dry-run --machine-id 0 --total-machines 8

    # Resume from checkpoint
    python run_phase2.py --resume --machine-id 0 --total-machines 8
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_preparation.config import PipelineConfig, get_default_config
from data_preparation.orchestrators.phase2_gpu import Phase2Orchestrator

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


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: GPU-based word-level alignment for Korean Moshi dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single machine with GPU 0
  python run_phase2.py --single-gpu --gpu 0

  # Distributed processing on machine 0 of 8
  python run_phase2.py --machine-id 0 --total-machines 8 --gpu 0

  # After all machines complete, merge results
  python merge_phase2_results.py --total-machines 8
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

    # GPU configuration
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use (default: 0)",
    )
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        default=True,
        help="Use single GPU mode (default: True)",
    )

    # Aligner configuration
    parser.add_argument(
        "--aligner",
        type=str,
        default="nfa",
        choices=["nfa", "whisper"],
        help="Alignment method: 'nfa' (NeMo Forced Aligner, recommended for Korean) "
             "or 'whisper' (whisper-timestamped)",
    )
    parser.add_argument(
        "--nfa-model",
        type=str,
        default="SungBeom/stt_kr_conformer_ctc_medium",
        help="NFA acoustic model (HuggingFace model name or local .nemo file path)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model to use (only for --aligner whisper)",
    )

    # Processing
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for NFA/Whisper processing (default: 64, A100 80GB: 64-128)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        default=True,
        help="Use optimized batched processing for better GPU utilization (default: True)",
    )
    parser.add_argument(
        "--no-batched",
        action="store_true",
        help="Disable batched processing (use sequential processing)",
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
        default=500,
        help="Save checkpoint every N conversations (default: 500)",
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
    config.phase2.machine_id = args.machine_id
    config.phase2.total_machines = args.total_machines
    config.phase2.gpu_id = args.gpu
    config.phase2.checkpoint_interval = args.checkpoint_interval
    config.phase2.resume_from_checkpoint = args.resume and not args.no_resume
    config.phase2.aligner_type = args.aligner
    config.phase2.batch_size = args.batch_size
    config.phase2.num_workers = args.num_workers

    # Set aligner-specific models
    if args.aligner == "nfa":
        config.nfa.acoustic_model = args.nfa_model
        config.nfa.batch_size = args.batch_size  # Apply to NFA config too
    else:
        config.whisper_alignment.whisper_model = args.whisper_model

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
    print("Phase 2 Processing Configuration")
    print("=" * 70)
    print(f"Machine:          {config.phase2.machine_id + 1}/{config.phase2.total_machines}")
    print(f"GPU ID:           {config.phase2.gpu_id}")
    print(f"Aligner:          {args.aligner.upper()}")
    if args.aligner == "nfa":
        print(f"  Model:          {config.nfa.acoustic_model}")
        print(f"  Language:       {config.nfa.language}")
    else:
        print(f"  Model:          {config.whisper_alignment.whisper_model}")
        print(f"  Language:       {config.whisper_alignment.language}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Batched mode:     {'Yes (optimized)' if args.batched and not args.no_batched else 'No (sequential)'}")
    print(f"Resume enabled:   {config.phase2.resume_from_checkpoint}")
    print(f"Checkpoint every: {config.phase2.checkpoint_interval}")
    print("")
    print("Datasets to process:")
    for d in config.datasets:
        print(f"  - {d.name} ({d.split})")
        print(f"    Input manifest: {d.manifest_path}")
        print(f"    Metadata dir: {d.metadata_dir}")
        print(f"    Output: {d.output_path / d.split}")
    print("=" * 70)
    print("")

    # Create orchestrator
    orchestrator = Phase2Orchestrator(config)

    if args.dry_run:
        logger.info("Dry run - showing task distribution")
        print("")

        for dataset_config in config.datasets:
            load_transcripts = (args.aligner == "nfa")
            tasks = orchestrator.load_phase1_manifest(
                dataset_config,
                load_transcripts=load_transcripts,
                machine_id=args.machine_id,
                total_machines=args.total_machines,
            )
            total_hours = sum(t.duration for t in tasks) / 3600
            with_transcripts = sum(1 for t in tasks if t.main_transcript or t.user_transcript)

            print(f"Dataset: {dataset_config.name}")
            print(f"  Tasks for this machine: {len(tasks)}")
            print(f"  Total duration: {total_hours:.1f} hours")
            if load_transcripts:
                print(f"  With transcripts: {with_transcripts}/{len(tasks)}")
            print("")

        print("To actually process, remove --dry-run flag")
        sys.exit(0)

    # Process datasets
    all_stats = []
    overall_start = time.time()

    for dataset_config in config.datasets:
        print("")
        print("=" * 70)
        print(f"Processing: {dataset_config.name}")
        print(f"Machine: {args.machine_id + 1}/{args.total_machines}")
        print("=" * 70)

        dataset_start = time.time()

        # Use batched processing for better GPU utilization (default)
        use_batched = args.batched and not args.no_batched

        if use_batched:
            # Optimized batched processing for better GPU utilization
            stats = orchestrator.process_single_gpu_batched(
                dataset_config,
                gpu_id=args.gpu,
                machine_id=args.machine_id,
                total_machines=args.total_machines,
                batch_size=args.batch_size,
            )
        else:
            # Sequential processing (legacy mode)
            stats = orchestrator.process_single_gpu(
                dataset_config,
                gpu_id=args.gpu,
                machine_id=args.machine_id,
                total_machines=args.total_machines,
            )

        all_stats.append((dataset_config.name, stats))

        # Print dataset results
        print("")
        print("-" * 40)
        print(f"Dataset Results: {dataset_config.name}")
        print("-" * 40)
        stats_dict = stats.to_dict()
        print(f"  Processed:      {stats_dict['processed_files']:,}")
        print(f"  Failed:         {stats_dict['failed_files']:,}")
        print(f"  Duration:       {stats_dict['total_duration_hours']:.1f} hours")
        print(f"  Processing:     {format_duration(stats_dict['processing_time_sec'])}")
        print(f"  Speed ratio:    {stats_dict['speed_ratio']:.1f}x realtime")
        print(f"  Avg quality:    {stats_dict['avg_quality_score']:.3f}")

    # Print overall summary
    overall_time = time.time() - overall_start
    print("")
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Machine:        {config.phase2.machine_id + 1}/{config.phase2.total_machines}")
    print(f"Total datasets: {len(all_stats)}")
    print("")

    total_processed = 0
    total_failed = 0
    total_hours = 0

    for name, stats in all_stats:
        s = stats.to_dict()
        total_processed += s["processed_files"]
        total_failed += s["failed_files"]
        total_hours += s["total_duration_hours"]
        print(f"  {name}: {s['processed_files']:,} files, {s['total_duration_hours']:.1f}h")

    print("")
    print("-" * 40)
    print(f"Total processed:  {total_processed:,} files")
    print(f"Total failed:     {total_failed:,} files")
    print(f"Total duration:   {total_hours:.1f} hours ({total_hours/24:.1f} days)")
    print(f"Processing time:  {format_duration(overall_time)}")
    print("=" * 70)

    if config.phase2.total_machines > 1:
        print("")
        print("Next steps:")
        print(f"  1. Wait for all {config.phase2.total_machines} machines to complete")
        print(f"  2. Run: python merge_phase2_results.py --total-machines {config.phase2.total_machines}")

    print("")
    logger.info("Phase 2 processing complete!")


if __name__ == "__main__":
    main()
