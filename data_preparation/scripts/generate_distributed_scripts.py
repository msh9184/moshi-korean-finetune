#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Generate independent execution scripts for distributed Phase 1 processing.

This script generates N machine-specific bash scripts that can be run
independently on separate Linux machines. Each script processes a
subset of shards via round-robin assignment.

Example usage:
    # Generate scripts for 10 machines
    python generate_distributed_scripts.py --total-machines 10 --output-dir ./run_scripts

    # Generate with custom config
    python generate_distributed_scripts.py \
        --total-machines 20 \
        --config /path/to/config.yaml \
        --output-dir /path/to

    # Generate scripts with specific dataset
    python generate_distributed_scripts.py \
        --total-machines 5 \
        --dataset aihub-broadcast-key463-839g-train \
        --output-dir ./run_scripts
"""

import argparse
import os
import stat
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_preparation.config import PipelineConfig, get_default_config


# Template for machine-specific run script
MACHINE_SCRIPT_TEMPLATE = '''#!/bin/bash
# Auto-generated Phase 1 script for Machine {machine_id}
# Generated: {timestamp}
# Total machines: {total_machines}
#
# This script processes shards assigned to this machine via round-robin.
# Shards: {shard_assignment}

set -e  # Exit on error

# Get absolute path of script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
MACHINE_ID={machine_id}
TOTAL_MACHINES={total_machines}
NUM_WORKERS={num_workers}
CONFIG_FILE="{config_path}"
MOSHI_DIR="{moshi_dir}"
LOG_DIR="$SCRIPT_DIR/logs"
DATASETS="{datasets}"

# Create log directory (using script's directory)
mkdir -p "$LOG_DIR"

# Activate virtual environment if exists
if [ -f "$MOSHI_DIR/venv/bin/activate" ]; then
    source "$MOSHI_DIR/venv/bin/activate"
elif [ -f "$MOSHI_DIR/.venv/bin/activate" ]; then
    source "$MOSHI_DIR/.venv/bin/activate"
fi

# Log file
LOG_FILE="$LOG_DIR/phase1_machine_{machine_id:03d}_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Phase 1 Processing - Machine {machine_id}"
echo "=========================================="
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo "Config: $CONFIG_FILE"
echo "Moshi dir: $MOSHI_DIR"
echo ""

# Run Phase 1 processing
cd "$MOSHI_DIR"

python -m data_preparation.scripts.run_phase1 \\
    --machine-id $MACHINE_ID \\
    --total-machines $TOTAL_MACHINES \\
    --num-workers $NUM_WORKERS \\
    {config_arg}\\
    {dataset_arg}\\
    --parallel \\
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Phase 1 Complete - Machine {machine_id}"
echo "=========================================="
echo "End time: $(date)"
echo "Log saved to: $LOG_FILE"
'''

# Template for master orchestration script
MASTER_SCRIPT_TEMPLATE = '''#!/bin/bash
# Master orchestration script for Phase 1 distributed processing
# Generated: {timestamp}
# Total machines: {total_machines}
#
# Usage:
#   1. Copy run_machine_XXX.sh to each machine
#   2. Run this script to verify all machines are ready
#   3. Start each machine's script independently
#   4. After all complete, run merge_results.py

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOTAL_MACHINES={total_machines}

echo "Phase 1 Distributed Processing - Master Script"
echo "==============================================="
echo ""
echo "Generated scripts for $TOTAL_MACHINES machines:"
echo ""

for i in $(seq 0 $((TOTAL_MACHINES - 1))); do
    script="$SCRIPT_DIR/run_machine_$(printf "%03d" $i).sh"
    if [ -f "$script" ]; then
        echo "  ✓ Machine $i: $script"
    else
        echo "  ✗ Machine $i: MISSING - $script"
    fi
done

echo ""
echo "Instructions:"
echo "1. Copy scripts to each machine:"
echo "   scp $SCRIPT_DIR/run_machine_*.sh user@machine:/path/to/"
echo ""
echo "2. Run on each machine:"
echo "   ./run_machine_XXX.sh"
echo ""
echo "3. After ALL machines complete, merge results:"
echo "   python merge_results.py --total-machines {total_machines}"
echo ""
echo "4. Monitor progress with:"
echo "   tail -f /path/to/logs/phase1_machine_*.log"
echo ""
'''

# Template for merge script
MERGE_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
# Auto-generated merge script for Phase 1 results
# Generated: {timestamp}
# Total machines: {total_machines}
"""
Merge results from all Phase 1 machines.

Run this after ALL machines have completed processing.

Usage:
    python merge_results.py
    python merge_results.py --verify-only
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_preparation.config import PipelineConfig, get_default_config, DatasetConfig
from data_preparation.orchestrators.phase1_cpu import Phase1Orchestrator, Phase1Stats


def verify_completion(dataset_config: DatasetConfig, total_machines: int) -> dict:
    """Verify all machines have completed."""
    status = {{
        "total_machines": total_machines,
        "completed": [],
        "missing": [],
        "partial": [],
    }}

    for machine_id in range(total_machines):
        checkpoint_path = dataset_config.get_machine_checkpoint_path(machine_id)
        manifest_path = dataset_config.get_machine_manifest_path(machine_id)

        if checkpoint_path.exists() and manifest_path.exists():
            status["completed"].append(machine_id)
        elif checkpoint_path.exists() or manifest_path.exists():
            status["partial"].append(machine_id)
        else:
            status["missing"].append(machine_id)

    return status


def main():
    parser = argparse.ArgumentParser(description="Merge Phase 1 results from all machines")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("{config_path}"),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--total-machines",
        type=int,
        default={total_machines},
        help="Total number of machines",
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

    args = parser.parse_args()

    # Load config
    if args.config.exists():
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = get_default_config()

    print("=" * 60)
    print("Phase 1 Results Merger")
    print("=" * 60)
    print(f"Total machines: {{args.total_machines}}")
    print(f"Datasets: {{len(config.datasets)}}")
    print("")

    all_complete = True

    for dataset_config in config.datasets:
        print(f"\\nDataset: {{dataset_config.name}}")
        print("-" * 40)

        # Verify completion
        status = verify_completion(dataset_config, args.total_machines)

        print(f"  Completed: {{len(status['completed'])}} / {{args.total_machines}}")
        if status["missing"]:
            print(f"  Missing: {{status['missing']}}")
            all_complete = False
        if status["partial"]:
            print(f"  Partial: {{status['partial']}}")
            all_complete = False

        if args.verify_only:
            continue

        if not all_complete and not args.force:
            print("\\n  ⚠️  Skipping merge - not all machines complete")
            print("     Use --force to merge anyway")
            continue

        # Merge manifests
        print("\\n  Merging manifests...")
        merged_manifest = Phase1Orchestrator.merge_manifests(
            dataset_config, args.total_machines
        )

        # Count entries
        with open(merged_manifest) as f:
            count = sum(1 for _ in f)
        print(f"  ✓ Created {{merged_manifest}}")
        print(f"    Total entries: {{count}}")

        # Merge stats
        print("\\n  Merging statistics...")
        merged_stats = Phase1Orchestrator.merge_stats(
            dataset_config, args.total_machines
        )

        # Save merged stats
        stats_path = dataset_config.stats_path.with_name("merged_stats.json")
        with open(stats_path, "w") as f:
            json.dump(merged_stats.to_dict(), f, indent=2)
        print(f"  ✓ Created {{stats_path}}")

        # Print summary
        print("\\n  Summary:")
        for k, v in merged_stats.to_dict().items():
            print(f"    {{k}}: {{v}}")

    print("\\n" + "=" * 60)
    if args.verify_only:
        if all_complete:
            print("✓ All machines complete. Ready to merge.")
        else:
            print("⚠️  Some machines incomplete. Check status above.")
    else:
        print("✓ Merge complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''


def generate_scripts(
    total_machines: int,
    output_dir: Path,
    config_path: Path = None,
    dataset_name: str = None,
    num_workers: int = 16,
    moshi_dir: Path = None,
):
    """Generate all distributed processing scripts.

    Args:
        total_machines: Number of machines for distributed processing
        output_dir: Directory to write scripts
        config_path: Path to config YAML (optional)
        dataset_name: Specific dataset to process (optional)
        num_workers: Workers per machine
        moshi_dir: Path to moshi-finetune directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine paths
    if moshi_dir is None:
        moshi_dir = Path(__file__).parent.parent.parent.resolve()
    else:
        moshi_dir = Path(moshi_dir).resolve()

    log_dir = output_dir / "logs"

    # Get dataset info
    if config_path and config_path.exists():
        config = PipelineConfig.from_yaml(config_path)
        config_arg = f"--config {config_path}"
    else:
        config = get_default_config()
        config_path = moshi_dir / "data_preparation" / "example_config.yaml"
        config_arg = ""

    datasets = [d.name for d in config.datasets]
    if dataset_name:
        datasets = [d for d in datasets if d == dataset_name]
        dataset_arg = f"--dataset {dataset_name}"
    else:
        dataset_arg = ""

    # Calculate shard assignment info
    shard_info = f"Shards {'{machine_id}'} mod {total_machines} = 0"

    print(f"Generating scripts for {total_machines} machines...")
    print(f"Output directory: {output_dir}")
    print(f"Moshi directory: {moshi_dir}")
    print(f"Datasets: {datasets}")
    print("")

    # Generate machine scripts
    for machine_id in range(total_machines):
        script_content = MACHINE_SCRIPT_TEMPLATE.format(
            machine_id=machine_id,
            total_machines=total_machines,
            num_workers=num_workers,
            config_path=str(config_path) if config_path else "",
            moshi_dir=str(moshi_dir),
            log_dir=str(log_dir),
            datasets=",".join(datasets),
            config_arg=config_arg + " \\\n    " if config_arg else "",
            dataset_arg=dataset_arg + " \\\n    " if dataset_arg else "",
            shard_assignment=f"i where i mod {total_machines} == {machine_id}",
            timestamp=timestamp,
        )

        script_path = output_dir / f"run_machine_{machine_id:03d}.sh"
        script_path.write_text(script_content)

        # Make executable
        script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        print(f"  Created: {script_path.name}")

    # Generate master script
    master_content = MASTER_SCRIPT_TEMPLATE.format(
        total_machines=total_machines,
        timestamp=timestamp,
    )
    master_path = output_dir / "master.sh"
    master_path.write_text(master_content)
    master_path.chmod(master_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  Created: {master_path.name}")

    # Generate merge script
    merge_content = MERGE_SCRIPT_TEMPLATE.format(
        total_machines=total_machines,
        config_path=str(config_path) if config_path else "",
        timestamp=timestamp,
    )
    merge_path = output_dir / "merge_results.py"
    merge_path.write_text(merge_content)
    merge_path.chmod(merge_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  Created: {merge_path.name}")

    print("")
    print("=" * 60)
    print("Script generation complete!")
    print("=" * 60)
    print("")
    print("Next steps:")
    print(f"  1. Review scripts in: {output_dir}")
    print(f"  2. Copy run_machine_XXX.sh to each machine")
    print(f"  3. Run each script independently on its machine")
    print(f"  4. After all complete, run: python {merge_path}")
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Generate distributed Phase 1 processing scripts"
    )

    parser.add_argument(
        "--total-machines",
        type=int,
        required=True,
        help="Total number of machines for distributed processing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./run_scripts"),
        help="Directory to write generated scripts",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Process specific dataset only",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of worker processes per machine",
    )
    parser.add_argument(
        "--moshi-dir",
        type=Path,
        help="Path to moshi-finetune directory on target machines",
    )

    args = parser.parse_args()

    generate_scripts(
        total_machines=args.total_machines,
        output_dir=args.output_dir,
        config_path=args.config,
        dataset_name=args.dataset,
        num_workers=args.num_workers,
        moshi_dir=args.moshi_dir,
    )


if __name__ == "__main__":
    main()
