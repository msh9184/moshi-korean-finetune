# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Phase 2 GPU-based orchestrator.

Handles word-level alignment using either:
- NeMo Forced Aligner (NFA) - CTC-based, uses transcripts from Phase 1
- whisper-timestamped - ASR-based, transcribes and aligns

NFA is recommended for Korean data when transcripts are available.

Designed for parallel execution across multiple GPUs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
import json
import logging
import multiprocessing as mp
import os
import queue
import time

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..config import PipelineConfig, DatasetConfig
from ..aligners import get_aligner
from ..aligners.nfa_aligner import cleanup_gpu_memory, get_gpu_memory_info
from ..writers.moshi_format import MoshiFormatWriter, ManifestEntry

logger = logging.getLogger(__name__)

# Constants for batch processing
DEFAULT_BATCH_SIZE = 32
MAX_BATCH_AUDIO_DURATION_SEC = 600  # Max total audio duration per batch


@dataclass
class Phase2Stats:
    """Statistics for Phase 2 processing."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_duration_hours: float = 0.0
    processing_time_sec: float = 0.0
    avg_quality_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "total_duration_hours": round(self.total_duration_hours, 2),
            "processing_time_sec": round(self.processing_time_sec, 1),
            "speed_ratio": round(
                self.total_duration_hours * 3600 / max(self.processing_time_sec, 1), 2
            ),
            "avg_quality_score": round(self.avg_quality_score, 3),
        }


@dataclass
class Phase2Task:
    """A task for Phase 2 processing."""
    conversation_id: str
    audio_path: Path
    segment_alignment_path: Path
    output_dir: Path
    duration: float
    # For NFA: transcripts loaded from Phase 1 metadata
    main_transcript: Optional[str] = None
    user_transcript: Optional[str] = None
    metadata_path: Optional[Path] = None


def gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config: PipelineConfig,
    stop_event: mp.Event,
    aligner_type: str = "nfa",
):
    """Worker process for GPU-based alignment.

    Args:
        gpu_id: GPU device ID
        task_queue: Queue of Phase2Task objects
        result_queue: Queue for results
        config: Pipeline configuration
        stop_event: Event to signal worker stop
        aligner_type: "nfa" or "whisper"
    """
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(f"GPU Worker {gpu_id} starting with {aligner_type} aligner")

    try:
        # Initialize aligner based on type (loads model once)
        AlignerClass = get_aligner(aligner_type)
        if aligner_type == "nfa":
            aligner = AlignerClass(config.nfa)
        else:
            aligner = AlignerClass(config.whisper_alignment)

        while not stop_event.is_set():
            try:
                task = task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:  # Poison pill
                break

            try:
                # Process the task - different parameters for NFA vs Whisper
                if aligner_type == "nfa":
                    result = aligner.align_stereo(
                        audio_path=task.audio_path,
                        conversation_id=task.conversation_id,
                        main_transcript=task.main_transcript,
                        user_transcript=task.user_transcript,
                        metadata_path=task.metadata_path,
                    )
                else:
                    result = aligner.align_stereo(
                        audio_path=task.audio_path,
                        conversation_id=task.conversation_id,
                    )

                # Save alignments
                if result.is_valid:
                    align01_path = task.output_dir / "alignment_speaker01" / f"{task.conversation_id}.json"
                    align02_path = task.output_dir / "alignment_speaker02" / f"{task.conversation_id}.json"

                    if result.main_alignment:
                        result.main_alignment.save(align01_path)
                    if result.user_alignment:
                        result.user_alignment.save(align02_path)

                result_queue.put({
                    "conversation_id": task.conversation_id,
                    "success": result.is_valid,
                    "error": result.error,
                    "quality_score": result.quality_score,
                    "duration": task.duration,
                })

            except Exception as e:
                logger.error(f"Worker {gpu_id} error on {task.conversation_id}: {e}")
                result_queue.put({
                    "conversation_id": task.conversation_id,
                    "success": False,
                    "error": str(e),
                    "quality_score": 0.0,
                    "duration": task.duration,
                })

    except Exception as e:
        logger.error(f"GPU Worker {gpu_id} fatal error: {e}")
    finally:
        logger.info(f"GPU Worker {gpu_id} stopping")


class Phase2Orchestrator:
    """Orchestrates Phase 2 GPU-based word alignment.

    Example usage:
        orchestrator = Phase2Orchestrator(config)

        # Process dataset using all GPUs
        stats = orchestrator.process_dataset(dataset_config)
        print(f"Processed {stats.processed_files} files")

        # Or process with specific GPUs
        stats = orchestrator.process_dataset(dataset_config, gpu_ids=[0, 1, 2, 3])
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the orchestrator.

        Args:
            config: Pipeline configuration
        """
        self.config = config

    def load_phase1_manifest(
        self,
        dataset_config: DatasetConfig,
        load_transcripts: bool = True,
        machine_id: Optional[int] = None,
        total_machines: Optional[int] = None,
    ) -> list[Phase2Task]:
        """Load Phase 1 manifest and create Phase 2 tasks.

        Args:
            dataset_config: Dataset configuration
            load_transcripts: Whether to load transcripts from metadata (for NFA)
            machine_id: Machine ID for distributed processing (0-indexed)
            total_machines: Total number of machines for distributed processing

        Returns:
            List of Phase2Task objects
        """
        manifest_path = dataset_config.manifest_path
        all_tasks = []

        if not manifest_path.exists():
            logger.error(f"Phase 1 manifest not found: {manifest_path}")
            return all_tasks

        output_dir = dataset_config.output_path / dataset_config.split
        metadata_dir = dataset_config.metadata_dir

        with open(manifest_path) as f:
            for line in f:
                data = json.loads(line)

                # Phase 1 manifest uses "audio_path" key
                # Support both "audio" (legacy) and "audio_path" (Phase 1 output)
                audio_rel_path = data.get("audio_path") or data.get("audio")
                if not audio_rel_path:
                    logger.warning(f"No audio path in manifest entry: {data.get('conversation_id', 'unknown')}")
                    continue

                audio_path = output_dir / audio_rel_path
                segment_path = output_dir / data.get("segment_alignment", "")

                if not audio_path.exists():
                    logger.warning(f"Audio not found: {audio_path}")
                    continue

                # Load transcripts from Phase 1 metadata if requested (for NFA)
                main_transcript = None
                user_transcript = None
                metadata_path = None

                if load_transcripts:
                    metadata_path = metadata_dir / f"{data['conversation_id']}.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, encoding="utf-8") as mf:
                                metadata = json.load(mf)

                            # Extract transcripts from segments
                            segments = metadata.get("segments", {})
                            main_segments = segments.get("main", [])
                            user_segments = segments.get("user", [])

                            main_transcript = " ".join(
                                s.get("text", "") for s in main_segments
                            ).strip()
                            user_transcript = " ".join(
                                s.get("text", "") for s in user_segments
                            ).strip()

                        except Exception as e:
                            logger.debug(f"Could not load metadata for {data['conversation_id']}: {e}")

                all_tasks.append(Phase2Task(
                    conversation_id=data["conversation_id"],
                    audio_path=audio_path,
                    segment_alignment_path=segment_path,
                    output_dir=output_dir,
                    duration=data["duration"],
                    main_transcript=main_transcript,
                    user_transcript=user_transcript,
                    metadata_path=metadata_path,
                ))

        # Sort by duration (longest first) for better load balancing
        if self.config.phase2.sort_by_duration:
            all_tasks.sort(key=lambda t: t.duration, reverse=True)

        # Distributed processing: filter tasks for this machine
        machine_id = machine_id if machine_id is not None else self.config.phase2.machine_id
        total_machines = total_machines if total_machines is not None else self.config.phase2.total_machines

        if total_machines > 1:
            # Round-robin assignment for better load balancing
            tasks = [t for i, t in enumerate(all_tasks) if i % total_machines == machine_id]
            logger.info(f"Machine {machine_id}/{total_machines}: {len(tasks)}/{len(all_tasks)} tasks assigned")
        else:
            tasks = all_tasks
            logger.info(f"Loaded {len(tasks)} tasks from manifest")

        if load_transcripts:
            with_transcripts = sum(1 for t in tasks if t.main_transcript or t.user_transcript)
            logger.info(f"  {with_transcripts}/{len(tasks)} tasks have transcripts")

        return tasks

    def process_dataset(
        self,
        dataset_config: DatasetConfig,
        gpu_ids: Optional[list[int]] = None,
        aligner_type: Optional[str] = None,
    ) -> Phase2Stats:
        """Process a dataset using multiple GPUs.

        Args:
            dataset_config: Dataset configuration
            gpu_ids: List of GPU IDs to use
            aligner_type: Override aligner type ("nfa" or "whisper")

        Returns:
            Processing statistics
        """
        start_time = time.time()
        stats = Phase2Stats()

        gpu_ids = gpu_ids or self.config.phase2.gpu_ids
        num_gpus = len(gpu_ids)
        aligner_type = aligner_type or self.config.phase2.aligner_type

        # Load tasks from Phase 1 manifest
        # Load transcripts for NFA aligner
        load_transcripts = (aligner_type == "nfa")
        tasks = self.load_phase1_manifest(dataset_config, load_transcripts=load_transcripts)
        stats.total_files = len(tasks)

        if not tasks:
            logger.warning("No tasks to process")
            return stats

        logger.info(f"Processing {stats.total_files} files with {num_gpus} GPUs using {aligner_type} aligner")

        # Create queues
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        stop_event = mp.Event()

        # Start worker processes
        workers = []
        for gpu_id in gpu_ids:
            p = mp.Process(
                target=gpu_worker,
                args=(gpu_id, task_queue, result_queue, self.config, stop_event, aligner_type),
            )
            p.start()
            workers.append(p)

        # Add tasks to queue
        for task in tasks:
            task_queue.put(task)

        # Add poison pills
        for _ in workers:
            task_queue.put(None)

        # Collect results
        quality_scores = []
        writer = MoshiFormatWriter(dataset_config)

        while stats.processed_files + stats.failed_files < stats.total_files:
            try:
                result = result_queue.get(timeout=60.0)

                if result["success"]:
                    stats.processed_files += 1
                    stats.total_duration_hours += result["duration"] / 3600
                    quality_scores.append(result["quality_score"])

                    # Create manifest entry
                    entry = writer.create_manifest_entry(
                        conversation_id=result["conversation_id"],
                        duration=result["duration"],
                    )
                    writer.add_manifest_entry(entry)
                else:
                    stats.failed_files += 1
                    logger.warning(
                        f"Failed {result['conversation_id']}: {result['error']}"
                    )

                # Log progress
                total_done = stats.processed_files + stats.failed_files
                if total_done % self.config.phase2.log_interval == 0:
                    elapsed = time.time() - start_time
                    rate = stats.total_duration_hours * 3600 / max(elapsed, 1)
                    logger.info(
                        f"Progress: {total_done}/{stats.total_files} "
                        f"({stats.processed_files} success, {stats.failed_files} failed) "
                        f"Speed: {rate:.1f}x realtime"
                    )

            except queue.Empty:
                # Check if workers are still alive
                alive = sum(1 for w in workers if w.is_alive())
                if alive == 0:
                    logger.warning("All workers have stopped")
                    break

        # Stop workers
        stop_event.set()
        for w in workers:
            w.join(timeout=10.0)
            if w.is_alive():
                w.terminate()

        # Finalize
        writer.finalize(phase=2)

        stats.processing_time_sec = time.time() - start_time
        if quality_scores:
            stats.avg_quality_score = sum(quality_scores) / len(quality_scores)

        return stats

    def process_single_gpu(
        self,
        dataset_config: DatasetConfig,
        gpu_id: int = 0,
        aligner_type: Optional[str] = None,
        machine_id: Optional[int] = None,
        total_machines: Optional[int] = None,
    ) -> Phase2Stats:
        """Process dataset on a single GPU (for distributed processing).

        Args:
            dataset_config: Dataset configuration
            gpu_id: GPU device ID
            aligner_type: Override aligner type ("nfa" or "whisper")
            machine_id: Machine ID for distributed processing (0-indexed)
            total_machines: Total number of machines

        Returns:
            Processing statistics
        """
        start_time = time.time()
        stats = Phase2Stats()
        aligner_type = aligner_type or self.config.phase2.aligner_type

        # Get machine settings
        machine_id = machine_id if machine_id is not None else self.config.phase2.machine_id
        total_machines = total_machines if total_machines is not None else self.config.phase2.total_machines

        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Load tasks with transcripts for NFA (with distributed filtering)
        load_transcripts = (aligner_type == "nfa")
        tasks = self.load_phase1_manifest(
            dataset_config,
            load_transcripts=load_transcripts,
            machine_id=machine_id,
            total_machines=total_machines,
        )
        stats.total_files = len(tasks)

        if not tasks:
            return stats

        logger.info(f"Machine {machine_id}/{total_machines}: Processing {stats.total_files} files using {aligner_type} aligner on GPU {gpu_id}")

        # Checkpoint handling
        checkpoint_path = self._get_checkpoint_path(dataset_config, machine_id)
        processed_ids = set()
        if self.config.phase2.resume_from_checkpoint and checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    checkpoint = json.load(f)
                    processed_ids = set(checkpoint.get("processed_ids", []))
                logger.info(f"Resuming from checkpoint: {len(processed_ids)} already processed")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

        # Filter out already processed tasks
        if processed_ids:
            original_count = len(tasks)
            tasks = [t for t in tasks if t.conversation_id not in processed_ids]
            stats.processed_files = original_count - len(tasks)
            logger.info(f"Skipping {original_count - len(tasks)} already processed tasks")

        # Initialize aligner based on type
        AlignerClass = get_aligner(aligner_type)
        if aligner_type == "nfa":
            aligner = AlignerClass(self.config.nfa)
        else:
            aligner = AlignerClass(self.config.whisper_alignment)

        # Create machine-specific manifest writer
        writer = MoshiFormatWriter(dataset_config, machine_id=machine_id)
        quality_scores = []

        for i, task in enumerate(tasks):
            try:
                # Different parameters for NFA vs Whisper
                if aligner_type == "nfa":
                    result = aligner.align_stereo(
                        audio_path=task.audio_path,
                        conversation_id=task.conversation_id,
                        main_transcript=task.main_transcript,
                        user_transcript=task.user_transcript,
                        metadata_path=task.metadata_path,
                    )
                else:
                    result = aligner.align_stereo(
                        audio_path=task.audio_path,
                        conversation_id=task.conversation_id,
                    )

                if result.is_valid:
                    # Save alignments
                    if result.main_alignment:
                        align01_path = task.output_dir / "alignment_speaker01" / f"{task.conversation_id}.json"
                        result.main_alignment.save(align01_path)
                    if result.user_alignment:
                        align02_path = task.output_dir / "alignment_speaker02" / f"{task.conversation_id}.json"
                        result.user_alignment.save(align02_path)

                    # Save Moshi format alignment (combined)
                    moshi_align_path = task.output_dir / "alignments" / f"{task.conversation_id}.json"
                    self._save_moshi_alignment(result, moshi_align_path)

                    stats.processed_files += 1
                    stats.total_duration_hours += task.duration / 3600
                    quality_scores.append(result.quality_score)
                    processed_ids.add(task.conversation_id)

                    entry = writer.create_manifest_entry(
                        conversation_id=task.conversation_id,
                        duration=task.duration,
                    )
                    writer.add_manifest_entry(entry)
                else:
                    stats.failed_files += 1
                    logger.warning(f"Failed {task.conversation_id}: {result.error}")

            except Exception as e:
                stats.failed_files += 1
                logger.error(f"Error processing {task.conversation_id}: {e}")

            # Log progress
            if (i + 1) % self.config.phase2.log_interval == 0:
                elapsed = time.time() - start_time
                rate = stats.total_duration_hours * 3600 / max(elapsed, 1)
                logger.info(
                    f"Progress: {i + 1}/{len(tasks)} "
                    f"({stats.processed_files} success, {stats.failed_files} failed) "
                    f"Speed: {rate:.1f}x realtime"
                )

            # Save checkpoint periodically
            if (i + 1) % self.config.phase2.checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_path, processed_ids, stats)

        # Finalize
        writer.finalize(phase=2, machine_id=machine_id)

        # Save final checkpoint
        self._save_checkpoint(checkpoint_path, processed_ids, stats)

        stats.processing_time_sec = time.time() - start_time
        if quality_scores:
            stats.avg_quality_score = sum(quality_scores) / len(quality_scores)

        return stats

    def _get_checkpoint_path(self, dataset_config: DatasetConfig, machine_id: int) -> Path:
        """Get checkpoint path for a machine."""
        checkpoint_dir = dataset_config.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / f"phase2_machine_{machine_id:03d}.json"

    def _save_checkpoint(self, checkpoint_path: Path, processed_ids: set, stats: Phase2Stats):
        """Save checkpoint for resume support."""
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "processed_ids": list(processed_ids),
                    "stats": stats.to_dict(),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f)
            logger.debug(f"Saved checkpoint: {len(processed_ids)} processed")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _save_moshi_alignment(self, result, output_path: Path):
        """Save alignment in Moshi format."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Combine alignments from both speakers
            alignments = []

            if result.main_alignment and result.main_alignment.words:
                for word in result.main_alignment.words:
                    alignments.append([
                        word.text,
                        [round(word.start, 3), round(word.end, 3)],
                        "SPEAKER_MAIN"
                    ])

            if result.user_alignment and result.user_alignment.words:
                for word in result.user_alignment.words:
                    alignments.append([
                        word.text,
                        [round(word.start, 3), round(word.end, 3)],
                        "SPEAKER_USER"
                    ])

            # Sort by start time
            alignments.sort(key=lambda x: x[1][0])

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"alignments": alignments}, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save Moshi alignment: {e}")

    def process_single_gpu_batched(
        self,
        dataset_config: DatasetConfig,
        gpu_id: int = 0,
        aligner_type: Optional[str] = None,
        machine_id: Optional[int] = None,
        total_machines: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Phase2Stats:
        """Process dataset on a single GPU with optimized batching.

        This method batches multiple conversations together for better GPU utilization.
        On A100 80GB, this can achieve 20-50x speedup compared to single-conversation processing.

        Args:
            dataset_config: Dataset configuration
            gpu_id: GPU device ID
            aligner_type: Override aligner type ("nfa" or "whisper")
            machine_id: Machine ID for distributed processing (0-indexed)
            total_machines: Total number of machines
            batch_size: Batch size for GPU processing (default: from config)

        Returns:
            Processing statistics
        """
        start_time = time.time()
        stats = Phase2Stats()
        aligner_type = aligner_type or self.config.phase2.aligner_type
        batch_size = batch_size or self.config.phase2.batch_size or DEFAULT_BATCH_SIZE

        # Get machine settings
        machine_id = machine_id if machine_id is not None else self.config.phase2.machine_id
        total_machines = total_machines if total_machines is not None else self.config.phase2.total_machines

        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Load tasks with transcripts for NFA (with distributed filtering)
        load_transcripts = (aligner_type == "nfa")
        tasks = self.load_phase1_manifest(
            dataset_config,
            load_transcripts=load_transcripts,
            machine_id=machine_id,
            total_machines=total_machines,
        )
        stats.total_files = len(tasks)

        if not tasks:
            return stats

        logger.info(
            f"Machine {machine_id}/{total_machines}: Processing {stats.total_files} files "
            f"using {aligner_type} aligner on GPU {gpu_id} (batch_size={batch_size})"
        )

        # Checkpoint handling
        checkpoint_path = self._get_checkpoint_path(dataset_config, machine_id)
        processed_ids = set()
        if self.config.phase2.resume_from_checkpoint and checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    checkpoint = json.load(f)
                    processed_ids = set(checkpoint.get("processed_ids", []))
                logger.info(f"Resuming from checkpoint: {len(processed_ids)} already processed")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

        # Filter out already processed tasks
        if processed_ids:
            original_count = len(tasks)
            tasks = [t for t in tasks if t.conversation_id not in processed_ids]
            stats.processed_files = original_count - len(tasks)
            logger.info(f"Skipping {original_count - len(tasks)} already processed tasks")

        # Initialize aligner with batch_size
        AlignerClass = get_aligner(aligner_type)
        if aligner_type == "nfa":
            # Ensure batch_size is set in config
            self.config.nfa.batch_size = batch_size
            aligner = AlignerClass(self.config.nfa)
        else:
            aligner = AlignerClass(self.config.whisper_alignment)

        # Create machine-specific manifest writer
        writer = MoshiFormatWriter(dataset_config, machine_id=machine_id)
        quality_scores = []

        # Group tasks into batches by total duration
        task_batches = self._create_task_batches(tasks, batch_size, MAX_BATCH_AUDIO_DURATION_SEC)
        logger.info(f"Created {len(task_batches)} batches from {len(tasks)} tasks")

        batch_idx = 0
        for batch in task_batches:
            batch_start = time.time()
            batch_results = []

            for task in batch:
                try:
                    # Different parameters for NFA vs Whisper
                    if aligner_type == "nfa":
                        result = aligner.align_stereo(
                            audio_path=task.audio_path,
                            conversation_id=task.conversation_id,
                            main_transcript=task.main_transcript,
                            user_transcript=task.user_transcript,
                            metadata_path=task.metadata_path,
                        )
                    else:
                        result = aligner.align_stereo(
                            audio_path=task.audio_path,
                            conversation_id=task.conversation_id,
                        )

                    batch_results.append((task, result, None))

                except Exception as e:
                    batch_results.append((task, None, str(e)))

            # Process batch results
            for task, result, error in batch_results:
                if result and result.is_valid:
                    # Save alignments
                    if result.main_alignment:
                        align01_path = task.output_dir / "alignment_speaker01" / f"{task.conversation_id}.json"
                        result.main_alignment.save(align01_path)
                    if result.user_alignment:
                        align02_path = task.output_dir / "alignment_speaker02" / f"{task.conversation_id}.json"
                        result.user_alignment.save(align02_path)

                    # Save Moshi format alignment
                    moshi_align_path = task.output_dir / "alignments" / f"{task.conversation_id}.json"
                    self._save_moshi_alignment(result, moshi_align_path)

                    stats.processed_files += 1
                    stats.total_duration_hours += task.duration / 3600
                    quality_scores.append(result.quality_score)
                    processed_ids.add(task.conversation_id)

                    entry = writer.create_manifest_entry(
                        conversation_id=task.conversation_id,
                        duration=task.duration,
                    )
                    writer.add_manifest_entry(entry)
                else:
                    stats.failed_files += 1
                    error_msg = error or (result.error if result else "Unknown error")
                    logger.warning(f"Failed {task.conversation_id}: {error_msg}")

            batch_idx += 1
            batch_elapsed = time.time() - batch_start
            batch_duration = sum(t.duration for t in batch)

            # Log batch progress
            if batch_idx % max(1, len(task_batches) // 20) == 0 or batch_idx == len(task_batches):
                elapsed = time.time() - start_time
                rate = stats.total_duration_hours * 3600 / max(elapsed, 1)
                logger.info(
                    f"Batch {batch_idx}/{len(task_batches)} | "
                    f"Progress: {stats.processed_files + stats.failed_files}/{len(tasks)} | "
                    f"Batch speed: {batch_duration/max(batch_elapsed, 0.1):.1f}x | "
                    f"Overall: {rate:.1f}x realtime"
                )

            # Save checkpoint periodically
            tasks_done = stats.processed_files + stats.failed_files
            if tasks_done % self.config.phase2.checkpoint_interval < len(batch):
                self._save_checkpoint(checkpoint_path, processed_ids, stats)

            # Periodic GPU memory cleanup to prevent OOM during long runs
            if batch_idx % 10 == 0:  # Every 10 batches
                cleanup_gpu_memory(force=True)
                if logger.isEnabledFor(logging.DEBUG):
                    mem_info = get_gpu_memory_info()
                    logger.debug(
                        f"GPU memory after batch {batch_idx}: "
                        f"{mem_info.get('allocated_gb', 0):.1f}GB allocated"
                    )

        # Finalize
        writer.finalize(phase=2, machine_id=machine_id)

        # Save final checkpoint
        self._save_checkpoint(checkpoint_path, processed_ids, stats)

        stats.processing_time_sec = time.time() - start_time
        if quality_scores:
            stats.avg_quality_score = sum(quality_scores) / len(quality_scores)

        # Log final summary
        logger.info(
            f"Machine {machine_id} completed: "
            f"{stats.processed_files} success, {stats.failed_files} failed | "
            f"{stats.total_duration_hours:.1f}h audio in {stats.processing_time_sec:.0f}s | "
            f"Speed: {stats.total_duration_hours * 3600 / max(stats.processing_time_sec, 1):.1f}x realtime"
        )

        return stats

    def _create_task_batches(
        self,
        tasks: List[Phase2Task],
        batch_size: int,
        max_duration_sec: float = 600.0,
    ) -> List[List[Phase2Task]]:
        """Create batches of tasks for efficient GPU processing.

        Batches are created to:
        1. Not exceed batch_size conversations
        2. Not exceed max_duration_sec total audio duration
        3. Group similar-length conversations for efficient padding

        Args:
            tasks: List of tasks to batch
            batch_size: Maximum conversations per batch
            max_duration_sec: Maximum total audio duration per batch

        Returns:
            List of task batches
        """
        if not tasks:
            return []

        # Sort by duration for more efficient batching (similar lengths together)
        sorted_tasks = sorted(tasks, key=lambda t: t.duration)

        batches = []
        current_batch = []
        current_duration = 0.0

        for task in sorted_tasks:
            # Check if adding this task would exceed limits
            if (len(current_batch) >= batch_size or
                current_duration + task.duration > max_duration_sec):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [task]
                current_duration = task.duration
            else:
                current_batch.append(task)
                current_duration += task.duration

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        return batches
