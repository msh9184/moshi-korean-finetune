import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import sphn
import torch.distributed as dist

from finetune.distributed import get_rank

from .interleaver import InterleavedTokenizer, Sample, get_filtering_statistics

logger = logging.getLogger("dataset")


AudioChunkPath = tuple[str, float]
_LOADED_DATASETS: dict[Path, list[AudioChunkPath]] = {}


def main_logger_info(message: str) -> None:
    if dist.is_initialized() and get_rank() == 0:
        logger.info(message)


def load_file(path: Path, world_size: int, rank: int) -> list[str]:
    lines = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if not idx % world_size == rank:
                continue
            lines.append(line)
    return lines


def maybe_load_local_dataset(
    path: Path, rank: int, world_size: int, instruct_tokenizer: InterleavedTokenizer
) -> list[AudioChunkPath]:
    if path in _LOADED_DATASETS:
        return _LOADED_DATASETS[path]

    duration = instruct_tokenizer.duration_sec
    main_logger_info(f"Loading {path} ...")
    lines: list[str] = load_file(path, rank=rank, world_size=world_size)

    chunks: list[AudioChunkPath] = []
    for line in lines:
        data = json.loads(line)
        start_sec = 0
        while start_sec < data["duration"]:
            chunks.append((data["path"], start_sec))
            start_sec += duration

    main_logger_info(f"{path} loaded and chunked.")
    _LOADED_DATASETS[path] = chunks

    return _LOADED_DATASETS[path]


@dataclass
class DataDir:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        jsonl_files = list(self.path.rglob("*jsonl"))
        assert len(jsonl_files) > 0, (
            f"{self.path} does not seem to have any files ending with '.jsonl'"
        )
        return jsonl_files


@dataclass
class DataFile:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        return [self.path]


def parse_data_sources(
    pretrain_data: str,
) -> tuple[list[DataDir | DataFile], list[float]]:
    seen: set[str] = set()
    sources: list[DataDir | DataFile] = []
    weights: list[float] = []

    sample_sources = pretrain_data

    for source in sample_sources.strip().split(","):
        if not source:
            continue

        source_items = source.strip().split(":")
        if len(source_items) == 1:
            path_ = source_items[0]
            weight = 1.0
        elif len(source_items) == 2:
            path_, weight_ = source_items
            weight = float(weight_)
        else:
            raise ValueError(
                f"{source} is not correctly formatted. Make sure to format each data source "
                "as <path/to/data>:<weight> or just <path/to/data>"
            )

        assert path_ not in seen, (
            f"{path_} seems to be duplicated. Make sure to only add it once."
        )
        assert weight > 0, (
            f"Make sure to define strictly positive data sampling weights, not {weight}"
        )

        data: DataDir | DataFile
        if Path(path_).is_dir():
            data = DataDir(path=Path(path_))
        elif Path(path_).is_file():
            data = DataFile(path=Path(path_))
        else:
            raise FileNotFoundError(
                f"The path {path_} does not exist. Make sure {path_} is either a file or directory "
                "that contains training data."
            )

        sources.append(data)
        weights.append(weight)

        seen.add(path_)

    sum_weights = sum(weights)
    n_weights = [weight / sum_weights for weight in weights]

    assert min(n_weights) > 0
    assert abs(1 - sum(n_weights)) < 1e-8, (
        f"Defined data sampling weights {weights} must sum to 1."
    )
    return sources, n_weights


def build_dataset(
    pretrain_data: str,
    instruct_tokenizer: InterleavedTokenizer,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
    shuffle_pretrain: bool = False,
) -> Iterator[Sample]:
    sources, probabilities = parse_data_sources(pretrain_data=pretrain_data)

    shuffle = not is_eval and shuffle_pretrain

    dataset_iterators = [
        get_dataset_iterator(
            source,
            instruct_tokenizer=instruct_tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            shuffle_at_epoch=shuffle,
        )
        for source in sources
    ]

    if is_eval:
        combined_iterator = itertools.chain.from_iterable(dataset_iterators)
    else:
        # make sure random_seed is different per rank and original seed
        random_seed = np.array((seed, rank))
        rng = np.random.RandomState(seed=random_seed)
        combined_iterator = interleave_iterators(
            dataset_iterators, probabilities=probabilities, rng=rng
        )

    return combined_iterator


def get_rng(seed: int, rank: int) -> np.random.RandomState:
    random_seed = np.array((seed, rank))
    rng = np.random.RandomState(seed=random_seed)
    return rng


def get_dataset_iterator(
    source: DataDir | DataFile,
    instruct_tokenizer: InterleavedTokenizer,
    rank: int,
    world_size: int,
    is_finite: bool,
    seed: int | None,
    shuffle_at_epoch: bool,
) -> Iterator[Sample]:
    epoch = 1
    first_sample_logged = False  # Debug flag for first sample

    while True:
        for jsonl_file in source.jsonl_files:
            # Set the parent directory of the JSONL file to the tokenizer
            # This allows JSON alignment files to be found correctly
            jsonl_base_dir = jsonl_file.parent
            instruct_tokenizer.jsonl_base_dir = jsonl_base_dir
            instruct_tokenizer._path_resolution_logged = False  # Reset log flag for each new JSONL

            # Using direct logger.warning for MPI visibility
            logger.warning(f"[DATASET] Processing JSONL: {jsonl_file}, base_dir: {jsonl_base_dir}")
            logger.warning(
                f"[DATASET DEBUG] mimi.sample_rate={instruct_tokenizer.mimi.sample_rate}, "
                f"duration_sec={instruct_tokenizer.duration_sec}"
            )

            import sys
            sys.stdout.flush()
            sys.stderr.flush()

            logger.warning(f"[DATASET] Calling sphn.dataset_jsonl()...")
            try:
                dataset = sphn.dataset_jsonl(
                    str(jsonl_file),
                    duration_sec=instruct_tokenizer.duration_sec,
                    num_threads=4,
                    sample_rate=instruct_tokenizer.mimi.sample_rate,
                    pad_last_segment=True,
                )
                logger.warning(f"[DATASET] sphn.dataset_jsonl() returned successfully.")
            except Exception as e:
                logger.error(f"[DATASET ERROR] sphn.dataset_jsonl() failed: {type(e).__name__}: {e}")
                raise
            if shuffle_at_epoch:
                dataset = dataset.shuffle(
                    with_replacement=False, skip=rank, step_by=world_size, seed=seed
                )
                seed += 1
            else:
                dataset = dataset.seq(skip=rank, step_by=world_size)

            logger.warning(f"[DATASET] Starting iteration over dataset...")
            sample_count = 0

            for sample in dataset:
                if sample_count == 0:
                    logger.warning(f"[DATASET] Received first sample from sphn iterator")

                raw_data = sample["data"]
                wav = raw_data[..., : sample["unpadded_len"]]

                sample_count += 1

                # CRITICAL DEBUG: Log first sample to verify sphn output shape
                # Using WARNING level because stdout (INFO) is not visible in MPI environment
                if not first_sample_logged:
                    logger.warning(f"[DATASET DEBUG] ===== FIRST SAMPLE FROM SPHN =====")
                    logger.warning(
                        f"[DATASET DEBUG] raw sample['data']: shape={raw_data.shape}, "
                        f"dtype={raw_data.dtype}"
                    )
                    logger.warning(
                        f"[DATASET DEBUG] wav after slice: shape={wav.shape}, "
                        f"ndim={wav.ndim}, min={wav.min():.4f}, max={wav.max():.4f}"
                    )
                    logger.warning(
                        f"[DATASET DEBUG] sample keys: {list(sample.keys())}"
                    )
                    logger.warning(
                        f"[DATASET DEBUG] path={sample['path']}, "
                        f"start_time_sec={sample['start_time_sec']}, "
                        f"unpadded_len={sample['unpadded_len']}"
                    )
                    # Check if stereo is preserved
                    if wav.ndim == 2 and wav.shape[0] == 2:
                        logger.warning("[DATASET DEBUG] ✓ Stereo audio detected (2 channels)")
                    elif wav.ndim == 2 and wav.shape[0] == 1:
                        logger.warning("[DATASET DEBUG] ⚠ Mono audio (1 channel) - stereo expected!")
                    elif wav.ndim == 1:
                        logger.warning("[DATASET DEBUG] ⚠ 1D audio array - stereo expected!")
                    else:
                        logger.warning(f"[DATASET DEBUG] ⚠ Unexpected shape: {wav.shape}")
                    logger.warning(f"[DATASET DEBUG] ===== END FIRST SAMPLE =====")
                    logger.warning(f"[DATASET DEBUG] Calling instruct_tokenizer()...")
                    first_sample_logged = True

                try:
                    result = instruct_tokenizer(wav, sample["start_time_sec"], sample["path"])

                    # Handle yield_both mode: result can be a list of samples
                    if isinstance(result, list):
                        for sample_item in result:
                            # Skip samples with skip_reason (filtered out)
                            if sample_item.skip_reason is not None:
                                if sample_count == 1:
                                    logger.warning(
                                        f"[DATASET] Sample skipped: {sample_item.skip_reason}"
                                    )
                                continue
                            if sample_count == 1:
                                logger.warning(
                                    f"[DATASET DEBUG] instruct_tokenizer() returned list "
                                    f"(yield_both mode, {len(result)} samples)"
                                )
                                logger.warning(f"[DATASET DEBUG] result.codes shape: {sample_item.codes.shape}")
                                if sample_item.is_role_swapped:
                                    logger.warning(f"[DATASET DEBUG] Sample is role-swapped")
                            yield sample_item
                    else:
                        # Single sample mode
                        # Skip samples with skip_reason (filtered out)
                        if result.skip_reason is not None:
                            if sample_count == 1:
                                logger.warning(
                                    f"[DATASET] Sample skipped: {result.skip_reason}"
                                )
                            continue

                        if sample_count == 1:
                            logger.warning(f"[DATASET DEBUG] instruct_tokenizer() returned successfully for first sample")
                            logger.warning(f"[DATASET DEBUG] result.codes shape: {result.codes.shape}")
                            if result.is_role_swapped:
                                logger.warning(f"[DATASET DEBUG] Sample is role-swapped")
                        yield result
                except Exception as e:
                    logger.error(f"[DATASET ERROR] instruct_tokenizer() failed: {type(e).__name__}: {e}")
                    raise
        if is_finite:
            break

        # Log filtering statistics at epoch end
        filter_stats = get_filtering_statistics()
        if filter_stats.total_segments > 0:
            filter_stats.log_summary(logger.info)

        print(f"Rank {rank} finished epoch {epoch}")
        epoch += 1


def interleave_iterators(iterators: list[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])
