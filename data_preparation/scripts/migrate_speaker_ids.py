#!/usr/bin/env python3
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Migrate Phase 1 metadata to include original speaker IDs per segment.

Version 3.1 - NEVER SKIP FILES (Use Placeholders for Unknown Speakers)

CRITICAL FIX in v3.1:
- Files with unknown main_speaker_id are NO LONGER SKIPPED
- Uses UNKNOWN_MAIN placeholder instead (ensures data consistency)
- User requirement: "화자 구분만 되면 된다" - speaker distinction is what matters

This script reads existing Phase 1 metadata JSON files and enriches each
segment with its original speaker ID by matching against diarization data.

EDGE CASES HANDLED:
1. Unknown speaker IDs: "?", "", null, "UNKNOWN", "N/A" → Uses placeholder
2. Unknown main speaker → Uses UNKNOWN_MAIN (NOT skipped)
3. Empty text segments
4. Missing or invalid timestamps
5. Missing diarization section
6. Empty speakers list
7. Floating point precision issues in timestamp matching
8. Duplicate segments with same timestamps
9. Structural variations in metadata format
10. Very large files with many speakers/segments
11. Metadata inconsistencies between sections

Usage:
    # Dry run with verbose output
    python migrate_speaker_ids.py \\
        --phase1-dir /path/to/phase1/output/train \\
        --dry-run --verbose --limit 100

    # Full migration with backup
    python migrate_speaker_ids.py \\
        --phase1-dir /path/to/phase1/output/train \\
        --backup

    # Output to separate directory with detailed report
    python migrate_speaker_ids.py \\
        --phase1-dir /path/to/phase1/output/train \\
        --output-dir /path/to/migrated/output \\
        --report
"""

import argparse
import json
import logging
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants for edge case handling
# ============================================================================

# Speaker IDs that indicate unknown/unidentified speakers
UNKNOWN_SPEAKER_PATTERNS = {
    "?",
    "",
    "unknown",
    "UNKNOWN",
    "n/a",
    "N/A",
    "none",
    "None",
    "null",
    "NULL",
    "-",
    "speaker_unknown",
    "unidentified",
}

# Default speaker ID to use when no match is found
DEFAULT_UNKNOWN_SPEAKER = "UNKNOWN_SPEAKER"

# Placeholder for unknown main speaker (never skip, always process)
UNKNOWN_MAIN_SPEAKER = "UNKNOWN_MAIN"

# Timestamp matching tolerance (in seconds)
TIMESTAMP_TOLERANCE_EXACT = 0.001  # 1ms for exact matching
TIMESTAMP_TOLERANCE_FUZZY = 0.05   # 50ms for fuzzy matching
TIMESTAMP_TOLERANCE_LOOSE = 0.1    # 100ms for loose matching


# ============================================================================
# Statistics and Reporting
# ============================================================================

@dataclass
class EdgeCaseStats:
    """Track edge cases encountered during migration."""
    unknown_speaker_ids: int = 0
    unknown_main_speaker: int = 0  # Files with unknown main speaker (recovered, not skipped)
    empty_text_segments: int = 0
    missing_timestamps: int = 0
    invalid_timestamps: int = 0
    no_diarization: int = 0
    empty_speakers_list: int = 0
    duplicate_timestamps: int = 0
    unmatched_segments: int = 0
    fuzzy_matched: int = 0
    loose_matched: int = 0
    fallback_used: int = 0

    def to_dict(self) -> dict:
        return {
            "unknown_speaker_ids": self.unknown_speaker_ids,
            "unknown_main_speaker": self.unknown_main_speaker,
            "empty_text_segments": self.empty_text_segments,
            "missing_timestamps": self.missing_timestamps,
            "invalid_timestamps": self.invalid_timestamps,
            "no_diarization": self.no_diarization,
            "empty_speakers_list": self.empty_speakers_list,
            "duplicate_timestamps": self.duplicate_timestamps,
            "unmatched_segments": self.unmatched_segments,
            "fuzzy_matched": self.fuzzy_matched,
            "loose_matched": self.loose_matched,
            "fallback_used": self.fallback_used,
        }

    def total_edge_cases(self) -> int:
        return sum(self.to_dict().values())


@dataclass
class MigrationStats:
    """Comprehensive statistics from migration process."""
    # File counts
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    error_files: int = 0

    # Segment counts
    total_main_segments: int = 0
    total_user_segments: int = 0
    main_segments_updated: int = 0
    user_segments_updated: int = 0
    already_has_speaker_id: int = 0

    # Multi-speaker tracking
    matched_via_diarization: int = 0
    multi_speaker_files: int = 0
    unique_speakers_found: int = 0
    max_speakers_in_file: int = 0

    # Edge cases
    edge_cases: EdgeCaseStats = field(default_factory=EdgeCaseStats)

    # Problem files tracking
    problem_files: list = field(default_factory=list)

    @property
    def total_segments(self) -> int:
        return self.total_main_segments + self.total_user_segments

    @property
    def total_updated(self) -> int:
        return self.main_segments_updated + self.user_segments_updated

    def update_rate(self) -> float:
        if self.total_segments == 0:
            return 0.0
        return self.total_updated / self.total_segments * 100

    def match_rate(self) -> float:
        """Rate of segments matched via diarization."""
        total_attempted = (
            self.matched_via_diarization +
            self.edge_cases.unmatched_segments +
            self.edge_cases.fuzzy_matched +
            self.edge_cases.loose_matched +
            self.edge_cases.fallback_used
        )
        if total_attempted == 0:
            return 100.0
        matched = (
            self.matched_via_diarization +
            self.edge_cases.fuzzy_matched +
            self.edge_cases.loose_matched
        )
        return matched / total_attempted * 100

    def to_dict(self) -> dict:
        return {
            "file_stats": {
                "total_files": self.total_files,
                "processed_files": self.processed_files,
                "skipped_files": self.skipped_files,
                "error_files": self.error_files,
                "multi_speaker_files": self.multi_speaker_files,
            },
            "segment_stats": {
                "total_main_segments": self.total_main_segments,
                "total_user_segments": self.total_user_segments,
                "main_segments_updated": self.main_segments_updated,
                "user_segments_updated": self.user_segments_updated,
                "already_has_speaker_id": self.already_has_speaker_id,
                "matched_via_diarization": self.matched_via_diarization,
            },
            "speaker_stats": {
                "unique_speakers_found": self.unique_speakers_found,
                "max_speakers_in_file": self.max_speakers_in_file,
            },
            "edge_cases": self.edge_cases.to_dict(),
            "rates": {
                "update_rate_percent": round(self.update_rate(), 2),
                "match_rate_percent": round(self.match_rate(), 2),
            },
            "problem_files_count": len(self.problem_files),
        }


# ============================================================================
# Helper Functions
# ============================================================================

def is_unknown_speaker(speaker_id: Optional[str]) -> bool:
    """Check if a speaker ID represents an unknown/unidentified speaker."""
    if speaker_id is None:
        return True
    if not isinstance(speaker_id, str):
        return True
    speaker_id_clean = speaker_id.strip().lower()
    return speaker_id_clean in {p.lower() for p in UNKNOWN_SPEAKER_PATTERNS}


def normalize_speaker_id(speaker_id: Optional[str]) -> str:
    """Normalize a speaker ID, handling edge cases."""
    if speaker_id is None:
        return DEFAULT_UNKNOWN_SPEAKER
    if not isinstance(speaker_id, str):
        return DEFAULT_UNKNOWN_SPEAKER

    speaker_id = speaker_id.strip()

    if is_unknown_speaker(speaker_id):
        return DEFAULT_UNKNOWN_SPEAKER

    # Limit length to prevent issues
    if len(speaker_id) > 100:
        return speaker_id[:100]

    return speaker_id


def validate_timestamp(start: Optional[float], end: Optional[float]) -> tuple[bool, str]:
    """Validate a timestamp pair.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if start is None or end is None:
        return False, "missing_value"

    try:
        start = float(start)
        end = float(end)
    except (ValueError, TypeError):
        return False, "invalid_type"

    if start < 0 or end < 0:
        return False, "negative_value"

    if start > end:
        return False, "start_after_end"

    # Very long segments might indicate issues
    if end - start > 3600:  # More than 1 hour
        return False, "too_long"

    return True, ""


def find_metadata_files(phase1_dir: Path) -> tuple[list[Path], Path]:
    """Find all metadata files in Phase 1 output.

    Handles multiple possible directory structures:
    1. phase1_dir/metadata/*.json
    2. phase1_dir/segment_alignment/*.json
    3. phase1_dir/*.json (direct)

    Returns:
        Tuple of (list of metadata file paths, metadata directory)
    """
    # Strategy 1: metadata/ directory (current structure)
    metadata_dir = phase1_dir / "metadata"
    if metadata_dir.exists():
        files = list(metadata_dir.glob("*.json"))
        if files:
            logger.info(f"Found {len(files)} files in {metadata_dir}")
            return sorted(files), metadata_dir

    # Strategy 2: segment_alignment/ directory (legacy structure)
    segment_dir = phase1_dir / "segment_alignment"
    if segment_dir.exists():
        files = list(segment_dir.glob("*.json"))
        if files:
            logger.info(f"Found {len(files)} files in {segment_dir}")
            return sorted(files), segment_dir

    # Strategy 3: Direct JSON files in phase1_dir
    files = list(phase1_dir.glob("*.json"))
    # Filter out stats/report files
    excluded = ["merge_report", "merged_stats", "stats", "manifest", "migration_report"]
    files = [f for f in files if not any(ex in f.name.lower() for ex in excluded)]
    if files:
        logger.info(f"Found {len(files)} files directly in {phase1_dir}")
        return sorted(files), phase1_dir

    return [], phase1_dir / "metadata"


# ============================================================================
# Core Migration Logic
# ============================================================================

def build_segment_to_speaker_map(
    metadata: dict,
    stats: MigrationStats
) -> dict[tuple[float, float], list[str]]:
    """Build a mapping from (start, end) to speaker_id(s) using diarization data.

    This handles:
    - Unknown speaker IDs ("?", "", null)
    - Missing diarization section
    - Empty speakers list
    - Duplicate timestamps

    Args:
        metadata: The full metadata dict
        stats: Statistics object to update

    Returns:
        Dict mapping (start, end) tuples to list of speaker_id strings
        (list because duplicates are possible)
    """
    segment_to_speaker: dict[tuple[float, float], list[str]] = defaultdict(list)

    diarization = metadata.get("diarization", {})

    # Edge case: No diarization section
    if not diarization:
        stats.edge_cases.no_diarization += 1
        return dict(segment_to_speaker)

    speakers = diarization.get("speakers", [])

    # Edge case: Empty speakers list
    if not speakers:
        stats.edge_cases.empty_speakers_list += 1
        return dict(segment_to_speaker)

    for speaker_info in speakers:
        speaker_id = speaker_info.get("speaker_id")

        # Track unknown speaker IDs
        if is_unknown_speaker(speaker_id):
            stats.edge_cases.unknown_speaker_ids += 1

        # Normalize the speaker ID
        normalized_id = normalize_speaker_id(speaker_id)

        for seg in speaker_info.get("segments", []):
            start = seg.get("start")
            end = seg.get("end")

            # Validate timestamp
            is_valid, error = validate_timestamp(start, end)
            if not is_valid:
                if error == "missing_value":
                    stats.edge_cases.missing_timestamps += 1
                else:
                    stats.edge_cases.invalid_timestamps += 1
                continue

            # Track empty text segments
            text = seg.get("text", "")
            if not text or not text.strip():
                stats.edge_cases.empty_text_segments += 1

            # Round for consistent matching
            key = (round(float(start), 3), round(float(end), 3))

            # Track duplicates
            if key in segment_to_speaker:
                stats.edge_cases.duplicate_timestamps += 1

            segment_to_speaker[key].append(normalized_id)

    return dict(segment_to_speaker)


def find_speaker_for_segment(
    start: float,
    end: float,
    segment_to_speaker: dict[tuple[float, float], list[str]],
    stats: MigrationStats,
    fallback_speaker: str = DEFAULT_UNKNOWN_SPEAKER
) -> tuple[str, str]:
    """Find the speaker ID for a segment with multi-level matching.

    Matching levels:
    1. Exact match (within 1ms)
    2. Fuzzy match (within 50ms)
    3. Loose match (within 100ms)
    4. Fallback to provided speaker

    Args:
        start: Segment start time
        end: Segment end time
        segment_to_speaker: Mapping from timestamps to speaker IDs
        stats: Statistics to update
        fallback_speaker: Speaker ID to use if no match found

    Returns:
        Tuple of (speaker_id, match_type)
        match_type: "exact", "fuzzy", "loose", "fallback"
    """
    key = (round(start, 3), round(end, 3))

    # Level 1: Exact match
    if key in segment_to_speaker:
        speakers = segment_to_speaker[key]
        # If multiple speakers (duplicate), take the first non-unknown
        for spk in speakers:
            if spk != DEFAULT_UNKNOWN_SPEAKER:
                stats.matched_via_diarization += 1
                return spk, "exact"
        # All are unknown, still return
        stats.matched_via_diarization += 1
        return speakers[0], "exact"

    # Level 2: Fuzzy match (50ms tolerance)
    for (s, e), speakers in segment_to_speaker.items():
        if abs(s - start) < TIMESTAMP_TOLERANCE_FUZZY and abs(e - end) < TIMESTAMP_TOLERANCE_FUZZY:
            for spk in speakers:
                if spk != DEFAULT_UNKNOWN_SPEAKER:
                    stats.edge_cases.fuzzy_matched += 1
                    return spk, "fuzzy"
            stats.edge_cases.fuzzy_matched += 1
            return speakers[0], "fuzzy"

    # Level 3: Loose match (100ms tolerance)
    for (s, e), speakers in segment_to_speaker.items():
        if abs(s - start) < TIMESTAMP_TOLERANCE_LOOSE and abs(e - end) < TIMESTAMP_TOLERANCE_LOOSE:
            for spk in speakers:
                if spk != DEFAULT_UNKNOWN_SPEAKER:
                    stats.edge_cases.loose_matched += 1
                    return spk, "loose"
            stats.edge_cases.loose_matched += 1
            return speakers[0], "loose"

    # Level 4: No match found, use fallback
    stats.edge_cases.fallback_used += 1
    return fallback_speaker, "fallback"


def migrate_metadata_file(
    file_path: Path,
    stats: MigrationStats,
    verbose: bool = False,
) -> Optional[dict]:
    """Migrate a single metadata file with comprehensive edge case handling.

    Args:
        file_path: Path to metadata JSON file
        stats: Migration statistics to update
        verbose: Enable verbose logging

    Returns:
        Migrated metadata dict, or None on error
    """
    conv_id = file_path.stem  # Default to filename

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error in {file_path}: {e}")
        stats.error_files += 1
        stats.problem_files.append({"file": str(file_path), "error": f"JSON parse: {e}"})
        return None
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        stats.error_files += 1
        stats.problem_files.append({"file": str(file_path), "error": str(e)})
        return None

    conv_id = metadata.get("conversation_id", file_path.stem)

    if not conv_id:
        logger.warning(f"No conversation_id in {file_path}")
        stats.skipped_files += 1
        return None

    # Get speaker IDs from multiple possible locations
    main_speaker_id = (
        metadata.get("diarization", {}).get("main_speaker_id") or
        metadata.get("main_speaker_id") or
        metadata.get("speakers", {}).get("main", {}).get("id")
    )

    # Normalize main speaker ID
    main_speaker_id = normalize_speaker_id(main_speaker_id)

    # IMPORTANT: Never skip files! Use placeholder for unknown main speaker
    # User requirement: "화자 구분만 되면 된다" - speaker distinction is what matters
    if main_speaker_id == DEFAULT_UNKNOWN_SPEAKER:
        main_speaker_id = UNKNOWN_MAIN_SPEAKER
        stats.edge_cases.unknown_main_speaker += 1
        logger.info(f"Using {UNKNOWN_MAIN_SPEAKER} placeholder for unknown main speaker in {file_path}")

    user_speaker_ids = (
        metadata.get("diarization", {}).get("user_speaker_ids") or
        metadata.get("user_speaker_ids") or
        metadata.get("speakers", {}).get("user", {}).get("ids") or
        []
    )

    # Filter out unknown speakers from user list for fallback purposes
    valid_user_speakers = [
        normalize_speaker_id(uid)
        for uid in user_speaker_ids
        if not is_unknown_speaker(uid)
    ]

    # Default fallback speaker
    fallback_user_speaker = (
        valid_user_speakers[0] if len(valid_user_speakers) == 1
        else DEFAULT_UNKNOWN_SPEAKER
    )

    # Check if multi-speaker file
    num_speakers = len(set(valid_user_speakers)) + 1  # +1 for main
    if num_speakers > 2:
        stats.multi_speaker_files += 1
    stats.max_speakers_in_file = max(stats.max_speakers_in_file, num_speakers)

    # Build segment -> speaker mapping
    segment_to_speaker = build_segment_to_speaker_map(metadata, stats)

    # Get segments
    segments = metadata.get("segments", {})

    # Handle different segment structures
    main_segments = []
    user_segments = []

    if isinstance(segments, dict):
        main_segments = segments.get("main", [])
        user_segments = segments.get("user", [])
    elif isinstance(segments, list):
        for seg in segments:
            speaker = seg.get("speaker", "")
            if speaker == "SPEAKER_MAIN":
                main_segments.append(seg)
            else:
                user_segments.append(seg)
    else:
        logger.warning(f"Unexpected segments format in {file_path}: {type(segments)}")
        stats.skipped_files += 1
        return None

    stats.total_main_segments += len(main_segments)
    stats.total_user_segments += len(user_segments)

    # Track for this file
    main_updated = 0
    user_updated = 0
    already_has = 0
    speakers_in_file = set()
    match_types = defaultdict(int)

    # Update main segments (all belong to main_speaker_id)
    for seg in main_segments:
        if "original_speaker_id" in seg:
            already_has += 1
            continue

        seg["original_speaker_id"] = main_speaker_id
        main_updated += 1
        speakers_in_file.add(main_speaker_id)

    # Update user segments with intelligent matching
    for seg in user_segments:
        if "original_speaker_id" in seg:
            already_has += 1
            continue

        start = seg.get("start")
        end = seg.get("end")

        # Validate timestamp
        is_valid, error = validate_timestamp(start, end)
        if not is_valid:
            # Use fallback for invalid timestamps
            seg["original_speaker_id"] = fallback_user_speaker
            seg["_speaker_match_type"] = f"fallback_{error}"
            user_updated += 1
            stats.edge_cases.fallback_used += 1
            continue

        # Find speaker using multi-level matching
        speaker_id, match_type = find_speaker_for_segment(
            float(start), float(end),
            segment_to_speaker,
            stats,
            fallback_speaker=fallback_user_speaker
        )

        seg["original_speaker_id"] = speaker_id
        seg["_speaker_match_type"] = match_type
        user_updated += 1
        speakers_in_file.add(speaker_id)
        match_types[match_type] += 1

    # Update global stats
    stats.main_segments_updated += main_updated
    stats.user_segments_updated += user_updated
    stats.already_has_speaker_id += already_has
    stats.unique_speakers_found += len(speakers_in_file)

    if verbose:
        match_summary = ", ".join(f"{k}={v}" for k, v in sorted(match_types.items()))
        logger.info(
            f"{conv_id}: speakers={num_speakers}, "
            f"main={len(main_segments)} ({main_updated} upd), "
            f"user={len(user_segments)} ({user_updated} upd), "
            f"matches=[{match_summary}]"
        )

    # Update segments in metadata
    if isinstance(metadata.get("segments"), dict):
        metadata["segments"]["main"] = main_segments
        metadata["segments"]["user"] = user_segments
    else:
        all_segments = []
        for seg in main_segments:
            seg["speaker"] = "SPEAKER_MAIN"
            all_segments.append(seg)
        for seg in user_segments:
            seg["speaker"] = "SPEAKER_USER"
            all_segments.append(seg)
        metadata["segments"] = sorted(all_segments, key=lambda s: s.get("start", 0))

    # Add comprehensive migration info
    metadata["_migration_info"] = {
        "migrated": True,
        "version": "3.1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "main_segments_updated": main_updated,
        "user_segments_updated": user_updated,
        "main_speaker_id": main_speaker_id,
        "user_speaker_ids": user_speaker_ids,
        "valid_user_speakers": valid_user_speakers,
        "num_speakers": num_speakers,
        "unique_speakers_in_file": list(speakers_in_file),
        "match_statistics": dict(match_types),
        "has_unknown_speakers": any(is_unknown_speaker(uid) for uid in user_speaker_ids),
    }

    stats.processed_files += 1
    return metadata


def migrate_dataset(
    phase1_dir: Path,
    output_dir: Optional[Path] = None,
    backup: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    limit: Optional[int] = None,
    generate_report: bool = False,
) -> MigrationStats:
    """Migrate all metadata files in a dataset.

    Args:
        phase1_dir: Phase 1 output directory (contains metadata/)
        output_dir: Output directory (if None, update in place)
        backup: Create backup before modifying
        dry_run: Don't modify files
        verbose: Enable verbose logging
        limit: Process only first N files (for testing)
        generate_report: Generate detailed JSON report

    Returns:
        Migration statistics
    """
    stats = MigrationStats()

    # Find metadata files
    metadata_files, metadata_dir = find_metadata_files(phase1_dir)

    if not metadata_files:
        logger.error(f"No metadata files found in {phase1_dir}")
        logger.error("Checked: metadata/, segment_alignment/, and direct JSON files")
        return stats

    stats.total_files = len(metadata_files)

    # Determine output directory
    if output_dir is None:
        out_metadata_dir = metadata_dir
    else:
        out_metadata_dir = output_dir / "metadata"
        out_metadata_dir.mkdir(parents=True, exist_ok=True)

    # Create backup if requested
    if backup and not dry_run and output_dir is None and metadata_dir.exists():
        backup_dir = metadata_dir.parent / f"{metadata_dir.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        if not backup_dir.exists():
            logger.info(f"Creating backup: {backup_dir}")
            shutil.copytree(metadata_dir, backup_dir)

    # Apply limit
    if limit:
        metadata_files = metadata_files[:limit]
        logger.info(f"Limited to {limit} files")

    logger.info(f"Processing {len(metadata_files)} metadata files...")

    # Process each file
    start_time = time.time()
    for i, file_path in enumerate(metadata_files):
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(metadata_files) - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {i + 1}/{len(metadata_files)} "
                f"({stats.match_rate():.1f}% match rate, "
                f"ETA: {eta:.0f}s)"
            )

        migrated = migrate_metadata_file(file_path, stats, verbose=verbose)

        if migrated and not dry_run:
            out_path = out_metadata_dir / file_path.name
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(migrated, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error writing {out_path}: {e}")
                stats.error_files += 1

    # Generate report if requested
    if generate_report:
        report_path = (output_dir or phase1_dir) / "migration_report.json"
        report = {
            "migration_info": {
                "version": "3.1",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "phase1_dir": str(phase1_dir),
                "output_dir": str(output_dir) if output_dir else "in-place",
                "dry_run": dry_run,
            },
            "statistics": stats.to_dict(),
            "problem_files": stats.problem_files[:100],  # Limit to first 100
        }

        if not dry_run:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"Report saved to: {report_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Phase 1 metadata to include original speaker IDs per segment (v3.1 - NEVER SKIP FILES)"
    )

    parser.add_argument(
        "--phase1-dir",
        type=Path,
        required=True,
        help="Phase 1 output directory (with metadata/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: update in place)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create timestamped backup before modifying files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't modify files, just show what would be done",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N files (for testing)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed JSON report",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    if not args.phase1_dir.exists():
        logger.error(f"Phase 1 directory not found: {args.phase1_dir}")
        sys.exit(1)

    # Print configuration
    print("=" * 70)
    print("Phase 1 Speaker ID Migration (v3.1 - NEVER SKIP FILES)")
    print("=" * 70)
    print(f"Phase 1 dir:    {args.phase1_dir}")
    print(f"Output dir:     {args.output_dir or 'in-place'}")
    print(f"Backup:         {args.backup}")
    print(f"Dry run:        {args.dry_run}")
    print(f"Limit:          {args.limit or 'None'}")
    print(f"Generate report:{args.report}")
    print("=" * 70)
    print()

    # Run migration
    start_time = time.time()
    stats = migrate_dataset(
        phase1_dir=args.phase1_dir,
        output_dir=args.output_dir,
        backup=args.backup,
        dry_run=args.dry_run,
        verbose=args.verbose,
        limit=args.limit,
        generate_report=args.report,
    )
    elapsed = time.time() - start_time

    # Print results
    print()
    print("=" * 70)
    print("Migration Results (v3.1 - NEVER SKIP FILES)")
    print("=" * 70)
    print()
    print("FILE STATISTICS:")
    print(f"  Total files:              {stats.total_files:,}")
    print(f"  Processed files:          {stats.processed_files:,}")
    print(f"  Skipped files:            {stats.skipped_files:,}")
    print(f"  Error files:              {stats.error_files:,}")
    print(f"  Multi-speaker files:      {stats.multi_speaker_files:,}")
    print(f"  Max speakers in file:     {stats.max_speakers_in_file}")
    print()
    print("SEGMENT STATISTICS:")
    print(f"  Total main segments:      {stats.total_main_segments:,}")
    print(f"  Total user segments:      {stats.total_user_segments:,}")
    print(f"  Main segments updated:    {stats.main_segments_updated:,}")
    print(f"  User segments updated:    {stats.user_segments_updated:,}")
    print(f"  Already had speaker_id:   {stats.already_has_speaker_id:,}")
    print()
    print("MATCHING STATISTICS:")
    print(f"  Exact matches:            {stats.matched_via_diarization:,}")
    print(f"  Fuzzy matches (50ms):     {stats.edge_cases.fuzzy_matched:,}")
    print(f"  Loose matches (100ms):    {stats.edge_cases.loose_matched:,}")
    print(f"  Fallback used:            {stats.edge_cases.fallback_used:,}")
    print(f"  Match rate:               {stats.match_rate():.2f}%")
    print()
    print("EDGE CASES ENCOUNTERED:")
    print(f"  Unknown main speaker:     {stats.edge_cases.unknown_main_speaker:,}  ← Recovered (not skipped)")
    print(f"  Unknown speaker IDs:      {stats.edge_cases.unknown_speaker_ids:,}")
    print(f"  Empty text segments:      {stats.edge_cases.empty_text_segments:,}")
    print(f"  Missing timestamps:       {stats.edge_cases.missing_timestamps:,}")
    print(f"  Invalid timestamps:       {stats.edge_cases.invalid_timestamps:,}")
    print(f"  No diarization section:   {stats.edge_cases.no_diarization:,}")
    print(f"  Empty speakers list:      {stats.edge_cases.empty_speakers_list:,}")
    print(f"  Duplicate timestamps:     {stats.edge_cases.duplicate_timestamps:,}")
    print()
    print(f"Update rate:              {stats.update_rate():.2f}%")
    print(f"Processing time:          {elapsed:.1f}s")
    print(f"Processing speed:         {stats.total_files/elapsed:.1f} files/sec")
    print("=" * 70)

    print()
    if args.dry_run:
        print("DRY RUN - No files were modified.")
    else:
        print("Migration complete!")

    if stats.problem_files:
        print()
        print(f"WARNING: {len(stats.problem_files)} problem files encountered.")
        print("First 5 problem files:")
        for pf in stats.problem_files[:5]:
            print(f"  - {pf['file']}: {pf['error']}")

    # Return appropriate exit code
    if stats.processed_files > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
