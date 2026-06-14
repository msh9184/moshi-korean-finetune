# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Korean Moshi Dataset Preparation Pipeline.

Two-phase pipeline for converting Korean broadcast data (Lhotse Shar format)
to Moshi finetuning format with word-level alignments.

Phase 1 (CPU): Lhotse Shar reading, speaker selection, stereo conversion
Phase 2 (GPU): Word-level alignment using whisper-timestamped
"""

__version__ = "0.1.0"
