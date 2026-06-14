# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Pipeline orchestrators for Phase 1 and Phase 2.

Phase 2 imports are lazy to avoid dependency issues when only Phase 1 is needed.
"""

from .phase1_cpu import Phase1Orchestrator

# Lazy import for Phase 2 (requires GPU dependencies)
def get_phase2_orchestrator():
    """Get Phase2Orchestrator (lazy import to avoid GPU dependency issues)."""
    from .phase2_gpu import Phase2Orchestrator
    return Phase2Orchestrator

__all__ = ["Phase1Orchestrator", "get_phase2_orchestrator"]
