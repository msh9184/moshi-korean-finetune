"""
Modular LLM Backbone System for K-Moshi.

This module provides a flexible, pluggable backbone architecture that allows
swapping between different LLM backends (Moshi, HFLM, etc.) via YAML configuration.

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     LMModelWrapper (Unified Interface)              │
    │  ┌────────────────────────────────────────────────────────────────┐ │
    │  │ AbstractBackbone (Pluggable)                                   │ │
    │  │  ├── MoshiBackbone   (d_model=4096, 32 layers, MHA)           │ │
    │  │  ├── HFLMBackbone  (d_model=3072, 30 layers, GQA)           │ │
    │  │  └── CustomBackbone  (user-defined)                           │ │
    │  └────────────────────────────────────────────────────────────────┘ │
    │  ┌────────────────────────────────────────────────────────────────┐ │
    │  │ DimensionAdapter (Optional)                                    │ │
    │  │  ├── input_proj:  4096 → backbone_dim                         │ │
    │  │  └── output_proj: backbone_dim → 4096                         │ │
    │  └────────────────────────────────────────────────────────────────┘ │
    │  ┌────────────────────────────────────────────────────────────────┐ │
    │  │ Shared Components (Fixed Moshi Dimension = 4096)               │ │
    │  │  ├── text_emb, audio_embs                                     │ │
    │  │  ├── Depformer (d_model=1024)                                 │ │
    │  │  └── text_linear, audio_linears                               │ │
    │  └────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    # Via YAML config
    backbone:
      type: "hf_lm"
      hf_lm:
        model_path: "/path/to/HF causal LM"
        hidden_size: 3072
      dimension_adapter:
        enable: true

    # Via Python API
    from finetune.backbone import BackboneFactory
    backbone = BackboneFactory.from_yaml(config)

Module Structure:
    backbone/
    ├── __init__.py           # This file
    ├── config.py             # Configuration dataclasses
    ├── base.py               # AbstractBackbone interface
    ├── adapters.py           # DimensionAdapter
    ├── moshi_backbone.py     # MoshiBackbone wrapper
    ├── hf_lm_backbone.py    # HFLMBackbone wrapper (Phase 2)
    └── factory.py            # BackboneFactory

Author: K-Moshi Development Team
Date: 2025-01-01
"""

from .config import (
    BackboneConfig,
    MoshiBackboneConfig,
    HFLMBackboneConfig,
    DimensionAdapterConfig,
    UnifiedBackboneConfig,
    BackboneType,
)
from .base import AbstractBackbone, BackboneOutput, BackboneState
from .adapters import DimensionAdapter, IdentityAdapter, create_dimension_adapter
from .moshi_backbone import MoshiBackbone, MoshiBackboneState
from .hf_lm_backbone import HFLMBackbone, HFLMBackboneState, create_hf_lm_backbone
from .factory import BackboneFactory
from .lm_model_wrapper import (
    LMModelWrapper,
    LMModelWrapperOutput,
    create_lm_model_wrapper,
    wrap_existing_lm_model,
)

__all__ = [
    # Configuration
    "BackboneConfig",
    "MoshiBackboneConfig",
    "HFLMBackboneConfig",
    "DimensionAdapterConfig",
    "UnifiedBackboneConfig",
    "BackboneType",
    # Base classes
    "AbstractBackbone",
    "BackboneOutput",
    "BackboneState",
    # Adapters
    "DimensionAdapter",
    "IdentityAdapter",
    "create_dimension_adapter",
    # Backbone implementations
    "MoshiBackbone",
    "MoshiBackboneState",
    "HFLMBackbone",
    "HFLMBackboneState",
    "create_hf_lm_backbone",
    # Factory
    "BackboneFactory",
    # LMModelWrapper (Phase 3)
    "LMModelWrapper",
    "LMModelWrapperOutput",
    "create_lm_model_wrapper",
    "wrap_existing_lm_model",
]
