"""
Backbone Factory Module for K-Moshi.

Provides factory pattern implementation for creating backbone instances
based on YAML configuration. Enables seamless switching between different
LLM backends (Moshi, HFLM, custom) without code changes.

Design Pattern: Abstract Factory + Strategy
    - Factory creates appropriate backbone based on config.type
    - Each backbone follows the AbstractBackbone interface (Strategy)
    - DimensionAdapter bridges dimension mismatches automatically

Usage:
    # From YAML config
    config = UnifiedBackboneConfig.load(yaml_path)
    backbone, adapter = BackboneFactory.create(config)

    # From existing LMModel (Moshi)
    backbone = BackboneFactory.from_lm_model(lm_model)

    # Programmatic creation
    backbone = BackboneFactory.create_moshi(transformer)
    backbone = BackboneFactory.create_hf_lm(model_path)

Registry Pattern:
    - Custom backbones can be registered at runtime
    - Enables extension without modifying factory code
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

from .base import AbstractBackbone
from .config import (
    UnifiedBackboneConfig,
    MoshiBackboneConfig,
    HFLMBackboneConfig,
    DimensionAdapterConfig,
)
from .adapters import DimensionAdapter, IdentityAdapter, create_dimension_adapter
from .moshi_backbone import MoshiBackbone
from .hf_lm_backbone import HFLMBackbone

logger = logging.getLogger(__name__)


# Type aliases for clarity
BackboneCreator = Callable[..., AbstractBackbone]
BackboneAndAdapter = Tuple[AbstractBackbone, nn.Module]


class BackboneFactory:
    """
    Factory for creating backbone instances from configuration.

    This factory supports:
    1. YAML-based configuration with UnifiedBackboneConfig
    2. Direct creation methods for each backbone type
    3. Runtime registration of custom backbone types
    4. Automatic dimension adapter creation when needed

    Class Attributes:
        _registry: Dict mapping backbone type names to creator functions
        _default_moshi_dim: Default Moshi dimension for adapter configuration

    Example:
        # From unified config
        config = UnifiedBackboneConfig(type="moshi")
        backbone, adapter = BackboneFactory.create(config)

        # Direct creation
        backbone = BackboneFactory.create_moshi(transformer)

        # With custom backbone
        BackboneFactory.register("my_backbone", MyBackboneCreator)
        backbone, adapter = BackboneFactory.create(config)
    """

    # Registry for backbone creators
    _registry: Dict[str, BackboneCreator] = {}

    # Default Moshi dimension (fixed by architecture)
    _default_moshi_dim: int = 4096

    @classmethod
    def create(
        cls,
        config: UnifiedBackboneConfig,
        lm_model: Optional[Any] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> BackboneAndAdapter:
        """
        Create backbone and dimension adapter from unified config.

        This is the main entry point for creating backbones from YAML config.

        Args:
            config: UnifiedBackboneConfig specifying backbone type and settings
            lm_model: Optional existing LMModel (required for type="moshi")
            device: Target device for model placement
            dtype: Target dtype for model parameters

        Returns:
            Tuple of (backbone, adapter):
                - backbone: AbstractBackbone implementation
                - adapter: DimensionAdapter or IdentityAdapter

        Raises:
            ValueError: If backbone type is not supported
        """
        backbone_type = config.type.lower()

        logger.info(f"[BackboneFactory] Creating backbone: type={backbone_type}")

        # Create backbone based on type
        if backbone_type == "moshi":
            backbone = cls._create_moshi_backbone(config.moshi, lm_model)
        elif backbone_type == "hf_lm":
            backbone = cls._create_hf_lm_backbone(config.hf_lm, device, dtype)
        elif backbone_type in cls._registry:
            # Use registered custom creator
            creator = cls._registry[backbone_type]
            backbone = creator(config, device=device, dtype=dtype)
        else:
            raise ValueError(
                f"Unknown backbone type: '{backbone_type}'. "
                f"Available types: moshi, hf_lm, {list(cls._registry.keys())}"
            )

        # Create dimension adapter if needed
        adapter = cls._create_adapter(config.dimension_adapter, backbone)

        # Move to device/dtype if specified
        if device is not None:
            backbone = backbone.to(device)
            if adapter.is_enabled:
                adapter = adapter.to(device)

        if dtype is not None:
            backbone = backbone.to(dtype)
            if adapter.is_enabled:
                adapter = adapter.to(dtype)

        return backbone, adapter

    @classmethod
    def _create_moshi_backbone(
        cls,
        config: MoshiBackboneConfig,
        lm_model: Optional[Any],
    ) -> MoshiBackbone:
        """Create MoshiBackbone from config and optional LMModel."""
        if lm_model is not None:
            logger.info("[BackboneFactory] Creating MoshiBackbone from LMModel")
            return MoshiBackbone.from_lm_model(lm_model)

        # If no lm_model provided, we cannot create MoshiBackbone
        # This is because Moshi requires the existing transformer weights
        raise ValueError(
            "MoshiBackbone requires an existing LMModel. "
            "Please provide lm_model parameter or use a checkpoint loader."
        )

    @classmethod
    def _create_hf_lm_backbone(
        cls,
        config: HFLMBackboneConfig,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> HFLMBackbone:
        """
        Create HFLMBackbone from config.

        Creates a HuggingFace Transformers-based backbone for HFLM/Mistral
        models. Supports pretrained loading and random initialization.

        Args:
            config: HFLMBackboneConfig with model settings
            device: Target device
            dtype: Target dtype

        Returns:
            HFLMBackbone instance

        Raises:
            ValueError: If model_path is not specified
            ImportError: If transformers package is not installed
        """
        if not config.model_path:
            raise ValueError(
                "hf_lm.model_path is required for HFLMBackbone. "
                "Specify the path to a HuggingFace model or local directory."
            )

        logger.info(
            f"[BackboneFactory] Creating HFLMBackbone: "
            f"model_path={config.model_path}, "
            f"hidden_dim={config.hidden_dim}, "
            f"num_layers={config.num_layers}, "
            f"num_heads={config.num_heads}/{config.num_key_value_heads} (GQA)"
        )

        try:
            backbone = HFLMBackbone(
                config=config,
                device=device,
                dtype=dtype,
            )
            return backbone

        except ImportError as e:
            logger.error(
                "[BackboneFactory] Failed to create HFLMBackbone. "
                "Ensure 'transformers' package is installed: pip install transformers"
            )
            raise

        except Exception as e:
            logger.error(f"[BackboneFactory] HFLMBackbone creation failed: {e}")
            raise

    @classmethod
    def _create_adapter(
        cls,
        config: DimensionAdapterConfig,
        backbone: AbstractBackbone,
    ) -> nn.Module:
        """Create dimension adapter based on config and backbone."""
        backbone_dim = backbone.get_hidden_dim()

        # Auto-enable adapter if dimensions don't match
        if not config.enable and backbone_dim != config.moshi_dim:
            logger.warning(
                f"[BackboneFactory] Backbone dimension ({backbone_dim}) differs from "
                f"Moshi dimension ({config.moshi_dim}). "
                "Consider enabling dimension_adapter in config."
            )
            return IdentityAdapter(config.moshi_dim)

        if config.enable:
            # Update backbone_dim in config if not set
            if config.backbone_dim is None:
                config.backbone_dim = backbone_dim

            return DimensionAdapter(config)

        return IdentityAdapter(config.moshi_dim)

    # =========================================================================
    # Direct Creation Methods
    # =========================================================================

    @classmethod
    def from_lm_model(cls, lm_model: Any) -> BackboneAndAdapter:
        """
        Create MoshiBackbone from existing LMModel.

        Convenience method that bypasses YAML config.

        Args:
            lm_model: Existing LMModel instance

        Returns:
            Tuple of (MoshiBackbone, IdentityAdapter)
        """
        backbone = MoshiBackbone.from_lm_model(lm_model)
        adapter = IdentityAdapter(backbone.get_hidden_dim())
        return backbone, adapter

    @classmethod
    def create_moshi(
        cls,
        transformer: Any,
        config: Optional[MoshiBackboneConfig] = None,
    ) -> MoshiBackbone:
        """
        Create MoshiBackbone from transformer directly.

        Args:
            transformer: StreamingTransformer instance
            config: Optional config (inferred if not provided)

        Returns:
            MoshiBackbone instance
        """
        return MoshiBackbone(transformer=transformer, config=config)

    # =========================================================================
    # Registry Methods
    # =========================================================================

    @classmethod
    def register(
        cls,
        name: str,
        creator: BackboneCreator,
        override: bool = False,
    ) -> None:
        """
        Register a custom backbone creator.

        Args:
            name: Backbone type name (used in config.type)
            creator: Callable that creates AbstractBackbone
            override: If True, allow overriding existing registration

        Raises:
            ValueError: If name already registered and override=False
        """
        if name in cls._registry and not override:
            raise ValueError(
                f"Backbone type '{name}' is already registered. "
                "Use override=True to replace."
            )

        cls._registry[name] = creator
        logger.info(f"[BackboneFactory] Registered backbone type: {name}")

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Remove a registered backbone type.

        Args:
            name: Backbone type name to remove

        Returns:
            True if removed, False if not found
        """
        if name in cls._registry:
            del cls._registry[name]
            logger.info(f"[BackboneFactory] Unregistered backbone type: {name}")
            return True
        return False

    @classmethod
    def list_types(cls) -> list:
        """
        List all available backbone types.

        Returns:
            List of backbone type names (built-in + registered)
        """
        built_in = ["moshi", "hf_lm"]
        registered = list(cls._registry.keys())
        return built_in + registered

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @classmethod
    def get_config_template(cls, backbone_type: str) -> str:
        """
        Get YAML config template for a backbone type.

        Args:
            backbone_type: Type of backbone

        Returns:
            YAML template string
        """
        templates = {
            "moshi": """
backbone:
  type: "moshi"
  moshi:
    hidden_dim: 4096
    num_layers: 32
    num_heads: 32
    gradient_checkpointing: true
  dimension_adapter:
    enable: false
""",
            "hf_lm": """
backbone:
  type: "hf_lm"
  hf_lm:
    model_path: "/path/to/HF-causal-LM"
    hidden_dim: 3072
    num_layers: 30
    num_heads: 24
    num_key_value_heads: 4  # GQA
    load_pretrained: true
  dimension_adapter:
    enable: true
    moshi_dim: 4096
    init_method: "xavier"
""",
        }

        if backbone_type not in templates:
            return f"# No template available for backbone type: {backbone_type}"

        return templates[backbone_type]

    @classmethod
    def validate_config(cls, config: UnifiedBackboneConfig) -> list:
        """
        Validate backbone configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of warning/error messages (empty if valid)
        """
        messages = []

        # Check backbone type
        valid_types = cls.list_types()
        if config.type not in valid_types:
            messages.append(f"ERROR: Unknown backbone type '{config.type}'")

        # Check dimension adapter configuration
        if config.type == "hf_lm" and not config.dimension_adapter.enable:
            if config.hf_lm.hidden_dim != config.dimension_adapter.moshi_dim:
                messages.append(
                    f"WARNING: HFLM hidden_dim ({config.hf_lm.hidden_dim}) "
                    f"differs from Moshi dim ({config.dimension_adapter.moshi_dim}). "
                    "Consider enabling dimension_adapter."
                )

        # Check HFLM model path
        if config.type == "hf_lm" and not config.hf_lm.model_path:
            messages.append("ERROR: hf_lm.model_path is required for HFLM backbone")

        # Check GQA configuration
        if config.type == "hf_lm":
            if config.hf_lm.num_heads % config.hf_lm.num_key_value_heads != 0:
                messages.append(
                    f"ERROR: num_heads ({config.hf_lm.num_heads}) must be divisible "
                    f"by num_key_value_heads ({config.hf_lm.num_key_value_heads})"
                )

        return messages
