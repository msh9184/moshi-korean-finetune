"""
Dimension Adapter Module for K-Moshi.

Provides linear projection layers to bridge different backbone dimensions
with the Moshi framework's fixed 4096-dimensional components.

Architecture:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    Data Flow with DimensionAdapter                    │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │  Moshi Embeddings     DimensionAdapter        Backbone Transformer   │
    │  (d=4096)             (Projection)            (d=3072 for HFLM)    │
    │       │                    │                         │               │
    │       ▼                    ▼                         ▼               │
    │  ┌─────────┐         ┌─────────┐              ┌─────────────┐        │
    │  │text_emb │    ──►  │input_   │    ──►      │  HFLM     │        │
    │  │audio_emb│         │proj    │              │  Backbone   │        │
    │  │(4096)   │         │(4096→  │              │  (3072)     │        │
    │  └─────────┘         │ 3072)  │              └──────┬──────┘        │
    │                      └─────────┘                    │               │
    │                                                     ▼               │
    │  ┌─────────┐         ┌─────────┐              ┌─────────────┐        │
    │  │Depformer│    ◄──  │output_  │    ◄──      │  Hidden     │        │
    │  │text_lin │         │proj    │              │  States     │        │
    │  │(4096)   │         │(3072→  │              │  (3072)     │        │
    │  └─────────┘         │ 4096)  │              └─────────────┘        │
    │                      └─────────┘                                    │
    └──────────────────────────────────────────────────────────────────────┘

Initialization Methods:
    - xavier: Xavier/Glorot uniform (default, best for general use)
    - kaiming: He initialization (good for ReLU-like activations)
    - normal: Normal distribution with configurable std
    - orthogonal: Orthogonal initialization (preserves gradient norms)

Usage:
    config = DimensionAdapterConfig(
        enable=True,
        moshi_dim=4096,
        backbone_dim=3072,
        init_method="xavier"
    )
    adapter = DimensionAdapter(config)

    # Project to backbone dimension
    backbone_input = adapter.project_input(moshi_embeddings)

    # Run backbone
    backbone_output = backbone(backbone_input)

    # Project back to Moshi dimension
    moshi_output = adapter.project_output(backbone_output.hidden_states)
"""

import logging
import math
from typing import Optional

import torch
from torch import nn, Tensor

from .config import DimensionAdapterConfig

logger = logging.getLogger(__name__)


class DimensionAdapter(nn.Module):
    """
    Linear projection layers for dimension adaptation between
    Moshi components (4096) and alternative backbones.

    This module provides bidirectional projection:
    - input_proj: Projects from Moshi dimension to backbone dimension
    - output_proj: Projects from backbone dimension back to Moshi dimension

    The projections are designed to:
    1. Preserve information as much as possible
    2. Support gradient flow for end-to-end training
    3. Optionally include residual connections for stability

    Attributes:
        config: DimensionAdapterConfig with all settings
        input_proj: Linear layer for input projection (moshi_dim → backbone_dim)
        output_proj: Linear layer for output projection (backbone_dim → moshi_dim)
    """

    def __init__(self, config: DimensionAdapterConfig):
        """
        Initialize dimension adapter.

        Args:
            config: DimensionAdapterConfig containing:
                - moshi_dim: Moshi embedding dimension (default 4096)
                - backbone_dim: Target backbone dimension
                - bias: Whether to use bias in projections
                - init_method: Weight initialization method
                - init_std: Std for normal initialization
                - residual_scale: Scale for residual connections (0 = no residual)
        """
        super().__init__()
        self.config = config

        if not config.enable:
            logger.info("[DimensionAdapter] Disabled - no projections created")
            self.input_proj = None
            self.output_proj = None
            return

        if config.backbone_dim is None:
            raise ValueError(
                "backbone_dim must be specified when dimension adapter is enabled"
            )

        moshi_dim = config.moshi_dim
        backbone_dim = config.backbone_dim

        logger.info(
            f"[DimensionAdapter] Creating projections: "
            f"{moshi_dim} <-> {backbone_dim}"
        )

        # Input projection: Moshi dimension → Backbone dimension
        self.input_proj = nn.Linear(
            moshi_dim,
            backbone_dim,
            bias=config.bias,
        )

        # Output projection: Backbone dimension → Moshi dimension
        self.output_proj = nn.Linear(
            backbone_dim,
            moshi_dim,
            bias=config.bias,
        )

        # Initialize weights
        self._initialize_weights()

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[DimensionAdapter] Total parameters: {total_params:,} "
            f"({total_params * 2 / 1e6:.2f}M in bf16)"
        )

    def _initialize_weights(self) -> None:
        """Initialize projection weights based on config.init_method."""
        method = self.config.init_method

        if self.input_proj is None or self.output_proj is None:
            return

        logger.info(f"[DimensionAdapter] Initializing weights with '{method}' method")

        for proj in [self.input_proj, self.output_proj]:
            if method == "xavier":
                nn.init.xavier_uniform_(proj.weight)
            elif method == "kaiming":
                nn.init.kaiming_uniform_(proj.weight, a=math.sqrt(5))
            elif method == "normal":
                nn.init.normal_(proj.weight, mean=0.0, std=self.config.init_std)
            elif method == "orthogonal":
                nn.init.orthogonal_(proj.weight)
            else:
                raise ValueError(f"Unknown init_method: {method}")

            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def project_input(self, x: Tensor) -> Tensor:
        """
        Project from Moshi dimension to backbone dimension.

        Args:
            x: Input tensor of shape [..., moshi_dim]

        Returns:
            Projected tensor of shape [..., backbone_dim]
        """
        if self.input_proj is None:
            return x
        return self.input_proj(x)

    def project_output(self, x: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        """
        Project from backbone dimension back to Moshi dimension.

        Optionally adds a scaled residual connection from the original input.

        Args:
            x: Backbone output tensor of shape [..., backbone_dim]
            residual: Optional residual tensor of shape [..., moshi_dim]
                      If provided and residual_scale > 0, adds scaled residual

        Returns:
            Projected tensor of shape [..., moshi_dim]
        """
        if self.output_proj is None:
            return x

        output = self.output_proj(x)

        # Add residual connection if configured
        if residual is not None and self.config.residual_scale > 0:
            output = output + self.config.residual_scale * residual

        return output

    def forward(
        self,
        x: Tensor,
        direction: str = "input",
        residual: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Generic forward pass with direction selection.

        Args:
            x: Input tensor
            direction: "input" for moshi→backbone, "output" for backbone→moshi
            residual: Optional residual for output direction

        Returns:
            Projected tensor
        """
        if direction == "input":
            return self.project_input(x)
        elif direction == "output":
            return self.project_output(x, residual)
        else:
            raise ValueError(f"Unknown direction: {direction}. Use 'input' or 'output'")

    @property
    def is_enabled(self) -> bool:
        """Check if adapter is enabled and has projections."""
        return self.input_proj is not None and self.output_proj is not None

    @property
    def input_dim(self) -> int:
        """Get the input (Moshi) dimension."""
        return self.config.moshi_dim

    @property
    def output_dim(self) -> int:
        """Get the output (backbone) dimension."""
        return self.config.backbone_dim or self.config.moshi_dim

    def extra_repr(self) -> str:
        """String representation for print."""
        if not self.is_enabled:
            return "disabled"
        return (
            f"moshi_dim={self.config.moshi_dim}, "
            f"backbone_dim={self.config.backbone_dim}, "
            f"bias={self.config.bias}, "
            f"init={self.config.init_method}"
        )


class _IdentityFunction(nn.Module):
    """Simple identity function wrapper that can be called like a layer."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class IdentityAdapter(nn.Module):
    """
    Identity adapter that passes through tensors unchanged.

    Used when no dimension adaptation is needed (same backbone dimension as Moshi).
    Provides the same interface as DimensionAdapter for compatibility.
    """

    def __init__(self, dim: int = 4096):
        super().__init__()
        self._dim = dim
        # Create identity function wrappers for compatibility with DimensionAdapter interface
        # These allow callers to use adapter.input_proj(x) and adapter.output_proj(x)
        self.input_proj = _IdentityFunction()
        self.output_proj = _IdentityFunction()

    def project_input(self, x: Tensor) -> Tensor:
        return x

    def project_output(self, x: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        return x

    def forward(
        self,
        x: Tensor,
        direction: str = "input",
        residual: Optional[Tensor] = None,
    ) -> Tensor:
        return x

    @property
    def is_enabled(self) -> bool:
        return False

    @property
    def input_dim(self) -> int:
        return self._dim

    @property
    def output_dim(self) -> int:
        return self._dim

    def extra_repr(self) -> str:
        return f"identity, dim={self._dim}"


def create_dimension_adapter(
    config: Optional[DimensionAdapterConfig] = None,
    moshi_dim: int = 4096,
    backbone_dim: Optional[int] = None,
) -> nn.Module:
    """
    Factory function to create appropriate dimension adapter.

    Args:
        config: DimensionAdapterConfig (if provided, other args ignored)
        moshi_dim: Moshi embedding dimension
        backbone_dim: Target backbone dimension (None = same as moshi_dim)

    Returns:
        DimensionAdapter if dimensions differ, IdentityAdapter otherwise
    """
    if config is not None:
        if not config.enable:
            return IdentityAdapter(config.moshi_dim)
        return DimensionAdapter(config)

    # Auto-configure based on dimensions
    if backbone_dim is None or backbone_dim == moshi_dim:
        return IdentityAdapter(moshi_dim)

    config = DimensionAdapterConfig(
        enable=True,
        moshi_dim=moshi_dim,
        backbone_dim=backbone_dim,
    )
    return DimensionAdapter(config)
