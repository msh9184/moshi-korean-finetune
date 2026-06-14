"""
Moshi Backbone Wrapper for K-Moshi.

Wraps the original Moshi StreamingTransformer as an AbstractBackbone implementation.
This allows the existing Moshi transformer to be used within the modular backbone system.

Key Features:
    - Compatible with existing Moshi checkpoint loading
    - Preserves StreamingModule interface for inference
    - Supports gradient checkpointing
    - No dimension adapter needed (native 4096 dim)

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      MoshiBackbone                                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  ┌───────────────────────────────────────────────────────────────┐  │
    │  │              StreamingTransformer (Wrapped)                    │  │
    │  │                                                                │  │
    │  │  • d_model: 4096                                              │  │
    │  │  • num_layers: 32                                             │  │
    │  │  • num_heads: 32 (MHA, no GQA)                                │  │
    │  │  • dim_feedforward: 16384                                     │  │
    │  │  • positional_embedding: rope                                 │  │
    │  │  • causal: True                                               │  │
    │  │                                                                │  │
    │  │  Streaming: KV Cache managed by StreamingTransformerLayer     │  │
    │  └───────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    # From existing LMModel
    lm_model = loaders.get_moshi_lm(...)
    backbone = MoshiBackbone.from_lm_model(lm_model)

    # Training forward
    output = backbone(hidden_states)  # BackboneOutput

    # Streaming inference
    with backbone.streaming(batch_size=1):
        output = backbone.streaming_forward(chunk)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Any, List

import torch
from torch import Tensor

from .base import AbstractBackbone, BackboneOutput, BackboneState
from .config import MoshiBackboneConfig

logger = logging.getLogger(__name__)


@dataclass
class MoshiBackboneState(BackboneState):
    """
    Streaming state for Moshi backbone.

    The KV cache is managed internally by StreamingTransformer's layers.
    This state primarily tracks position and execution mask.
    """
    # Note: Moshi's StreamingTransformer manages its own KV cache internally
    # via _streaming_state in each StreamingTransformerLayer

    # Reference to the transformer's streaming context for proper cleanup
    transformer_streaming_context: Optional[Any] = None

    def __post_init__(self):
        """Initialize state with proper parent initialization."""
        super().__post_init__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Cleanup transformer streaming context on exit."""
        if self.transformer_streaming_context is not None:
            self.transformer_streaming_context.__exit__(exc_type, exc_value, traceback)
        super().__exit__(exc_type, exc_value, traceback)


class MoshiBackbone(AbstractBackbone[MoshiBackboneState]):
    """
    Wrapper for Moshi's StreamingTransformer as AbstractBackbone.

    This class adapts the existing Moshi transformer to the AbstractBackbone
    interface, enabling it to be used interchangeably with other backbones
    like HFLM.

    The wrapper delegates most functionality to the underlying transformer
    while providing a consistent interface for the modular backbone system.

    Attributes:
        config: MoshiBackboneConfig with transformer settings
        transformer: The wrapped StreamingTransformer instance
    """

    def __init__(
        self,
        transformer: Any,  # StreamingTransformer from moshi.modules.transformer
        config: Optional[MoshiBackboneConfig] = None,
    ):
        """
        Initialize MoshiBackbone wrapper.

        Args:
            transformer: Existing StreamingTransformer instance
            config: Optional MoshiBackboneConfig (inferred from transformer if None)
        """
        super().__init__()

        self.transformer = transformer

        # Infer config from transformer if not provided
        if config is None:
            config = MoshiBackboneConfig(
                hidden_dim=transformer.layers[0].self_attn.embed_dim
                if hasattr(transformer, 'layers') else 4096,
                num_layers=len(transformer.layers)
                if hasattr(transformer, 'layers') else 32,
                gradient_checkpointing=getattr(transformer, 'checkpointing', False),
            )

        self.config = config
        self._hidden_dim = config.hidden_dim

        logger.info(
            f"[MoshiBackbone] Initialized with d_model={self._hidden_dim}, "
            f"layers={config.num_layers}"
        )

    @classmethod
    def from_lm_model(cls, lm_model: Any) -> "MoshiBackbone":
        """
        Create MoshiBackbone from an existing LMModel.

        Args:
            lm_model: LMModel instance from moshi.models.lm

        Returns:
            MoshiBackbone wrapping the LMModel's transformer
        """
        if not hasattr(lm_model, 'transformer'):
            raise ValueError("LMModel must have 'transformer' attribute")

        config = MoshiBackboneConfig(
            hidden_dim=lm_model.dim,
            num_layers=len(lm_model.transformer.layers),
            gradient_checkpointing=getattr(
                lm_model.transformer, 'checkpointing', False
            ),
        )

        return cls(transformer=lm_model.transformer, config=config)

    # =========================================================================
    # AbstractBackbone Implementation
    # =========================================================================

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> BackboneOutput:
        """
        Forward pass through the Moshi transformer.

        Args:
            hidden_states: Input tensor [B, T, D] or [B, D, T]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_values: Optional KV cache (managed internally by transformer)
            use_cache: Whether to use KV cache (handled by transformer's streaming)
            output_hidden_states: Whether to return all layer outputs
            output_attentions: Whether to return attention weights

        Returns:
            BackboneOutput with transformer output
        """
        # Moshi transformer expects [B, T, D] format
        # We assume input is always [B, T, D] for consistency with AbstractBackbone interface
        # If transposition is needed, it should be handled by the caller
        if hidden_states.dim() == 3:
            B, dim1, dim2 = hidden_states.shape
            # Validate that one of the dimensions matches hidden_dim
            if dim2 == self._hidden_dim:
                # Standard [B, T, D] format
                x = hidden_states
            elif dim1 == self._hidden_dim and dim2 != self._hidden_dim:
                # Input appears to be [B, D, T], transpose to [B, T, D]
                x = hidden_states.transpose(1, 2)
                logger.debug(
                    f"[MoshiBackbone] Transposed input from [{B}, {dim1}, {dim2}] "
                    f"to [{B}, {dim2}, {dim1}]"
                )
            else:
                # Ambiguous or unexpected shape, use as-is and let transformer handle it
                x = hidden_states
        else:
            x = hidden_states

        # Forward through transformer
        # StreamingTransformer.forward signature: (x, *args, **kwargs)
        output = self.transformer(x)

        # Collect all hidden states if requested
        all_hidden_states = None
        if output_hidden_states:
            # Note: StreamingTransformer doesn't expose intermediate states by default
            # This would require modification to the underlying transformer
            logger.warning(
                "output_hidden_states=True not fully supported for MoshiBackbone. "
                "Returning only final hidden states."
            )

        return BackboneOutput(
            hidden_states=output,
            all_hidden_states=all_hidden_states,
            attentions=None,  # Not exposed by default
        )

    def _init_streaming_state(self, batch_size: int) -> MoshiBackboneState:
        """
        Initialize streaming state for Moshi backbone.

        The actual KV cache is managed by the underlying StreamingTransformer.
        This method ensures the transformer is put into streaming mode.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            MoshiBackboneState instance
        """
        device = self.get_device()

        # Start the transformer's streaming mode and capture the context
        transformer_ctx = self.transformer.streaming(batch_size)
        transformer_ctx.__enter__()

        return MoshiBackboneState(
            batch_size=batch_size,
            device=device,
            transformer_streaming_context=transformer_ctx,
        )

    def get_hidden_dim(self) -> int:
        """Return Moshi's native hidden dimension (4096)."""
        return self._hidden_dim

    # =========================================================================
    # Extended Methods
    # =========================================================================

    def streaming(self, batch_size: int):
        """
        Context manager for streaming mode.

        Uses the parent's streaming implementation which properly manages
        both the AbstractBackbone state and the underlying transformer's
        streaming context via MoshiBackboneState.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            Context manager that handles streaming state
        """
        # Use parent's streaming() which calls _init_streaming_state
        # This ensures both AbstractBackbone._streaming_state and
        # the transformer's streaming context are properly managed
        return super().streaming(batch_size)

    def streaming_forward(
        self,
        hidden_states: Tensor,
        **kwargs,
    ) -> BackboneOutput:
        """
        Single step forward for streaming inference.

        Args:
            hidden_states: Input tensor [B, 1, D]

        Returns:
            BackboneOutput with single step output
        """
        # Streaming mode is handled by the transformer's internal state
        output = self.transformer(hidden_states)

        return BackboneOutput(hidden_states=output)

    def get_num_layers(self) -> int:
        """Return number of transformer layers."""
        return self.config.num_layers

    def get_num_heads(self) -> int:
        """Return number of attention heads."""
        return self.config.num_heads

    def supports_gradient_checkpointing(self) -> bool:
        """Moshi supports gradient checkpointing."""
        return True

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing in the transformer."""
        if hasattr(self.transformer, 'checkpointing'):
            self.transformer.checkpointing = True
            logger.info("[MoshiBackbone] Gradient checkpointing enabled")
        else:
            logger.warning(
                "[MoshiBackbone] Cannot enable gradient checkpointing - "
                "attribute not found"
            )

    def get_parameter_count(self) -> dict:
        """Return parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    @property
    def layers(self):
        """Access underlying transformer layers."""
        return self.transformer.layers if hasattr(self.transformer, 'layers') else []

    def __repr__(self) -> str:
        return (
            f"MoshiBackbone(\n"
            f"  hidden_dim={self._hidden_dim},\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  num_heads={self.config.num_heads},\n"
            f"  gradient_checkpointing={self.config.gradient_checkpointing}\n"
            f")"
        )
