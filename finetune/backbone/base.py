"""
Abstract Backbone Interface for K-Moshi.

Defines the contract that all backbone implementations must follow.
This enables swappable LLM backends while maintaining compatibility
with the Moshi framework's streaming architecture.

Design Principles:
    1. Compatible with Moshi's StreamingModule pattern
    2. Supports both training and inference modes
    3. Enables dimension adaptation for non-Moshi backbones
    4. Preserves KV cache functionality for streaming inference

Architecture:
    AbstractBackbone
    ├── forward()           → Training mode
    ├── streaming()         → Context manager for streaming inference
    ├── _init_streaming_state() → Initialize KV cache
    └── get_hidden_dim()    → Return backbone's native dimension
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from contextlib import ExitStack
from typing import Optional, List, Any, Dict, TypeVar, Generic

import torch
from torch import nn, Tensor


@dataclass
class BackboneOutput:
    """
    Standard output format for backbone forward pass.

    Attributes:
        hidden_states: Final hidden representations [B, T, D]
        all_hidden_states: Optional list of hidden states from each layer
        attentions: Optional attention weights for analysis
        past_key_values: KV cache for streaming inference
    """
    # Primary output: final layer hidden states
    hidden_states: Tensor  # [batch_size, seq_len, hidden_dim]

    # Optional: intermediate hidden states from all layers
    # Used for analysis, probing, or layer-wise loss computation
    all_hidden_states: Optional[List[Tensor]] = None

    # Optional: attention weights from all layers
    # Used for attention visualization and analysis
    attentions: Optional[List[Tensor]] = None

    # KV cache for streaming/generation
    # Format depends on backbone implementation
    past_key_values: Optional[Any] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackboneState:
    """
    Streaming state for backbone inference.

    Compatible with Moshi's State pattern for streaming context management.
    Each backbone implementation should extend this with specific KV cache format.

    Attributes:
        batch_size: Current batch size
        device: Device for tensors
        kv_cache: Key-value cache for attention layers
        position: Current position in sequence (for positional embeddings)
        exec_mask: Execution mask for selective batch processing
    """
    batch_size: int
    device: torch.device

    # KV cache - format depends on backbone
    # Moshi: List of (key, value) tuples per layer
    # HFLM/HuggingFace: DynamicCache or similar
    kv_cache: Optional[Any] = None

    # Current sequence position for streaming
    position: int = 0

    # Execution mask for selective batch processing
    exec_mask: Optional[Tensor] = None

    def __post_init__(self):
        if self.exec_mask is None:
            self.exec_mask = torch.ones(
                self.batch_size, dtype=torch.bool, device=self.device
            )

    def reset(self, reset_mask: Optional[Tensor] = None) -> None:
        """Reset state for specified batch indices."""
        if reset_mask is not None and self.kv_cache is not None:
            # Subclasses should implement specific KV cache reset logic
            pass
        self.position = 0

    def __enter__(self) -> "BackboneState":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Cleanup if needed
        pass


StateT = TypeVar("StateT", bound=BackboneState)


class AbstractBackbone(ABC, nn.Module, Generic[StateT]):
    """
    Abstract base class for all backbone implementations.

    This class defines the interface that all backbone models must implement
    to be used within the K-Moshi framework. It follows Moshi's StreamingModule
    pattern for compatibility with streaming inference.

    Subclasses must implement:
        - forward(): Main forward pass for training
        - _init_streaming_state(): Initialize streaming state
        - get_hidden_dim(): Return native hidden dimension

    Optional overrides:
        - streaming_forward(): Streaming inference step
        - _apply_streaming_state(): Apply state to model

    Example:
        class MoshiBackbone(AbstractBackbone):
            def __init__(self, transformer):
                super().__init__()
                self.transformer = transformer

            def forward(self, x, **kwargs):
                output = self.transformer(x)
                return BackboneOutput(hidden_states=output)

            def get_hidden_dim(self):
                return 4096
    """

    def __init__(self):
        super().__init__()
        self._streaming_state: Optional[StateT] = None
        self._streaming_detached: bool = False

    @property
    def is_streaming(self) -> bool:
        """Check if model is in streaming mode."""
        return self._streaming_state is not None

    @property
    def streaming_state(self) -> Optional[StateT]:
        """Get current streaming state."""
        return self._streaming_state

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
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
        Forward pass for training.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len] or
                           [batch_size, 1, seq_len, seq_len]
            position_ids: Position indices [batch_size, seq_len]
            past_key_values: Optional KV cache for streaming inference
            use_cache: Whether to return and use KV cache
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights

        Returns:
            BackboneOutput with hidden_states and optional extras
        """
        ...

    @abstractmethod
    def _init_streaming_state(self, batch_size: int) -> StateT:
        """
        Initialize streaming state for inference.

        Creates the initial state including empty KV cache.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            Initialized BackboneState subclass instance
        """
        ...

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """
        Return the native hidden dimension of the backbone.

        This is used to configure DimensionAdapter if needed.

        Returns:
            Hidden dimension (e.g., 4096 for Moshi, 3072 for HFLM)
        """
        ...

    # =========================================================================
    # Streaming Methods - Compatible with Moshi's pattern
    # =========================================================================

    def streaming(self, batch_size: int) -> ExitStack:
        """
        Context manager to enter streaming mode.

        Usage:
            with backbone.streaming(batch_size=1):
                for chunk in audio_chunks:
                    output = backbone.streaming_forward(chunk)

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            ExitStack context manager that resets state on exit
        """
        exit_stack = ExitStack()
        self._start_streaming(batch_size, exit_stack)
        exit_stack.callback(self._stop_streaming)
        return exit_stack

    def streaming_forever(self, batch_size: int) -> None:
        """Enter streaming mode without automatic exit."""
        self.streaming(batch_size).__enter__()

    def _start_streaming(self, batch_size: int, exit_stack: ExitStack) -> None:
        """Initialize streaming mode."""
        assert self._streaming_state is None, "Already in streaming mode!"
        state = self._init_streaming_state(batch_size)
        exit_stack.enter_context(state)
        self._streaming_state = state

    def _stop_streaming(self) -> None:
        """Exit streaming mode and cleanup."""
        self._streaming_state = None

    def reset_streaming(self, reset_mask: Optional[Tensor] = None) -> None:
        """Reset streaming state for specific batch indices."""
        if self._streaming_state is None:
            raise ValueError("Not in streaming mode")
        self._streaming_state.reset(reset_mask)

    def streaming_forward(
        self,
        hidden_states: Tensor,
        **kwargs,
    ) -> BackboneOutput:
        """
        Single step forward pass for streaming inference.

        Uses the stored streaming state (KV cache) for efficient generation.

        Args:
            hidden_states: Input tensor [batch_size, 1, hidden_dim]
            **kwargs: Additional arguments passed to forward

        Returns:
            BackboneOutput with hidden_states
        """
        assert self.is_streaming, "Must be in streaming mode"
        return self.forward(
            hidden_states,
            past_key_values=self._streaming_state.kv_cache,
            use_cache=True,
            **kwargs,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_num_layers(self) -> int:
        """Return number of transformer layers (if applicable)."""
        raise NotImplementedError("Subclass should implement if needed")

    def get_num_heads(self) -> int:
        """Return number of attention heads (if applicable)."""
        raise NotImplementedError("Subclass should implement if needed")

    def supports_gradient_checkpointing(self) -> bool:
        """Return whether backbone supports gradient checkpointing."""
        return False

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing if supported."""
        if not self.supports_gradient_checkpointing():
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support gradient checkpointing"
            )

    def get_dtype(self) -> torch.dtype:
        """Return the dtype of model parameters."""
        for param in self.parameters():
            return param.dtype
        return torch.float32

    def get_device(self) -> torch.device:
        """Return the device of model parameters."""
        for param in self.parameters():
            return param.device
        return torch.device("cpu")
