"""
HFLM Backbone Wrapper for K-Moshi.

Wraps a Hugging Face causal LM (Mistral-based architecture) as an AbstractBackbone
implementation. This enables using HFLM as a drop-in replacement for the Moshi
transformer backbone.

Key Features:
    - HuggingFace Transformers integration
    - GQA (Grouped Query Attention) support: 24 heads, 4 KV heads
    - Sliding window attention (4096 window)
    - DynamicCache-based KV cache for streaming inference
    - Gradient checkpointing support
    - Dimension adapter integration (3072 ↔ 4096)

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      HFLMBackbone                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  ┌───────────────────────────────────────────────────────────────┐  │
    │  │           MistralModel (HuggingFace Transformers)              │  │
    │  │                                                                │  │
    │  │  • d_model: 3072                                              │  │
    │  │  • num_layers: 30                                             │  │
    │  │  • num_heads: 24                                              │  │
    │  │  • num_kv_heads: 4 (GQA: 6x repetition)                       │  │
    │  │  • intermediate_size: 8192                                    │  │
    │  │  • sliding_window: 4096                                       │  │
    │  │  • rope_theta: 500000                                         │  │
    │  │                                                                │  │
    │  │  Streaming: DynamicCache managed externally                   │  │
    │  └───────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    # From config
    config = HFLMBackboneConfig(model_path="/path/to/HF causal LM")
    backbone = HFLMBackbone(config)

    # Training forward
    output = backbone(hidden_states)  # BackboneOutput

    # Streaming inference
    with backbone.streaming(batch_size=1):
        output = backbone.streaming_forward(chunk)

Note:
    When using HFLMBackbone with Moshi components (embeddings, depformer),
    a DimensionAdapter is required to bridge the 3072 ↔ 4096 dimension gap.

Author: K-Moshi Development Team
Date: 2025-01-01
"""

import logging
from dataclasses import dataclass
from typing import Optional, Any, List, Tuple, Union

import torch
from torch import Tensor, nn

from .base import AbstractBackbone, BackboneOutput, BackboneState
from .config import HFLMBackboneConfig

logger = logging.getLogger(__name__)


# =============================================================================
# HFLM Streaming State
# =============================================================================

@dataclass
class HFLMBackboneState(BackboneState):
    """
    Streaming state for HFLM backbone.

    Uses HuggingFace's DynamicCache for KV cache management.
    This is compatible with Mistral-based models.

    Attributes:
        kv_cache: DynamicCache instance for storing key-value pairs
        past_seen_tokens: Number of tokens already processed
        attention_mask: Accumulated attention mask for causal attention
    """
    # Number of tokens seen so far (for position computation)
    past_seen_tokens: int = 0

    # Accumulated attention mask for proper causal masking
    attention_mask: Optional[Tensor] = None

    # Reference to the DynamicCache instance
    # Set via kv_cache from parent class

    def __post_init__(self):
        """Initialize with lazy cache creation."""
        super().__post_init__()
        # kv_cache will be set when streaming starts

    def update_position(self, num_new_tokens: int = 1) -> None:
        """Update position after processing tokens."""
        self.past_seen_tokens += num_new_tokens
        self.position = self.past_seen_tokens

    def reset(self, reset_mask: Optional[Tensor] = None) -> None:
        """Reset state for specific batch indices or all."""
        super().reset(reset_mask)
        if reset_mask is None:
            # Full reset
            self.past_seen_tokens = 0
            self.attention_mask = None
            if self.kv_cache is not None:
                # Clear cache for all sequences
                try:
                    if hasattr(self.kv_cache, 'reset'):
                        self.kv_cache.reset()
                    elif hasattr(self.kv_cache, 'key_cache'):
                        # DynamicCache style
                        self.kv_cache.key_cache.clear()
                        self.kv_cache.value_cache.clear()
                except Exception as e:
                    logger.warning(f"[HFLMState] Cache reset failed: {e}")


# =============================================================================
# HFLM Backbone Implementation
# =============================================================================

class HFLMBackbone(AbstractBackbone[HFLMBackboneState]):
    """
    Wrapper for HFLM/Mistral-based models as AbstractBackbone.

    This class adapts HuggingFace Transformers models to the AbstractBackbone
    interface, enabling them to be used within the K-Moshi modular backbone system.

    The backbone extracts and uses only the transformer layers (MistralModel),
    not the full causal LM head, since logits are computed by the shared
    Moshi depformer and linears.

    Attributes:
        config: HFLMBackboneConfig with model settings
        model: The underlying HuggingFace model (without LM head)
        gradient_checkpointing: Whether gradient checkpointing is enabled
    """

    def __init__(
        self,
        config: HFLMBackboneConfig,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize HFLMBackbone.

        Args:
            config: HFLMBackboneConfig with model settings
            model: Optional pre-loaded model (if None, loads from config.model_path)
            device: Target device for model
            dtype: Target dtype for model parameters
        """
        super().__init__()

        self.config = config
        self._hidden_dim = config.hidden_dim
        self._num_layers = config.num_layers
        self._num_heads = config.num_heads
        self._num_kv_heads = config.num_key_value_heads
        self._gradient_checkpointing = config.gradient_checkpointing

        # Load model if not provided
        if model is not None:
            self.model = model
        else:
            self.model = self._load_model(config, device, dtype)

        # Track device and dtype
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype or config.get_torch_dtype()

        logger.info(
            f"[HFLMBackbone] Initialized: "
            f"hidden_dim={self._hidden_dim}, "
            f"layers={self._num_layers}, "
            f"heads={self._num_heads}/{self._num_kv_heads} (GQA)"
        )

    def _load_model(
        self,
        config: HFLMBackboneConfig,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ) -> nn.Module:
        """
        Load the HFLM/Mistral model from HuggingFace.

        Args:
            config: Configuration with model path and settings
            device: Target device
            dtype: Target dtype

        Returns:
            Loaded model (transformer layers only, no LM head)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError:
            raise ImportError(
                "transformers package is required for HFLMBackbone. "
                "Install it with: pip install transformers"
            )

        model_path = config.model_path
        if not model_path:
            raise ValueError("config.model_path is required for HFLMBackbone")

        logger.info(f"[HFLMBackbone] Loading model from: {model_path}")

        # Determine dtype
        torch_dtype = dtype or config.get_torch_dtype()

        try:
            if config.load_pretrained:
                # Load pretrained model
                full_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map="auto" if device is None else None,
                    trust_remote_code=config.trust_remote_code,
                    attn_implementation="flash_attention_2" if (config.use_flash_attention and self._check_flash_attn()) else "eager",
                )

                # Extract the base model (without LM head)
                # For Mistral-based models, this is typically model.model
                if hasattr(full_model, 'model'):
                    model = full_model.model
                else:
                    # Fallback: use the full model
                    model = full_model
                    logger.warning(
                        "[HFLMBackbone] Could not extract base model. "
                        "Using full model including LM head."
                    )

                # Move to device if specified
                if device is not None:
                    model = model.to(device)

            else:
                # Random initialization
                hf_config = AutoConfig.from_pretrained(
                    model_path,
                    trust_remote_code=config.trust_remote_code,
                )
                # Override config values if needed
                hf_config.hidden_size = config.hidden_dim
                hf_config.num_hidden_layers = config.num_layers
                hf_config.num_attention_heads = config.num_heads
                hf_config.num_key_value_heads = config.num_key_value_heads
                hf_config.intermediate_size = config.intermediate_size
                hf_config.rope_theta = config.rope_theta

                full_model = AutoModelForCausalLM.from_config(hf_config)
                model = full_model.model if hasattr(full_model, 'model') else full_model

                if device is not None:
                    model = model.to(device)
                model = model.to(torch_dtype)

            # Enable gradient checkpointing if configured
            # CRITICAL: Use use_reentrant=False for FSDP compatibility
            # The default use_reentrant=True can cause dimension errors during backward
            # when combined with FSDP sharding. Non-reentrant mode is FSDP-safe.
            if config.gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    # Check transformers version for kwargs support
                    import inspect
                    sig = inspect.signature(model.gradient_checkpointing_enable)
                    if 'gradient_checkpointing_kwargs' in sig.parameters:
                        # Newer transformers (>=4.31) support kwargs
                        model.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False}
                        )
                        logger.info(
                            "[HFLMBackbone] Gradient checkpointing enabled "
                            "(non-reentrant mode for FSDP compatibility)"
                        )
                    else:
                        # Older transformers - enable but warn about potential issues
                        model.gradient_checkpointing_enable()
                        logger.warning(
                            "[HFLMBackbone] Gradient checkpointing enabled with default settings. "
                            "Consider upgrading transformers>=4.31 for FSDP compatibility (use_reentrant=False)."
                        )

            return model

        except Exception as e:
            logger.error(f"[HFLMBackbone] Failed to load model: {e}")
            raise

    def _check_flash_attn(self) -> bool:
        """Check if Flash Attention 2 is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False

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
        Forward pass through the HFLM transformer.

        Args:
            hidden_states: Input tensor [B, T, D] where D=3072
            attention_mask: Optional attention mask [B, T] or [B, 1, T, T]
            position_ids: Optional position indices [B, T]
            past_key_values: Optional KV cache (DynamicCache or tuple)
            use_cache: Whether to return and use KV cache
            output_hidden_states: Whether to return all layer outputs
            output_attentions: Whether to return attention weights

        Returns:
            BackboneOutput with hidden states and optional extras
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Validate dimension
        if hidden_dim != self._hidden_dim:
            raise ValueError(
                f"Input hidden_dim ({hidden_dim}) does not match "
                f"backbone hidden_dim ({self._hidden_dim}). "
                "Ensure DimensionAdapter is properly configured."
            )

        # Prepare position_ids if not provided
        if position_ids is None:
            past_len = 0
            if past_key_values is not None:
                if hasattr(past_key_values, 'get_seq_length'):
                    past_len = past_key_values.get_seq_length()
                elif isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                    # Legacy tuple format
                    past_len = past_key_values[0][0].shape[2]

            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=hidden_states.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Prepare causal attention mask if needed
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_len,
                device=hidden_states.device,
                dtype=torch.long
            )

        # Forward through the model
        # HuggingFace MistralModel expects inputs_embeds for hidden states
        outputs = self.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # Extract outputs
        final_hidden_states = outputs.last_hidden_state

        # Collect all hidden states if requested
        all_hidden_states = None
        if output_hidden_states and hasattr(outputs, 'hidden_states'):
            all_hidden_states = list(outputs.hidden_states)

        # Collect attentions if requested
        attentions = None
        if output_attentions and hasattr(outputs, 'attentions'):
            attentions = list(outputs.attentions)

        # Get past_key_values for caching
        new_past = outputs.past_key_values if use_cache else None

        return BackboneOutput(
            hidden_states=final_hidden_states,
            all_hidden_states=all_hidden_states,
            attentions=attentions,
            past_key_values=new_past,
        )

    def _init_streaming_state(self, batch_size: int) -> HFLMBackboneState:
        """
        Initialize streaming state for HFLM backbone.

        Creates a DynamicCache for efficient KV caching during streaming.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            HFLMBackboneState instance with initialized cache
        """
        device = self.get_device()

        # Create DynamicCache for HuggingFace models
        try:
            from transformers import DynamicCache
            kv_cache = DynamicCache()
        except ImportError:
            # Fallback for older transformers versions
            kv_cache = None
            logger.warning(
                "[HFLMBackbone] DynamicCache not available. "
                "Streaming may not work optimally."
            )

        return HFLMBackboneState(
            batch_size=batch_size,
            device=device,
            kv_cache=kv_cache,
            past_seen_tokens=0,
        )

    def get_hidden_dim(self) -> int:
        """Return the HF LM's native hidden dimension (3072)."""
        return self._hidden_dim

    # =========================================================================
    # Streaming Methods
    # =========================================================================

    def streaming_forward(
        self,
        hidden_states: Tensor,
        **kwargs,
    ) -> BackboneOutput:
        """
        Single step forward for streaming inference.

        Uses the stored streaming state (KV cache) for efficient generation.

        Args:
            hidden_states: Input tensor [B, 1, D] for single step

        Returns:
            BackboneOutput with hidden states for this step
        """
        if not self.is_streaming:
            raise RuntimeError("Must be in streaming mode. Use backbone.streaming(batch_size).")

        state = self._streaming_state

        # Compute position_ids based on past tokens
        batch_size = hidden_states.shape[0]
        position_ids = torch.tensor(
            [[state.past_seen_tokens]],
            device=hidden_states.device
        ).expand(batch_size, -1)

        # Forward with cache
        output = self.forward(
            hidden_states,
            position_ids=position_ids,
            past_key_values=state.kv_cache,
            use_cache=True,
            **kwargs,
        )

        # Update state
        state.update_position(hidden_states.shape[1])

        # Update cache reference
        if output.past_key_values is not None:
            state.kv_cache = output.past_key_values

        return output

    # =========================================================================
    # Extended Methods
    # =========================================================================

    def get_num_layers(self) -> int:
        """Return number of transformer layers."""
        return self._num_layers

    def get_num_heads(self) -> int:
        """Return number of attention heads."""
        return self._num_heads

    def get_num_kv_heads(self) -> int:
        """Return number of key-value heads (for GQA)."""
        return self._num_kv_heads

    @property
    def kv_repeat(self) -> int:
        """Return GQA repetition factor."""
        return self._num_heads // self._num_kv_heads

    def supports_gradient_checkpointing(self) -> bool:
        """HFLM supports gradient checkpointing."""
        return True

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing in the model.

        Uses non-reentrant mode (use_reentrant=False) for FSDP compatibility.
        This is critical to avoid dimension errors during backward pass when
        combined with FSDP sharding.
        """
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            import inspect
            sig = inspect.signature(self.model.gradient_checkpointing_enable)
            if 'gradient_checkpointing_kwargs' in sig.parameters:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                logger.info(
                    "[HFLMBackbone] Gradient checkpointing enabled "
                    "(non-reentrant mode for FSDP compatibility)"
                )
            else:
                self.model.gradient_checkpointing_enable()
                logger.warning(
                    "[HFLMBackbone] Gradient checkpointing enabled with default settings. "
                    "Consider upgrading transformers>=4.31 for FSDP compatibility."
                )
            self._gradient_checkpointing = True
        else:
            logger.warning(
                "[HFLMBackbone] Cannot enable gradient checkpointing - "
                "method not found"
            )

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing in the model."""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
            self._gradient_checkpointing = False
            logger.info("[HFLMBackbone] Gradient checkpointing disabled")

    def get_device(self) -> torch.device:
        """Return the device of model parameters."""
        for param in self.model.parameters():
            return param.device
        return self._device

    def get_dtype(self) -> torch.dtype:
        """Return the dtype of model parameters."""
        for param in self.model.parameters():
            return param.dtype
        return self._dtype

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
        if hasattr(self.model, 'layers'):
            return self.model.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        return []

    def get_embedding_layer(self) -> Optional[nn.Module]:
        """
        Get the embedding layer from the model.

        This can be useful for weight tying or analysis.
        """
        if hasattr(self.model, 'embed_tokens'):
            return self.model.embed_tokens
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens
        return None

    def freeze_layers(self, num_layers: Optional[int] = None) -> int:
        """
        Freeze transformer layers (useful for efficient finetuning).

        Args:
            num_layers: Number of layers to freeze from bottom.
                       If None, freezes all layers.

        Returns:
            Number of layers actually frozen
        """
        layers = self.layers
        if not layers:
            logger.warning("[HFLMBackbone] No layers found to freeze")
            return 0

        num_to_freeze = num_layers if num_layers is not None else len(layers)
        num_to_freeze = min(num_to_freeze, len(layers))

        frozen_count = 0
        for i, layer in enumerate(layers):
            if i < num_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                frozen_count += 1

        logger.info(f"[HFLMBackbone] Frozen {frozen_count}/{len(layers)} layers")
        return frozen_count

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("[HFLMBackbone] All parameters unfrozen")

    def __repr__(self) -> str:
        return (
            f"HFLMBackbone(\n"
            f"  model_path='{self.config.model_path}',\n"
            f"  hidden_dim={self._hidden_dim},\n"
            f"  num_layers={self._num_layers},\n"
            f"  num_heads={self._num_heads},\n"
            f"  num_kv_heads={self._num_kv_heads} (GQA repeat={self.kv_repeat}),\n"
            f"  gradient_checkpointing={self._gradient_checkpointing}\n"
            f")"
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_hf_lm_backbone(
    model_path: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    load_pretrained: bool = True,
    gradient_checkpointing: bool = True,
    **kwargs,
) -> HFLMBackbone:
    """
    Convenience function to create HFLMBackbone.

    Args:
        model_path: Path to HuggingFace model or model name
        device: Target device
        dtype: Target dtype
        load_pretrained: Whether to load pretrained weights
        gradient_checkpointing: Whether to enable gradient checkpointing
        **kwargs: Additional config parameters

    Returns:
        Initialized HFLMBackbone
    """
    config = HFLMBackboneConfig(
        model_path=model_path,
        load_pretrained=load_pretrained,
        gradient_checkpointing=gradient_checkpointing,
        **kwargs,
    )
    return HFLMBackbone(config, device=device, dtype=dtype)
