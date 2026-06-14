"""
Backbone Configuration Module.

Defines configuration dataclasses for all backbone types and dimension adapters.
These configs are designed to be YAML-serializable via simple_parsing.

Configuration Hierarchy:
    BackboneConfig (base)
    ├── MoshiBackboneConfig   (Moshi 7B default)
    ├── HFLMBackboneConfig  (HF-causal-LM)
    └── DimensionAdapterConfig (optional projection)

Example YAML:
    backbone:
      type: "hf_lm"

      moshi:
        # Used when type="moshi" (default)
        use_streaming_transformer: true

      hf_lm:
        # Used when type="hf_lm"
        model_path: "/path/to/HF-causal-LM"
        hidden_size: 3072
        num_attention_heads: 24
        num_key_value_heads: 4
        num_hidden_layers: 30

      dimension_adapter:
        enable: true
        moshi_dim: 4096
        bias: false
        init_method: "xavier"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict
import logging
import torch

from simple_parsing.helpers import Serializable


class BackboneType(str, Enum):
    """Supported backbone types."""
    MOSHI = "moshi"
    HFLM = "hf_lm"
    CUSTOM = "custom"


@dataclass
class BackboneConfig(Serializable):
    """
    Base backbone configuration.

    All backbone-specific configs inherit from this class.
    This provides common settings shared across all backbone types.
    """
    # Hidden dimension of the backbone transformer
    hidden_dim: int = 4096

    # Number of transformer layers
    num_layers: int = 32

    # Number of attention heads
    num_heads: int = 32

    # Maximum sequence length supported
    max_seq_len: int = 4096

    # Data type for model parameters
    dtype: str = "bfloat16"

    # Device placement ("cuda", "cpu", "meta")
    device: str = "cuda"

    # Enable gradient checkpointing (memory vs compute tradeoff)
    gradient_checkpointing: bool = True

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.dtype not in dtype_map:
            raise ValueError(f"Unknown dtype: {self.dtype}. Supported: {list(dtype_map.keys())}")
        return dtype_map[self.dtype]


@dataclass
class MoshiBackboneConfig(BackboneConfig):
    """
    Moshi-specific backbone configuration.

    Default values match Moshi 7B architecture:
    - 4096 hidden dimension
    - 32 transformer layers
    - 32 attention heads (MHA, no GQA)
    - 16384 feedforward dimension (4x hidden)

    This is used when wrapping the existing Moshi StreamingTransformer.
    """
    # Override defaults to match Moshi 7B
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    max_seq_len: int = 4096

    # Moshi-specific: feedforward dimension
    dim_feedforward: int = 16384  # 4 * hidden_dim

    # Moshi-specific: use streaming transformer
    use_streaming_transformer: bool = True

    # Moshi-specific: context window for cross-attention (None = unlimited)
    context: Optional[int] = None

    # Moshi-specific: causal attention
    causal: bool = True

    # Moshi-specific: normalization type
    norm: str = "layer_norm"

    # Moshi-specific: normalize embeddings
    norm_emb: bool = False

    # Moshi-specific: positional embedding type
    positional_embedding: str = "rope"

    # Moshi-specific: RoPE base frequency
    rope_base: float = 10000.0


@dataclass
class HFLMBackboneConfig(BackboneConfig):
    """
    HF-causal-LM backbone configuration.

    Key differences from Moshi:
    - Smaller dimension: 3072 (vs 4096)
    - GQA: 24 heads, 4 KV heads (kv_repeat=6)
    - 30 layers (vs 32)
    - Sliding window attention
    - Higher RoPE theta (500000)

    This requires a DimensionAdapter when used with Moshi components.
    """
    # Override for HFLM architecture
    hidden_dim: int = 3072
    num_layers: int = 30
    num_heads: int = 24
    max_seq_len: int = 32768

    # HFLM-specific: GQA configuration
    num_key_value_heads: int = 4  # GQA: 24/4 = 6x repetition

    # HFLM-specific: feedforward dimension
    intermediate_size: int = 8192

    # HFLM-specific: RoPE configuration
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict] = None  # For extended context

    # HFLM-specific: sliding window attention
    # Set sliding_window to None for full attention (no window)
    use_sliding_window: bool = True
    sliding_window: Optional[int] = 4096  # None = full attention

    # HFLM-specific: activation function
    hidden_act: str = "silu"

    # HFLM-specific: RMS norm epsilon
    rms_norm_eps: float = 1e-6

    # HFLM-specific: tie word embeddings
    tie_word_embeddings: bool = False

    # Path to HuggingFace model directory or model name
    model_path: str = ""

    # Whether to load pretrained weights (False = random init)
    load_pretrained: bool = True

    # Trust remote code for HuggingFace models
    trust_remote_code: bool = True

    # Model precision (bfloat16, float16, float32)
    torch_dtype: str = "bfloat16"

    # Flash Attention 2 (requires flash-attn package)
    use_flash_attention: bool = False

    def get_torch_dtype(self) -> torch.dtype:
        """
        Convert string torch_dtype to torch.dtype.

        Override the parent method to use HFLM-specific torch_dtype field.
        This is compatible with HuggingFace Transformers naming convention.
        """
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.torch_dtype not in dtype_map:
            raise ValueError(
                f"Unknown torch_dtype: {self.torch_dtype}. "
                f"Supported: {list(dtype_map.keys())}"
            )
        return dtype_map[self.torch_dtype]

    @property
    def kv_repeat(self) -> int:
        """Number of times to repeat KV heads for GQA."""
        return self.num_heads // self.num_key_value_heads

    # Internal flag to track if this config is actually being used
    _is_active: bool = field(default=False, repr=False)

    def __post_init__(self):
        # Validate GQA configuration
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

        # Note: GQA logging is deferred to when the config is actually used
        # This prevents spurious logs when HFLMBackboneConfig is instantiated
        # as a default field in UnifiedBackboneConfig but Moshi is selected

    def log_gqa_config(self):
        """Log GQA configuration when this backbone is actually being used."""
        if self.num_key_value_heads != self.num_heads:
            logging.info(
                f"[HFLM] GQA enabled: {self.num_heads} heads, "
                f"{self.num_key_value_heads} KV heads (repeat={self.kv_repeat})"
            )


@dataclass
class DimensionAdapterConfig(Serializable):
    """
    Dimension adapter configuration for bridging different backbone dimensions.

    When using a backbone with different hidden dimension than Moshi (4096),
    linear projections are used to bridge the dimensions.

    Architecture:
        Embeddings (4096) → input_proj → Backbone → output_proj → Depformer (4096)

    Example:
        HFLM (3072):
        - input_proj:  Linear(4096 → 3072)
        - output_proj: Linear(3072 → 4096)
    """
    # Enable dimension adaptation
    enable: bool = False

    # Moshi embedding/depformer dimension (fixed)
    moshi_dim: int = 4096

    # Target backbone dimension (auto-detected if not specified)
    backbone_dim: Optional[int] = None

    # Use bias in projection layers
    bias: bool = False

    # Dropout rate for projection layers (0.0 = no dropout)
    dropout: float = 0.0

    # Initialization method for projection weights
    # Options: "xavier", "kaiming", "normal", "orthogonal"
    init_method: str = "xavier"

    # Initialization scale factor (multiplier for initialized weights)
    init_scale: float = 1.0

    # Standard deviation for "normal" initialization
    init_std: float = 0.02

    # Scale factor for residual connections (0 = no residual)
    residual_scale: float = 0.0

    def __post_init__(self):
        valid_init_methods = ("xavier", "kaiming", "normal", "orthogonal")
        if self.init_method not in valid_init_methods:
            raise ValueError(
                f"init_method must be one of {valid_init_methods}, "
                f"got '{self.init_method}'"
            )


@dataclass
class UnifiedBackboneConfig(Serializable):
    """
    Unified backbone configuration for YAML-based model selection.

    This is the main configuration class used in training args.
    It contains all backbone-specific configs and the type selector.

    Example YAML:
        backbone:
          type: "hf_lm"

          moshi:
            use_streaming_transformer: true
            gradient_checkpointing: true

          hf_lm:
            model_path: "/path/to/HF causal LM"
            hidden_size: 3072

          dimension_adapter:
            enable: true
            init_method: "xavier"
    """
    # Backbone type selector
    type: str = "moshi"

    # Moshi-specific configuration
    moshi: MoshiBackboneConfig = field(default_factory=MoshiBackboneConfig)

    # HFLM-specific configuration
    hf_lm: HFLMBackboneConfig = field(default_factory=HFLMBackboneConfig)

    # Dimension adapter configuration
    dimension_adapter: DimensionAdapterConfig = field(default_factory=DimensionAdapterConfig)

    def __post_init__(self):
        # Validate type
        valid_types = ("moshi", "hf_lm", "custom")
        if self.type not in valid_types:
            raise ValueError(
                f"backbone.type must be one of {valid_types}, "
                f"got '{self.type}'"
            )

        # Auto-configure dimension adapter based on backbone type
        if self.type == "hf_lm" and not self.dimension_adapter.enable:
            if self.hf_lm.hidden_dim != self.dimension_adapter.moshi_dim:
                logging.warning(
                    f"HFLM hidden_dim ({self.hf_lm.hidden_dim}) differs from "
                    f"Moshi dim ({self.dimension_adapter.moshi_dim}). "
                    f"Consider enabling dimension_adapter."
                )

        # Set backbone_dim automatically if not specified
        if self.dimension_adapter.enable and self.dimension_adapter.backbone_dim is None:
            if self.type == "moshi":
                self.dimension_adapter.backbone_dim = self.moshi.hidden_dim
            elif self.type == "hf_lm":
                self.dimension_adapter.backbone_dim = self.hf_lm.hidden_dim

        # Log configuration
        logging.info(f"[Backbone] Type: {self.type}")
        if self.dimension_adapter.enable:
            logging.info(
                f"[Backbone] DimensionAdapter: "
                f"{self.dimension_adapter.moshi_dim} <-> {self.dimension_adapter.backbone_dim}"
            )

    def get_active_config(self) -> BackboneConfig:
        """Get the configuration for the selected backbone type."""
        if self.type == "moshi":
            return self.moshi
        elif self.type == "hf_lm":
            # Log GQA config only when HFLM is actually being used
            self.hf_lm.log_gqa_config()
            self.hf_lm._is_active = True
            return self.hf_lm
        else:
            raise ValueError(f"Unknown backbone type: {self.type}")
