"""
LMModelWrapper for Modular Backbone System.

This module provides a unified wrapper that combines:
1. Modular backbone (Moshi, HFLM, custom)
2. Dimension adapter (for non-4096 backbones)
3. Shared Moshi components (embeddings, depformer, linears)
4. Speaker conditioning via sum_condition (optional)

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     LMModelWrapper                                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  Input: codes [B, 9/17, T]                                          │
    │                                                                      │
    │  ┌─────────────────────────────────────────────────────────────────┐ │
    │  │ Embeddings (from LMModel, fixed at d=4096)                      │ │
    │  │  - text_emb: [32000, 4096]                                      │ │
    │  │  - audio_embs: n_q x [2048, 4096]  (n_q=16 for full-duplex)     │ │
    │  └──────────────────────────┬──────────────────────────────────────┘ │
    │                             │                                        │
    │                             ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────┐ │
    │  │ DimensionAdapter.input_proj (optional, for HFLM)              │ │
    │  │  - Linear(4096 → 3072) for HFLM                               │ │
    │  │  - Identity for Moshi                                           │ │
    │  └──────────────────────────┬──────────────────────────────────────┘ │
    │                             │                                        │
    │                             ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────┐ │
    │  │ AbstractBackbone (pluggable)                                    │ │
    │  │  ├── MoshiBackbone:  StreamingTransformer (32L, 4096d, 32h)    │ │
    │  │  ├── HFLMBackbone: Mistral-like (30L, 3072d, 24h, GQA)       │ │
    │  │  └── CustomBackbone: User-defined                              │ │
    │  └──────────────────────────┬──────────────────────────────────────┘ │
    │                             │                                        │
    │                             ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────┐ │
    │  │ DimensionAdapter.output_proj (optional, for HFLM)             │ │
    │  │  - Linear(3072 → 4096) for HFLM                               │ │
    │  │  - Identity for Moshi                                           │ │
    │  └──────────────────────────┬──────────────────────────────────────┘ │
    │                             │                                        │
    │                             ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────┐ │
    │  │ Depformer (from LMModel, fixed at d=1024)                       │ │
    │  │  - Depth transformer for autoregressive codebook prediction    │ │
    │  └──────────────────────────┬──────────────────────────────────────┘ │
    │                             │                                        │
    │                             ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────┐ │
    │  │ Output Linears (from LMModel)                                   │ │
    │  │  - text_linear: [4096, 32000]                                   │ │
    │  │  - audio_linears: 8 x [1024, 2048]                             │ │
    │  └─────────────────────────────────────────────────────────────────┘ │
    │                                                                      │
    │  Output: LMOutput (text_logits, audio_logits)                       │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    # From config
    config = UnifiedBackboneConfig(type="hf_lm", ...)
    wrapper = LMModelWrapper.from_config(config, base_lm_model)

    # Training forward
    output = wrapper(codes)

    # Streaming inference
    with wrapper.streaming(batch_size=1):
        output = wrapper.step(codes_step)

Author: K-Moshi Development Team
Date: 2025-01-01
"""

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, List

import torch
from torch import nn, Tensor

from .base import AbstractBackbone, BackboneOutput
from .config import UnifiedBackboneConfig, BackboneType
from .factory import BackboneFactory
from .adapters import DimensionAdapter, IdentityAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# Delay Utility Functions (matching original lm_utils.py exactly)
# =============================================================================

def _delay_sequence(
    delays: List[int],
    tensor: torch.Tensor,
    padding: torch.Tensor,
) -> torch.Tensor:
    """
    Apply delays to a sequence tensor.

    This exactly mirrors moshi/models/lm_utils.py:_delay_sequence

    Args:
        delays: List of delays for each codebook [K]
        tensor: Input tensor [B, K, T]
        padding: Padding tensor for delayed positions [B, K, 1]

    Returns:
        Delayed tensor [B, K, T]
    """
    B, K, T = tensor.shape
    assert len(delays) == K, (len(delays), K)
    outs = []

    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(delay, dims=1)
        if delay > 0:
            line[:, :delay] = padding[:, k]
        outs.append(line)
    return torch.stack(outs, dim=1)


def _undelay_sequence(
    delays: List[int],
    tensor: torch.Tensor,
    fill_value: float = float('NaN'),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Undo delays on a sequence tensor and create validity mask.

    This exactly mirrors moshi/models/lm_utils.py:_undelay_sequence

    Args:
        delays: List of delays for each codebook [K]
        tensor: Input tensor [B, K, T, ...]
        fill_value: Value to fill invalid positions

    Returns:
        Tuple of:
            - Undelayed tensor [B, K, T, ...]
            - Validity mask [B, K, T] where True = valid position
    """
    B, K, T, *rest = tensor.shape
    assert len(delays) == K
    mask = torch.ones(B, K, T, dtype=torch.bool, device=tensor.device)
    outs = []

    if all([delay == 0 for delay in delays]):
        return tensor, mask

    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(-delay, dims=1)
        if delay > 0:
            line[:, -delay:] = fill_value
            mask[:, k, -delay:] = 0
        outs.append(line)
    return torch.stack(outs, dim=1), mask


@dataclass
class LMModelWrapperOutput:
    """
    Output from LMModelWrapper forward pass.

    Mirrors the output structure of LMModel for compatibility with
    existing training and evaluation code.

    Attributes:
        text_logits: Text prediction logits [B, T, vocab_size]
        logits: Audio prediction logits [B, dep_q, T, codebook_size]
        text_mask: Mask for text loss computation [B, T]
        mask: Mask for audio loss computation [B, dep_q, T]
    """
    text_logits: Tensor  # [B, T, text_vocab_size]
    logits: Tensor  # [B, dep_q, T, audio_vocab_size]
    text_mask: Optional[Tensor] = None  # [B, T]
    mask: Optional[Tensor] = None  # [B, dep_q, T]


class LMModelWrapper(nn.Module):
    """
    Unified wrapper for modular backbone system.

    This class provides a drop-in replacement for LMModel that supports:
    1. Pluggable backbones (Moshi, HFLM, custom)
    2. Automatic dimension adaptation
    3. Streaming inference compatibility
    4. Gradient checkpointing support

    The wrapper preserves the LMModel interface while enabling
    backbone swapping via configuration.

    Attributes:
        config: UnifiedBackboneConfig specifying backbone type and settings
        backbone: AbstractBackbone implementation
        adapter: DimensionAdapter or IdentityAdapter
        embeddings: Shared embedding layers from base LMModel
        depformer: Shared depth transformer from base LMModel
        linears: Shared output linear layers from base LMModel
    """

    def __init__(
        self,
        config: UnifiedBackboneConfig,
        backbone: AbstractBackbone,
        adapter: nn.Module,
        text_emb: nn.Module,
        audio_embs: nn.ModuleList,
        depformer: nn.Module,
        depformer_in: nn.ModuleList,
        depformer_emb: nn.ModuleList,
        depformer_text_emb: nn.Module,
        text_linear: nn.Module,
        audio_linears: nn.ModuleList,
        n_q: int = 16,
        dep_q: int = 8,
        dim: int = 4096,
        text_card: int = 32000,
        audio_card: int = 2048,
        **kwargs,
    ):
        """
        Initialize LMModelWrapper.

        Args:
            config: Unified backbone configuration
            backbone: Initialized backbone instance
            adapter: Dimension adapter (DimensionAdapter or IdentityAdapter)
            text_emb: Text embedding layer
            audio_embs: List of audio embedding layers
            depformer: Depth transformer
            depformer_in: Depformer input projections
            depformer_emb: Depformer embeddings
            depformer_text_emb: Depformer text embedding for teacher forcing
            text_linear: Text output linear
            audio_linears: Audio output linears
            n_q: Total number of audio codebooks (input)
            dep_q: Number of depformer codebooks (output)
            dim: Model dimension (4096 for Moshi)
            text_card: Text vocabulary size
            audio_card: Audio codebook size
        """
        super().__init__()

        self.config = config
        self.backbone = backbone
        self.adapter = adapter

        # Shared components from base LMModel
        self.text_emb = text_emb
        self.audio_embs = audio_embs
        self.depformer = depformer
        self.depformer_in = depformer_in
        self.depformer_emb = depformer_emb
        self.depformer_text_emb = depformer_text_emb
        self.text_linear = text_linear
        self.audio_linears = audio_linears

        # Output normalization (applied after backbone, before text_linear/depformer)
        # Only used when backbone doesn't have built-in output normalization
        self.out_norm = kwargs.get('out_norm', None)

        # Dimensions
        self._n_q = n_q
        self._dep_q = dep_q
        self._dim = dim
        self._text_card = text_card
        self._audio_card = audio_card

        # Special token IDs (required by train.py)
        self._text_padding_token_id = kwargs.get('text_padding_token_id', 3)
        self._end_of_text_padding_id = kwargs.get('end_of_text_padding_id', 0)
        self._zero_token_id = kwargs.get('zero_token_id', -1)  # Match original LMModel
        self._audio_offset = 1  # Always 1 (text is first, audio starts at index 1)

        # Initial token IDs for sequence start
        self._initial_token_id = audio_card  # Same as LMModel.initial_token_id
        self._text_initial_token_id = text_card  # Same as LMModel.text_initial_token_id

        # Delays for each codebook (CRITICAL for correct training)
        # Default delays from original moshi: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        # For n_q=16 (full-duplex): 1 text + 8 moshi audio + 8 user audio = 17 codebooks
        # For n_q=8 (monologue): 1 text + 8 audio = 9 codebooks
        default_delays = [0] + [0] + [1] * (dep_q - 1)  # Simplified default
        if n_q > dep_q:
            # Full-duplex: add user stream delays
            user_audio_delays = [0] + [1] * (n_q - dep_q - 1)
            default_delays = default_delays + user_audio_delays
        self._delays = kwargs.get('delays', default_delays)

        # Validate delays match codebook count
        num_codebooks = n_q + 1  # text + audio
        if len(self._delays) != num_codebooks:
            logger.warning(
                f"[LMModelWrapper] Delays length {len(self._delays)} != num_codebooks {num_codebooks}. "
                f"Adjusting delays to match."
            )
            if len(self._delays) < num_codebooks:
                self._delays = self._delays + [0] * (num_codebooks - len(self._delays))
            else:
                self._delays = self._delays[:num_codebooks]

        # Depformer norms (required by train.py for model verification)
        self._depformer_norms = kwargs.get('depformer_norms', None)

        # Store additional attributes
        self._extra_attrs = kwargs

        # Speaker conditioning (optional, for zero-shot speaker adaptation)
        # This is set via set_speaker_conditioner() after initialization
        self.speaker_conditioner: Optional[nn.Module] = None
        self._speaker_conditioning_enabled = False

        logger.info(
            f"[LMModelWrapper] Initialized with backbone={config.type}, "
            f"adapter_enabled={getattr(adapter, 'is_enabled', False)}, "
            f"n_q={n_q}, dep_q={dep_q}, dim={dim}, delays={self._delays[:5]}..."
        )

    @classmethod
    def from_config(
        cls,
        config: UnifiedBackboneConfig,
        base_lm_model: Any,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "LMModelWrapper":
        """
        Create LMModelWrapper from configuration and base LMModel.

        This factory method:
        1. Creates the backbone using BackboneFactory
        2. Extracts shared components from base_lm_model
        3. Assembles the wrapper

        Args:
            config: Unified backbone configuration
            base_lm_model: Existing LMModel to extract components from
            device: Target device for model placement
            dtype: Target dtype for model parameters

        Returns:
            Initialized LMModelWrapper
        """
        logger.info(f"[LMModelWrapper] Creating wrapper with backbone type: {config.type}")

        # Extract LMModel attributes
        n_q = getattr(base_lm_model, 'n_q', 16)
        dep_q = getattr(base_lm_model, 'dep_q', 8)
        dim = getattr(base_lm_model, 'dim', 4096)
        text_card = getattr(base_lm_model, 'text_card', 32000)
        audio_card = getattr(base_lm_model, 'audio_card', 2048)

        # Create backbone and adapter using factory
        backbone, adapter = BackboneFactory.create(
            config,
            lm_model=base_lm_model,
            device=device,
            dtype=dtype,
        )

        # Extract shared components from base LMModel
        text_emb = base_lm_model.text_emb
        audio_embs = base_lm_model.emb

        # Extract depformer components
        depformer = base_lm_model.depformer
        depformer_in = base_lm_model.depformer_in
        depformer_emb = base_lm_model.depformer_emb
        depformer_text_emb = getattr(base_lm_model, 'depformer_text_emb', None)
        if depformer_text_emb is None:
            raise ValueError(
                "base_lm_model must have 'depformer_text_emb' attribute. "
                "Please ensure you're using the correct Moshi model version."
            )

        # Extract output linears
        text_linear = base_lm_model.text_linear
        audio_linears = base_lm_model.linears

        # Extract special token IDs (required by train.py)
        text_padding_token_id = getattr(base_lm_model, 'text_padding_token_id', 3)
        end_of_text_padding_id = getattr(base_lm_model, 'end_of_text_padding_id', 0)
        zero_token_id = getattr(base_lm_model, 'zero_token_id', -1)

        # Extract delays from base LMModel (CRITICAL for correct training)
        # Original LMModel stores delays as self.delays
        delays = getattr(base_lm_model, 'delays', None)
        if delays is None:
            logger.warning(
                "[LMModelWrapper] Could not extract delays from base_lm_model. "
                "Using default delays. This may cause training issues."
            )

        # Extract depformer norms if available (required by train.py for verification)
        depformer_norms = getattr(base_lm_model, 'depformer_norms', None)
        if depformer_norms is None and hasattr(depformer, 'norms'):
            depformer_norms = depformer.norms

        # Extract output normalization (applied after transformer, before text_linear/depformer)
        # This is important for proper normalization matching original LMModel
        out_norm = getattr(base_lm_model, 'out_norm', None)

        # Create wrapper
        wrapper = cls(
            config=config,
            backbone=backbone,
            adapter=adapter,
            text_emb=text_emb,
            audio_embs=audio_embs,
            depformer=depformer,
            depformer_in=depformer_in,
            depformer_emb=depformer_emb,
            depformer_text_emb=depformer_text_emb,
            text_linear=text_linear,
            audio_linears=audio_linears,
            n_q=n_q,
            dep_q=dep_q,
            dim=dim,
            text_card=text_card,
            audio_card=audio_card,
            text_padding_token_id=text_padding_token_id,
            end_of_text_padding_id=end_of_text_padding_id,
            zero_token_id=zero_token_id,
            delays=delays,
            depformer_norms=depformer_norms,
            out_norm=out_norm,
        )

        # Move to device/dtype if specified
        if device is not None:
            wrapper = wrapper.to(device)
        if dtype is not None:
            wrapper = wrapper.to(dtype)

        return wrapper

    @classmethod
    def from_lm_model(
        cls,
        lm_model: Any,
        config: Optional[UnifiedBackboneConfig] = None,
    ) -> "LMModelWrapper":
        """
        Create LMModelWrapper by wrapping an existing LMModel.

        This is the simplest way to create a wrapper for an existing Moshi model.
        The wrapper will use MoshiBackbone and IdentityAdapter.

        Args:
            lm_model: Existing LMModel instance
            config: Optional config (defaults to Moshi config if None)

        Returns:
            LMModelWrapper wrapping the existing LMModel
        """
        if config is None:
            config = UnifiedBackboneConfig(type="moshi")

        return cls.from_config(config, lm_model)

    # =========================================================================
    # LMModel-compatible Interface
    # =========================================================================

    @property
    def n_q(self) -> int:
        """Number of audio codebook positions (input)."""
        return self._n_q

    @property
    def dep_q(self) -> int:
        """Number of depformer codebooks (output)."""
        return self._dep_q

    @property
    def dim(self) -> int:
        """Model dimension (always 4096 for Moshi compatibility)."""
        return self._dim

    @property
    def num_codebooks(self) -> int:
        """Total number of codebooks (text + audio)."""
        return self._n_q + 1

    @property
    def text_card(self) -> int:
        """Text vocabulary size."""
        return self._text_card

    @property
    def audio_card(self) -> int:
        """Audio codebook size."""
        return self._audio_card

    @property
    def text_padding_token_id(self) -> int:
        """Text padding token ID (required by train.py for interleaver)."""
        return self._text_padding_token_id

    @property
    def end_of_text_padding_id(self) -> int:
        """End of text padding ID (required by train.py for interleaver)."""
        return self._end_of_text_padding_id

    @property
    def zero_token_id(self) -> int:
        """Zero/silence token ID (required by train.py for interleaver)."""
        return self._zero_token_id

    @property
    def audio_offset(self) -> int:
        """Audio token offset in vocabulary (required by train.py)."""
        return self._audio_offset

    @property
    def linears(self) -> nn.ModuleList:
        """Alias for audio_linears (required by train.py for model verification)."""
        return self.audio_linears

    @property
    def emb(self) -> nn.ModuleList:
        """Alias for audio_embs (required by train.py for LMModel compatibility)."""
        return self.audio_embs

    @property
    def depformer_norms(self) -> Optional[nn.ModuleList]:
        """Depformer layer norms (required by train.py for model verification)."""
        return self._depformer_norms

    @property
    def backbone_dim(self) -> int:
        """Backbone hidden dimension (e.g., 3072 for HFLM, 4096 for Moshi)."""
        return self.backbone.get_hidden_dim()

    @property
    def moshi_dim(self) -> int:
        """Moshi fixed dimension (always 4096 for embeddings and output heads)."""
        return self._dim

    @property
    def delays(self) -> List[int]:
        """Delay pattern for each codebook (required for proper sequence alignment)."""
        return self._delays

    @property
    def num_codebooks(self) -> int:
        """Total number of codebooks (1 text + n_q audio)."""
        return self._n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        """Total number of audio codebooks in input (n_q)."""
        return self._n_q

    @property
    def num_audio_embs(self) -> int:
        """
        Number of audio embeddings available for processing.

        This returns len(self.audio_embs), which is typically n_q (16 for full-duplex).
        When created via from_config(), audio_embs is extracted from base_lm_model.emb
        which has n_q embeddings (16 in Moshi's full-duplex architecture).

        In full-duplex mode:
        - n_q = 16: Total audio codebooks in input (8 moshi + 8 user)
        - dep_q = 8: Audio codebooks predicted by depformer (moshi only)
        - audio_embs has n_q (16) embeddings for Temporal Transformer input
        - This matches original Moshi LMModel.emb which has n_q embeddings

        Note: The Temporal Transformer should embed ALL n_q audio codebooks
        as input context. Only the Depth Transformer output is limited to dep_q.
        """
        return len(self.audio_embs)

    @property
    def initial_token_id(self) -> int:
        """Token ID for start of audio sequence."""
        return self._initial_token_id

    @property
    def text_initial_token_id(self) -> int:
        """Token ID for start of text sequence."""
        return self._text_initial_token_id

    # =========================================================================
    # Initial Token Generation (matching LMModel._get_initial_token)
    # =========================================================================

    def _get_initial_token(self) -> torch.Tensor:
        """
        Generate initial tokens for sequence start.

        This exactly mirrors LMModel._get_initial_token()

        Returns:
            Initial token tensor [1, K, 1] where K = num_codebooks
        """
        device = next(iter(self.parameters())).device
        zero = torch.full(
            [1, 1, 1], self._zero_token_id, device=device, dtype=torch.long
        )
        special = torch.full_like(zero, self._initial_token_id)
        text_special = torch.full_like(zero, self._text_initial_token_id)

        audio_token = special.expand(-1, self.num_audio_codebooks, -1)
        text_token = text_special
        token = torch.cat([text_token, audio_token], dim=1)
        return token

    # =========================================================================
    # Speaker Conditioning Methods (Zero-Shot Speaker Adaptation)
    # =========================================================================

    def set_speaker_conditioner(self, conditioner: nn.Module) -> None:
        """
        Set the speaker conditioner module for zero-shot speaker adaptation.

        The speaker conditioner transforms speaker embeddings to sum_condition
        format, which is added to the Temporal Transformer input:
            input_ = text_emb[MOSHI] + Σaudio_emb[MOSHI] + speaker_condition

        Args:
            conditioner: SpeakerConditioner or SpeakerConditioningModule instance

        Usage:
            from finetune.modules import SpeakerConditioner, SpeakerConditionerConfig

            config = SpeakerConditionerConfig(input_dim=192, output_dim=4096)
            conditioner = SpeakerConditioner(config)
            wrapper.set_speaker_conditioner(conditioner)
        """
        self.speaker_conditioner = conditioner
        self._speaker_conditioning_enabled = True
        logger.info("[LMModelWrapper] Speaker conditioner set and enabled")

    def enable_speaker_conditioning(self) -> None:
        """Enable speaker conditioning (if conditioner is set)."""
        if self.speaker_conditioner is None:
            raise ValueError(
                "Cannot enable speaker conditioning: no conditioner set. "
                "Call set_speaker_conditioner() first."
            )
        self._speaker_conditioning_enabled = True
        logger.info("[LMModelWrapper] Speaker conditioning enabled")

    def disable_speaker_conditioning(self) -> None:
        """Disable speaker conditioning (keeps conditioner, just doesn't use it)."""
        self._speaker_conditioning_enabled = False
        logger.info("[LMModelWrapper] Speaker conditioning disabled")

    @property
    def speaker_conditioning_enabled(self) -> bool:
        """Check if speaker conditioning is currently enabled."""
        return self._speaker_conditioning_enabled and self.speaker_conditioner is not None

    def get_speaker_stats(self) -> dict:
        """Get statistics from speaker conditioner for monitoring."""
        if self.speaker_conditioner is not None and hasattr(self.speaker_conditioner, 'get_stats'):
            return self.speaker_conditioner.get_stats()
        return {}

    # =========================================================================
    # Forward Methods (matching LMModel exactly)
    # =========================================================================

    def forward(
        self,
        codes: Tensor,
        condition_tensors: Optional[Any] = None,
        sum_condition: Optional[Tensor] = None,
        speaker_embedding: Optional[Tensor] = None,
        **kwargs,
    ) -> LMModelWrapperOutput:
        """
        Forward pass matching LMModel.forward() exactly.

        This implementation follows the original LMModel flow:
        1. Apply delays to input codes
        2. Prepend initial tokens
        3. Compute embeddings (forward_text style)
        4. Process through backbone
        5. Generate text and audio logits
        6. Undelay output logits to align with input codes

        Args:
            codes: Input codes [B, K, T] where K = num_codebooks
                   - codes[:, 0]: Text tokens
                   - codes[:, 1:]: Audio codebook tokens
            condition_tensors: Optional conditioning (unused in current impl)
            sum_condition: Optional pre-computed sum_condition [B, 1, D] for
                          speaker conditioning. If provided, added to combined_input.
            speaker_embedding: Optional speaker embedding [B, D_spk] from speaker
                              encoder. Will be transformed via speaker_conditioner
                              if set. Mutually exclusive with sum_condition.

        Returns:
            LMModelWrapperOutput with properly aligned logits and masks

        Note on Speaker Conditioning:
            Speaker conditioning is applied via sum_condition mechanism:
            combined_input = text_emb + audio_emb + speaker_condition

            There are two ways to provide speaker conditioning:
            1. Direct: Pass sum_condition [B, 1, D] tensor
            2. Via conditioner: Pass speaker_embedding [B, D_spk], requires
               speaker_conditioner to be set via set_speaker_conditioner()
        """
        B, K, T = codes.shape
        assert K == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {K}"

        # =====================================================================
        # Step 1: Apply delays and prepend initial tokens (matching LMModel)
        # =====================================================================
        initial = self._get_initial_token().expand(B, -1, -1)  # [B, K, 1]
        delayed_codes = _delay_sequence(self._delays, codes, initial)  # [B, K, T]
        # Prepend initial tokens: delayed_codes becomes [B, K, T+1]
        delayed_codes = torch.cat([initial, delayed_codes], dim=2)

        # =====================================================================
        # Step 2: Forward through text embeddings and backbone (forward_text)
        # Input: delayed_codes[:, :, :-1] = positions 0 to T-1
        # =====================================================================
        input_sequence = delayed_codes[:, :, :-1]  # [B, K, T]
        B_seq, K_seq, S = input_sequence.shape

        # Compute audio embeddings (sum embeddings for ALL audio codebooks)
        # NOTE: Use num_audio_embs which equals n_q (16 for full-duplex).
        # This matches original Moshi LMModel.forward_text() which loops over
        # range(self.num_audio_codebooks) where num_audio_codebooks = n_q.
        # Both Moshi and User audio are embedded into the Temporal Transformer.
        audio_input = None
        n_audio_embs = self.num_audio_embs  # = n_q = 16 (typically)
        for cb_index in range(n_audio_embs):
            audio_codes = input_sequence[:, cb_index + self._audio_offset]  # [B, S]
            audio_emb = self.audio_embs[cb_index](audio_codes)  # [B, S, D]
            audio_input = audio_emb if audio_input is None else audio_input + audio_emb

        # Compute text embedding
        text_codes = input_sequence[:, 0]  # [B, S]
        text_emb = self.text_emb(text_codes)  # [B, S, D]

        # Combine embeddings
        combined_input = text_emb if audio_input is None else text_emb + audio_input  # [B, S, D=4096]

        # =====================================================================
        # Step 2.5: Apply speaker conditioning via sum_condition
        # =====================================================================
        # Speaker conditioning is added to combined_input before backbone:
        #   combined_input = text_emb[MOSHI] + Σaudio_emb[MOSHI] + speaker_condition
        #
        # Priority: sum_condition > speaker_embedding > speaker_conditioner(disabled)
        effective_sum_condition = None

        if sum_condition is not None:
            # Direct sum_condition provided
            effective_sum_condition = sum_condition
        elif speaker_embedding is not None and self.speaker_conditioning_enabled:
            # Transform speaker embedding via conditioner
            effective_sum_condition = self.speaker_conditioner(speaker_embedding)

        if effective_sum_condition is not None:
            # Broadcast and add: [B, 1, D] + [B, S, D] → [B, S, D]
            combined_input = combined_input + effective_sum_condition.to(combined_input)

        # Validate shape before backbone
        if combined_input.size(-1) != self._dim:
            raise ValueError(
                f"[LMModelWrapper] Combined input dimension {combined_input.size(-1)} != expected {self._dim}. "
                f"text_emb shape: {text_emb.shape}, audio_input shape: {audio_input.shape if audio_input is not None else None}"
            )

        # Apply input projection for backbone dimension adaptation
        backbone_input = self.adapter.input_proj(combined_input)  # [B, S, backbone_dim]

        # Process through backbone
        backbone_output = self.backbone(backbone_input)
        transformer_out = backbone_output.hidden_states  # [B, S, backbone_dim]

        # Validate backbone output shape
        if transformer_out.dim() != 3 or transformer_out.size(-1) == 0:
            raise ValueError(
                f"[LMModelWrapper] Invalid backbone output shape: {transformer_out.shape}. "
                f"Expected [B, S, backbone_dim] with backbone_dim > 0"
            )

        # Apply output projection for Moshi dimension
        transformer_out = self.adapter.output_proj(transformer_out)  # [B, S, D=4096]

        # Apply output normalization (matching LMModel.forward_text exactly)
        if self.out_norm is not None:
            transformer_out = self.out_norm(transformer_out)

        # Generate text logits
        text_logits = self.text_linear(transformer_out)  # [B, S, text_card]
        text_logits = text_logits[:, None]  # [B, 1, S, text_card] to match LMModel output shape

        # =====================================================================
        # Step 3: Forward through depformer for audio logits
        # Teacher forcing input: delayed_codes[:, :, 1:] = positions 1 to T
        # =====================================================================
        depformer_sequence = delayed_codes[:, :, 1:]  # [B, K, S] for teacher forcing
        audio_logits = self._forward_depformer_training(depformer_sequence, transformer_out)
        # audio_logits shape: [B, dep_q, S, audio_card]

        # =====================================================================
        # Step 4: Undelay output logits to align with original input codes
        # This is CRITICAL for correct loss computation
        # =====================================================================
        # Undelay audio logits
        audio_delays = self._delays[self._audio_offset:self._audio_offset + self._dep_q]
        audio_logits, audio_mask = _undelay_sequence(audio_delays, audio_logits, fill_value=float('NaN'))
        # Mask out zero tokens
        audio_mask = audio_mask & (codes[:, self._audio_offset:self._audio_offset + self._dep_q] != self._zero_token_id)

        # Undelay text logits
        text_delays = self._delays[:1]
        text_logits, text_mask = _undelay_sequence(text_delays, text_logits, fill_value=float('NaN'))
        # Mask out zero tokens
        text_mask = text_mask & (codes[:, :1] != self._zero_token_id)

        # Squeeze text dimensions for compatibility
        text_logits = text_logits.squeeze(1)  # [B, S, text_card]
        text_mask = text_mask.squeeze(1)  # [B, S]

        return LMModelWrapperOutput(
            text_logits=text_logits,
            logits=audio_logits,
            text_mask=text_mask,
            mask=audio_mask,
        )

    def _forward_depformer_training(
        self,
        sequence: Tensor,
        transformer_out: Tensor,
    ) -> Tensor:
        """
        Forward through depformer for training (matching LMModel.forward_depformer_training exactly).

        This follows the original LMModel.forward_depformer_training() pattern:
        1. Build depformer inputs for all dep_q codebook steps
        2. Stack and reshape to [B*T, Ka, D] so depformer processes Ka dimension
        3. Apply depformer transformer
        4. Apply per-codebook norms and linears

        Args:
            sequence: Delayed codes for teacher forcing [B, K, T]
                      (delayed_codes[:, :, 1:] from forward())
                      - sequence[:, 0]: text tokens (delayed)
                      - sequence[:, 1:]: audio codebook tokens (delayed)
            transformer_out: Backbone output [B, T, D=4096]

        Returns:
            Audio logits [B, dep_q, T, audio_card]
        """
        B, K, T = sequence.shape
        Ka = self._dep_q  # Number of audio codebooks to predict

        assert K == self.num_codebooks, (
            f"Codebooks for Depformer training should be passed all at once, got {K}."
        )

        # Determine if we have multi-linear depformer_in (one per codebook)
        depformer_multi_linear = len(self.depformer_in) == Ka

        # Build depformer inputs for all codebook positions
        depformer_inputs = []
        for cb_index in range(Ka):
            # Project backbone hidden states through depformer_in
            # Matching original: handles both multi-linear and single-linear modes
            if depformer_multi_linear:
                transformer_in = self.depformer_in[cb_index](transformer_out)
            else:
                transformer_in = self.depformer_in[0](transformer_out)

            # Add teacher forcing embedding
            # Matching original exactly:
            #   cb_index=0: text token from sequence[:, 0]
            #   cb_index>0: audio token from sequence[:, cb_index + audio_offset - 1]
            if cb_index == 0:
                token_in = self.depformer_text_emb(sequence[:, 0])
            else:
                # Original: sequence[:, cb_index + self.audio_offset - 1]
                # With audio_offset=1: sequence[:, cb_index]
                token_in = self.depformer_emb[cb_index - 1](
                    sequence[:, cb_index + self._audio_offset - 1]
                )

            depformer_inputs.append(token_in + transformer_in)

        # Stack along time dimension: [B, T, Ka, depformer_dim]
        depformer_input = torch.stack(depformer_inputs, dim=2)

        # Reshape to [B * T, Ka, depformer_dim]
        # This makes the depformer iterate over Ka (codebook) dimension
        depformer_input = depformer_input.view(B * T, Ka, -1)

        # Process through depformer transformer
        depformer_output = self.depformer(depformer_input)  # [B*T, Ka, depformer_dim]

        # Generate logits for each codebook (matching original exactly)
        all_logits = []
        for cb_index in range(Ka):
            # Original: self.linears[cb_index](self.depformer_norms[cb_index](depformer_output[:, cb_index]))
            cb_output = depformer_output[:, cb_index]  # [B*T, depformer_dim]

            # Apply depformer norm (required, not optional)
            if self._depformer_norms is not None:
                cb_output = self._depformer_norms[cb_index](cb_output)

            # Project to audio vocabulary
            logits = self.audio_linears[cb_index](cb_output)  # [B*T, audio_card]

            # Reshape back to [B, T, audio_card]
            all_logits.append(logits.view(B, T, -1))

        # Stack to [B, Ka, T, audio_card]
        logits = torch.stack(all_logits, dim=1)

        assert logits.dim() == 4, logits.shape  # [B, Ka, T, card]
        return logits

    # =========================================================================
    # Streaming Interface
    # =========================================================================

    def streaming(self, batch_size: int) -> ExitStack:
        """
        Context manager for streaming inference.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            Context manager that handles streaming state
        """
        exit_stack = ExitStack()

        # Enter backbone streaming mode
        backbone_ctx = self.backbone.streaming(batch_size)
        exit_stack.enter_context(backbone_ctx)

        # Enter depformer streaming mode if applicable
        if hasattr(self.depformer, 'streaming'):
            depformer_ctx = self.depformer.streaming(batch_size)
            exit_stack.enter_context(depformer_ctx)

        return exit_stack

    def step(
        self,
        codes_step: Tensor,
        **kwargs,
    ) -> LMModelWrapperOutput:
        """
        Single step forward for streaming inference.

        Args:
            codes_step: Input codes for single step [B, num_codebooks, 1]

        Returns:
            LMModelWrapperOutput for single step
        """
        B, num_cb, _ = codes_step.shape

        # 1. Compute embeddings for this step
        text_codes = codes_step[:, 0]  # [B, 1]
        text_embedded = self.text_emb(text_codes)  # [B, 1, D]

        audio_embedded = torch.zeros_like(text_embedded)
        n_audio_embs = len(self.audio_embs)

        for k in range(min(num_cb - 1, n_audio_embs)):
            audio_codes = codes_step[:, k + 1]
            audio_embedded = audio_embedded + self.audio_embs[k](audio_codes)

        x = text_embedded + audio_embedded  # [B, 1, D]

        # 2. Apply input projection
        x = self.adapter.input_proj(x)

        # 3. Process through backbone (streaming mode)
        backbone_output = self.backbone.streaming_forward(x)
        hidden_states = backbone_output.hidden_states

        # 4. Apply output projection
        hidden_states = self.adapter.output_proj(hidden_states)

        # 4.5 Apply output normalization
        if self.out_norm is not None:
            hidden_states = self.out_norm(hidden_states)

        # 5. Generate text logits
        text_logits = self.text_linear(hidden_states)

        # 6. Process through depformer
        audio_logits = self._forward_depformer_training(codes_step, hidden_states)

        return LMModelWrapperOutput(
            text_logits=text_logits,
            logits=audio_logits,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_backbone(self) -> AbstractBackbone:
        """Return the underlying backbone."""
        return self.backbone

    def get_adapter(self) -> nn.Module:
        """Return the dimension adapter."""
        return self.adapter

    def get_parameter_count(self) -> dict:
        """
        Return parameter count breakdown.

        Returns:
            Dictionary with total, trainable, and per-component counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters()) if hasattr(self.adapter, 'parameters') else 0
        embedding_params = (
            sum(p.numel() for p in self.text_emb.parameters()) +
            sum(p.numel() for emb in self.audio_embs for p in emb.parameters())
        )
        depformer_params = sum(p.numel() for p in self.depformer.parameters())
        linear_params = (
            sum(p.numel() for p in self.text_linear.parameters()) +
            sum(p.numel() for lin in self.audio_linears for p in lin.parameters())
        )

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "backbone": backbone_params,
            "adapter": adapter_params,
            "embeddings": embedding_params,
            "depformer": depformer_params,
            "linears": linear_params,
        }

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing in backbone if supported."""
        if self.backbone.supports_gradient_checkpointing():
            self.backbone.enable_gradient_checkpointing()
            logger.info("[LMModelWrapper] Gradient checkpointing enabled in backbone")
        else:
            logger.warning(
                f"[LMModelWrapper] Backbone {self.backbone.__class__.__name__} "
                "does not support gradient checkpointing"
            )

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        logger.info(f"[LMModelWrapper] Backbone {'frozen' if freeze else 'unfrozen'}")

    def freeze_embeddings(self, freeze: bool = True) -> None:
        """Freeze or unfreeze embedding parameters."""
        for param in self.text_emb.parameters():
            param.requires_grad = not freeze
        for emb in self.audio_embs:
            for param in emb.parameters():
                param.requires_grad = not freeze
        logger.info(f"[LMModelWrapper] Embeddings {'frozen' if freeze else 'unfrozen'}")

    def __repr__(self) -> str:
        adapter_status = "enabled" if getattr(self.adapter, 'is_enabled', False) else "disabled"
        return (
            f"LMModelWrapper(\n"
            f"  backbone_type={self.config.type},\n"
            f"  backbone={self.backbone.__class__.__name__},\n"
            f"  adapter={adapter_status},\n"
            f"  n_q={self._n_q},\n"
            f"  dep_q={self._dep_q},\n"
            f"  dim={self._dim}\n"
            f")"
        )


# =============================================================================
# Helper Functions
# =============================================================================

def create_lm_model_wrapper(
    config: UnifiedBackboneConfig,
    base_lm_model: Any,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> LMModelWrapper:
    """
    Convenience function to create LMModelWrapper.

    Args:
        config: Unified backbone configuration
        base_lm_model: Existing LMModel to extract components from
        device: Target device
        dtype: Target dtype

    Returns:
        Initialized LMModelWrapper
    """
    return LMModelWrapper.from_config(config, base_lm_model, device, dtype)


def wrap_existing_lm_model(
    lm_model: Any,
    config: Optional[UnifiedBackboneConfig] = None,
) -> LMModelWrapper:
    """
    Wrap an existing LMModel for modular backbone compatibility.

    This is useful for:
    1. Transitioning existing code to use the wrapper interface
    2. Enabling future backbone swapping without code changes
    3. Testing the wrapper with existing Moshi models

    Args:
        lm_model: Existing LMModel instance
        config: Optional config (defaults to Moshi config)

    Returns:
        LMModelWrapper wrapping the existing model
    """
    return LMModelWrapper.from_lm_model(lm_model, config)
