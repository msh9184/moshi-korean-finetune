# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Speaker Encoder Module for K-Moshi Zero-Shot Speaker Conditioning

This module provides speaker embedding extraction from reference audio.
The extracted embeddings are used to condition the Temporal Transformer
via sum_condition mechanism.

Architecture Overview:
    Reference Audio (16kHz) → Speaker Encoder → Speaker Embedding (D_spk)

Supported Encoders:
    - ECAPA-TDNN: Pre-trained on VoxCeleb, 192-dim output
    - (Future) Custom Korean Speaker Encoder from team

Usage:
    config = SpeakerEncoderConfig(
        encoder_type="ecapa_tdnn",
        pretrained_path="speechbrain/spkrec-ecapa-voxceleb",
        freeze=True,
        output_dim=192,
    )
    encoder = ECAPATDNNSpeakerEncoder(config)
    embedding = encoder(reference_audio)  # [B, 192]

References:
    - ECAPA-TDNN: https://arxiv.org/abs/2005.07143
    - SpeechBrain: https://github.com/speechbrain/speechbrain
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Literal
import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SpeakerEncoderConfig:
    """Configuration for Speaker Encoder.

    Attributes:
        encoder_type: Type of speaker encoder
            - "ecapa_tdnn": ECAPA-TDNN from SpeechBrain (192-dim, default)
            - "w2v_bert2": W2v-BERT 2.0 Speaker Verification (256-dim, SOTA)
            - "dummy": Dummy encoder for testing without dependencies
            - "custom": Placeholder for team's custom model
        pretrained_path: Path or HuggingFace ID for pretrained model
        freeze: Whether to freeze encoder weights during training
        output_dim: Output embedding dimension (192 for ECAPA-TDNN, 256 for w2v_bert2)
        sample_rate: Expected input sample rate (16000 for most encoders)
        normalize_embedding: Whether to L2-normalize output embeddings
    """
    encoder_type: Literal["ecapa_tdnn", "w2v_bert2", "dummy", "custom"] = "ecapa_tdnn"
    pretrained_path: str = "speechbrain/spkrec-ecapa-voxceleb"
    freeze: bool = True
    output_dim: int = 192
    sample_rate: int = 16000
    normalize_embedding: bool = True

    # For custom encoder (future use with team's model)
    custom_encoder_path: Optional[str] = None
    custom_encoder_config: dict = field(default_factory=dict)


class BaseSpeakerEncoder(ABC, nn.Module):
    """Abstract base class for speaker encoders.

    All speaker encoders must implement:
        - forward(): Extract speaker embedding from audio
        - output_dim: Return the embedding dimension

    The encoder should handle:
        - Variable length audio input
        - Batch processing
        - Optional L2 normalization
    """

    def __init__(self, config: SpeakerEncoderConfig):
        super().__init__()
        self.config = config
        self._output_dim = config.output_dim
        self._sample_rate = config.sample_rate
        self._normalize = config.normalize_embedding

    @property
    def output_dim(self) -> int:
        """Return the speaker embedding dimension."""
        return self._output_dim

    @property
    def sample_rate(self) -> int:
        """Return the expected input sample rate."""
        return self._sample_rate

    @abstractmethod
    def forward(self, audio: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract speaker embedding from audio.

        Args:
            audio: Input audio tensor [B, T] at self.sample_rate
            lengths: Optional tensor of actual lengths [B] for padded batches

        Returns:
            Speaker embedding tensor [B, output_dim]
        """
        pass

    def normalize(self, embedding: torch.Tensor) -> torch.Tensor:
        """L2-normalize embedding if configured."""
        if self._normalize:
            return nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info(f"Froze {self.__class__.__name__} parameters")

    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info(f"Unfroze {self.__class__.__name__} parameters")


class ECAPATDNNSpeakerEncoder(BaseSpeakerEncoder):
    """ECAPA-TDNN Speaker Encoder using SpeechBrain.

    ECAPA-TDNN is a state-of-the-art speaker verification model
    that extracts robust speaker embeddings. We use the pre-trained
    model from SpeechBrain trained on VoxCeleb.

    Key Features:
        - 192-dimensional speaker embedding
        - Robust to noise and channel variations
        - Pre-trained on large-scale speaker data

    Architecture:
        Audio → TDNN Blocks → Attentive Statistics Pooling → FC → Embedding

    Reference:
        Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention,
        Propagation and Aggregation in TDNN Based Speaker Verification"
        https://arxiv.org/abs/2005.07143
    """

    def __init__(self, config: SpeakerEncoderConfig):
        super().__init__(config)

        self.encoder = None
        self._load_pretrained()

        if config.freeze:
            self.freeze()

    def _load_pretrained(self) -> None:
        """Load pre-trained ECAPA-TDNN from SpeechBrain."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            logger.info(f"Loading ECAPA-TDNN from {self.config.pretrained_path}")

            # SpeechBrain's EncoderClassifier handles model loading
            self.encoder = EncoderClassifier.from_hparams(
                source=self.config.pretrained_path,
                savedir=f"pretrained_models/{self.config.pretrained_path.replace('/', '_')}",
                run_opts={"device": "cpu"},  # Will move to correct device later
            )

            logger.info("ECAPA-TDNN loaded successfully")

        except ImportError:
            logger.error(
                "SpeechBrain not installed. Install with: pip install speechbrain"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load ECAPA-TDNN: {e}")
            raise

    def forward(self, audio: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract speaker embedding using ECAPA-TDNN.

        Args:
            audio: Input audio tensor [B, T] at 16kHz
            lengths: Optional tensor of actual lengths [B]

        Returns:
            Speaker embedding tensor [B, 192]
        """
        if self.encoder is None:
            raise RuntimeError("ECAPA-TDNN encoder not loaded")

        # SpeechBrain expects [B, T] at 16kHz
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Get device from audio
        device = audio.device

        # SpeechBrain's encode_batch handles the forward pass
        # It returns embeddings of shape [B, 1, 192]
        with torch.set_grad_enabled(not self.config.freeze):
            embeddings = self.encoder.encode_batch(audio.to(device), lengths)

        # Remove the middle dimension: [B, 1, 192] → [B, 192]
        embeddings = embeddings.squeeze(1)

        # Normalize if configured
        embeddings = self.normalize(embeddings)

        return embeddings

    def to(self, device: torch.device) -> "ECAPATDNNSpeakerEncoder":
        """Move encoder to device."""
        super().to(device)
        if self.encoder is not None:
            # SpeechBrain models need special handling
            self.encoder.device = device
            self.encoder.mods.to(device)
        return self


class DummySpeakerEncoder(BaseSpeakerEncoder):
    """Dummy speaker encoder for testing without SpeechBrain dependency.

    This encoder returns random embeddings and is useful for:
        - Testing the integration pipeline
        - Running without GPU/SpeechBrain
        - Debugging speaker conditioning flow
    """

    def __init__(self, config: SpeakerEncoderConfig):
        super().__init__(config)

        # Simple linear projection to simulate encoding
        self.projection = nn.Linear(1, config.output_dim)

        logger.warning(
            "Using DummySpeakerEncoder - embeddings are random! "
            "Install SpeechBrain for real speaker embeddings."
        )

    def forward(self, audio: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return dummy speaker embeddings.

        Args:
            audio: Input audio tensor [B, T]
            lengths: Ignored

        Returns:
            Random embedding tensor [B, output_dim]
        """
        batch_size = audio.shape[0] if audio.dim() > 1 else 1
        device = audio.device
        dtype = audio.dtype

        # Generate deterministic-ish embedding based on audio statistics
        # This helps with reproducibility in testing
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Use mean amplitude as seed for embedding
        mean_amp = audio.abs().mean(dim=-1, keepdim=True)  # [B, 1]

        # Project to embedding dimension
        embedding = self.projection(mean_amp)  # [B, output_dim]

        # Normalize
        embedding = self.normalize(embedding)

        return embedding


class W2vBERT2SpeakerEncoder(BaseSpeakerEncoder):
    """W2v-BERT 2.0 Speaker Verification Encoder.

    State-of-the-art speaker encoder based on w2v-BERT 2.0 with:
    - Multi-layer Feature Aggregation (MFA)
    - Attentive Statistics Pooling (ASP)
    - Knowledge distillation guided structured pruning

    This encoder achieves 0.14% EER on VoxCeleb1-O test set.

    Key Features:
        - 256-dimensional speaker embedding (configurable)
        - 16kHz input sample rate
        - Support for frozen/unfrozen training
        - Compatible with pruned/distilled models

    Architecture:
        Audio → w2v-BERT 2.0 → MFA (multi-layer) → ASP Pooling → Bottleneck → Embedding

    Reference:
        Li et al., "Enhancing Speaker Verification with w2v-BERT 2.0 and
        Knowledge Distillation guided Structured Pruning"
        https://arxiv.org/abs/2510.04213

    Model Download:
        https://huggingface.co/zl389/w2v-bert-2.0_SV/blob/main/model_lmft_0.14.pth

    Usage:
        config = SpeakerEncoderConfig(
            encoder_type="w2v_bert2",
            pretrained_path="path/to/model_lmft_0.14.pth",
            output_dim=256,
        )
        encoder = W2vBERT2SpeakerEncoder(config)
        embedding = encoder(audio)  # [B, 256]
    """

    def __init__(self, config: SpeakerEncoderConfig):
        # Override output_dim if not explicitly set for w2v-bert2
        if config.output_dim == 192:  # default ECAPA-TDNN dim
            config.output_dim = 256  # w2v-bert2 default
        super().__init__(config)

        self.encoder = None
        self.pooling = None
        self.bottleneck = None
        self._d_model = None
        self._n_mfa_layers = None

        self._load_pretrained()

        if config.freeze:
            self.freeze()

    def _load_pretrained(self) -> None:
        """Load pre-trained W2v-BERT 2.0 Speaker Verification model."""
        import os

        pretrained_path = self.config.pretrained_path

        # Check if path exists
        if not os.path.exists(pretrained_path):
            logger.error(
                f"W2v-BERT 2.0 SV model not found at {pretrained_path}. "
                "Please download from: https://huggingface.co/zl389/w2v-bert-2.0_SV/blob/main/model_lmft_0.14.pth"
            )
            raise FileNotFoundError(f"Model not found: {pretrained_path}")

        logger.info(f"Loading W2v-BERT 2.0 SV from {pretrained_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)

            # Extract model configuration from checkpoint
            # The checkpoint contains 'modules' dict with 'spk_model' state dict
            if "modules" in checkpoint and "spk_model" in checkpoint["modules"]:
                state_dict = checkpoint["modules"]["spk_model"]
            else:
                state_dict = checkpoint

            # Determine model dimensions from state_dict
            self._infer_model_config(state_dict)

            # Build model architecture
            self._build_model_architecture()

            # Load weights
            self._load_weights(state_dict)

            logger.info(
                f"W2v-BERT 2.0 SV loaded successfully: "
                f"d_model={self._d_model}, n_mfa_layers={self._n_mfa_layers}, "
                f"output_dim={self.config.output_dim}"
            )

        except Exception as e:
            logger.error(f"Failed to load W2v-BERT 2.0 SV: {e}")
            raise

    def _infer_model_config(self, state_dict: dict) -> None:
        """Infer model configuration from state dict keys and shapes.

        This method analyzes the checkpoint weights to determine the exact
        architecture configuration, ensuring compatibility with pretrained models.
        """
        # Initialize with None to detect if values were inferred
        self._feat_dim = None
        self._pooling_hidden_dim = None

        # Try to find bottleneck layer to determine output dim and feat_dim
        for key, value in state_dict.items():
            if "bottleneck" in key and "weight" in key:
                # bottleneck.weight shape: [output_dim, pooling_output_dim]
                inferred_output_dim = value.shape[0]
                pooling_output_dim = value.shape[1]

                # pooling output = feat_dim * expansion (expansion=2 for ASP)
                self._feat_dim = pooling_output_dim // 2
                self._output_dim = inferred_output_dim

                logger.info(
                    f"[W2v-BERT 2.0 SV] Inferred from bottleneck weights: "
                    f"output_dim={inferred_output_dim}, pooling_output_dim={pooling_output_dim}, "
                    f"feat_dim={self._feat_dim}"
                )
                break

        # Try to find pooling attention layer to determine hidden_dim
        for key, value in state_dict.items():
            if "pooling" in key and "attention" in key and "0.weight" in key:
                # attention.0.weight shape: [hidden_dim, feat_dim, 1] (Conv1d)
                self._pooling_hidden_dim = value.shape[0]
                inferred_feat_dim = value.shape[1]

                # Verify feat_dim consistency
                if self._feat_dim is not None and self._feat_dim != inferred_feat_dim:
                    logger.warning(
                        f"[W2v-BERT 2.0 SV] feat_dim mismatch: bottleneck says {self._feat_dim}, "
                        f"attention says {inferred_feat_dim}. Using attention value."
                    )
                    self._feat_dim = inferred_feat_dim
                elif self._feat_dim is None:
                    self._feat_dim = inferred_feat_dim

                logger.info(
                    f"[W2v-BERT 2.0 SV] Inferred from attention weights: "
                    f"hidden_dim={self._pooling_hidden_dim}, feat_dim={self._feat_dim}"
                )
                break

        # Set default d_model based on w2v-bert-2.0 architecture
        self._d_model = 1024  # w2v-bert-2.0 hidden size

        # Calculate n_mfa_layers from feat_dim if possible
        if self._feat_dim is not None:
            if self._feat_dim % self._d_model == 0:
                self._n_mfa_layers = self._feat_dim // self._d_model
                logger.info(
                    f"[W2v-BERT 2.0 SV] Calculated n_mfa_layers={self._n_mfa_layers} "
                    f"from feat_dim={self._feat_dim} / d_model={self._d_model}"
                )
            else:
                # feat_dim is not a multiple of d_model
                # This might indicate a different architecture or projection
                logger.warning(
                    f"[W2v-BERT 2.0 SV] feat_dim={self._feat_dim} is not divisible by "
                    f"d_model={self._d_model}. Using feat_dim directly without MFA layer calculation."
                )
                self._n_mfa_layers = 1  # Will use feat_dim directly
        else:
            # Fallback: check for adapter layers
            self._n_mfa_layers = 25  # default for w2v-bert-2.0
            adapter_count = 0
            for key in state_dict.keys():
                if "adapter_layers" in key and ".0.weight" in key:
                    adapter_count += 1

            if adapter_count > 0:
                self._n_mfa_layers = adapter_count
                logger.info(f"[W2v-BERT 2.0 SV] Detected {adapter_count} adapter layers")

        # Set default pooling hidden dim if not inferred
        if self._pooling_hidden_dim is None:
            self._pooling_hidden_dim = self._d_model
            logger.info(
                f"[W2v-BERT 2.0 SV] Using default pooling_hidden_dim={self._pooling_hidden_dim}"
            )

    def _build_model_architecture(self) -> None:
        """Build the speaker model architecture.

        Uses the configuration inferred from checkpoint weights to ensure
        architecture compatibility with the pretrained model.
        """
        # Use feat_dim inferred from checkpoint, or calculate from d_model * n_mfa_layers
        if self._feat_dim is not None:
            feat_dim = self._feat_dim
            logger.info(f"[W2v-BERT 2.0 SV] Using inferred feat_dim={feat_dim}")
        else:
            feat_dim = self._d_model * self._n_mfa_layers if self._n_mfa_layers > 1 else self._d_model
            logger.info(
                f"[W2v-BERT 2.0 SV] Calculated feat_dim={feat_dim} "
                f"(d_model={self._d_model} * n_mfa_layers={self._n_mfa_layers})"
            )

        # Use pooling hidden dim inferred from checkpoint, or default to d_model
        pooling_hidden_dim = self._pooling_hidden_dim if self._pooling_hidden_dim else self._d_model

        # Attentive Statistics Pooling
        self.pooling = _ASP(feat_dim, pooling_hidden_dim)

        # Bottleneck layer - output dim from config, input dim is feat_dim * 2 (ASP expansion)
        self.bottleneck = nn.Linear(feat_dim * 2, self._output_dim)

        logger.info(
            f"[W2v-BERT 2.0 SV] Built architecture: "
            f"ASP(input={feat_dim}, hidden={pooling_hidden_dim}) -> "
            f"Bottleneck(input={feat_dim * 2}, output={self._output_dim})"
        )

        # Try to load the full w2v-bert-2.0 encoder
        self._load_w2v_bert_encoder()

    def _load_w2v_bert_encoder(self) -> None:
        """Try to load the w2v-bert-2.0 encoder for full forward pass."""
        try:
            from transformers import Wav2Vec2BertModel, AutoFeatureExtractor

            # Try to load from HuggingFace
            model_id = "facebook/w2v-bert-2.0"
            local_path = self.config.custom_encoder_config.get("w2v_bert_path", None)

            if local_path and os.path.exists(local_path):
                logger.info(f"Loading w2v-bert-2.0 from local path: {local_path}")
                self.w2v_bert = Wav2Vec2BertModel.from_pretrained(local_path)
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(local_path)
            else:
                logger.info(f"Loading w2v-bert-2.0 from HuggingFace: {model_id}")
                self.w2v_bert = Wav2Vec2BertModel.from_pretrained(model_id)
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

            # Remove masked_spec_embed for DDP compatibility
            if hasattr(self.w2v_bert, 'masked_spec_embed'):
                delattr(self.w2v_bert, 'masked_spec_embed')

            self._has_full_encoder = True
            logger.info("Full w2v-bert-2.0 encoder loaded successfully")

        except ImportError:
            logger.warning(
                "transformers not installed or w2v-bert-2.0 not available. "
                "Using pre-computed embeddings mode."
            )
            self.w2v_bert = None
            self.feature_extractor = None
            self._has_full_encoder = False

        except Exception as e:
            logger.warning(f"Could not load full w2v-bert-2.0 encoder: {e}")
            self.w2v_bert = None
            self.feature_extractor = None
            self._has_full_encoder = False

    def _load_weights(self, state_dict: dict) -> None:
        """Load weights into the model components.

        This method carefully loads weights from the checkpoint, reporting
        exactly which weights were loaded and any mismatches.
        """
        # Load pooling weights
        pooling_state = {}
        for key, value in state_dict.items():
            if key.startswith("pooling."):
                new_key = key.replace("pooling.", "")
                pooling_state[new_key] = value

        if pooling_state:
            try:
                # Use strict=True to ensure exact match
                load_result = self.pooling.load_state_dict(pooling_state, strict=True)
                logger.info(
                    f"[W2v-BERT 2.0 SV] Loaded pooling layer weights successfully. "
                    f"Keys: {list(pooling_state.keys())}"
                )
            except RuntimeError as e:
                # If strict loading fails, try with strict=False and log details
                logger.warning(f"[W2v-BERT 2.0 SV] Strict pooling weight loading failed: {e}")
                try:
                    load_result = self.pooling.load_state_dict(pooling_state, strict=False)
                    logger.warning(
                        f"[W2v-BERT 2.0 SV] Loaded pooling weights with strict=False. "
                        f"Some weights may not have been loaded correctly."
                    )
                except Exception as e2:
                    logger.error(f"[W2v-BERT 2.0 SV] Could not load pooling weights: {e2}")

        # Load bottleneck weights
        bottleneck_state = {}
        for key, value in state_dict.items():
            if key.startswith("bottleneck."):
                new_key = key.replace("bottleneck.", "")
                bottleneck_state[new_key] = value

        if bottleneck_state:
            try:
                # Use strict=True to ensure exact match
                load_result = self.bottleneck.load_state_dict(bottleneck_state, strict=True)
                logger.info(
                    f"[W2v-BERT 2.0 SV] Loaded bottleneck layer weights successfully. "
                    f"Keys: {list(bottleneck_state.keys())}"
                )
            except RuntimeError as e:
                logger.warning(f"[W2v-BERT 2.0 SV] Strict bottleneck weight loading failed: {e}")
                try:
                    load_result = self.bottleneck.load_state_dict(bottleneck_state, strict=False)
                    logger.warning(
                        f"[W2v-BERT 2.0 SV] Loaded bottleneck weights with strict=False. "
                        f"Some weights may not have been loaded correctly."
                    )
                except Exception as e2:
                    logger.error(f"[W2v-BERT 2.0 SV] Could not load bottleneck weights: {e2}")

    def forward(self, audio: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract speaker embedding using W2v-BERT 2.0.

        Args:
            audio: Input audio tensor [B, T] at 16kHz
            lengths: Optional tensor of actual lengths [B]

        Returns:
            Speaker embedding tensor [B, output_dim]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = audio.device

        if self._has_full_encoder and self.w2v_bert is not None:
            # Full forward pass through w2v-bert-2.0
            embeddings = self._forward_full(audio)
        else:
            # Fallback: use simple feature extraction
            embeddings = self._forward_simple(audio)

        # Normalize if configured
        embeddings = self.normalize(embeddings)

        return embeddings

    def _forward_full(self, audio: torch.Tensor) -> torch.Tensor:
        """Full forward pass through w2v-bert-2.0 encoder.

        Handles MFA (Multi-layer Feature Aggregation) to match the checkpoint's
        expected input dimensions for pooling and bottleneck layers.
        """
        device = audio.device
        dtype = next(self.parameters()).dtype if len(list(self.parameters())) > 0 else audio.dtype

        # Process with feature extractor
        features = self.feature_extractor(
            audio.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        input_features = features.input_features.to(device).to(dtype)

        # Forward through w2v-bert-2.0
        with torch.set_grad_enabled(not self.config.freeze):
            outputs = self.w2v_bert(
                input_features,
                output_hidden_states=True,
            )

        # Multi-layer feature aggregation
        # Use _feat_dim to determine how many layers to concatenate
        if self._n_mfa_layers == 1:
            x = outputs.last_hidden_state  # [B, T, d_model]
        else:
            # Concatenate the last n_mfa_layers hidden states
            x = torch.cat(
                outputs.hidden_states[-self._n_mfa_layers:],
                dim=-1,
            )  # [B, T, d_model * n_mfa_layers]

        # Check if dimension matches expected feat_dim
        actual_feat_dim = x.shape[-1]
        expected_feat_dim = self._feat_dim

        if actual_feat_dim != expected_feat_dim:
            # Dimension mismatch: need to project to expected dimension
            # This can happen when checkpoint was trained with different MFA configuration
            if not hasattr(self, 'mfa_projection'):
                logger.warning(
                    f"[W2v-BERT 2.0 SV] Feature dimension mismatch: "
                    f"encoder output={actual_feat_dim}, expected={expected_feat_dim}. "
                    f"Creating projection layer."
                )
                self.mfa_projection = nn.Linear(actual_feat_dim, expected_feat_dim).to(device).to(dtype)

            x = self.mfa_projection(x)

        # Pooling and bottleneck
        x = self.pooling(x)
        x = self.bottleneck(x)

        return x

    def _forward_simple(self, audio: torch.Tensor) -> torch.Tensor:
        """Simple forward pass without full encoder (for testing)."""
        batch_size = audio.shape[0]
        device = audio.device
        dtype = audio.dtype

        # Generate simple embedding based on audio statistics
        # This is a fallback when full encoder is not available
        mean_amp = audio.abs().mean(dim=-1, keepdim=True)
        std_amp = audio.std(dim=-1, keepdim=True)

        # Create a simple feature vector
        simple_features = torch.cat([mean_amp, std_amp], dim=-1)

        # Project to output dimension
        embedding = torch.zeros(batch_size, self.config.output_dim, device=device, dtype=dtype)
        embedding[:, :2] = simple_features

        logger.warning(
            "Using simple fallback embedding. Install transformers and download "
            "w2v-bert-2.0 for full functionality."
        )

        return embedding

    def to(self, device: torch.device) -> "W2vBERT2SpeakerEncoder":
        """Move encoder to device."""
        super().to(device)
        if hasattr(self, 'w2v_bert') and self.w2v_bert is not None:
            self.w2v_bert.to(device)
        return self


class _ASP(nn.Module):
    """Attentive Statistics Pooling for speaker embedding.

    Computes attention-weighted mean and standard deviation
    for robust speaker representation.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.expansion = 2
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, D]

        Returns:
            Pooled tensor [B, 2*D]
        """
        # x: [B, T, D] -> transpose for conv1d
        w = self.attention(x.transpose(1, 2)).transpose(1, 2)  # [B, T, D]
        mu = torch.sum(x * w, dim=1)  # [B, D]
        sg = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))  # [B, D]
        out = torch.cat([mu, sg], dim=1)  # [B, 2D]
        return out


def create_speaker_encoder(config: SpeakerEncoderConfig) -> BaseSpeakerEncoder:
    """Factory function to create speaker encoder based on config.

    Args:
        config: SpeakerEncoderConfig instance

    Returns:
        Appropriate speaker encoder instance
    """
    if config.encoder_type == "ecapa_tdnn":
        try:
            return ECAPATDNNSpeakerEncoder(config)
        except ImportError:
            logger.warning(
                "SpeechBrain not available, falling back to DummySpeakerEncoder"
            )
            return DummySpeakerEncoder(config)
    elif config.encoder_type == "w2v_bert2":
        return W2vBERT2SpeakerEncoder(config)
    elif config.encoder_type == "dummy":
        return DummySpeakerEncoder(config)
    elif config.encoder_type == "custom":
        # Placeholder for team's custom encoder
        raise NotImplementedError(
            "Custom speaker encoder not yet implemented. "
            "This will be added when the team's model is ready."
        )
    else:
        raise ValueError(f"Unknown encoder type: {config.encoder_type}")


# Convenience function for quick testing
def get_default_speaker_encoder(
    encoder_type: str = "ecapa_tdnn",
    freeze: bool = True,
    pretrained_path: Optional[str] = None,
) -> BaseSpeakerEncoder:
    """Get default speaker encoder with standard settings.

    Args:
        encoder_type: Type of encoder ("ecapa_tdnn", "w2v_bert2")
        freeze: Whether to freeze encoder weights
        pretrained_path: Optional path to pretrained model (required for w2v_bert2)

    Returns:
        Configured speaker encoder
    """
    if encoder_type == "ecapa_tdnn":
        config = SpeakerEncoderConfig(
            encoder_type="ecapa_tdnn",
            pretrained_path="speechbrain/spkrec-ecapa-voxceleb",
            freeze=freeze,
            output_dim=192,
            sample_rate=16000,
            normalize_embedding=True,
        )
    elif encoder_type == "w2v_bert2":
        if pretrained_path is None:
            raise ValueError(
                "pretrained_path is required for w2v_bert2 encoder. "
                "Download from: https://huggingface.co/zl389/w2v-bert-2.0_SV/blob/main/model_lmft_0.14.pth"
            )
        config = SpeakerEncoderConfig(
            encoder_type="w2v_bert2",
            pretrained_path=pretrained_path,
            freeze=freeze,
            output_dim=256,
            sample_rate=16000,
            normalize_embedding=True,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return create_speaker_encoder(config)
