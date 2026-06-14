// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! HF backbone Language Model for K-Moshi
//!
//! This module provides a variant of LmModel that uses HF backbone (Mistral-based)
//! as the main transformer backbone instead of the original Moshi transformer.
//!
//! Architecture:
//! ```text
//! Input Tokens → Embeddings (4096) → DimensionAdapter (4096→3072)
//!     → HfBackbone/Mistral Backbone (3072) → DimensionAdapter (3072→4096)
//!     → Output Norm → Text Logits
//!     → DepFormer → Audio Logits
//! ```
//!
//! Key features:
//! - Reuses Moshi's embedding and depformer components
//! - Dimension adapter bridges between Moshi (4096) and HfBackbone (3072)
//! - Compatible with existing K-Moshi finetuned checkpoints

use candle::{DType, Device, IndexOp, Module, Result, Tensor};

use crate::dimension_adapter::{DimensionAdapter, DimensionAdapterConfig};
use crate::lm::{BackendType, Config, DepFormer, DepFormerConfig};
use crate::mistral_backend::{MistralBackend, MistralBackendConfig};
use crate::nn::{linear, MaybeQuantizedEmbedding, MaybeQuantizedLinear, MaybeQuantizedVarBuilder};
use crate::transformer::{self, Norm};
use crate::StreamMask;

/// Configuration for HfBackbone-based LM model
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct HfBackboneLmConfig {
    /// Base LM config (embeddings, depformer, vocab sizes)
    pub base: Config,
    /// Mistral/HfBackbone backbone configuration
    pub mistral: MistralBackendConfig,
    /// Dimension adapter configuration
    pub adapter: DimensionAdapterConfig,
}

impl Default for HfBackboneLmConfig {
    fn default() -> Self {
        Self::v0_1()
    }
}

impl HfBackboneLmConfig {
    /// Default configuration for K-Moshi with HF backbone backbone
    pub fn v0_1() -> Self {
        let mut base = Config::v0_1_hf_backbone();
        base.backend_type = BackendType::HfBackbone;

        Self {
            base,
            mistral: MistralBackendConfig::hf_backbone_3b(),
            adapter: DimensionAdapterConfig::moshi_to_hf_backbone(),
        }
    }

    /// Streaming configuration with 8 depformer slices
    pub fn v0_1_streaming() -> Self {
        let mut cfg = Self::v0_1();
        cfg.base = Config::v0_1_hf_backbone_streaming(8);
        cfg
    }
}

/// HfBackbone-based Language Model
///
/// This model uses HF backbone (Mistral architecture) as the main backbone,
/// with dimension adapters to bridge between Moshi's 4096 dimension and
/// HfBackbone's 3072 dimension.
#[derive(Debug)]
pub struct HfBackboneLmModel {
    /// HfBackbone/Mistral backbone
    backbone: MistralBackend,
    /// Dimension adapter (4096 ↔ 3072)
    adapter: DimensionAdapter,
    /// Text embedding
    text_emb: MaybeQuantizedEmbedding,
    /// Audio embeddings (one per codebook)
    audio_embs: Vec<MaybeQuantizedEmbedding>,
    /// Text output linear
    text_linear: MaybeQuantizedLinear,
    /// Output normalization
    out_norm: Norm,
    /// Depth transformer for audio generation
    depformer: Option<DepFormer>,
    /// Audio vocabulary size
    audio_vocab_size: usize,
    /// Text input vocabulary size
    text_in_vocab_size: usize,
    /// Data type
    dtype: DType,
    /// Device
    device: Device,
}

impl HfBackboneLmModel {
    /// Create a new HfBackbone LM model
    pub fn new(cfg: &HfBackboneLmConfig, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let d_model = cfg.base.transformer.d_model; // 4096 for embeddings
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Load embeddings (same as Moshi)
        let text_emb =
            MaybeQuantizedEmbedding::new(cfg.base.text_in_vocab_size, d_model, vb.pp("text_emb"))?;

        let vb_e = vb.pp("emb");
        let mut audio_embs = Vec::with_capacity(cfg.base.audio_codebooks);
        for i in 0..cfg.base.audio_codebooks {
            let emb =
                MaybeQuantizedEmbedding::new(cfg.base.audio_vocab_size, d_model, vb_e.pp(i))?;
            audio_embs.push(emb);
        }

        // Load output layers
        let out_norm = Norm::new(d_model, &cfg.base.transformer, vb.pp("out_norm"))?;
        let text_linear =
            linear(d_model, cfg.base.text_out_vocab_size, false, vb.pp("text_linear"))?;

        // Load depformer (same as Moshi)
        let depformer = match &cfg.base.depformer {
            None => None,
            Some(depformer_cfg) => {
                let depformer = DepFormer::new(
                    cfg.base.text_in_vocab_size,
                    cfg.base.audio_vocab_size,
                    d_model,
                    depformer_cfg,
                    vb.pp("depformer"),
                )?;
                Some(depformer)
            }
        };

        // Initialize dimension adapter
        let adapter = DimensionAdapter::new(&cfg.adapter, vb.pp("adapter"))?;

        // Load HfBackbone/Mistral backbone
        let backbone = MistralBackend::new(&cfg.mistral, vb.pp("backbone").into())?;

        Ok(Self {
            backbone,
            adapter,
            text_emb,
            audio_embs,
            text_linear,
            out_norm,
            depformer,
            audio_vocab_size: cfg.base.audio_vocab_size,
            text_in_vocab_size: cfg.base.text_in_vocab_size,
            dtype,
            device,
        })
    }

    /// Load from a safetensors file with separate backbone
    ///
    /// This loads:
    /// - Moshi-compatible weights for embeddings, text_linear, out_norm, depformer
    /// - HfBackbone weights for the backbone
    /// - Optionally pre-trained dimension adapter weights
    pub fn load<P: AsRef<std::path::Path>>(
        cfg: &HfBackboneLmConfig,
        moshi_weights: P,
        backbone_weights: P,
        adapter_weights: Option<P>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // Load Moshi-compatible components
        let vb_moshi = unsafe {
            MaybeQuantizedVarBuilder::Real(candle_nn::VarBuilder::from_mmaped_safetensors(
                &[moshi_weights],
                dtype,
                device,
            )?)
        };

        let d_model = cfg.base.transformer.d_model;

        // Load embeddings
        let text_emb = MaybeQuantizedEmbedding::new(
            cfg.base.text_in_vocab_size,
            d_model,
            vb_moshi.pp("text_emb"),
        )?;

        let vb_e = vb_moshi.pp("emb");
        let mut audio_embs = Vec::with_capacity(cfg.base.audio_codebooks);
        for i in 0..cfg.base.audio_codebooks {
            let emb =
                MaybeQuantizedEmbedding::new(cfg.base.audio_vocab_size, d_model, vb_e.pp(i))?;
            audio_embs.push(emb);
        }

        // Load output layers
        let out_norm = Norm::new(d_model, &cfg.base.transformer, vb_moshi.pp("out_norm"))?;
        let text_linear =
            linear(d_model, cfg.base.text_out_vocab_size, false, vb_moshi.pp("text_linear"))?;

        // Load depformer
        let depformer = match &cfg.base.depformer {
            None => None,
            Some(depformer_cfg) => {
                let depformer = DepFormer::new(
                    cfg.base.text_in_vocab_size,
                    cfg.base.audio_vocab_size,
                    d_model,
                    depformer_cfg,
                    vb_moshi.pp("depformer"),
                )?;
                Some(depformer)
            }
        };

        // Load dimension adapter
        let adapter = match adapter_weights {
            Some(adapter_path) => {
                let vb_adapter = unsafe {
                    candle_nn::VarBuilder::from_mmaped_safetensors(&[adapter_path], dtype, device)?
                };
                DimensionAdapter::new(&cfg.adapter, vb_adapter)?
            }
            None => {
                // Initialize with random weights if no pre-trained adapter
                DimensionAdapter::init(&cfg.adapter, device, dtype)?
            }
        };

        // Load HfBackbone/Mistral backbone
        let backbone = MistralBackend::load(&cfg.mistral, backbone_weights, dtype, device)?;

        Ok(Self {
            backbone,
            adapter,
            text_emb,
            audio_embs,
            text_linear,
            out_norm,
            depformer,
            audio_vocab_size: cfg.base.audio_vocab_size,
            text_in_vocab_size: cfg.base.text_in_vocab_size,
            dtype,
            device: device.clone(),
        })
    }

    /// Reset model state (clears KV cache)
    pub fn reset_state(&mut self) {
        self.backbone.clear_kv_cache();
    }

    /// Get number of input audio codebooks
    pub fn in_audio_codebooks(&self) -> usize {
        self.audio_embs.len()
    }

    /// Get audio padding token ID
    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    /// Get text start token ID
    pub fn text_start_token(&self) -> u32 {
        self.text_in_vocab_size as u32 - 1
    }

    /// Get number of generated audio codebooks
    pub fn generated_audio_codebooks(&self) -> usize {
        self.depformer.as_ref().map_or(0, |v| v.num_slices())
    }

    /// Check if model is quantized
    pub fn is_quantized(&self) -> bool {
        match self.text_linear {
            MaybeQuantizedLinear::Quantized(_) => true,
            MaybeQuantizedLinear::Real(_) => false,
        }
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Forward pass
    ///
    /// Returns (text_logits, hidden_states)
    pub fn forward(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        _mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        // Compute embeddings (dimension: 4096)
        let mut emb = match text_ids.as_ref() {
            Some(text_ids) => text_ids.apply(&self.text_emb)?,
            None => {
                let hidden_size = self.text_emb.hidden_size()?;
                Tensor::zeros((1, 1, hidden_size), self.dtype, &self.device)?
            }
        };

        for (audio_emb, audio_ids) in self.audio_embs.iter().zip(audio_ids.iter()) {
            if let Some(audio_ids) = audio_ids {
                let e = audio_ids.apply(audio_emb)?;
                emb = (emb + e)?;
            }
        }

        // Project down: 4096 → 3072
        let emb_down = self.adapter.project_down(&emb)?;

        // Run through HfBackbone backbone
        let ys_down = self.backbone.forward_streaming(&emb_down)?;

        // Project up: 3072 → 4096
        let ys = self.adapter.project_up(&ys_down)?;

        // Apply output norm and text linear
        let ys_normed = ys.apply(&self.out_norm)?;
        let logits = ys_normed.apply(&self.text_linear)?;

        Ok((logits, ys_normed))
    }

    /// Sample from depformer
    pub fn depformer_sample(
        &mut self,
        xs: &Tensor,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        let sample = match self.depformer.as_mut() {
            None => None,
            Some(m) => {
                let sample = m.sample(xs, text_token, forced_audio_tokens, lp)?;
                Some(sample)
            }
        };
        Ok(sample)
    }

    /// Sample from depformer with classifier-free guidance
    ///
    /// Note: CFG is not fully optimized for HfBackbone, uses basic sampling
    pub fn depformer_sample_cfg(
        &mut self,
        xs: &Tensor,
        _cfg_alpha: f64,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        // HfBackbone doesn't have native CFG support, fall back to regular sampling
        // TODO: Implement proper CFG for HfBackbone if needed
        let sample = match self.depformer.as_mut() {
            None => None,
            Some(m) => {
                let sample = m.sample(xs, text_token, forced_audio_tokens, lp)?;
                Some(sample)
            }
        };
        Ok(sample)
    }

    /// Forward pass with conditioning
    ///
    /// Note: HfBackbone doesn't support conditioning, conditions are ignored
    pub fn forward_cond(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &crate::StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        if conditions.is_some() {
            // Log warning but continue - HfBackbone doesn't support conditioning
            tracing::warn!("HfBackbone backend does not support conditioning, ignoring conditions");
        }
        self.forward(text_ids, audio_ids, mask)
    }

    /// Forward pass with cross-attention
    ///
    /// Note: HfBackbone doesn't support cross-attention, this returns an error
    pub fn forward_ca(
        &mut self,
        _text_ids: Option<Tensor>,
        _audio_ids: Vec<Option<Tensor>>,
        _ca_src: &crate::transformer::CaSrc,
        _conditions: Option<&crate::conditioner::Condition>,
        _mask: &crate::StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        candle::bail!("HfBackbone backend does not support cross-attention (forward_ca)")
    }
}

/// Load a HfBackbone-based LM model from files
pub fn load_hf_backbone_lm<P: AsRef<std::path::Path>>(
    moshi_weights: P,
    backbone_weights: P,
    adapter_weights: Option<P>,
    dtype: DType,
    device: &Device,
) -> Result<HfBackboneLmModel> {
    let cfg = HfBackboneLmConfig::v0_1_streaming();
    HfBackboneLmModel::load(&cfg, moshi_weights, backbone_weights, adapter_weights, dtype, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_backbone_config() {
        let cfg = HfBackboneLmConfig::v0_1();
        assert_eq!(cfg.base.backend_type, BackendType::HfBackbone);
        assert_eq!(cfg.mistral.hidden_size, 3072);
        assert_eq!(cfg.adapter.source_dim, 4096);
        assert_eq!(cfg.adapter.target_dim, 3072);
    }

    #[test]
    fn test_streaming_config() {
        let cfg = HfBackboneLmConfig::v0_1_streaming();
        assert_eq!(cfg.base.audio_codebooks, 16);
    }
}
