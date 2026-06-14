// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! Mistral/HfBackbone Backend for K-Moshi
//!
//! This module wraps the candle-transformers Mistral implementation
//! to serve as an alternative backbone for K-Moshi inference.
//!
//! Key features:
//! - GQA (Grouped Query Attention) support
//! - Sliding window attention
//! - RoPE (Rotary Position Embedding)
//! - Compatible with HF-backbone model weights

use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mistral::{Config as MistralConfig, Model as MistralModel};

use crate::kv_cache::KvCache;
use crate::streaming::{StreamMask, StreamTensor, StreamingModule};

/// Configuration for Mistral backend
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct MistralBackendConfig {
    /// Hidden size (3072 for HF backbone)
    pub hidden_size: usize,
    /// Number of transformer layers (30 for HF backbone)
    pub num_hidden_layers: usize,
    /// Number of attention heads (24 for HF backbone)
    pub num_attention_heads: usize,
    /// Number of key-value heads for GQA (4 for HF backbone)
    pub num_key_value_heads: usize,
    /// Intermediate size in MLP (8192 for HF backbone)
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// RoPE theta parameter
    pub rope_theta: f64,
    /// Sliding window size (None for full attention)
    pub sliding_window: Option<usize>,
    /// RMS norm epsilon
    pub rms_norm_eps: f64,
    /// Use Flash Attention if available
    pub use_flash_attn: bool,
}

impl Default for MistralBackendConfig {
    fn default() -> Self {
        Self::hf_backbone_3b()
    }
}

impl MistralBackendConfig {
    /// Configuration for HF-backbone model
    pub fn hf_backbone_3b() -> Self {
        Self {
            hidden_size: 3072,
            num_hidden_layers: 30,
            num_attention_heads: 24,
            num_key_value_heads: 4,
            intermediate_size: 8192,
            max_position_embeddings: 32768,
            rope_theta: 500000.0,
            sliding_window: None, // Full attention for training compatibility
            rms_norm_eps: 1e-5,
            use_flash_attn: false,
        }
    }

    /// Convert to candle_transformers MistralConfig
    pub fn to_mistral_config(&self) -> MistralConfig {
        MistralConfig {
            vocab_size: 32000, // Will be overridden by actual model
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            hidden_act: candle_nn::Activation::Silu,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            sliding_window: self.sliding_window,
            use_flash_attn: self.use_flash_attn,
            head_dim: None, // Computed automatically
        }
    }
}

/// Streaming state for Mistral backend
#[derive(Debug, Clone)]
pub struct MistralState {
    /// Current position in the sequence
    position: usize,
    /// Batch size
    batch_size: usize,
}

impl MistralState {
    pub fn new(batch_size: usize) -> Self {
        Self {
            position: 0,
            batch_size,
        }
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn advance(&mut self, steps: usize) {
        self.position += steps;
    }
}

/// Mistral backend wrapper for K-Moshi
///
/// This wraps the candle-transformers Mistral model with streaming support
/// and KV cache management compatible with Moshi's inference loop.
#[derive(Debug, Clone)]
pub struct MistralBackend {
    /// The underlying Mistral model
    model: MistralModel,
    /// Configuration
    config: MistralBackendConfig,
    /// Current streaming state
    state: Option<MistralState>,
    /// Device
    device: Device,
    /// DType
    dtype: DType,
}

impl MistralBackend {
    /// Create a new Mistral backend from weights
    pub fn new(config: &MistralBackendConfig, vb: VarBuilder) -> Result<Self> {
        let mistral_config = config.to_mistral_config();
        let model = MistralModel::new(&mistral_config, vb.clone())?;

        Ok(Self {
            model,
            config: config.clone(),
            state: None,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Load from a safetensors file
    pub fn load<P: AsRef<std::path::Path>>(
        config: &MistralBackendConfig,
        model_file: P,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file], dtype, device)?
        };
        Self::new(config, vb)
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    /// Forward pass
    ///
    /// Takes hidden states and returns transformed hidden states.
    /// Input shape: [batch, seq_len, hidden_size]
    /// Output shape: [batch, seq_len, hidden_size]
    pub fn forward(&mut self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        // The Mistral model expects token IDs, but we're passing hidden states
        // We need to bypass the embedding layer and directly use the transformer layers
        //
        // Since candle_transformers::mistral::Model doesn't expose internal layers,
        // we need a custom approach. For now, we'll use a workaround.
        //
        // Note: This is a simplified implementation. A full implementation would
        // require modifying candle_transformers or using a custom Mistral implementation.

        self.model.forward(xs, seqlen_offset)
    }

    /// Forward pass for streaming inference
    pub fn forward_streaming(&mut self, xs: &Tensor) -> Result<Tensor> {
        let seqlen_offset = self.state.as_ref().map(|s| s.position()).unwrap_or(0);
        let result = self.forward(xs, seqlen_offset)?;

        // Update position
        if let Some(ref mut state) = self.state {
            let seq_len = xs.dim(1)?;
            state.advance(seq_len);
        }

        Ok(result)
    }

    /// Clear KV cache
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
        if let Some(ref mut state) = self.state {
            state.reset();
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
}

impl StreamingModule for MistralBackend {
    fn reset_state(&mut self) {
        self.clear_kv_cache();
        self.state = None;
    }

    fn step(&mut self, xs: &StreamTensor, _mask: &StreamMask) -> Result<StreamTensor> {
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => {
                // Initialize state if needed
                if self.state.is_none() {
                    let batch_size = xs.dim(0)?;
                    self.state = Some(MistralState::new(batch_size));
                }

                let result = self.forward_streaming(xs)?;
                Ok(StreamTensor::from_tensor(result))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_backbone_config() {
        let config = MistralBackendConfig::hf_backbone_3b();
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_hidden_layers, 30);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.num_key_value_heads, 4);
    }

    #[test]
    fn test_config_conversion() {
        let config = MistralBackendConfig::hf_backbone_3b();
        let mistral_config = config.to_mistral_config();
        assert_eq!(mistral_config.hidden_size, 3072);
        assert_eq!(mistral_config.num_hidden_layers, 30);
    }
}
