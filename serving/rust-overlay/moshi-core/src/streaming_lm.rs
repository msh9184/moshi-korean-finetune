// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! Streaming Language Model Trait and Unified Enum
//!
//! This module provides a common interface for different LM backends (Moshi, HfBackbone)
//! through the `StreamingLm` trait and the `LmModelEnum` wrapper.

use candle::{DType, Device, Result, Tensor};
use crate::StreamMask;
use crate::transformer::CaSrc;

/// Common trait for streaming language models.
///
/// This trait defines the interface that both `LmModel` (Moshi) and `HfBackboneLmModel`
/// must implement to be used in the streaming inference pipeline.
pub trait StreamingLm: Send + Sync {
    /// Reset the model state (clear KV caches)
    fn reset_state(&mut self);

    /// Get the number of input audio codebooks
    fn in_audio_codebooks(&self) -> usize;

    /// Get the audio padding token ID
    fn audio_pad_token(&self) -> u32;

    /// Get the text start token ID
    fn text_start_token(&self) -> u32;

    /// Get the number of generated audio codebooks (depformer slices)
    fn generated_audio_codebooks(&self) -> usize;

    /// Check if the model uses quantization
    fn is_quantized(&self) -> bool;

    /// Get the device the model is on
    fn device(&self) -> &Device;

    /// Get the model's data type
    fn dtype(&self) -> DType;

    /// Forward pass through the model
    ///
    /// Returns (text_logits, hidden_states)
    fn forward(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)>;

    /// Sample from the depformer
    fn depformer_sample(
        &mut self,
        xs: &Tensor,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>>;

    /// Sample from the depformer with classifier-free guidance
    fn depformer_sample_cfg(
        &mut self,
        xs: &Tensor,
        cfg_alpha: f64,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>>;

    /// Forward pass with conditioning
    fn forward_cond(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)>;

    /// Forward pass with cross-attention
    fn forward_ca(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        ca_src: &CaSrc,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)>;
}

/// Unified enum for different LM backends.
///
/// This enum wraps both Moshi and HfBackbone models to provide a unified interface
/// for the streaming inference pipeline.
#[derive(Debug)]
pub enum LmModelEnum {
    /// Original Moshi 7B model
    Moshi(crate::lm::LmModel),
    /// HF backbone model with Mistral backbone
    HfBackbone(crate::hf_backbone_lm::HfBackboneLmModel),
}

impl Clone for LmModelEnum {
    fn clone(&self) -> Self {
        match self {
            LmModelEnum::Moshi(m) => LmModelEnum::Moshi(m.clone()),
            LmModelEnum::HfBackbone(m) => {
                // HfBackboneLmModel doesn't implement Clone, so we panic
                // This should be handled differently in production
                panic!("HfBackboneLmModel clone not supported - use separate instances")
            }
        }
    }
}

impl LmModelEnum {
    /// Create a new Moshi variant
    pub fn moshi(model: crate::lm::LmModel) -> Self {
        LmModelEnum::Moshi(model)
    }

    /// Create a new HfBackbone variant
    pub fn hf_backbone(model: crate::hf_backbone_lm::HfBackboneLmModel) -> Self {
        LmModelEnum::HfBackbone(model)
    }

    /// Check if this is a Moshi model
    pub fn is_moshi(&self) -> bool {
        matches!(self, LmModelEnum::Moshi(_))
    }

    /// Check if this is a HfBackbone model
    pub fn is_hf_backbone(&self) -> bool {
        matches!(self, LmModelEnum::HfBackbone(_))
    }

    /// Get mutable reference to the underlying Moshi model
    pub fn as_moshi_mut(&mut self) -> Option<&mut crate::lm::LmModel> {
        match self {
            LmModelEnum::Moshi(m) => Some(m),
            _ => None,
        }
    }

    /// Get mutable reference to the underlying HfBackbone model
    pub fn as_hf_backbone_mut(&mut self) -> Option<&mut crate::hf_backbone_lm::HfBackboneLmModel> {
        match self {
            LmModelEnum::HfBackbone(m) => Some(m),
            _ => None,
        }
    }
}

// Implement StreamingLm for LmModelEnum by delegating to the inner model
impl StreamingLm for LmModelEnum {
    fn reset_state(&mut self) {
        match self {
            LmModelEnum::Moshi(m) => m.reset_state(),
            LmModelEnum::HfBackbone(m) => m.reset_state(),
        }
    }

    fn in_audio_codebooks(&self) -> usize {
        match self {
            LmModelEnum::Moshi(m) => m.in_audio_codebooks(),
            LmModelEnum::HfBackbone(m) => m.in_audio_codebooks(),
        }
    }

    fn audio_pad_token(&self) -> u32 {
        match self {
            LmModelEnum::Moshi(m) => m.audio_pad_token(),
            LmModelEnum::HfBackbone(m) => m.audio_pad_token(),
        }
    }

    fn text_start_token(&self) -> u32 {
        match self {
            LmModelEnum::Moshi(m) => m.text_start_token(),
            LmModelEnum::HfBackbone(m) => m.text_start_token(),
        }
    }

    fn generated_audio_codebooks(&self) -> usize {
        match self {
            LmModelEnum::Moshi(m) => m.generated_audio_codebooks(),
            LmModelEnum::HfBackbone(m) => m.generated_audio_codebooks(),
        }
    }

    fn is_quantized(&self) -> bool {
        match self {
            LmModelEnum::Moshi(m) => m.is_quantized(),
            LmModelEnum::HfBackbone(m) => m.is_quantized(),
        }
    }

    fn device(&self) -> &Device {
        match self {
            LmModelEnum::Moshi(m) => m.device(),
            LmModelEnum::HfBackbone(m) => m.device(),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            LmModelEnum::Moshi(m) => m.dtype(),
            LmModelEnum::HfBackbone(m) => m.dtype(),
        }
    }

    fn forward(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            LmModelEnum::Moshi(m) => m.forward(text_ids, audio_ids, mask),
            LmModelEnum::HfBackbone(m) => m.forward(text_ids, audio_ids, mask),
        }
    }

    fn depformer_sample(
        &mut self,
        xs: &Tensor,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        match self {
            LmModelEnum::Moshi(m) => m.depformer_sample(xs, text_token, forced_audio_tokens, lp),
            LmModelEnum::HfBackbone(m) => m.depformer_sample(xs, text_token, forced_audio_tokens, lp),
        }
    }

    fn depformer_sample_cfg(
        &mut self,
        xs: &Tensor,
        cfg_alpha: f64,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        match self {
            LmModelEnum::Moshi(m) => m.depformer_sample_cfg(xs, cfg_alpha, text_token, forced_audio_tokens, lp),
            LmModelEnum::HfBackbone(m) => m.depformer_sample_cfg(xs, cfg_alpha, text_token, forced_audio_tokens, lp),
        }
    }

    fn forward_cond(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            LmModelEnum::Moshi(m) => m.forward_cond(text_ids, audio_ids, conditions, mask),
            LmModelEnum::HfBackbone(m) => m.forward_cond(text_ids, audio_ids, conditions, mask),
        }
    }

    fn forward_ca(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        ca_src: &CaSrc,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            LmModelEnum::Moshi(m) => m.forward_ca(text_ids, audio_ids, ca_src, conditions, mask),
            LmModelEnum::HfBackbone(m) => m.forward_ca(text_ids, audio_ids, ca_src, conditions, mask),
        }
    }
}

// Implement StreamingLm for LmModel
impl StreamingLm for crate::lm::LmModel {
    fn reset_state(&mut self) {
        crate::lm::LmModel::reset_state(self)
    }

    fn in_audio_codebooks(&self) -> usize {
        crate::lm::LmModel::in_audio_codebooks(self)
    }

    fn audio_pad_token(&self) -> u32 {
        crate::lm::LmModel::audio_pad_token(self)
    }

    fn text_start_token(&self) -> u32 {
        crate::lm::LmModel::text_start_token(self)
    }

    fn generated_audio_codebooks(&self) -> usize {
        crate::lm::LmModel::generated_audio_codebooks(self)
    }

    fn is_quantized(&self) -> bool {
        crate::lm::LmModel::is_quantized(self)
    }

    fn device(&self) -> &Device {
        crate::lm::LmModel::device(self)
    }

    fn dtype(&self) -> DType {
        crate::lm::LmModel::dtype(self)
    }

    fn forward(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        crate::lm::LmModel::forward(self, text_ids, audio_ids, mask)
    }

    fn depformer_sample(
        &mut self,
        xs: &Tensor,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        crate::lm::LmModel::depformer_sample(self, xs, text_token, forced_audio_tokens, lp)
    }

    fn depformer_sample_cfg(
        &mut self,
        xs: &Tensor,
        cfg_alpha: f64,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        crate::lm::LmModel::depformer_sample_cfg(self, xs, cfg_alpha, text_token, forced_audio_tokens, lp)
    }

    fn forward_cond(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        crate::lm::LmModel::forward_cond(self, text_ids, audio_ids, conditions, mask)
    }

    fn forward_ca(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        ca_src: &CaSrc,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        crate::lm::LmModel::forward_ca(self, text_ids, audio_ids, ca_src, conditions, mask)
    }
}

// Implement StreamingLm for HfBackboneLmModel
impl StreamingLm for crate::hf_backbone_lm::HfBackboneLmModel {
    fn reset_state(&mut self) {
        crate::hf_backbone_lm::HfBackboneLmModel::reset_state(self)
    }

    fn in_audio_codebooks(&self) -> usize {
        crate::hf_backbone_lm::HfBackboneLmModel::in_audio_codebooks(self)
    }

    fn audio_pad_token(&self) -> u32 {
        crate::hf_backbone_lm::HfBackboneLmModel::audio_pad_token(self)
    }

    fn text_start_token(&self) -> u32 {
        crate::hf_backbone_lm::HfBackboneLmModel::text_start_token(self)
    }

    fn generated_audio_codebooks(&self) -> usize {
        crate::hf_backbone_lm::HfBackboneLmModel::generated_audio_codebooks(self)
    }

    fn is_quantized(&self) -> bool {
        crate::hf_backbone_lm::HfBackboneLmModel::is_quantized(self)
    }

    fn device(&self) -> &Device {
        crate::hf_backbone_lm::HfBackboneLmModel::device(self)
    }

    fn dtype(&self) -> DType {
        crate::hf_backbone_lm::HfBackboneLmModel::dtype(self)
    }

    fn forward(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        crate::hf_backbone_lm::HfBackboneLmModel::forward(self, text_ids, audio_ids, mask)
    }

    fn depformer_sample(
        &mut self,
        xs: &Tensor,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        crate::hf_backbone_lm::HfBackboneLmModel::depformer_sample(self, xs, text_token, forced_audio_tokens, lp)
    }

    fn depformer_sample_cfg(
        &mut self,
        xs: &Tensor,
        cfg_alpha: f64,
        text_token: Option<u32>,
        forced_audio_tokens: &[Option<u32>],
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        crate::hf_backbone_lm::HfBackboneLmModel::depformer_sample_cfg(self, xs, cfg_alpha, text_token, forced_audio_tokens, lp)
    }

    fn forward_cond(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        crate::hf_backbone_lm::HfBackboneLmModel::forward_cond(self, text_ids, audio_ids, conditions, mask)
    }

    fn forward_ca(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        ca_src: &CaSrc,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &StreamMask,
    ) -> Result<(Tensor, Tensor)> {
        crate::hf_backbone_lm::HfBackboneLmModel::forward_ca(self, text_ids, audio_ids, ca_src, conditions, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enum_variants() {
        // Just test that the enum compiles correctly
        // Actual model tests require weights
    }
}
