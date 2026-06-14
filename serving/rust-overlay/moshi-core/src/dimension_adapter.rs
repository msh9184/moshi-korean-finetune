// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! Dimension Adapter for K-Moshi
//!
//! This module provides linear projection layers to bridge between
//! different hidden dimensions in the Moshi architecture.
//!
//! Key use case:
//! - Moshi original backbone: 4096 hidden size
//! - HF backbone backbone: 3072 hidden size
//!
//! The adapter provides bidirectional projection to seamlessly
//! integrate alternative backbones without modifying the rest of
//! the Moshi architecture (embeddings, depformer, etc.)

use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::streaming::{StreamMask, StreamTensor, StreamingModule};

/// Configuration for dimension adapter
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct DimensionAdapterConfig {
    /// Source dimension (e.g., 4096 for Moshi)
    pub source_dim: usize,
    /// Target dimension (e.g., 3072 for HfBackbone)
    pub target_dim: usize,
    /// Whether to use bias in linear projections
    pub use_bias: bool,
    /// Whether to apply layer normalization after projection
    pub use_layer_norm: bool,
    /// Epsilon for layer normalization
    pub layer_norm_eps: f64,
}

impl Default for DimensionAdapterConfig {
    fn default() -> Self {
        Self::moshi_to_hf_backbone()
    }
}

impl DimensionAdapterConfig {
    /// Configuration for Moshi (4096) → HfBackbone (3072) adapter
    pub fn moshi_to_hf_backbone() -> Self {
        Self {
            source_dim: 4096,
            target_dim: 3072,
            use_bias: false,
            use_layer_norm: true,
            layer_norm_eps: 1e-5,
        }
    }

    /// Configuration for identity (no dimension change)
    pub fn identity(dim: usize) -> Self {
        Self {
            source_dim: dim,
            target_dim: dim,
            use_bias: false,
            use_layer_norm: false,
            layer_norm_eps: 1e-5,
        }
    }

    /// Check if this is an identity adapter (no dimension change)
    pub fn is_identity(&self) -> bool {
        self.source_dim == self.target_dim
    }
}

/// RMS Layer Normalization (used in Moshi/Mistral)
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // RMS norm: x / sqrt(mean(x^2) + eps) * weight
        let variance = xs.sqr()?.mean_keepdim(candle::D::Minus1)?;
        let xs_normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        xs_normed.broadcast_mul(&self.weight)
    }
}

/// Bidirectional dimension adapter
///
/// Provides projection layers to convert between different hidden dimensions.
/// Used to integrate HfBackbone (3072) backbone into Moshi (4096) architecture.
#[derive(Debug, Clone)]
pub struct DimensionAdapter {
    /// Project from source to target dimension (e.g., 4096 → 3072)
    down_proj: Option<candle_nn::Linear>,
    /// Project from target back to source dimension (e.g., 3072 → 4096)
    up_proj: Option<candle_nn::Linear>,
    /// Optional layer norm after down projection
    down_norm: Option<RmsNorm>,
    /// Optional layer norm after up projection
    up_norm: Option<RmsNorm>,
    /// Configuration
    config: DimensionAdapterConfig,
    /// Device
    device: Device,
    /// Data type
    dtype: DType,
}

impl DimensionAdapter {
    /// Create a new dimension adapter from configuration
    pub fn new(config: &DimensionAdapterConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // If identity, don't create any projection layers
        if config.is_identity() {
            return Ok(Self {
                down_proj: None,
                up_proj: None,
                down_norm: None,
                up_norm: None,
                config: config.clone(),
                device,
                dtype,
            });
        }

        // Create down projection (source → target)
        let down_proj = if config.use_bias {
            candle_nn::linear(config.source_dim, config.target_dim, vb.pp("down_proj"))?
        } else {
            candle_nn::linear_no_bias(config.source_dim, config.target_dim, vb.pp("down_proj"))?
        };

        // Create up projection (target → source)
        let up_proj = if config.use_bias {
            candle_nn::linear(config.target_dim, config.source_dim, vb.pp("up_proj"))?
        } else {
            candle_nn::linear_no_bias(config.target_dim, config.source_dim, vb.pp("up_proj"))?
        };

        // Optional layer norms
        let down_norm = if config.use_layer_norm {
            Some(RmsNorm::new(config.target_dim, config.layer_norm_eps, vb.pp("down_norm"))?)
        } else {
            None
        };

        let up_norm = if config.use_layer_norm {
            Some(RmsNorm::new(config.source_dim, config.layer_norm_eps, vb.pp("up_norm"))?)
        } else {
            None
        };

        Ok(Self {
            down_proj: Some(down_proj),
            up_proj: Some(up_proj),
            down_norm,
            up_norm,
            config: config.clone(),
            device,
            dtype,
        })
    }

    /// Create an identity adapter (no dimension change)
    pub fn identity(dim: usize, device: &Device, dtype: DType) -> Self {
        Self {
            down_proj: None,
            up_proj: None,
            down_norm: None,
            up_norm: None,
            config: DimensionAdapterConfig::identity(dim),
            device: device.clone(),
            dtype,
        }
    }

    /// Initialize a new dimension adapter with random weights
    pub fn init(config: &DimensionAdapterConfig, device: &Device, dtype: DType) -> Result<Self> {
        if config.is_identity() {
            return Ok(Self::identity(config.source_dim, device, dtype));
        }

        // Initialize down projection with Kaiming initialization
        let down_weight = Self::kaiming_init(config.source_dim, config.target_dim, device, dtype)?;
        let down_proj = candle_nn::Linear::new(down_weight, None);

        // Initialize up projection with Kaiming initialization
        let up_weight = Self::kaiming_init(config.target_dim, config.source_dim, device, dtype)?;
        let up_proj = candle_nn::Linear::new(up_weight, None);

        // Initialize layer norms with ones
        let down_norm = if config.use_layer_norm {
            let weight = Tensor::ones(config.target_dim, dtype, device)?;
            Some(RmsNorm::from_weight(weight, config.layer_norm_eps))
        } else {
            None
        };

        let up_norm = if config.use_layer_norm {
            let weight = Tensor::ones(config.source_dim, dtype, device)?;
            Some(RmsNorm::from_weight(weight, config.layer_norm_eps))
        } else {
            None
        };

        Ok(Self {
            down_proj: Some(down_proj),
            up_proj: Some(up_proj),
            down_norm,
            up_norm,
            config: config.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Kaiming uniform initialization for linear layer weights
    fn kaiming_init(in_d: usize, out_d: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let bound = (3.0_f64 / in_d as f64).sqrt();
        Tensor::rand(0f32, 1f32, (out_d, in_d), device)?
            .to_dtype(dtype)?
            .affine(2.0 * bound, -bound)
    }

    /// Project from source dimension to target dimension
    /// Input: [batch, seq_len, source_dim]
    /// Output: [batch, seq_len, target_dim]
    pub fn project_down(&self, xs: &Tensor) -> Result<Tensor> {
        if self.config.is_identity() {
            return Ok(xs.clone());
        }

        let xs = self.down_proj.as_ref().unwrap().forward(xs)?;
        match &self.down_norm {
            Some(norm) => norm.forward(&xs),
            None => Ok(xs),
        }
    }

    /// Project from target dimension back to source dimension
    /// Input: [batch, seq_len, target_dim]
    /// Output: [batch, seq_len, source_dim]
    pub fn project_up(&self, xs: &Tensor) -> Result<Tensor> {
        if self.config.is_identity() {
            return Ok(xs.clone());
        }

        let xs = self.up_proj.as_ref().unwrap().forward(xs)?;
        match &self.up_norm {
            Some(norm) => norm.forward(&xs),
            None => Ok(xs),
        }
    }

    /// Get source dimension
    pub fn source_dim(&self) -> usize {
        self.config.source_dim
    }

    /// Get target dimension
    pub fn target_dim(&self) -> usize {
        self.config.target_dim
    }

    /// Check if this is an identity adapter
    pub fn is_identity(&self) -> bool {
        self.config.is_identity()
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

/// Streaming wrapper for dimension adapter (down projection)
#[derive(Debug, Clone)]
pub struct StreamingDownAdapter {
    adapter: DimensionAdapter,
}

impl StreamingDownAdapter {
    pub fn new(adapter: DimensionAdapter) -> Self {
        Self { adapter }
    }
}

impl StreamingModule for StreamingDownAdapter {
    fn reset_state(&mut self) {
        // No state to reset for linear projection
    }

    fn step(&mut self, xs: &StreamTensor, _mask: &StreamMask) -> Result<StreamTensor> {
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => {
                let result = self.adapter.project_down(xs)?;
                Ok(StreamTensor::from_tensor(result))
            }
        }
    }
}

/// Streaming wrapper for dimension adapter (up projection)
#[derive(Debug, Clone)]
pub struct StreamingUpAdapter {
    adapter: DimensionAdapter,
}

impl StreamingUpAdapter {
    pub fn new(adapter: DimensionAdapter) -> Self {
        Self { adapter }
    }
}

impl StreamingModule for StreamingUpAdapter {
    fn reset_state(&mut self) {
        // No state to reset for linear projection
    }

    fn step(&mut self, xs: &StreamTensor, _mask: &StreamMask) -> Result<StreamTensor> {
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => {
                let result = self.adapter.project_up(xs)?;
                Ok(StreamTensor::from_tensor(result))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_config() {
        let config = DimensionAdapterConfig::identity(4096);
        assert!(config.is_identity());
        assert_eq!(config.source_dim, 4096);
        assert_eq!(config.target_dim, 4096);
    }

    #[test]
    fn test_moshi_to_hf_backbone_config() {
        let config = DimensionAdapterConfig::moshi_to_hf_backbone();
        assert!(!config.is_identity());
        assert_eq!(config.source_dim, 4096);
        assert_eq!(config.target_dim, 3072);
    }

    #[test]
    fn test_identity_adapter() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let adapter = DimensionAdapter::identity(4096, &device, dtype);

        assert!(adapter.is_identity());
        assert_eq!(adapter.source_dim(), 4096);
        assert_eq!(adapter.target_dim(), 4096);
    }

    #[test]
    fn test_adapter_init() -> Result<()> {
        let config = DimensionAdapterConfig::moshi_to_hf_backbone();
        let device = Device::Cpu;
        let dtype = DType::F32;

        let adapter = DimensionAdapter::init(&config, &device, dtype)?;

        assert!(!adapter.is_identity());
        assert_eq!(adapter.source_dim(), 4096);
        assert_eq!(adapter.target_dim(), 3072);

        // Test forward pass
        let input = Tensor::randn(0f32, 1f32, (1, 10, 4096), &device)?;
        let down = adapter.project_down(&input)?;
        assert_eq!(down.dims(), &[1, 10, 3072]);

        let up = adapter.project_up(&down)?;
        assert_eq!(up.dims(), &[1, 10, 4096]);

        Ok(())
    }
}
