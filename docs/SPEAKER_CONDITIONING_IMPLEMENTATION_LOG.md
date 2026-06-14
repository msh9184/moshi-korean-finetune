# K-Moshi Zero-Shot Speaker Conditioning Implementation Log

**Created**: 2026-01-21
**Last Updated**: 2026-01-21
**Status**: Phase 1 Complete, Phase 2 In Progress

---

## Overview

This document tracks all changes made for implementing Zero-Shot Speaker Conditioning in K-Moshi.

---

## Phase 1: Speaker Encoder Integration (COMPLETED)

### New Files Created

#### 1. `finetune/modules/__init__.py`
```
Purpose: Module package initialization for speaker conditioning
Exports: BaseSpeakerEncoder, ECAPATDNNSpeakerEncoder, SpeakerEncoderConfig,
         SpeakerConditioner, SpeakerConditionerConfig
```

#### 2. `finetune/modules/speaker_encoder.py`
```
Purpose: Speaker embedding extraction from reference audio
Classes:
  - SpeakerEncoderConfig: Configuration dataclass
  - BaseSpeakerEncoder: Abstract base class with freeze/unfreeze
  - ECAPATDNNSpeakerEncoder: SpeechBrain ECAPA-TDNN integration (192-dim output)
  - DummySpeakerEncoder: Testing without SpeechBrain dependency
Functions:
  - create_speaker_encoder(): Factory function
  - get_default_speaker_encoder(): Convenience function
Lines: ~280
```

#### 3. `finetune/modules/speaker_conditioner.py`
```
Purpose: Transform speaker embeddings to Temporal TF sum_condition format
Classes:
  - SpeakerConditionerConfig: Configuration dataclass
  - SpeakerConditioner: Projection layer (192 -> 4096) with learnable scale
  - SpeakerConditioningModule: Combined encoder + conditioner
  - ReferenceSampler: Training-time reference audio sampling
Features:
  - Learnable scale parameter (initial: 0.1)
  - Optional LayerNorm
  - "multiply" or "gated" scale modes
  - Reference sampling avoiding target segment overlap
Lines: ~350
```

#### 4. `tests/__init__.py`
```
Purpose: Test suite package initialization
```

#### 5. `tests/test_speaker_conditioning.py`
```
Purpose: Unit tests for speaker conditioning modules
Test Classes:
  - TestSpeakerEncoderConfig: Config validation tests
  - TestDummySpeakerEncoder: Encoder functionality tests
  - TestSpeakerConditioner: Projection and scaling tests
  - TestReferenceSampler: Segment extraction tests
  - TestSpeakerConditioningIntegration: Full pipeline tests
  - TestSpeakerConditioningArgs: Configuration args tests
Lines: ~350
```

### Modified Files

#### 1. `finetune/backbone/lm_model_wrapper.py`

**Changes Made**:

1. **Module docstring update** (line 7):
   ```python
   # Added: "4. Speaker conditioning via sum_condition (optional)"
   ```

2. **__init__ method** (after line 313):
   ```python
   # Added speaker conditioning attributes
   self.speaker_conditioner: Optional[nn.Module] = None
   self._speaker_conditioning_enabled = False
   ```

3. **New Speaker Conditioning Methods** (lines 614-664):
   ```python
   def set_speaker_conditioner(self, conditioner: nn.Module) -> None
   def enable_speaker_conditioning(self) -> None
   def disable_speaker_conditioning(self) -> None
   @property speaker_conditioning_enabled -> bool
   def get_speaker_stats(self) -> dict
   ```

4. **forward() signature update** (line 670):
   ```python
   def forward(
       self,
       codes: Tensor,
       condition_tensors: Optional[Any] = None,
       sum_condition: Optional[Tensor] = None,      # NEW
       speaker_embedding: Optional[Tensor] = None,  # NEW
       **kwargs,
   ) -> LMModelWrapperOutput:
   ```

5. **forward() docstring update** (lines 689-710):
   ```
   Added documentation for sum_condition and speaker_embedding parameters
   Added Note on Speaker Conditioning section
   ```

6. **Step 2.5: Speaker conditioning application** (lines 747-765):
   ```python
   # Step 2.5: Apply speaker conditioning via sum_condition
   effective_sum_condition = None
   if sum_condition is not None:
       effective_sum_condition = sum_condition
   elif speaker_embedding is not None and self.speaker_conditioning_enabled:
       effective_sum_condition = self.speaker_conditioner(speaker_embedding)
   if effective_sum_condition is not None:
       combined_input = combined_input + effective_sum_condition.to(combined_input)
   ```

#### 2. `finetune/args.py`

**Changes Made** (after line 725):

1. **SpeakerEncoderArgs** (lines 732-773):
   ```python
   @dataclass
   class SpeakerEncoderArgs(Serializable):
       encoder_type: str = "ecapa_tdnn"
       pretrained_path: str = "speechbrain/spkrec-ecapa-voxceleb"
       freeze: bool = True
       output_dim: int = 192
       sample_rate: int = 16000
       normalize_embedding: bool = True
       custom_encoder_path: str | None = None
   ```

2. **SpeakerConditionerArgs** (lines 776-812):
   ```python
   @dataclass
   class SpeakerConditionerArgs(Serializable):
       output_dim: int = 4096
       initial_scale: float = 0.1
       use_layernorm: bool = True
       dropout: float = 0.0
       learnable_scale: bool = True
       scale_mode: str = "multiply"
   ```

3. **ReferenceSamplerArgs** (lines 815-835):
   ```python
   @dataclass
   class ReferenceSamplerArgs(Serializable):
       min_duration_sec: float = 3.0
       max_duration_sec: float = 10.0
       sample_rate: int = 24000
       target_sample_rate: int = 16000
   ```

4. **SpeakerConditioningArgs** (lines 838-891):
   ```python
   @dataclass
   class SpeakerConditioningArgs(Serializable):
       enabled: bool = False
       method: str = "encoder"
       encoder: SpeakerEncoderArgs
       conditioner: SpeakerConditionerArgs
       reference_sampler: ReferenceSamplerArgs
       inference_reference_path: str | None = None
       inference_reference_text: str | None = None
   ```

5. **TrainArgs update** (line 1499-1501):
   ```python
   # Added to TrainArgs
   speaker: SpeakerConditioningArgs = field(default_factory=SpeakerConditioningArgs)
   ```

#### 3. `docs/ZERO_SHOT_SPEAKER_CONDITIONING_SPECIFICATION.md`

**Changes Made** (lines 1051-1070):

Updated Phase 1 implementation status to COMPLETED with list of implemented files.

---

## Phase 2: Reference Sampling Pipeline (COMPLETED)

### Completed Tasks

| Task | Description | Status |
|------|-------------|--------|
| 2.1 | Extend interleaver.py with reference sampling | ✅ COMPLETED |
| 2.2 | Modify batch collation for reference data | ✅ COMPLETED |
| 2.3 | Update train.py for speaker conditioning training | ✅ COMPLETED |
| 2.4 | Integration testing | Pending |

### Modified Files

#### 1. `finetune/data/interleaver.py`

**Extended Sample class** (lines 637-665):
```python
# Added fields for speaker conditioning
speaker_reference_audio: torch.Tensor | None = None  # [T] at 16kHz
speaker_reference_text: str | None = None            # Optional reference text
```

**Extended Batch class** (lines 668-714):
```python
# Added fields for batch-level speaker conditioning
speaker_reference_audios: list[torch.Tensor] | None = None
speaker_reference_texts: list[str] | None = None

# Updated collate() to collect speaker reference data
```

**Extended InterleavedTokenizer.__init__** (lines 1271-1340):
```python
# Added speaker_conditioning_config parameter
# Added attributes:
# - speaker_conditioning_enabled: bool
# - speaker_ref_min_duration_sec: float (default 3.0)
# - speaker_ref_max_duration_sec: float (default 10.0)
# - speaker_ref_target_sample_rate: int (default 16000)
```

**Added `_sample_speaker_reference` method to InterleavedTokenizer** (lines 1397-1521):
```python
def _sample_speaker_reference(
    self,
    full_audio: np.ndarray,
    alignments: list[Alignment],
    target_start_sec: float,
    target_end_sec: float,
    main_speaker_label: str = "SPEAKER_MAIN",
) -> tuple[torch.Tensor | None, str | None]:
    """
    Sample speaker reference audio from MOSHI channel, avoiding target segment.
    Returns (reference_audio_16khz, reference_text).
    """
```

**Extended StereoInterleavedTokenizer.__init__** (lines 1758-1831):
```python
# Added speaker_conditioning_config parameter (same as InterleavedTokenizer)
```

**Added `_sample_speaker_reference` method to StereoInterleavedTokenizer** (lines 1956-2099):
```python
# Same logic as InterleavedTokenizer but handles stereo audio
# Extracts from MOSHI channel (channel 0 = left)
```

**Updated `get_interleaved_tokenizer` factory function** (lines 2991-3125):
```python
# Added speaker_conditioning_config parameter
# Passes config to both InterleavedTokenizer and StereoInterleavedTokenizer
```

#### 2. `train.py`

**Added speaker conditioning setup** (lines 511-583):
```python
# Speaker conditioning configuration extraction
speaker_conditioning_config = {
    "enabled": True,
    "min_duration_sec": args.speaker.reference_sampler.min_duration_sec,
    "max_duration_sec": args.speaker.reference_sampler.max_duration_sec,
    "target_sample_rate": args.speaker.reference_sampler.target_sample_rate,
}

# Speaker encoder and conditioner initialization
speaker_encoder = create_speaker_encoder(encoder_config)
speaker_conditioner = SpeakerConditioner(conditioner_config)

# Integration with model wrapper
unwrapped_model.set_speaker_conditioner(speaker_conditioner)
unwrapped_model.enable_speaker_conditioning()
```

**Updated training loop** (lines 1025-1049):
```python
# Extract speaker embeddings in training loop
speaker_embedding = None
if speaker_encoder is not None and batch.speaker_reference_audios is not None:
    ref_audios = batch.speaker_reference_audios
    valid_embeddings = []
    for ref_audio in ref_audios:
        if ref_audio is not None and ref_audio.numel() > 0:
            ref_audio_gpu = ref_audio.cuda()
            with torch.no_grad():
                emb = speaker_encoder(ref_audio_gpu)
            valid_embeddings.append(emb)
    speaker_embedding = torch.cat(valid_embeddings, dim=0)

# Pass to model
output = model(codes=codes, condition_tensors=condition_tensors, speaker_embedding=speaker_embedding)
```

---

## Configuration Example

```yaml
# example/korean_speaker_conditioning.yaml

speaker:
  enabled: true
  method: "encoder"

  encoder:
    encoder_type: "ecapa_tdnn"
    pretrained_path: "speechbrain/spkrec-ecapa-voxceleb"
    freeze: true
    output_dim: 192
    sample_rate: 16000
    normalize_embedding: true

  conditioner:
    output_dim: 4096
    initial_scale: 0.1
    use_layernorm: true
    dropout: 0.0
    learnable_scale: true
    scale_mode: "multiply"

  reference_sampler:
    min_duration_sec: 3.0
    max_duration_sec: 10.0
    sample_rate: 24000
    target_sample_rate: 16000
```

---

## Usage Example

```python
from finetune.modules import (
    ECAPATDNNSpeakerEncoder,
    SpeakerEncoderConfig,
    SpeakerConditioner,
    SpeakerConditionerConfig,
)

# 1. Create speaker encoder
encoder_config = SpeakerEncoderConfig(
    encoder_type="ecapa_tdnn",
    freeze=True,
    output_dim=192,
)
encoder = ECAPATDNNSpeakerEncoder(encoder_config)

# 2. Create speaker conditioner
conditioner_config = SpeakerConditionerConfig(
    input_dim=192,
    output_dim=4096,
    initial_scale=0.1,
    learnable_scale=True,
)
conditioner = SpeakerConditioner(conditioner_config)

# 3. Set conditioner on model wrapper
model_wrapper.set_speaker_conditioner(conditioner)

# 4. During training
reference_audio = load_reference_audio()  # [B, T] at 16kHz
speaker_emb = encoder(reference_audio)     # [B, 192]
output = model_wrapper(codes, speaker_embedding=speaker_emb)

# OR directly with sum_condition
sum_condition = conditioner(speaker_emb)   # [B, 1, 4096]
output = model_wrapper(codes, sum_condition=sum_condition)
```

---

## File Change Summary

### Phase 1 Files

| File | Action | Lines Changed |
|------|--------|---------------|
| `finetune/modules/__init__.py` | Created | 30 |
| `finetune/modules/speaker_encoder.py` | Created | 280 |
| `finetune/modules/speaker_conditioner.py` | Created | 350 |
| `tests/__init__.py` | Created | 2 |
| `tests/test_speaker_conditioning.py` | Created | 350 |
| `finetune/backbone/lm_model_wrapper.py` | Modified | +80 |
| `finetune/args.py` | Modified | +170 |
| `docs/ZERO_SHOT_SPEAKER_CONDITIONING_SPECIFICATION.md` | Modified | +20 |
| **Phase 1 Total** | | **~1280** |

### Phase 2 Files

| File | Action | Lines Changed |
|------|--------|---------------|
| `finetune/data/interleaver.py` | Modified | +350 |
| `train.py` | Modified | +80 |
| `docs/SPEAKER_CONDITIONING_IMPLEMENTATION_LOG.md` | Updated | +120 |
| **Phase 2 Total** | | **~550** |

### Overall Total

| Phase | Lines |
|-------|-------|
| Phase 1 | ~1280 |
| Phase 2 | ~550 |
| **Total** | **~1830** |

---

## Dependencies Added

```
# Required for ECAPA-TDNN speaker encoder
speechbrain>=0.5.15

# Already included
torch>=2.0.0
torchaudio>=2.0.0
```

---

## Implementation Status Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ COMPLETED | Speaker Encoder Integration |
| Phase 2 | ✅ COMPLETED | Reference Sampling Pipeline |
| Phase 3 | 📋 Planned | VALL-E Style Audio Prompt Method |
| Phase 4 | 📋 Planned | Training and Evaluation |
| Phase 5 | 📋 Planned | Inference API and Rust Server Integration |

## Next Steps

1. **Integration Testing**: Run end-to-end tests with speaker conditioning enabled
2. ~~**Phase 3**: Implement VALL-E style audio prompt method~~ → ✅ **COMPLETED (V2)**
3. **Phase 4**: Training experiments with different speaker conditioning configurations
4. **Phase 5**: Inference API integration and Rust server support

---

## Phase 3: V2 Update - W2v-BERT 2.0 SV + VALL-E Style Prompting (COMPLETED)

**Updated**: 2026-01-22

### New Files Created

#### 1. `finetune/modules/audio_prompt.py`
```
Purpose: VALL-E style audio/text prompting for zero-shot speaker adaptation
Classes:
  - AudioPromptConfig: Configuration dataclass
  - AudioPromptSample: Sampled prompt data structure
  - AudioPromptSampler: Reference segment sampling from codes
  - AudioPromptEncoder: Learnable prompt encoding (future)
  - AudioPromptModule: Main interface combining sampler + encoder
Functions:
  - create_audio_prompt_module(): Factory function
  - get_default_audio_prompt_config(): Default config
Lines: ~450
Features:
  - Three modes: "speaker_embedding", "audio_only", "audio_text"
  - Configurable prompt duration (3-10s)
  - Overlap avoidance with training segment
  - Prompt masking for loss computation
```

### Modified Files

#### 1. `finetune/modules/speaker_encoder.py`

**New W2vBERT2SpeakerEncoder class** (added):
```python
class W2vBERT2SpeakerEncoder(BaseSpeakerEncoder):
    """W2v-BERT 2.0 Speaker Verification Encoder.
    - 0.14% EER on VoxCeleb1-O (State-of-the-Art)
    - 256-dimensional speaker embedding
    - Uses Attentive Statistics Pooling (ASP)
    Reference: https://arxiv.org/abs/2510.04213
    """
```

**Helper class _ASP** (added):
```python
class _ASP(nn.Module):
    """Attentive Statistics Pooling for W2v-BERT 2.0."""
```

**Updated create_speaker_encoder()** to support "w2v_bert2" type.

**Updated get_default_speaker_encoder()** to accept encoder_type parameter.

#### 2. `finetune/modules/__init__.py`

**Updated exports** to include:
- `W2vBERT2SpeakerEncoder`
- `AudioPromptConfig`
- `AudioPromptSample`
- `AudioPromptSampler`
- `AudioPromptEncoder`
- `AudioPromptModule`
- `create_audio_prompt_module`
- `get_default_audio_prompt_config`

#### 3. `finetune/args.py`

**New AudioPromptArgs dataclass** (lines 863-920):
```python
@dataclass
class AudioPromptArgs(Serializable):
    enable: bool = False
    mode: str = "speaker_embedding"  # or "audio_only", "audio_text"
    min_duration_sec: float = 3.0
    max_duration_sec: float = 10.0
    sample_strategy: str = "random"
    avoid_overlap: bool = True
```

**Updated SpeakerEncoderArgs**:
- Added `encoder_type` options: "ecapa_tdnn", "w2v_bert2", "dummy", "custom"
- Added `w2v_bert2_n_mfa_layers: int = -1`
- Added `w2v_bert2_pooling: str = "ASP"`
- Auto-adjustment of output_dim for w2v_bert2 (192 → 256)

**Updated SpeakerConditioningArgs**:
- Added `method` options: "encoder", "prompt", "both"
- Added `audio_prompt: AudioPromptArgs` field
- Auto-enable audio_prompt when method is "prompt" or "both"

### V2 Configuration Example

```yaml
# example/korean_v4_fsdp_speaker_v2.yaml

speaker:
  enabled: true
  method: "both"  # Use both encoder and audio prompt

  encoder:
    encoder_type: "w2v_bert2"  # SOTA 0.14% EER
    pretrained_path: "/path/to/model_lmft_0.14.pth"
    output_dim: 256
    freeze: true
    w2v_bert2_n_mfa_layers: -1  # Use all layers
    w2v_bert2_pooling: "ASP"

  conditioner:
    output_dim: 4096
    initial_scale: 0.1
    use_layernorm: true

  audio_prompt:
    enable: true
    mode: "audio_text"  # PersonaPlex style
    min_duration_sec: 3.0
    max_duration_sec: 10.0
    sample_strategy: "random"
    avoid_overlap: true

  reference_sampler:
    min_duration_sec: 3.0
    max_duration_sec: 10.0
```

### V2 File Change Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `finetune/modules/audio_prompt.py` | Created | ~450 |
| `finetune/modules/speaker_encoder.py` | Modified | +150 |
| `finetune/modules/__init__.py` | Modified | +25 |
| `finetune/args.py` | Modified | +80 |
| `docs/SPEAKER_CONDITIONING_IMPLEMENTATION_V2.md` | Created | ~250 |
| **Phase 3 Total** | | **~955** |

### Dependencies Added (V2)

```
# For W2v-BERT 2.0 SV speaker encoder
transformers>=4.30.0

# Already required
speechbrain>=0.5.15  # For ECAPA-TDNN
torch>=2.0.0
torchaudio>=2.0.0
```

### Updated Implementation Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ COMPLETED | Speaker Encoder Integration |
| Phase 2 | ✅ COMPLETED | Reference Sampling Pipeline |
| Phase 3 | ✅ **COMPLETED** | **V2: W2v-BERT 2.0 SV + VALL-E Style Prompting** |
| Phase 4 | 📋 Planned | Training and Evaluation |
| Phase 5 | 📋 Planned | Inference API and Rust Server Integration |

---

*Document maintained by K-Moshi Development Team*
*Last Updated: 2026-01-22*
