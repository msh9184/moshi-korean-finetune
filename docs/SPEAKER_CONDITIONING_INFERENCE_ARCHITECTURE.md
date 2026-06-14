# Speaker Conditioning Inference Architecture
## K-Moshi Zero-Shot Speaker Adaptation - Evaluation & Inference Design

**Author**: K-Moshi Development Team
**Date**: 2026-01-23
**Version**: 1.0.0
**Status**: Architecture Design Document

---

## 1. Executive Summary

이 문서는 K-Moshi의 Speaker Conditioning 기능이 Training에서 Evaluation/Inference로 전환될 때 필요한 아키텍처 설계를 정의합니다.

### 핵심 요구사항
1. **Deterministic Reference Sampling**: 평가/추론 시 랜덤성 완전 제거
2. **Rust Backend Integration**: Candle 기반 추론 엔진에 Speaker Conditioning 통합
3. **Multiple Reference Input Methods**: 파일 경로, 문자열, 웹 업로드 지원
4. **Default Reference Configuration**: 설정 파일 기반 기본 참조 오디오/텍스트

---

## 2. Current Architecture Analysis

### 2.1 Training Phase (Current Implementation)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE (Current)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐   │
│  │  Full Audio File │───▶│  Random Sampling │───▶│  Reference      │   │
│  │  (dialogue.wav)  │    │  (train mode)    │    │  Audio/Text     │   │
│  └──────────────────┘    └──────────────────┘    └────────┬────────┘   │
│                                                           │            │
│                                                           ▼            │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐   │
│  │  Speaker Encoder │◀───│  16kHz Resample  │◀───│  Reference      │   │
│  │  (W2v-BERT 2.0)  │    │                  │    │  3-10 sec       │   │
│  └────────┬─────────┘    └──────────────────┘    └─────────────────┘   │
│           │                                                            │
│           ▼                                                            │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐   │
│  │  Speaker         │───▶│  sum_condition   │───▶│  LMModel        │   │
│  │  Conditioner     │    │  [B, 1, 4096]    │    │  Forward        │   │
│  └──────────────────┘    └──────────────────┘    └─────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Files & Current Sampling Logic

#### `finetune/modules/audio_prompt.py` (Lines 179-229)
```python
# Current: Random sampling with random duration
sample_strategy: Literal["random", "start", "end", "voiced"] = "random"

# Duration is randomly sampled
duration_frames = torch.randint(
    self.min_frames,
    max_possible_frames + 1,
    (1,),
).item()

# Position is randomly sampled from valid regions
start_frame = self._sample_from_regions(valid_regions, duration_frames)
```

**문제점**:
- `duration_frames`가 `torch.randint`로 랜덤 결정
- `start_frame`도 랜덤 샘플링
- 평가/추론 시 동일 입력에 대해 다른 결과 발생

#### `finetune/args.py` (Lines 1002-1026)
```python
# Available strategies (already defined)
valid_strategies = ("random", "start", "end", "voiced")
sample_strategy: str = "random"
```

**"start" 전략 존재하나 미구현**:
- 옵션은 정의되어 있으나 실제 deterministic 구현 없음

---

## 3. Deterministic Reference Sampling Design

### 3.1 Evaluation/Inference Mode Protocol

```
┌─────────────────────────────────────────────────────────────────────────┐
│              DETERMINISTIC REFERENCE SAMPLING PROTOCOL                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Strategy: "start" (RECOMMENDED for eval/inference)                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Full Audio: [==============================================]    │   │
│  │              ^                                                   │   │
│  │              │                                                   │   │
│  │              └─ Always start from position 0                     │   │
│  │                                                                   │   │
│  │  Reference:  [==========]                                        │   │
│  │              │← fixed_duration_sec (e.g., 10.0) →│               │   │
│  │                                                                   │   │
│  │  Parameters:                                                      │   │
│  │    - start_position: 0 (always)                                  │   │
│  │    - duration: config.fixed_duration_sec (not random)            │   │
│  │    - no random number generation                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Configuration Changes

#### New Args in `finetune/args.py`
```python
@dataclass
class AudioPromptArgs(Serializable):
    # ... existing fields ...

    # NEW: Deterministic mode for eval/inference
    deterministic: bool = False  # Enable deterministic sampling
    fixed_duration_sec: float = 10.0  # Fixed duration when deterministic=True

    # NEW: First N words mode (alternative to fixed duration)
    use_word_count: bool = False  # Use word count instead of duration
    fixed_word_count: int = 20  # Number of words when use_word_count=True
```

#### Evaluation YAML Example
```yaml
speaker:
  enabled: true
  method: "both"

  audio_prompt:
    enable: true
    mode: "audio_text"
    sample_strategy: "start"      # CRITICAL: Use "start" for eval
    deterministic: true           # NEW: Enable deterministic mode
    fixed_duration_sec: 10.0      # NEW: Fixed 10 seconds
    avoid_overlap: false          # Overlap OK for eval (same file)
```

### 3.3 Implementation Changes Required

#### `finetune/modules/audio_prompt.py` Modifications

```python
def sample_single(
    self,
    codes: torch.Tensor,
    exclude_start: Optional[int] = None,
    exclude_end: Optional[int] = None,
    deterministic: bool = False,  # NEW parameter
) -> AudioPromptSample:
    """Sample a single reference segment from codes.

    Args:
        codes: Full sequence codes [9, T]
        exclude_start: Start frame to avoid (training only)
        exclude_end: End frame to avoid (training only)
        deterministic: If True, use fixed sampling (eval/inference)
    """
    num_codebooks, total_frames = codes.shape

    if deterministic or self.config.sample_strategy == "start":
        # DETERMINISTIC: Always start from 0, fixed duration
        start_frame = 0
        duration_frames = min(
            int(self.config.fixed_duration_sec * self.frame_rate),
            total_frames
        )
    elif self.config.sample_strategy == "end":
        # DETERMINISTIC: Always from end, fixed duration
        duration_frames = min(
            int(self.config.fixed_duration_sec * self.frame_rate),
            total_frames
        )
        start_frame = max(0, total_frames - duration_frames)
    else:
        # RANDOM: Original behavior for training
        duration_frames = torch.randint(...)
        start_frame = self._sample_from_regions(...)

    end_frame = min(start_frame + duration_frames, total_frames)

    # Extract codes
    audio_codes = codes[1:9, start_frame:end_frame]
    text_tokens = codes[0, start_frame:end_frame]

    return AudioPromptSample(
        audio_codes=audio_codes,
        text_tokens=text_tokens,
        duration_sec=(end_frame - start_frame) / self.frame_rate,
        start_idx=start_frame,
        end_idx=end_frame,
    )
```

---

## 4. Evaluation Pipeline Integration

### 4.1 Current eval.py Gap Analysis

```python
# finetune/eval.py (Current)
def evaluate(
    model: FSDP | torch.nn.Module,
    data_loader: Generator,
    ...
) -> EvalResults:
    # Missing: speaker_embedding parameter
    # Missing: audio_prompt integration
    output = model(codes=codes, condition_tensors=condition_tensors)
```

### 4.2 Required eval.py Modifications

```python
def evaluate(
    model: FSDP | torch.nn.Module,
    data_loader: Generator,
    mimi: MimiModel,
    args: TrainArgs,
    speaker_encoder: Optional[BaseSpeakerEncoder] = None,  # NEW
    audio_prompt_sampler: Optional[AudioPromptSampler] = None,  # NEW
    **kwargs,
) -> EvalResults:
    """FSDP-compatible evaluation with speaker conditioning.

    Speaker conditioning in evaluation:
        1. Reference is sampled deterministically (start of file)
        2. Speaker embedding is computed once per sample
        3. Audio prompt (if enabled) uses same reference segment
    """

    for batch in data_loader:
        codes = batch.codes  # [B, 9, T]

        # NEW: Speaker conditioning for evaluation
        speaker_embedding = None
        prompted_codes = codes
        prompt_mask = None

        if args.speaker.enabled:
            # Sample reference DETERMINISTICALLY
            if audio_prompt_sampler is not None:
                prompt_samples = audio_prompt_sampler.sample_batch(
                    codes,
                    current_start=0,
                    current_end=codes.shape[-1],
                    deterministic=True,  # CRITICAL
                )
                prompted_codes, prompt_mask = audio_prompt_sampler.apply_prompts(
                    codes, prompt_samples
                )

            # Compute speaker embedding from reference audio
            if speaker_encoder is not None and batch.reference_audio is not None:
                speaker_embedding = speaker_encoder(
                    batch.reference_audio,  # [B, T_ref] at 16kHz
                )

        # Forward pass with conditioning
        output = model(
            codes=prompted_codes,
            condition_tensors=condition_tensors,
            speaker_embedding=speaker_embedding,  # NEW
            prompt_mask=prompt_mask,  # NEW
        )
```

### 4.3 Sample Saver Integration

#### `finetune/monitoring/sample_saver.py` Enhancements

```python
def save_samples(
    self,
    codes: torch.Tensor,
    output: LMModelOutput,
    step: int,
    split: str,
    speaker_embedding: Optional[torch.Tensor] = None,  # NEW
    reference_audio: Optional[torch.Tensor] = None,  # NEW
    reference_text: Optional[str] = None,  # NEW
) -> None:
    """Save samples with speaker conditioning metadata.

    New files saved when speaker conditioning is enabled:
        - sample_XX_reference.wav: Reference audio used for conditioning
        - sample_XX_reference.txt: Reference text (if available)
        - sample_XX_metadata.json: Speaker embedding norm, conditioning info
    """

    # Save reference audio
    if reference_audio is not None:
        ref_path = save_dir / f"sample_{idx:02d}_reference.wav"
        torchaudio.save(ref_path, reference_audio, 24000)

    # Save metadata
    metadata = {
        "speaker_embedding_norm": speaker_embedding.norm().item() if speaker_embedding else None,
        "reference_duration_sec": reference_audio.shape[-1] / 24000 if reference_audio else None,
        "conditioning_method": args.speaker.method,
    }
```

---

## 5. Rust Backend Inference Architecture

### 5.1 Current Rust Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CURRENT RUST BACKEND ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  moshi-backend/src/stream_both.rs                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Config {                                                        │   │
│  │    instance_name: String,                                        │   │
│  │    lm_model_file: String,                                        │   │
│  │    mimi_model_file: String,                                      │   │
│  │    text_tokenizer_file: String,                                  │   │
│  │    // NO speaker conditioning fields                             │   │
│  │  }                                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  moshi-core/src/conditioner.rs                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  enum Condition {                                                │   │
│  │    AddToInput(Tensor),  // Ready for speaker embedding!         │   │
│  │  }                                                               │   │
│  │                                                                  │   │
│  │  ConditionProvider {                                             │   │
│  │    conditioners: HashMap<String, Conditioner>,                   │   │
│  │    // Lut: Categorical conditioning                              │   │
│  │    // ContinuousAttribute: Scalar conditioning                   │   │
│  │    // MISSING: Speaker embedding conditioner                     │   │
│  │  }                                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Proposed Rust Architecture Extensions

```
┌─────────────────────────────────────────────────────────────────────────┐
│               PROPOSED RUST SPEAKER CONDITIONING ARCHITECTURE           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  NEW: moshi-core/src/speaker_conditioner.rs                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  /// Speaker Conditioner for Zero-Shot Voice Cloning             │   │
│  │  pub struct SpeakerConditioner {                                 │   │
│  │      encoder: SpeakerEncoder,        // W2v-BERT 2.0 or ECAPA   │   │
│  │      projection: Linear,             // D_spk → D_model          │   │
│  │      scale: f32,                     // Initial scale (0.1)     │   │
│  │      layer_norm: Option<LayerNorm>,  // Optional normalization  │   │
│  │  }                                                               │   │
│  │                                                                  │   │
│  │  impl SpeakerConditioner {                                       │   │
│  │      pub fn condition(&self, audio: &Tensor) -> Result<Condition>│   │
│  │      {                                                           │   │
│  │          // 1. Extract speaker embedding                         │   │
│  │          let embedding = self.encoder.forward(audio)?;           │   │
│  │          // 2. Project to model dimension                        │   │
│  │          let projected = self.projection.forward(&embedding)?;   │   │
│  │          // 3. Apply scale and layer norm                        │   │
│  │          let scaled = (projected * self.scale)?;                 │   │
│  │          let output = match &self.layer_norm {                   │   │
│  │              Some(ln) => ln.forward(&scaled)?,                   │   │
│  │              None => scaled,                                      │   │
│  │          };                                                       │   │
│  │          // 4. Return as AddToInput condition                    │   │
│  │          Ok(Condition::AddToInput(output.unsqueeze(1)?))         │   │
│  │      }                                                           │   │
│  │  }                                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  MODIFIED: moshi-backend/src/stream_both.rs                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]   │   │
│  │  pub struct Config {                                             │   │
│  │      // ... existing fields ...                                  │   │
│  │                                                                  │   │
│  │      // NEW: Speaker conditioning                                │   │
│  │      pub speaker_encoder_file: Option<String>,                   │   │
│  │      pub speaker_conditioner_file: Option<String>,               │   │
│  │      pub default_reference_audio: Option<String>,                │   │
│  │      pub default_reference_text: Option<String>,                 │   │
│  │      pub reference_duration_sec: f32,  // Default: 10.0          │   │
│  │  }                                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  NEW: WebSocket Message Types                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  pub enum MsgType {                                              │   │
│  │      Handshake = 0,                                              │   │
│  │      Audio = 1,                                                  │   │
│  │      Text = 2,                                                   │   │
│  │      Control = 3,                                                │   │
│  │      Metadata = 4,                                               │   │
│  │      Error = 5,                                                  │   │
│  │      Ping = 6,                                                   │   │
│  │      SpeakerReference = 7,  // NEW: Upload reference audio       │   │
│  │  }                                                               │   │
│  │                                                                  │   │
│  │  #[derive(serde::Deserialize)]                                   │   │
│  │  pub struct SpeakerReferenceReq {                                │   │
│  │      pub audio_base64: Option<String>,  // Base64 encoded PCM    │   │
│  │      pub audio_path: Option<String>,    // File path on server   │   │
│  │      pub reference_text: Option<String>,// Optional text         │   │
│  │      pub use_default: bool,             // Use config default    │   │
│  │  }                                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Reference Input Methods

#### Method 1: File Path (Server-Side)
```json
{
  "type": "SpeakerReference",
  "audio_path": "/path/to/reference.wav",
  "reference_text": "안녕하세요, 저는 한국어 음성 합성 모델입니다."
}
```

#### Method 2: Base64 Upload (Browser)
```json
{
  "type": "SpeakerReference",
  "audio_base64": "UklGRiQAAABXQVZFZm10...",
  "reference_text": "참조 텍스트"
}
```

#### Method 3: Default Reference (Config)
```json
{
  "type": "SpeakerReference",
  "use_default": true
}
```

### 5.4 Rust Backend Config JSON

```json
{
  "instance_name": "k-moshi-korean",
  "lm_model_file": "${MODEL_DIR}/lm_model.safetensors",
  "mimi_model_file": "${MODEL_DIR}/mimi.safetensors",
  "text_tokenizer_file": "${MODEL_DIR}/tokenizer.model",

  "speaker_encoder_file": "${MODEL_DIR}/speaker_encoder.safetensors",
  "speaker_conditioner_file": "${MODEL_DIR}/speaker_conditioner.safetensors",
  "default_reference_audio": "${MODEL_DIR}/default_reference.wav",
  "default_reference_text": "안녕하세요, K-Moshi 입니다.",
  "reference_duration_sec": 10.0,

  "lm_config": {
    "text_start_token": 32000,
    "text_pad_token": 3,
    "text_eop_token": 0
  }
}
```

---

## 6. Web Frontend Integration

### 6.1 Reference Audio Upload Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WEB FRONTEND REFERENCE UPLOAD FLOW                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │  User Browser   │    │  K-Moshi        │    │  Rust Backend   │     │
│  │                 │    │  Frontend       │    │                 │     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│           │                      │                      │              │
│           │  1. Select file      │                      │              │
│           │  (drag & drop or     │                      │              │
│           │   file picker)       │                      │              │
│           │─────────────────────▶│                      │              │
│           │                      │                      │              │
│           │                      │  2. Read file as     │              │
│           │                      │     ArrayBuffer      │              │
│           │                      │  3. Convert to       │              │
│           │                      │     Base64           │              │
│           │                      │                      │              │
│           │                      │  4. WebSocket send   │              │
│           │                      │  { type: 7,          │              │
│           │                      │    audio_base64: ... │              │
│           │                      │    reference_text:.. │              │
│           │                      │  }                   │              │
│           │                      │─────────────────────▶│              │
│           │                      │                      │              │
│           │                      │                      │  5. Decode   │
│           │                      │                      │     Base64   │
│           │                      │                      │  6. Resample │
│           │                      │                      │     to 16kHz │
│           │                      │                      │  7. Extract  │
│           │                      │                      │     embedding│
│           │                      │                      │  8. Cache    │
│           │                      │                      │     condition│
│           │                      │                      │              │
│           │                      │  9. Ack message      │              │
│           │                      │◀─────────────────────│              │
│           │                      │                      │              │
│           │  10. "Voice ready"   │                      │              │
│           │◀─────────────────────│                      │              │
│           │                      │                      │              │
│           │  11. Start speaking  │                      │              │
│           │─────────────────────▶│                      │              │
│           │                      │  12. Audio stream    │              │
│           │                      │      + condition     │              │
│           │                      │─────────────────────▶│              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Frontend Code Example (TypeScript)

```typescript
// serving/client/src/lib/speakerReference.ts

interface SpeakerReferenceRequest {
  type: 'SpeakerReference';
  audio_base64?: string;
  audio_path?: string;
  reference_text?: string;
  use_default?: boolean;
}

export async function uploadSpeakerReference(
  websocket: WebSocket,
  audioFile: File,
  referenceText?: string,
): Promise<void> {
  // 1. Read file as ArrayBuffer
  const arrayBuffer = await audioFile.arrayBuffer();

  // 2. Convert to Base64
  const base64 = btoa(
    String.fromCharCode(...new Uint8Array(arrayBuffer))
  );

  // 3. Send via WebSocket
  const message: SpeakerReferenceRequest = {
    type: 'SpeakerReference',
    audio_base64: base64,
    reference_text: referenceText,
  };

  // 4. Send as binary message (MsgType = 7)
  const msgType = new Uint8Array([7]);
  const payload = new TextEncoder().encode(JSON.stringify(message));
  const combined = new Uint8Array(msgType.length + payload.length);
  combined.set(msgType);
  combined.set(payload, msgType.length);

  websocket.send(combined);
}

export function useDefaultReference(websocket: WebSocket): void {
  const message: SpeakerReferenceRequest = {
    type: 'SpeakerReference',
    use_default: true,
  };

  // Same binary format
  const msgType = new Uint8Array([7]);
  const payload = new TextEncoder().encode(JSON.stringify(message));
  const combined = new Uint8Array(msgType.length + payload.length);
  combined.set(msgType);
  combined.set(payload, msgType.length);

  websocket.send(combined);
}
```

---

## 7. Implementation Roadmap

### Phase 1: Python Training/Eval Integration (Week 1-2)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Add `deterministic` mode to AudioPromptSampler | P0 | 4h | None |
| Implement "start" strategy properly | P0 | 4h | None |
| Modify eval.py to accept speaker_embedding | P0 | 8h | Sampler changes |
| Enhance sample_saver.py for reference saving | P1 | 4h | eval.py changes |
| Add eval YAML examples | P1 | 2h | All above |

### Phase 2: Rust Backend Speaker Conditioning (Week 3-4)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Create speaker_conditioner.rs module | P0 | 16h | None |
| Add SpeakerReference message type | P0 | 8h | conditioner module |
| Extend Config with speaker fields | P0 | 4h | None |
| Implement default reference loading | P1 | 4h | Config extension |
| Add Base64 audio decoding | P1 | 4h | Message type |

### Phase 3: Web Frontend Integration (Week 5-6)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Add reference audio upload UI | P0 | 8h | Rust backend |
| Implement drag & drop file upload | P1 | 4h | Upload UI |
| Add reference text input field | P1 | 4h | Upload UI |
| "Use Default Voice" button | P2 | 2h | All above |
| Voice preview before conversation | P2 | 8h | All above |

### Phase 4: Testing & Documentation (Week 7)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| E2E deterministic evaluation tests | P0 | 8h | Phase 1 |
| Rust unit tests for speaker conditioning | P0 | 8h | Phase 2 |
| Integration tests (frontend → backend) | P1 | 8h | Phase 3 |
| User documentation | P1 | 4h | All above |

---

## 8. Key Design Decisions

### 8.1 Deterministic Protocol Summary

| Aspect | Training | Evaluation/Inference |
|--------|----------|---------------------|
| Sample Strategy | `random` | `start` |
| Duration | Random in [min, max] | Fixed `fixed_duration_sec` |
| Position | Random from valid regions | Always 0 (file start) |
| Random Seed | N/A (truly random) | N/A (no randomness) |

### 8.2 Reference Input Priority

1. **WebSocket Upload** (highest): User-provided reference via browser
2. **Session Config**: Reference specified in session init
3. **Config File Default** (lowest): Fallback reference audio/text

### 8.3 Condition Application

```rust
// Rust pseudocode for condition application
fn forward_with_speaker(&self, input: Tensor, speaker_condition: Option<Condition>) -> Tensor {
    let mut hidden = self.embed(input);

    // Apply speaker condition via AddToInput
    if let Some(Condition::AddToInput(cond)) = speaker_condition {
        hidden = hidden + cond.broadcast_to(hidden.shape())?;
    }

    // Continue with transformer layers
    self.transformer.forward(hidden)
}
```

---

## 9. Testing Strategy

### 9.1 Deterministic Evaluation Test

```python
# tests/test_deterministic_eval.py

def test_deterministic_reference_sampling():
    """Ensure same input always produces same reference."""

    config = AudioPromptConfig(
        enable=True,
        mode="audio_text",
        sample_strategy="start",
        deterministic=True,
        fixed_duration_sec=10.0,
    )
    sampler = AudioPromptSampler(config)

    # Same codes
    codes = torch.randn(9, 1000)  # [9, T]

    # Sample multiple times
    sample1 = sampler.sample_single(codes, deterministic=True)
    sample2 = sampler.sample_single(codes, deterministic=True)
    sample3 = sampler.sample_single(codes, deterministic=True)

    # All must be identical
    assert sample1.start_idx == sample2.start_idx == sample3.start_idx == 0
    assert sample1.end_idx == sample2.end_idx == sample3.end_idx
    assert torch.equal(sample1.audio_codes, sample2.audio_codes)
    assert torch.equal(sample1.audio_codes, sample3.audio_codes)
```

### 9.2 Rust Speaker Conditioning Test

```rust
// moshi-core/src/speaker_conditioner_test.rs

#[test]
fn test_speaker_conditioning_deterministic() {
    let conditioner = SpeakerConditioner::load("test_model.safetensors").unwrap();

    // Same reference audio
    let audio = Tensor::randn(&[1, 160000], DType::F32, &Device::Cpu).unwrap();

    // Condition multiple times
    let cond1 = conditioner.condition(&audio).unwrap();
    let cond2 = conditioner.condition(&audio).unwrap();

    // Must be identical (no randomness in inference)
    match (cond1, cond2) {
        (Condition::AddToInput(t1), Condition::AddToInput(t2)) => {
            assert!(tensors_equal(&t1, &t2));
        }
    }
}
```

---

## 10. References

### Internal Documentation
- [ZERO_SHOT_SPEAKER_CONDITIONING_SPECIFICATION.md](./ZERO_SHOT_SPEAKER_CONDITIONING_SPECIFICATION.md)
- [SPEAKER_CONDITIONING_ARCHITECTURE_DIAGRAM.md](./SPEAKER_CONDITIONING_ARCHITECTURE_DIAGRAM.md)
- [STAGE_PRETRAINED_LOADING_DESIGN.md](./STAGE_PRETRAINED_LOADING_DESIGN.md)

### External References
- [Moshi Paper](https://arxiv.org/abs/2410.00037)
- [J-Moshi Paper](https://arxiv.org/abs/2506.02979)
- [W2v-BERT 2.0 SV](https://arxiv.org/abs/2510.04213)
- [PersonaPlex Paper](https://arxiv.org/abs/2407.13447)

---

## Appendix A: Full Configuration Example

### A.1 Training Configuration (Random Sampling)
```yaml
# example/korean_moshi_stage2_speaker_train.yaml
speaker:
  enabled: true
  method: "both"

  encoder:
    encoder_type: "w2v_bert2"
    pretrained_path: "/path/to/w2v-bert-2.0_SV/model.pth"
    freeze: true
    output_dim: 256

  conditioner:
    output_dim: 4096
    initial_scale: 0.1
    use_layernorm: true

  audio_prompt:
    enable: true
    mode: "audio_text"
    min_duration_sec: 10.0
    max_duration_sec: 15.0
    sample_strategy: "random"      # Random for training
    deterministic: false           # Allow randomness
    avoid_overlap: true            # Avoid training segment
```

### A.2 Evaluation Configuration (Deterministic Sampling)
```yaml
# example/korean_moshi_stage2_speaker_eval.yaml
speaker:
  enabled: true
  method: "both"

  encoder:
    encoder_type: "w2v_bert2"
    pretrained_path: "/path/to/w2v-bert-2.0_SV/model.pth"
    freeze: true
    output_dim: 256

  conditioner:
    output_dim: 4096
    initial_scale: 0.1
    use_layernorm: true

  audio_prompt:
    enable: true
    mode: "audio_text"
    fixed_duration_sec: 10.0       # FIXED duration
    sample_strategy: "start"       # Always from start
    deterministic: true            # NO randomness
    avoid_overlap: false           # N/A for eval
```

---

*Last Updated: 2026-01-23*
*Document Version: 1.0.0*
