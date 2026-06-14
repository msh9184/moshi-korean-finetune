# Full-Duplex Speech Model 아키텍처 심층 분석

> **분석 일자**: 2026-01-20
> **목적**: K-Moshi 아키텍처 설계를 위한 최신 Full-Duplex 모델 비교 분석
> **분석 모델**: Moshi, PersonaPlex, F-actor, LUCY, VITA-Audio, Fun-Audio-Chat, MiMo-Audio

---

## 1. Executive Summary

### 1.1 핵심 발견

| 모델 | Voice Control | System Prompt | Full-Duplex 방식 | 학습 데이터 | 특이점 |
|------|--------------|---------------|-----------------|------------|--------|
| **Moshi** | ❌ 없음 (데이터 분포 의존) | ❌ | Dual-stream (User+System) | 7M시간 | 원조, RVQ+Depformer |
| **PersonaPlex** | ✅ Voice Embedding (.pt) | ✅ Text Prompt | Moshi 기반 | 3,467시간 | Hybrid Prompting |
| **F-actor** | ✅ ECAPA-TDNN | ✅ Instruction Prefix | FSQ (병렬 예측) | 2,000시간 | 단일 스테이지 학습 |
| **LUCY** | ❌ Speaker Label만 | ❌ | 미공개 | AudioQA-1M | 감정 제어 |
| **VITA-Audio** | ❌ 미지원 | Task Prompt | MCTP (병렬 토큰) | 미공개 | 3-5x 속도 향상 |
| **Fun-Audio-Chat** | CosyVoice (별도 TTS) | ❌ | Dual-Resolution (5Hz+25Hz) | Core-Cocktail | 50% GPU 절감 |
| **MiMo-Audio** | RVQ 토큰 기반 | ❌ | Patch Encoder (6.25Hz) | 100M시간 | Few-shot 학습 |

### 1.2 K-Moshi를 위한 핵심 인사이트

1. **F-actor가 가장 유망한 참조 모델**: 2,000시간 데이터로 instruction-following 달성
2. **Speaker Embedding 방식**: ECAPA-TDNN이 가장 검증된 방법
3. **FSQ vs RVQ**: FSQ가 병렬 예측에 유리하여 효율적
4. **System Prompt 통합**: Prefix token으로 concatenation하는 것이 효과적

---

## 2. Moshi 원본 아키텍처 분석

### 2.1 핵심 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Moshi Architecture                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: codes [B, K, T]  (K = 1 text + 8 audio codebooks = 9 total)        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Embedding Layer                               │    │
│  │  ┌──────────────┐  ┌──────────────────────────────────────────────┐ │    │
│  │  │ text_emb     │  │ emb[0..7] (audio codebook embeddings)        │ │    │
│  │  │ (32K vocab)  │  │ (1024 vocab each)                            │ │    │
│  │  └──────────────┘  └──────────────────────────────────────────────┘ │    │
│  │                                                                     │    │
│  │  input_ = text_emb + sum(emb[i] for i in 0..7)  # 모두 더함        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Main Transformer (7B)                             │    │
│  │  - d_model: 4096, num_heads: 32, num_layers: 32                     │    │
│  │  - Streaming Transformer with causal attention                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ├──────────────────────┐                       │
│                              ▼                      ▼                       │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │    text_linear                  │  │         Depformer               │  │
│  │  (dim → text_card = 32K)        │  │  - 8 output codebooks           │  │
│  │  → Text Logits                  │  │  - Per-step processing          │  │
│  └─────────────────────────────────┘  │  → Audio Logits                 │  │
│                                       └─────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Moshi의 Voice Control 부재 이유

```python
# moshi/models/lm.py:390-398
for cb_index in range(self.num_audio_codebooks):
    audio_emb = self.emb[cb_index](
        input_sequence[:, cb_index + self.audio_offset]
    )
    input_ = audio_emb if input_ is None else input_ + audio_emb
text_emb = self.text_emb(input_sequence[:, 0])
input_ = text_emb if input_ is None else input_ + text_emb
```

**문제점**:
- 모든 입력이 단순 합산 (`input_ + audio_emb`)
- **Speaker identity를 명시적으로 제어할 수 있는 메커니즘 없음**
- 음성 특성은 **학습 데이터의 분포에 전적으로 의존**
- Text stream에도 speaker 정보 없음

### 2.3 Moshi Data Format

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Stereo WAV (24kHz)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Left Channel (SPEAKER_MAIN = Moshi)    Right Channel (SPEAKER_USER)       │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐ │
│  │ AI가 생성할 음성                │    │ 사용자 입력 음성               │ │
│  │ - 단일 화자로 학습됨            │    │ - 다양한 화자 가능             │ │
│  │ - 목소리 = 데이터 분포          │    │ - Inference: 마이크 입력       │ │
│  └─────────────────────────────────┘    └─────────────────────────────────┘ │
│                                                                             │
│  → Mimi Encoder → codes [B, 9, T] (1 text + 8 audio)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. PersonaPlex 아키텍처 분석

### 3.1 Voice Embedding 메커니즘

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PersonaPlex Hybrid Prompting                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Voice Prompt (Audio Tokens)                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ load_voice_prompt("NATF2.pt")                                       │    │
│  │                                                                     │    │
│  │ Option A: Audio File → Mimi Encode → Sequential Replay              │    │
│  │ Option B: Pre-computed Embedding (.pt) → Direct Cache Load          │    │
│  │                                                                     │    │
│  │ _step_voice_prompt_core():                                          │    │
│  │   - Replay cached embeddings through model                          │    │
│  │   - state.cache.copy_(self.voice_prompt_cache)  ← 캐시 동기화      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  2. Text Prompt (Role Definition)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ text_prompt_tokens = tokenize("You work for First Neuron Bank...")  │    │
│  │                                                                     │    │
│  │ Processing Order:                                                   │    │
│  │   voice_prompt → audio_silence → text_prompt → audio_silence        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  → Voice + Text jointly condition all subsequent generation                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 PersonaPlex의 Voice Embedding 저장 방식

```python
# PersonaPlex Voice Embeddings (18개)
voices = {
    # Natural (대화 품질 최적화)
    "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",  # Female
    "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",  # Male

    # Variety (음향 다양성)
    "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",  # Female
    "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",  # Male
}

# 저장 형식: Mimi codec으로 인코딩된 audio tokens
# 사용 시: LM의 cache에 직접 로드하여 conditioning
```

### 3.3 PersonaPlex의 한계

1. **Voice Embedding이 고정됨**: 새로운 화자 추가 시 재학습 필요
2. **Moshi 아키텍처 종속**: 기본 구조 변경 불가
3. **학습 데이터 필요**: 각 voice embedding에 대한 학습 데이터 필요

---

## 4. F-actor 아키텍처 분석 (핵심!)

### 4.1 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         F-actor Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   Instruction Prefix                                 │    │
│  │  ┌─────────────────────────┐   ┌─────────────────────────────────┐  │    │
│  │  │ Speaker Embedding       │ + │ Text Instruction                 │  │    │
│  │  │ (ECAPA-TDNN, 5초 샘플)  │   │ (Role, Topic, Behavior)          │  │    │
│  │  │ → LLM token space 투영  │   │ → Tokenized                      │  │    │
│  │  └─────────────────────────┘   └─────────────────────────────────┘  │    │
│  │                              │                                      │    │
│  │                              ▼                                      │    │
│  │  [Speaker_Emb] [Text_Tokens...] ← Concatenated Prefix               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                Audio Streams (FSQ Quantization)                      │    │
│  │                                                                     │    │
│  │  User Stream:   [DAU₁ᵘ] [DAU₂ᵘ] [DAU₃ᵘ] [DAU₄ᵘ] (4 codebooks)     │    │
│  │  System Stream: [DAU₁ˢ] [DAU₂ˢ] [DAU₃ˢ] [DAU₄ˢ] (4 codebooks)     │    │
│  │                                                                     │    │
│  │  Embedding: x = Σ(user,sys) Σ(i=1..4) Embed(DAUᵢˢ)                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              LLM Backbone (Llama3.2-1B-Instruct)                     │    │
│  │              - Audio encoder FROZEN                                 │    │
│  │              - LLM만 fine-tuning                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    8 Linear Output Heads                             │    │
│  │  User: [Head₁ᵘ] [Head₂ᵘ] [Head₃ᵘ] [Head₄ᵘ] (학습 시에만 사용)      │    │
│  │  System: [Head₁ˢ] [Head₂ˢ] [Head₃ˢ] [Head₄ˢ] (추론 시 사용)        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 F-actor의 핵심 혁신

#### 4.2.1 Speaker Embedding 방식

```python
# F-actor Speaker Embedding Process
# 1. ECAPA-TDNN으로 화자 임베딩 추출 (5초 샘플)
speaker_embedding = ecapa_tdnn(reference_audio[:5_seconds])

# 2. LLM token space로 투영
projected_speaker = speaker_projection_layer(speaker_embedding)  # [1, dim]

# 3. Text instruction과 concatenation
instruction_tokens = tokenize("""
    Conversation narrative: Customer service call about account inquiry
    Initiation: System starts
    Backchannel frequency: Medium
    Voice: Use provided speaker embedding
""")

# 4. Prefix 구성
prefix = torch.cat([projected_speaker, instruction_tokens], dim=1)
```

#### 4.2.2 FSQ vs RVQ 선택 이유

| 특성 | RVQ (Moshi) | FSQ (F-actor) |
|------|-------------|---------------|
| 예측 방식 | 순차적 (Depformer) | 병렬 가능 |
| 계산 효율 | 낮음 (8단계) | 높음 (1단계) |
| 품질 | 높음 | 약간 낮음 |
| 학습 복잡도 | 높음 | 낮음 |

**F-actor 선택**: FSQ를 사용하여 Depformer 없이 병렬 예측 → 효율성 대폭 향상

#### 4.2.3 Instruction Following 능력

```yaml
# F-actor가 제어 가능한 요소들
controllable_dimensions:
  - narrative_adherence: "대화 시나리오 준수"
  - initiation_control: "누가 먼저 말할지"
  - backchannel_frequency: "맞장구 빈도"
  - interruption_frequency: "끼어들기 빈도"
  - speaker_voice: "목소리 특성"

# 성능 결과
performance:
  initiation_accuracy: 100%  # (text stream 포함 시)
  backchannel_correlation: 0.54
  interruption_correlation: 0.25
  voice_similarity: 0.54  # (target vs generated)
```

### 4.3 F-actor 학습 효율성

```yaml
# F-actor Training Recipe
training:
  data: "Behavior-SD (2,164 hours)"
  gpu: "4x A100-40GB"
  time: "~48 hours"
  steps: "100,000 (early stopping)"

  frozen:
    - audio_encoder: "Nemo-nano-codec-22khz"

  trainable:
    - llm_backbone: "Llama3.2-1B-Instruct"
    - output_heads: "8 linear layers"
    - speaker_projection: "Linear layer"

# 핵심: 2,000시간 데이터로 instruction-following 달성!
```

---

## 5. 기타 모델 분석

### 5.1 LUCY

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LUCY: 감정 제어 + 외부 도구 사용                                        │
├─────────────────────────────────────────────────────────────────────────┤
│ - Speaker label만 사용 (F10, M05 등)                                    │
│ - SNAC codec 사용                                                       │
│ - 3-stage 학습: Audio alignment → AudioQA → Emotion/Function calling   │
│ - Full-duplex 구현 방식 미공개                                          │
│ - Voice embedding 메커니즘 없음                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 VITA-Audio

```
┌─────────────────────────────────────────────────────────────────────────┐
│ VITA-Audio: MCTP로 병렬 토큰 생성                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ - Multiple Cross-modal Token Prediction (MCTP) 모듈                     │
│ - 한 번의 forward pass로 다수 audio token 생성                          │
│ - 3-5x inference speedup                                                │
│ - First-audio latency: 236ms → 53ms                                     │
│ - 4-stage 학습: Alignment → Single MCTP → Multi MCTP → SFT              │
│ - Speaker control 없음                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Fun-Audio-Chat

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Fun-Audio-Chat: Dual-Resolution 효율화                                  │
├─────────────────────────────────────────────────────────────────────────┤
│ - 5Hz shared backbone + 25Hz refined head                               │
│ - 50% GPU 절감 vs 12.5Hz/25Hz 모델                                     │
│ - Voice: Fun-CosyVoice3 (별도 TTS 모델)                                │
│ - Core-Cocktail 학습 방법론                                            │
│ - Duplex 세부 구현 미공개                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4 MiMo-Audio

```
┌─────────────────────────────────────────────────────────────────────────┐
│ MiMo-Audio: Patch 기반 효율적 처리                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ - Patch Encoder: 4 timesteps → 1 patch (6.25Hz)                        │
│ - Patch Decoder: Autoregressive 25Hz 복원                              │
│ - 100M+ 시간 데이터로 few-shot 학습 능력                               │
│ - Voice conversion, style transfer 가능 (RVQ 토큰 기반)                │
│ - Full-duplex 대화 미지원 (sequential processing)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 아키텍처 비교 종합

### 6.1 Voice/Speaker Control 방식 비교

| 모델 | 방식 | 장점 | 단점 |
|------|------|------|------|
| **Moshi** | 없음 (데이터 의존) | 단순함 | 제어 불가 |
| **PersonaPlex** | Audio token cache | Moshi 호환 | 고정된 voice set |
| **F-actor** | ECAPA-TDNN + Projection | 유연함, 효율적 | 추가 모델 필요 |
| **LUCY** | Speaker label | 간단함 | 연속적 제어 불가 |
| **Fun-Audio-Chat** | 별도 TTS | 고품질 | 레이턴시 증가 |

### 6.2 Full-Duplex 구현 방식 비교

| 모델 | 방식 | User Stream 처리 | System Stream 처리 |
|------|------|------------------|-------------------|
| **Moshi** | Dual-stream RVQ | 별도 codebook | 별도 codebook |
| **PersonaPlex** | Moshi 동일 | Moshi 동일 | Moshi 동일 |
| **F-actor** | Dual-stream FSQ | 4 codebooks | 4 codebooks |
| **VITA-Audio** | MCTP | 단일 스트림 | 병렬 토큰 |
| **Fun-Audio-Chat** | Dual-Resolution | 5Hz | 25Hz |

### 6.3 학습 효율성 비교

| 모델 | 데이터 | GPU | 시간 | 특이점 |
|------|--------|-----|------|--------|
| **Moshi** | 7M시간 | 미공개 | 미공개 | 대규모 사전학습 |
| **PersonaPlex** | 3,467시간 | 미공개 | 미공개 | Fine-tuning |
| **F-actor** | 2,000시간 | 4xA100-40G | 48시간 | **가장 효율적** |
| **LUCY** | AudioQA-1M | 미공개 | 미공개 | 3-stage |
| **MiMo-Audio** | 100M시간 | 미공개 | 미공개 | 대규모 |

---

## 7. K-Moshi 아키텍처 설계 제안

### 7.1 Option A: F-actor 스타일 (권장)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    K-Moshi Option A: F-actor 스타일                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Korean Speaker Embedding]     +     [Korean Text Instruction]             │
│         │                                      │                            │
│         │ ECAPA-TDNN                          │ Korean Tokenizer            │
│         │ (한국어 화자 특화)                   │ (Custom 또는 기존)          │
│         ▼                                      ▼                            │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │              Concatenated Instruction Prefix                       │      │
│  │  [Spk_Emb] [당신은 케이모시입니다...] [대화를 시작하세요]         │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                   Audio Streams (FSQ 또는 RVQ)                     │      │
│  │  User: [Korean input audio tokens]                                │      │
│  │  System: [K-Moshi output audio tokens]                            │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │           LLM Backbone (Moshi 7B or Korean LLM)                    │      │
│  │           - Audio Encoder: Mimi (frozen)                          │      │
│  │           - LLM: Fine-tuning with LoRA                            │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                    Output Heads                                    │      │
│  │  Text: Korean text logits                                         │      │
│  │  Audio: Korean audio codebook logits                              │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

장점:
- 2,000시간 수준 데이터로 학습 가능
- Speaker embedding으로 화자 제어
- Instruction following 능력 획득
- 기존 Moshi 호환 가능

단점:
- ECAPA-TDNN 추가 필요
- FSQ 사용 시 아키텍처 수정 필요
```

### 7.2 Option B: PersonaPlex 스타일 (보수적)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  K-Moshi Option B: PersonaPlex 스타일                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. K-Moshi Voice Embedding 생성 (1회)                                      │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │  - 한국어 참조 음성 (10-30초)                                      │      │
│  │  - Mimi encode → audio tokens → .pt 저장                          │      │
│  │  - KMOSHI_V1.pt (단일 화자)                                       │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  2. Text Prompt 정의                                                        │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │  "당신은 케이모시(K-Moshi)입니다.                                  │      │
│  │   한국어 AI 음성 비서입니다.                 │      │
│  │   친절하고 따뜻한 말투를 사용합니다..."                            │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  3. Inference Time                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │  load_voice_prompt("KMOSHI_V1.pt")                                │      │
│  │  set_text_prompt(korean_instruction)                              │      │
│  │  → 원본 Moshi 아키텍처 그대로 사용                                │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

장점:
- Moshi 아키텍처 수정 최소화
- 검증된 PersonaPlex 방식 활용
- 빠른 구현 가능

단점:
- Voice 다양성 제한 (고정 embedding)
- Instruction following 능력 약함
- 새 화자 추가 시 재학습 필요
```

### 7.3 Option C: 하이브리드 (최적)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    K-Moshi Option C: 하이브리드 (권장)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: PersonaPlex 스타일로 빠른 프로토타입                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │  - 기존 Moshi 아키텍처 유지                                        │      │
│  │  - K-Moshi voice embedding (.pt) 생성                             │      │
│  │  - 한국어 데이터로 fine-tuning                                     │      │
│  │  - 목표: 기본 한국어 대화 능력                                     │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                              │                                              │
│                              ▼                                              │
│  Phase 2: F-actor 스타일로 확장                                             │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │  - ECAPA-TDNN speaker encoder 추가                                │      │
│  │  - Instruction prefix 시스템 구현                                 │      │
│  │  - Instruction-following 데이터 추가                              │      │
│  │  - 목표: 화자 제어 + 행동 제어                                     │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                              │                                              │
│                              ▼                                              │
│  Phase 3: 최적화 (선택)                                                     │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │  - FSQ 전환 검토 (RVQ → FSQ)                                      │      │
│  │  - MCTP 모듈 적용 검토 (VITA-Audio 참고)                          │      │
│  │  - Dual-Resolution 검토 (Fun-Audio-Chat 참고)                     │      │
│  │  - 목표: 추론 속도 최적화                                          │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 데이터셋 구축 전략

### 8.1 F-actor 참고 데이터 구성

```yaml
# F-actor Behavior-SD 데이터셋 구성
behavior_sd:
  total_hours: 2,164
  format: "Synthetic dialogues with behavioral annotations"

  annotations:
    - conversation_narrative: "대화 시나리오 설명"
    - initiation: "System 또는 User"
    - backchannel_count: "맞장구 횟수"
    - interruption_count: "끼어들기 횟수"
    - speaker_id: "52명의 화자 중 선택"

  processing:
    - alignment: "Kaldi forced alignment"
    - filtering: "74.4% 유지 (text-speech 일치)"
    - narrative_rewriting: "Gemma-1.1-7b로 시스템 관점 변환"
```

### 8.2 K-Moshi 데이터셋 제안

```yaml
# K-Moshi Dataset Strategy (F-actor 참고)
k_moshi_dataset:
  target_hours: 1,000  # F-actor의 절반 수준으로 시작

  composition:
    real_korean_dialogue:
      ratio: 40%
      hours: 400
      source: "KSponSpeech, AI Hub"
      processing: "Back-annotation for instruction"

    instruction_following:
      ratio: 30%
      hours: 300
      type: "Synthetic with behavioral control"
      annotations:
        - narrative: "시나리오 설명"
        - speaker_control: "화자 특성"
        - behavior_control: "대화 행동"

    customer_service:
      ratio: 20%
      hours: 200
      scenarios: "은행, 병원, 음식점 등"

    identity_qa:
      ratio: 10%
      hours: 100
      content: "K-Moshi 자기소개, 능력 설명"

  speaker_strategy:
    k_moshi_voice: "단일 참조 음성 (고정)"
    user_voices: "다양한 화자 (TTS 합성)"
    speaker_embedding: "ECAPA-TDNN 추출"
```

---

## 9. 구현 우선순위 및 로드맵

### 9.1 Phase 1: 기본 한국어 능력 (4-6주)

```
목표: PersonaPlex 스타일로 기본 한국어 대화 능력 확보

Task 1.1: K-Moshi Voice Embedding 생성
- 한국어 참조 음성 선정 (10-30초)
- Mimi encode → .pt 저장
- 품질 검증 (WER, Speaker Similarity)

Task 1.2: 한국어 데이터 준비
- KSponSpeech/AI Hub 데이터 확보
- Stereo WAV 변환 (L=AI, R=User)
- Whisper 전사 + 정렬

Task 1.3: Fine-tuning
- 기존 Moshi 아키텍처 사용
- LoRA fine-tuning
- 평가: 한국어 WER, 대화 품질
```

### 9.2 Phase 2: Speaker Control 추가 (4-6주)

```
목표: F-actor 스타일 speaker embedding 시스템 추가

Task 2.1: ECAPA-TDNN 통합
- 한국어 speaker embedding 모델 준비
- LLM token space projection layer 추가
- Speaker embedding extraction pipeline

Task 2.2: Instruction Prefix 시스템
- Korean instruction tokenizer
- Prefix concatenation 메커니즘
- Text prompt templates

Task 2.3: Instruction-following 데이터 생성
- Behavioral annotation 추가
- 다양한 시나리오 생성
- 화자 제어 데이터 구축
```

### 9.3 Phase 3: 최적화 (선택, 4-6주)

```
목표: 추론 속도 및 품질 최적화

Option A: FSQ 전환
- RVQ → FSQ 아키텍처 수정
- Depformer 제거 또는 간소화
- 병렬 예측 구현

Option B: MCTP 적용
- Multiple token prediction
- First-audio latency 감소
- Streaming 최적화

Option C: Dual-Resolution
- 5Hz backbone + 25Hz head
- GPU 메모리 최적화
```

---

## 10. 결론 및 권장사항

### 10.1 핵심 결론

1. **Voice Control의 핵심은 Speaker Embedding**
   - Moshi 원본: 제어 메커니즘 없음 → 데이터 분포 의존
   - F-actor: ECAPA-TDNN + Projection → 유연한 제어
   - PersonaPlex: Audio token cache → 고정된 voice set

2. **F-actor가 가장 효율적인 참조 모델**
   - 2,000시간 데이터로 instruction-following 달성
   - 단일 스테이지 학습 (48시간, 4xA100)
   - Audio encoder frozen → LLM만 학습

3. **K-Moshi 권장 접근법**
   - Phase 1: PersonaPlex 스타일로 빠른 프로토타입
   - Phase 2: F-actor 스타일로 speaker control 추가
   - Phase 3: 필요시 FSQ/MCTP 최적화

### 10.2 즉시 실행 가능한 작업

```yaml
immediate_actions:
  1_voice_embedding:
    - "K-Moshi 참조 음성 선정 (한국어 10-30초)"
    - "Mimi encode → KMOSHI_V1.pt 생성"

  2_data_preparation:
    - "KSponSpeech 데이터 확보 (수동 다운로드)"
    - "Stereo WAV 변환 파이프라인"
    - "Whisper 한국어 전사"

  3_baseline_training:
    - "기존 Moshi 아키텍처로 한국어 fine-tuning"
    - "LoRA 설정 (rank=128)"
    - "1,000시간 목표로 점진적 학습"
```

---

## 참고 문헌

1. [Moshi Paper](https://arxiv.org/abs/2410.00037)
2. [PersonaPlex](https://github.com/NVIDIA/personaplex)
3. [F-actor](https://arxiv.org/html/2601.11329)
4. [LUCY](https://github.com/VITA-MLLM/LUCY)
5. [VITA-Audio](https://github.com/VITA-MLLM/VITA-Audio)
6. [Fun-Audio-Chat](https://github.com/FunAudioLLM/Fun-Audio-Chat)
7. [MiMo-Audio](https://github.com/XiaomiMiMo/MiMo-Audio)
8. [FullDuplexBench](https://arxiv.org/abs/2503.04721)

---

*Last Updated: 2026-01-20*
*Document: Full-Duplex Speech Model Architecture Analysis*
*Author: K-Moshi Research Team*
