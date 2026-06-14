# K-Moshi Zero-Shot Speaker Ability 아키텍처 심층 분석

> **분석 일자**: 2026-01-20
> **목적**: Moshi 구조에서 Zero-shot Speaker Control 능력 구현 방안 분석
> **핵심 요구사항**: RVQ+Depformer (Mimi codec) 유지, Reference 음성(10-20초)으로 화자 특성 전달

---

## 1. Executive Summary

### 1.1 핵심 질문

```
Q: Moshi 아키텍처에서 reference 음성으로 zero-shot voice cloning이 가능한가?
A: 가능하다. 다만 여러 접근법의 trade-off를 신중히 고려해야 한다.
```

### 1.2 주요 발견

| 접근법 | 구현 복잡도 | 품질 기대 | Moshi 호환성 | 권장도 |
|--------|------------|----------|-------------|--------|
| **A: Speaker Encoder (ECAPA-TDNN)** | 중간 | 높음 | 높음 (sum_condition) | ⭐⭐⭐⭐⭐ |
| **B: Audio Token Prompt (VALL-E 스타일)** | 낮음 | 중간 | 매우 높음 (prepend) | ⭐⭐⭐⭐ |
| **C: Cross-Attention Speaker** | 높음 | 매우 높음 | 중간 | ⭐⭐⭐ |
| **D: Hybrid (A+B)** | 높음 | 매우 높음 | 높음 | ⭐⭐⭐⭐⭐ |

### 1.3 권장 전략

```
Phase 1: Audio Token Prompt (B) - 빠른 검증
Phase 2: Speaker Encoder (A) - 품질 향상
Phase 3: Hybrid (D) - 최적화
```

---

## 2. Zero-Shot TTS 기술 현황 분석

### 2.1 주요 모델 아키텍처 비교

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Zero-Shot TTS 아키텍처 스펙트럼                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│  │ VALL-E      │     │ CosyVoice2  │     │ Chatterbox  │     │ F-actor   │ │
│  │ (Microsoft) │     │ (Alibaba)   │     │ (Resemble)  │     │ (2025)    │ │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └─────┬─────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  Audio Token          Reference Audio     Speaker Embedding   Speaker Emb   │
│  Prompt (3초)         + Flow Matching     (Few seconds)       + Instruction │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│  특징:                                                                      │
│  ────────────────────────────────────────────────────────────────────────── │
│  • 8 codebooks        • ERes2Net speaker   • LLaMA backbone   • ECAPA-TDNN  │
│  • EnCodec            • Supervised tokens   • 350M-500M       • 투영 layer │
│  • AR + NAR           • CFM decoder         • Emotion control • Instruction │
│  • 60K hrs data       • 166K hrs data       • Cross-lingual   • 2K hrs data │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Speaker Conditioning 방식 분류

#### Type 1: Audio Token Prompt (VALL-E 계열)

```python
# VALL-E 방식: Reference audio를 codec token으로 변환하여 prefix로 사용
reference_audio = load_audio("speaker_ref.wav")  # 3-10초
reference_tokens = encodec.encode(reference_audio)  # [8, T_ref]

# Text tokens과 concatenation
text_tokens = tokenize(text)
prompt = concat([reference_tokens, BOS, text_tokens])

# Language model이 speaker 특성을 암묵적으로 학습
output_tokens = llm.generate(prompt)
```

**장점**:
- 구현 단순 (codec encoder만 필요)
- 별도 speaker encoder 학습 불필요
- Moshi의 기존 구조와 높은 호환성

**단점**:
- Reference 길이에 따른 context 소비
- Speaker 특성 추출이 암묵적 (명시적 제어 어려움)
- 긴 reference 필요 (품질 확보 위해)

#### Type 2: Speaker Encoder (F-actor, CosyVoice 계열)

```python
# Speaker Encoder 방식: 별도 모델로 speaker embedding 추출
speaker_encoder = ECAPA_TDNN()  # 또는 ERes2Net, WavLM 등
reference_audio = load_audio("speaker_ref.wav")  # 5-20초

# Speaker embedding 추출 (고정 벡터)
speaker_embedding = speaker_encoder(reference_audio)  # [1, D_spk]

# LLM token space로 projection
speaker_condition = projection_layer(speaker_embedding)  # [1, D_llm]

# Conditioning 방식:
# Option A: Sum conditioning (모든 step에 더함)
input_hidden = input_hidden + speaker_condition

# Option B: Cross-attention
output = transformer(input, cross_attention_src=speaker_condition)
```

**장점**:
- 짧은 reference로도 안정적인 품질
- 명시적인 speaker 제어
- Context 효율적 (고정 크기)

**단점**:
- 별도 speaker encoder 필요
- 추가 projection layer 학습 필요
- Speaker encoder 품질이 전체 품질에 영향

#### Type 3: Hybrid (Chatterbox, 최신 모델)

```python
# Hybrid 방식: Audio prompt + Speaker embedding 결합
reference_audio = load_audio("speaker_ref.wav")

# 1. Speaker embedding (global)
speaker_emb = speaker_encoder(reference_audio)

# 2. Audio token prompt (local detail)
audio_tokens = codec.encode(reference_audio[:3_seconds])

# 3. 결합
model_input = concat([audio_tokens, text_tokens])
model_condition = speaker_emb  # sum 또는 cross-attention

output = llm.generate(model_input, condition=model_condition)
```

**장점**:
- Global + Local 정보 모두 활용
- 최고 수준의 speaker similarity
- 유연한 제어

**단점**:
- 가장 복잡한 구현
- 두 정보원 간 balance 학습 필요

---

## 3. Moshi 아키텍처에서의 Speaker Conditioning 삽입 지점

### 3.1 Moshi의 기존 Conditioning 시스템

```python
# moshi/conditioners/base.py에서 발견한 핵심 구조

class ConditionFuser(nn.Module):
    """세 가지 conditioning 방식 지원"""
    FUSING_METHODS = ["sum", "prepend", "cross"]

    def get_sum(self, conditions) -> torch.Tensor | None:
        """모든 timestep에 더해지는 condition (global)"""
        # speaker_embedding에 적합

    def get_cross(self, conditions) -> torch.Tensor | None:
        """Cross-attention source로 사용되는 condition"""
        # 가변 길이 speaker context에 적합

    def get_prepend(self, conditions) -> torch.Tensor | None:
        """입력 시퀀스 앞에 붙는 condition"""
        # audio token prompt에 적합
```

### 3.2 Moshi LMModel에서의 Conditioning 처리

```python
# moshi/models/lm.py:380-408

def forward_text(
    self,
    sequence: torch.Tensor,
    sum_condition: torch.Tensor | None = None,      # ← Speaker embedding 삽입 가능!
    cross_attention_src: torch.Tensor | None = None, # ← Speaker context 삽입 가능!
) -> tuple[torch.Tensor, torch.Tensor]:

    # Audio + Text embedding
    input_ = audio_emb + text_emb

    # Sum conditioning (이미 지원됨!)
    if sum_condition is not None:
        input_ = input_ + sum_condition.to(input_)

    # Cross-attention (이미 지원됨!)
    if cross_attention_src is not None:
        cross_attention_src = cross_attention_src.to(input_)

    transformer_out = self.transformer(input_, cross_attention_src=cross_attention_src)
```

### 3.3 Moshi TTS에서의 Speaker Conditioning 예시

```python
# moshi/models/tts.py:654-667 - 기존 multi-speaker TTS 구현

# Voice embedding 로드
emb = load_file(voices[idx], device='cpu')['speaker_wavs']

# TensorCondition으로 wrapping
tensors = {
    'speaker_wavs': TensorCondition(voice_tensor, mask)
}

# ConditionAttributes로 전달
return ConditionAttributes(text=text, tensor=tensors)
```

### 3.4 삽입 가능 지점 요약

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Moshi Speaker Conditioning 삽입 지점                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Sequence                                                             │
│  [Text tokens] + [Audio tokens (user)] + [Audio tokens (moshi)]            │
│        │               │                       │                           │
│        ▼               ▼                       ▼                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Embedding Layer                                 │    │
│  │  text_emb + audio_emb[0..7] → input_                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ★ 삽입 지점 1: Sum Conditioning                                     │    │
│  │                                                                     │    │
│  │  input_ = input_ + speaker_condition   ← 여기!                      │    │
│  │                                                                     │    │
│  │  • Speaker encoder output (ECAPA-TDNN) 투영 후 사용                 │    │
│  │  • 모든 timestep에 동일한 speaker 정보 제공                         │    │
│  │  • 구현: ConditionFuser.get_sum() 활용                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ★ 삽입 지점 2: Cross-Attention                                      │    │
│  │                                                                     │    │
│  │  transformer(input_, cross_attention_src=speaker_context) ← 여기!  │    │
│  │                                                                     │    │
│  │  • Reference audio를 별도 encoder로 처리한 결과                     │    │
│  │  • Attention으로 필요한 정보 선택적 참조                            │    │
│  │  • 구현: ConditionFuser.get_cross() 활용                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   Main Transformer (7B)                              │    │
│  │  StreamingTransformer with cross_attention support                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│                        Depformer → Audio Logits                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

★ 삽입 지점 3: Input Prepend (Audio Token Prompt)

┌─────────────────────────────────────────────────────────────────────────────┐
│  Original: [Text] [User Audio] [Moshi Audio]                                │
│                                                                             │
│  Modified: [Speaker Ref Tokens] [Text] [User Audio] [Moshi Audio]          │
│            ↑                                                                │
│            Reference audio를 Mimi로 encode한 토큰                          │
│            (VALL-E prompt 방식)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 접근법 상세 분석

### 4.1 Option A: Speaker Encoder 기반 (권장)

#### 4.1.1 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Option A: Speaker Encoder Approach                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Reference Audio (10-20초)                                                  │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                ECAPA-TDNN / ERes2Net / WavLM                        │    │
│  │                                                                     │    │
│  │  • Pre-trained speaker verification model                          │    │
│  │  • Output: 192-512 dim speaker embedding                           │    │
│  │  • Frozen during K-Moshi training                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼  speaker_emb: [B, D_spk]                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                Speaker Projection Layer (trainable)                  │    │
│  │                                                                     │    │
│  │  Linear(D_spk, D_model) → [B, D_model]                             │    │
│  │                                                                     │    │
│  │  (D_spk=192, D_model=4096 for Moshi 7B)                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼  speaker_condition: [B, 1, D_model]                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Moshi LMModel                                    │    │
│  │                                                                     │    │
│  │  forward_text(..., sum_condition=speaker_condition)                │    │
│  │                                                                     │    │
│  │  # 내부에서:                                                         │    │
│  │  input_ = text_emb + audio_emb + speaker_condition                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.1.2 구현 코드 스케치

```python
# k-moshi-finetune/finetune/speaker_encoder.py

import torch
import torch.nn as nn
from speechbrain.pretrained import EncoderClassifier

class SpeakerConditioner(nn.Module):
    """Speaker embedding을 Moshi conditioning으로 변환"""

    def __init__(self, speaker_dim: int = 192, model_dim: int = 4096):
        super().__init__()
        # Pre-trained ECAPA-TDNN (frozen)
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda"}
        )
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False

        # Trainable projection
        self.projection = nn.Linear(speaker_dim, model_dim, bias=False)

    def forward(self, reference_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reference_audio: [B, T] waveform at 16kHz
        Returns:
            speaker_condition: [B, 1, D_model] for sum_condition
        """
        with torch.no_grad():
            speaker_emb = self.speaker_encoder.encode_batch(reference_audio)
            # speaker_emb: [B, D_spk]

        speaker_condition = self.projection(speaker_emb)  # [B, D_model]
        return speaker_condition.unsqueeze(1)  # [B, 1, D_model]
```

#### 4.1.3 학습 데이터 구성

```yaml
# 데이터 형식
training_data_format:
  each_sample:
    audio_path: "dialogue_001.wav"           # Stereo: L=K-Moshi, R=User
    speaker_reference: "speaker_ref_001.wav" # K-Moshi와 동일 화자의 별도 녹음 (10-20초)
    alignments: [...]

  important_constraint:
    - speaker_reference는 audio_path의 K-Moshi(Left channel)와 동일 화자여야 함
    - 다양한 화자의 데이터로 학습해야 zero-shot 능력 획득
    - 각 화자당 최소 30분 이상의 대화 데이터 권장

# Ground Truth 설정
ground_truth:
  audio_output: "audio_path의 Left channel (K-Moshi 음성)"
  text_output: "alignments에서 SPEAKER_MAIN의 텍스트"
  speaker_reference: "동일 화자의 다른 발화"

# 학습 목표
training_objective:
  - 주어진 speaker_reference의 화자 특성을 가진 음성 생성
  - Text content는 입력 text에 따름
  - 대화 능력은 기존 Moshi 학습과 동일
```

### 4.2 Option B: Audio Token Prompt 기반 (VALL-E 스타일)

#### 4.2.1 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Option B: Audio Token Prompt Approach                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Reference Audio (10-20초)                                                  │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Mimi Encoder (frozen)                             │    │
│  │                                                                     │    │
│  │  • 기존 Moshi의 codec encoder 그대로 사용                           │    │
│  │  • Output: 8 codebook tokens at 12.5Hz                             │    │
│  │  • 10초 audio → ~125 frames                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼  ref_tokens: [B, 8, T_ref]                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              Input Sequence Construction                             │    │
│  │                                                                     │    │
│  │  Original Moshi input:                                              │    │
│  │  [Text_stream] [User_audio] [Moshi_audio]                          │    │
│  │                                                                     │    │
│  │  Modified input:                                                    │    │
│  │  [Speaker_ref_tokens] [SEP] [Text_stream] [User_audio] [Moshi_audio]│    │
│  │  ↑                                                                  │    │
│  │  Reference audio의 Mimi tokens (prepend)                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Moshi LMModel                                    │    │
│  │                                                                     │    │
│  │  • Autoregressive하게 reference 토큰 "재생"                         │    │
│  │  • Model이 speaker 특성을 context에서 학습                          │    │
│  │  • In-context learning 방식                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 장단점 분석

```yaml
pros:
  - 구현 단순: Mimi encoder만 사용, 추가 모델 불필요
  - Moshi 호환성 최고: 기존 인터리빙 구조 그대로 활용
  - Fine-grained 정보: Codec token에 prosody, pitch 등 세부 정보 포함
  - 검증된 방식: VALL-E, CosyVoice에서 효과 입증

cons:
  - Context 소비: 10초 reference → ~125 tokens 소비
  - 암묵적 학습: Speaker 특성이 명시적으로 분리되지 않음
  - Reference 품질 의존: Noisy reference에 취약할 수 있음
  - 학습 데이터 요구량: In-context learning을 위해 더 많은 데이터 필요

trade_offs:
  reference_length_vs_quality:
    - 짧은 reference (3-5초): 빠른 inference, 낮은 speaker similarity
    - 긴 reference (15-20초): 높은 similarity, context 많이 소비
    - 권장: 10초 (balance)
```

### 4.3 Option C: Cross-Attention 기반

#### 4.3.1 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Option C: Cross-Attention Approach                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Reference Audio                                                            │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │            Speaker Context Encoder (trainable)                       │    │
│  │                                                                     │    │
│  │  • WavLM / HuBERT / Custom encoder                                 │    │
│  │  • Output: [B, T_ref, D_context] 가변 길이 context                 │    │
│  │  • Speaker의 detailed acoustic features 보존                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                    │
│        ▼  speaker_context: [B, T_ref, D_context]                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Moshi Transformer                                │    │
│  │                                                                     │    │
│  │  for each layer:                                                    │    │
│  │      self_attn_out = self_attention(x)                             │    │
│  │      cross_attn_out = cross_attention(                             │    │
│  │          query=self_attn_out,                                      │    │
│  │          key=speaker_context,                                      │    │
│  │          value=speaker_context                                     │    │
│  │      )                                                             │    │
│  │      x = self_attn_out + cross_attn_out                           │    │
│  │      x = ffn(x)                                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 4.3.2 Moshi 지원 현황

```python
# moshi/modules/transformer.py에서 cross_attention 지원 확인 필요
# StreamingTransformer가 cross_attention_src를 받아 처리

# lm.py:400-402
if cross_attention_src is not None:
    cross_attention_src = cross_attention_src.to(input_)
transformer_out = self.transformer(input_, cross_attention_src=cross_attention_src)
```

**주의**: Moshi transformer가 cross_attention을 지원하지만, Depformer에는 전달되지 않음 (`kwargs_dep["cross_attention"] = False`). 따라서 main transformer에서만 speaker context 활용 가능.

### 4.4 Option D: Hybrid (권장)

#### 4.4.1 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Option D: Hybrid Approach                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Reference Audio (10-20초)                                                  │
│        │                                                                    │
│        ├────────────────────────┬────────────────────────┐                  │
│        ▼                        ▼                        ▼                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │ ECAPA-TDNN   │    │    Mimi Encoder  │    │ (Optional)       │          │
│  │ (Global ID)  │    │ (Local Detail)   │    │ WavLM Context    │          │
│  └──────────────┘    └──────────────────┘    └──────────────────┘          │
│        │                        │                        │                  │
│        ▼                        ▼                        ▼                  │
│  speaker_emb           ref_tokens              speaker_context             │
│  [B, D_spk]            [B, 8, T_ref]           [B, T, D_ctx]               │
│        │                        │                        │                  │
│        ▼                        │                        │                  │
│  ┌──────────────┐               │                        │                  │
│  │  Projection  │               │                        │                  │
│  └──────────────┘               │                        │                  │
│        │                        │                        │                  │
│        ▼                        │                        │                  │
│  sum_condition                  │                        │                  │
│  [B, 1, D_model]                │                        │                  │
│        │                        │                        │                  │
│        │    ┌───────────────────┘                        │                  │
│        │    │                                            │                  │
│        │    ▼                                            │                  │
│        │  Input prepend                                  │                  │
│        │  [ref_tokens] + [original_input]                │                  │
│        │    │                                            │                  │
│        │    │    ┌───────────────────────────────────────┘                  │
│        │    │    │                                                          │
│        ▼    ▼    ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Moshi LMModel                               │    │
│  │                                                                     │    │
│  │  forward_text(                                                      │    │
│  │      sequence=[ref_tokens] + [input],     # Local detail           │    │
│  │      sum_condition=speaker_emb_projected, # Global ID              │    │
│  │      cross_attention_src=speaker_context  # Rich context (optional)│    │
│  │  )                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Benefits:                                                                  │
│  • Global speaker ID: 안정적인 화자 특성 유지                              │
│  • Local detail: Prosody, pitch 등 세부 특성 전달                          │
│  • Rich context: 필요시 더 detailed 정보 참조                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 학습 전략

### 5.1 데이터 요구사항

```yaml
# Zero-shot speaker ability를 위한 데이터 구성

minimum_requirements:
  total_hours: 500-1000
  unique_speakers: 50-100명 이상
  per_speaker_minimum: 10분 이상

data_format:
  - audio_path: "dialogue.wav"          # Stereo (L=AI, R=User)
  - speaker_reference: "speaker_ref.wav" # Same speaker as L channel
  - alignments: [...]                   # Word-level alignment

key_principle: |
  학습 시 speaker_reference와 audio_path의 Left channel은
  동일 화자이지만 다른 발화여야 한다.
  이를 통해 모델이 "reference의 화자 특성을 추출하여
  새로운 content에 적용"하는 것을 학습한다.
```

### 5.2 Ground Truth 설정

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      학습 시 Ground Truth 설정                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input:                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. User audio stream (Right channel)                                │    │
│  │ 2. Text content (from alignments)                                  │    │
│  │ 3. Speaker reference (별도 파일, same speaker as Left)             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Target (Ground Truth):                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. K-Moshi audio stream (Left channel) → audio loss               │    │
│  │ 2. K-Moshi text stream (from alignments) → text loss              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  핵심: Speaker reference ≠ Target audio (내용 다름, 화자 동일)             │
│                                                                             │
│  학습 목표:                                                                 │
│  • Speaker reference에서 화자 특성 추출                                    │
│  • 추출한 특성 + 입력 content → Target audio 생성                         │
│  • Cross-entropy loss on text tokens                                       │
│  • Cross-entropy loss on audio tokens (8 codebooks)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 학습 파이프라인

```python
# 학습 시 데이터 처리 pseudo-code

def training_step(batch):
    # 1. Reference audio에서 speaker conditioning 추출
    speaker_ref = batch['speaker_reference']  # [B, T_ref] waveform

    # Option A: Speaker encoder
    speaker_condition = speaker_encoder(speaker_ref)  # [B, 1, D_model]

    # Option B: Audio token prompt
    ref_tokens = mimi.encode(speaker_ref)  # [B, 8, T_ref_frames]

    # 2. 대화 데이터 처리 (기존 Moshi와 동일)
    dialogue_audio = batch['audio']  # Stereo
    alignments = batch['alignments']
    codes = interleaver.build_token_stream(dialogue_audio, alignments)

    # 3. Forward pass with speaker conditioning
    if use_speaker_encoder:
        output = model(codes, sum_condition=speaker_condition)
    elif use_token_prompt:
        # ref_tokens을 codes 앞에 prepend
        codes_with_ref = prepend_reference(ref_tokens, codes)
        output = model(codes_with_ref)

    # 4. Loss 계산 (기존과 동일)
    text_loss = compute_text_loss(output.text_logits, target_text)
    audio_loss = compute_audio_loss(output.logits, target_audio)
    loss = text_loss + audio_loss

    return loss
```

### 5.4 Inference 시나리오

```python
# Inference 시 zero-shot speaker 사용

def inference(user_input_audio, target_speaker_reference, text_response):
    """
    Args:
        user_input_audio: 사용자 음성 입력 (실시간)
        target_speaker_reference: 원하는 AI 목소리 reference (10-20초)
        text_response: (optional) 텍스트 응답 (TTS mode) 또는 자동 생성 (dialogue mode)
    """

    # 1. Speaker conditioning 준비
    speaker_condition = speaker_encoder(target_speaker_reference)

    # 2. Streaming inference
    with model.streaming():
        for audio_chunk in user_input_audio:
            # User audio 처리
            user_codes = mimi.encode(audio_chunk)

            # Model step with speaker conditioning
            output = model.step(
                user_codes,
                sum_condition=speaker_condition
            )

            # AI audio 생성 (target speaker voice)
            ai_audio = mimi.decode(output.audio_codes)
            yield ai_audio
```

---

## 6. 기술적 도전 과제

### 6.1 Speaker-Content Disentanglement

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Challenge: Speaker-Content 분리                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  문제:                                                                      │
│  Speaker reference의 "content"가 생성 결과에 영향을 미침                   │
│                                                                             │
│  예시:                                                                      │
│  Reference: "안녕하세요, 반갑습니다"                                       │
│  Target text: "오늘 날씨가 좋네요"                                         │
│  ❌ Bad output: "안녕하세요" 같은 단어가 섞여 나옴                         │
│  ✓ Good output: Reference 화자 목소리로 "오늘 날씨가 좋네요"               │
│                                                                             │
│  해결책:                                                                    │
│  ────────────────────────────────────────────────────────────────────────── │
│  1. Speaker Encoder 사용 (Option A)                                        │
│     - ECAPA-TDNN은 speaker verification용으로 학습됨                       │
│     - Content-independent한 speaker 특성만 추출하도록 설계됨               │
│     - 가장 깔끔한 분리                                                     │
│                                                                             │
│  2. Multiple Reference 사용                                                 │
│     - 동일 화자의 여러 발화를 reference로 사용                             │
│     - Content 정보가 평균화되어 희석                                       │
│     - CosyVoice의 multi-reference approach                                 │
│                                                                             │
│  3. Contrastive Learning                                                    │
│     - 동일 화자, 다른 content → speaker embedding 유사하게                 │
│     - 다른 화자, 같은 content → speaker embedding 다르게                   │
│     - 추가 학습 필요                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Streaming 호환성

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Challenge: Streaming Compatibility                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Moshi의 강점: Full-duplex streaming                                        │
│  도전: Speaker conditioning을 streaming에 어떻게 통합?                     │
│                                                                             │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│  Option A (Speaker Encoder) - Streaming 친화적 ✓                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Initialization:                                                    │    │
│  │    speaker_condition = speaker_encoder(reference)  # 1회 계산       │    │
│  │                                                                     │    │
│  │  Streaming loop:                                                    │    │
│  │    for chunk in audio_stream:                                       │    │
│  │        output = model.step(chunk, sum_condition=speaker_condition)  │    │
│  │        # speaker_condition은 매 step 동일하게 사용                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Option B (Token Prompt) - Streaming 복잡 ⚠                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  문제:                                                              │    │
│  │    - Reference tokens을 매번 context에 포함해야 함                  │    │
│  │    - Context window 소비 증가                                       │    │
│  │    - 긴 대화에서 context overflow 위험                              │    │
│  │                                                                     │    │
│  │  해결책:                                                            │    │
│  │    - KV cache에 reference tokens의 attention 결과 저장              │    │
│  │    - Sliding window에서 reference 부분은 유지                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Multi-Speaker Dialogue

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                Challenge: Multi-Speaker Dialogue Handling                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  시나리오: K-Moshi가 여러 화자 역할을 수행해야 할 때                        │
│                                                                             │
│  예: Customer service bot이 여러 persona 사용                               │
│  - 친절한 상담원 A (여성)                                                  │
│  - 전문 기술 상담원 B (남성)                                               │
│                                                                             │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                             │
│  해결책 1: Turn-level Speaker Switching                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  [Turn 1] Speaker A condition → Response in voice A                 │    │
│  │  [Turn 2] Speaker B condition → Response in voice B                 │    │
│  │  ...                                                                │    │
│  │                                                                     │    │
│  │  구현: turn 시작 시 speaker_condition 교체                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  해결책 2: Speaker ID Embedding                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  각 speaker reference에 ID 부여                                     │    │
│  │  speaker_bank = {                                                   │    │
│  │      "agent_A": speaker_encoder(ref_A),                            │    │
│  │      "agent_B": speaker_encoder(ref_B),                            │    │
│  │  }                                                                 │    │
│  │                                                                     │    │
│  │  Inference 시 ID로 speaker condition 선택                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 권장 구현 로드맵

### 7.1 Phase 1: 기본 검증 (2-3주)

```yaml
goal: "Audio Token Prompt 방식으로 빠르게 feasibility 검증"

tasks:
  1_data_preparation:
    - 소규모 multi-speaker 데이터셋 준비 (10-20명, 각 30분)
    - Speaker reference + dialogue audio 쌍 구성
    - Validation set 별도 분리 (unseen speakers)

  2_implementation:
    - Interleaver 수정: reference tokens prepend 기능 추가
    - Training loop 수정: reference audio 처리
    - 기존 Moshi 코드 최소 변경

  3_training:
    - 소규모 학습 (100-200시간)
    - Speaker similarity 평가 (reference vs generated)

  4_evaluation:
    - Seen speakers: 학습에 포함된 화자
    - Unseen speakers: 학습에 미포함된 화자 (zero-shot)
    - Metrics: Speaker similarity, WER, MOS

expected_outcome:
  - Zero-shot speaker cloning 가능 여부 확인
  - Reference 길이 vs 품질 trade-off 파악
  - 다음 단계 방향 결정
```

### 7.2 Phase 2: Speaker Encoder 통합 (3-4주)

```yaml
goal: "ECAPA-TDNN 기반 speaker conditioning으로 품질 향상"

tasks:
  1_speaker_encoder:
    - ECAPA-TDNN 통합 (SpeechBrain)
    - Projection layer 구현 및 초기화
    - Speaker embedding extraction pipeline

  2_conditioning_integration:
    - ConditionProvider에 speaker conditioner 추가
    - ConditionFuser에서 sum_condition으로 전달
    - Forward pass 수정

  3_training:
    - Phase 1 대비 큰 데이터셋 (500-1000시간)
    - Speaker diversity 확보 (50-100명)
    - LoRA fine-tuning + projection layer 학습

  4_evaluation:
    - Phase 1 대비 품질 비교
    - Streaming inference 테스트
    - Latency 측정

expected_outcome:
  - 안정적인 speaker similarity (> 0.7)
  - Streaming 호환성 확인
  - 제품화 가능 수준 품질
```

### 7.3 Phase 3: Hybrid 최적화 (2-3주)

```yaml
goal: "Speaker Encoder + Audio Prompt hybrid로 최고 품질 달성"

tasks:
  1_hybrid_architecture:
    - Global (speaker encoder) + Local (audio prompt) 결합
    - 두 정보원 간 balance weight 학습
    - Cross-attention 추가 검토

  2_advanced_training:
    - Contrastive learning 적용
    - Multi-reference training
    - Data augmentation (noise, reverb)

  3_optimization:
    - Inference 속도 최적화
    - Memory 효율화
    - 양자화 적용

expected_outcome:
  - State-of-the-art speaker similarity
  - Production-ready quality
  - Efficient inference
```

---

## 8. 결론

### 8.1 핵심 결론

1. **Moshi 아키텍처는 zero-shot speaker ability 구현에 적합하다**
   - 기존 conditioning 시스템 (`sum_condition`, `cross_attention`) 활용 가능
   - TTS 모델에서 이미 `speaker_wavs` conditioner 사용 중

2. **Speaker Encoder (ECAPA-TDNN) 방식이 가장 권장됨**
   - Content와 speaker 분리가 명확
   - Streaming 친화적
   - 짧은 reference로도 안정적 품질

3. **점진적 구현이 효과적**
   - Phase 1: Audio token prompt로 빠른 검증
   - Phase 2: Speaker encoder로 품질 향상
   - Phase 3: Hybrid로 최적화

### 8.2 예상 성과

```yaml
expected_metrics:
  speaker_similarity:
    seen_speakers: "> 0.85"
    unseen_speakers: "> 0.70"  # zero-shot

  audio_quality:
    mos_score: "> 3.5"
    wer: "< 15%"

  inference:
    first_token_latency: "< 500ms"
    streaming_rtf: "< 1.0"  # real-time factor
```

### 8.3 주요 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| Speaker-content 분리 실패 | 중 | 높음 | Contrastive learning, multi-reference |
| 학습 데이터 부족 | 중 | 높음 | TTS 합성 데이터 활용 |
| Streaming 호환성 문제 | 낮 | 중 | Speaker encoder 방식 우선 |
| Speaker similarity 미달 | 중 | 높음 | Hybrid 방식으로 보완 |

---

## 참고 문헌

1. [VALL-E: Neural Codec Language Models](https://arxiv.org/abs/2301.02111)
2. [CosyVoice 2: Scalable Streaming Speech Synthesis](https://arxiv.org/abs/2412.10117)
3. [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
4. [F-actor: Full-Duplex Spoken Dialogue](https://arxiv.org/abs/2601.11329)
5. [Voice Cloning Survey](https://arxiv.org/abs/2505.00579)
6. [OpenAudio S1](https://huggingface.co/fishaudio/openaudio-s1-mini)
7. [ECAPA-TDNN Speaker Recognition](https://arxiv.org/abs/2005.07143)

---

*Last Updated: 2026-01-20*
*Document: K-Moshi Zero-Shot Speaker Architecture Analysis*
*Author: K-Moshi Research Team*
