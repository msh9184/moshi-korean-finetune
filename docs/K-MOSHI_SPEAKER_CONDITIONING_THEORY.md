# K-Moshi Zero-Shot Speaker Conditioning: 이론적 기반과 아키텍처

**Version**: 1.0
**Created**: 2026-01-21
**Author**: K-Moshi Development Team

---

## 목차

1. [Input Embedding 개념 명확화](#1-input-embedding-개념-명확화)
2. [Zero-Shot Speaker Conditioning 기법 이론화](#2-zero-shot-speaker-conditioning-기법-이론화)
3. [Depth Transformer 구조 개선 Future Work](#3-depth-transformer-구조-개선-future-work)
4. [K-Moshi 아키텍처 다이어그램](#4-k-moshi-아키텍처-다이어그램)
5. [개발 히스토리 및 현재 상태](#5-개발-히스토리-및-현재-상태)

---

## 1. Input Embedding 개념 명확화

### 1.1 기본 수식 정의

Moshi의 Temporal Transformer에 입력되는 `combined_input`은 다음과 같이 정의됩니다:

```
combined_input[t] = text_emb[t] + Σᵢ audio_emb[i][t] + speaker_condition
```

**수학적 표현:**

$$
\mathbf{h}_t = \mathbf{E}_{text}(x_t^{text}) + \sum_{i=0}^{K-1} \mathbf{E}_{audio}^{(i)}(x_t^{audio,i}) + \mathbf{s}
$$

여기서:
- $\mathbf{h}_t \in \mathbb{R}^D$: 시간 $t$에서의 combined input (D=4096)
- $\mathbf{E}_{text}$: Text embedding layer (vocab_size → D)
- $\mathbf{E}_{audio}^{(i)}$: i번째 audio codebook embedding layer (2048 → D)
- $x_t^{text}$: 시간 $t$의 text token
- $x_t^{audio,i}$: 시간 $t$의 i번째 audio codebook token
- $K$: Audio codebook 수 (dep_q=8)
- $\mathbf{s} \in \mathbb{R}^D$: Speaker conditioning vector (time-invariant)

### 1.2 각 컴포넌트의 역할

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INPUT EMBEDDING BREAKDOWN                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   text_emb[t]   │  "Inner Monologue" - MOSHI가 말할 텍스트의 의미적 표현   │
│  │   [B, T, 4096]  │  • SentencePiece 토큰 → 4096차원 dense vector           │
│  └────────┬────────┘  • Moshi의 "생각"을 인코딩                              │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │ Σ audio_emb[t]  │  MOSHI의 8개 오디오 코드북 임베딩의 합                   │
│  │   [B, T, 4096]  │  • Codebook 0: Semantic (의미/prosody) - 가중치 100x    │
│  └────────┬────────┘  • Codebook 1-7: Acoustic (음향 세부사항)               │
│           │           • 각 코드북이 다른 주파수 대역/특성 인코딩              │
│           │           • Sum pooling: 모든 오디오 정보 통합                   │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │ speaker_cond[t] │  화자 특성 조건화 벡터 (Zero-Shot에서 핵심!)            │
│  │   [B, 1, 4096]  │  • Reference audio에서 추출한 speaker identity          │
│  └────────┬────────┘  • Time-invariant: 모든 시간 step에 동일하게 적용       │
│           │           • Broadcasting: [B,1,D] → [B,T,D]                     │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     combined_input = text + audio + speaker              │ │
│  │                            [B, T, 4096]                                  │ │
│  │                                 ↓                                        │ │
│  │                       Temporal Transformer                               │ │
│  │                       (32 layers, 4096 dim)                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 코드 레벨 매핑 (lm_model_wrapper.py:730-765)

```python
# Step 1: Audio embedding summation (8 codebooks)
audio_input = None
for cb_index in range(n_audio_embs):  # n_audio_embs = dep_q = 8
    audio_codes = input_sequence[:, cb_index + self._audio_offset]  # [B, S]
    audio_emb = self.audio_embs[cb_index](audio_codes)  # [B, S, D=4096]
    audio_input = audio_emb if audio_input is None else audio_input + audio_emb

# Step 2: Text embedding
text_codes = input_sequence[:, 0]  # [B, S]
text_emb = self.text_emb(text_codes)  # [B, S, D=4096]

# Step 3: Combine text + audio
combined_input = text_emb + audio_input  # [B, S, D=4096]

# Step 4: Add speaker conditioning (NEW in K-Moshi)
if effective_sum_condition is not None:
    # Broadcasting: [B, 1, D] + [B, S, D] → [B, S, D]
    combined_input = combined_input + effective_sum_condition.to(combined_input)
```

### 1.4 Embedding 차원 흐름도

```
Input Tokens                    Embedding                 Combined
─────────────                   ─────────                 ────────

text_token[t]    ─────────►  text_emb[t]
(vocab=32000)      Linear      [4096]          ┐
                                               │
audio_code[0,t]  ─────────►  audio_emb[0,t]    │
(card=2048)        Linear      [4096]          │
                                               │
audio_code[1,t]  ─────────►  audio_emb[1,t]    ├──► SUM ──► combined_input[t]
(card=2048)        Linear      [4096]          │            [4096]
                                               │
  ...              ...         ...             │
                                               │
audio_code[7,t]  ─────────►  audio_emb[7,t]    │
(card=2048)        Linear      [4096]          │
                                               │
speaker_emb      ─────────►  speaker_cond      │
(192)              MLP         [4096]          ┘
                 + scale
```

---

## 2. Zero-Shot Speaker Conditioning 기법 이론화

K-Moshi는 **2가지 상호 보완적인 Speaker Conditioning 방법**을 지원합니다.

### 2.1 기법 분류 체계

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              ZERO-SHOT SPEAKER CONDITIONING TAXONOMY                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Method 1: ENCODER-BASED (현재 구현됨)                                  │  │
│  │  ═══════════════════════════════════════                               │  │
│  │                                                                         │  │
│  │  Reference Audio ──► Speaker Encoder ──► Projection ──► sum_condition  │  │
│  │  [3-10초, 16kHz]     (ECAPA-TDNN)        (192→4096)      [B, 1, 4096]  │  │
│  │                       [192-dim]          + scale                       │  │
│  │                                                                         │  │
│  │  장점:                                                                  │  │
│  │  • 명시적 speaker embedding space                                      │  │
│  │  • Pretrained speaker recognition 모델 활용                            │  │
│  │  • 메모리 효율적 (고정 길이 embedding)                                  │  │
│  │  • 다른 화자와의 interpolation 가능                                    │  │
│  │                                                                         │  │
│  │  단점:                                                                  │  │
│  │  • Speaker encoder의 표현력 한계                                       │  │
│  │  • Fine-grained prosody 캡처 어려움                                    │  │
│  │  • Encoder-LM domain gap                                               │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Method 2: AUDIO-PROMPT (VALL-E style, 향후 구현 예정)                  │  │
│  │  ═══════════════════════════════════════════════════════               │  │
│  │                                                                         │  │
│  │  Reference Audio ──► Mimi Encode ──► Prepend to sequence               │  │
│  │  [3-10초, 24kHz]     [8 codebooks]    [prefix, target]                 │  │
│  │                                                                         │  │
│  │  장점:                                                                  │  │
│  │  • End-to-end (no external encoder)                                    │  │
│  │  • Fine-grained prosody preservation                                   │  │
│  │  • In-context learning via attention                                   │  │
│  │                                                                         │  │
│  │  단점:                                                                  │  │
│  │  • 더 긴 context 필요 (메모리 증가)                                     │  │
│  │  • 추론 시 latency 증가                                                │  │
│  │  • Reference 품질에 민감                                               │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Method 3: HYBRID (연구 방향)                                          │  │
│  │  ═════════════════════════════                                         │  │
│  │                                                                         │  │
│  │  Encoder + Audio Prompt 결합:                                          │  │
│  │  • Global speaker identity: Encoder → sum_condition                    │  │
│  │  • Local prosody/style: Audio prompt → cross-attention                 │  │
│  │                                                                         │  │
│  │  잠재적 이점:                                                           │  │
│  │  • Global + local 정보 모두 활용                                       │  │
│  │  • 긴 reference 없이도 fine-grained control                            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Method 1: Encoder-Based (상세 이론)

#### 2.2.1 수학적 정의

**Speaker Encoder Function:**
$$
\mathbf{e}_{spk} = f_{enc}(\mathbf{x}_{ref}) \in \mathbb{R}^{192}
$$

**Speaker Conditioner Function:**
$$
\mathbf{s} = \alpha \cdot \mathbf{W} \cdot \text{LayerNorm}(\mathbf{e}_{spk}) \in \mathbb{R}^{4096}
$$

여기서:
- $f_{enc}$: ECAPA-TDNN speaker encoder (pretrained)
- $\mathbf{x}_{ref}$: Reference audio waveform [T_ref samples]
- $\mathbf{W} \in \mathbb{R}^{4096 \times 192}$: Learnable projection matrix
- $\alpha \in \mathbb{R}$: Learnable scale parameter (초기값 0.1)

#### 2.2.2 Architecture Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENCODER-BASED SPEAKER CONDITIONING                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Reference Audio                                                            │
│  ┌────────────────┐                                                         │
│  │ [48000~160000] │  3-10 seconds @ 16kHz                                   │
│  │    samples     │                                                         │
│  └───────┬────────┘                                                         │
│          │                                                                  │
│          ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │                    ECAPA-TDNN Speaker Encoder                   │        │
│  │  ┌──────────┐ ┌────────────┐ ┌─────────────┐ ┌───────────────┐ │        │
│  │  │ 1D Conv  │→│ SE-Res2Net │→│ Attentive   │→│ FC + BN       │ │        │
│  │  │ (k=5)    │ │ Blocks     │ │ Statistics  │ │ (L2 Norm)     │ │        │
│  │  └──────────┘ └────────────┘ │ Pooling     │ └───────────────┘ │        │
│  │                              └─────────────┘                    │        │
│  │                                                                 │        │
│  │  Output: [B, 192]  (L2-normalized speaker embedding)            │        │
│  │                                                                 │        │
│  │  ✓ FROZEN during training (pretrained on VoxCeleb)              │        │
│  └─────────────────────────────────────────┬──────────────────────┘        │
│                                            │                                │
│                                            ▼                                │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    Speaker Conditioner (Learnable)               │       │
│  │                                                                  │       │
│  │  ┌──────────────┐    ┌────────────────┐    ┌─────────────────┐  │       │
│  │  │  LayerNorm   │ →  │  Linear        │ →  │   Scale         │  │       │
│  │  │  (192)       │    │  (192 → 4096)  │    │   (× α)         │  │       │
│  │  └──────────────┘    └────────────────┘    └────────┬────────┘  │       │
│  │                                                     │           │       │
│  │                                                     ▼           │       │
│  │                              Output: [B, 1, 4096]               │       │
│  │                                                                  │       │
│  │  Learnable parameters:                                          │       │
│  │  • Linear weight: 192 × 4096 = 786,432 params                   │       │
│  │  • Linear bias: 4096 params                                     │       │
│  │  • LayerNorm: 2 × 192 = 384 params                              │       │
│  │  • Scale α: 1 param                                             │       │
│  │  Total: ~790K params (vs 7B model = 0.01%)                      │       │
│  └─────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.2.3 Scale Parameter 설계 원리

**왜 작은 초기 scale (α=0.1)?**

1. **안정적 학습 시작**: 큰 speaker conditioning은 초기에 불안정
2. **점진적 통합**: 모델이 먼저 text+audio 관계 학습 후 speaker 정보 통합
3. **Residual connection 유사**: skip connection처럼 점진적 기여

```python
# Scale mode options
scale_mode = "multiply"  # Default: α × projection(emb)
# OR
scale_mode = "gated"     # Advanced: σ(gate(emb)) × projection(emb)
```

### 2.3 Method 2: Audio-Prompt (VALL-E Style) - 이론적 설계

#### 2.3.1 수학적 정의

Reference audio를 Mimi로 인코딩하여 sequence에 prefix로 추가:

$$
\mathbf{X}_{input} = [\mathbf{X}_{ref}; \mathbf{X}_{target}]
$$

여기서:
- $\mathbf{X}_{ref} \in \mathbb{R}^{K \times T_{ref}}$: Reference audio codes
- $\mathbf{X}_{target} \in \mathbb{R}^{K \times T_{target}}$: Target audio codes
- $K = 9$: Total codebooks (1 text + 8 audio)

#### 2.3.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUDIO-PROMPT SPEAKER CONDITIONING                         │
│                         (VALL-E Style, Future)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Reference Audio                    Target Prompt                           │
│  ┌────────────────┐                ┌────────────────┐                      │
│  │ [T_ref samples]│                │ "안녕하세요"    │                      │
│  │   @ 24kHz      │                │                │                      │
│  └───────┬────────┘                └───────┬────────┘                      │
│          │                                 │                                │
│          ▼                                 ▼                                │
│  ┌────────────────┐                ┌────────────────┐                      │
│  │  Mimi Encoder  │                │ Text Tokenizer │                      │
│  │  (8 codebooks) │                │ (SentencePiece)│                      │
│  └───────┬────────┘                └───────┬────────┘                      │
│          │                                 │                                │
│          ▼                                 ▼                                │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │                    CONCATENATED SEQUENCE                        │        │
│  │                                                                 │        │
│  │  Time:    │← T_ref →│←      T_target (to generate)     →│      │        │
│  │           │         │                                    │      │        │
│  │  Text:    │ [PAD]   │ [안] [녕] [하] [세] [요] [PAD]... │      │        │
│  │  Audio 0: │ [c0,0]..│ [PRED] [PRED] [PRED] [PRED]...    │      │        │
│  │  Audio 1: │ [c1,0]..│ [PRED] [PRED] [PRED] [PRED]...    │      │        │
│  │    ...    │   ...   │  ...                              │      │        │
│  │  Audio 7: │ [c7,0]..│ [PRED] [PRED] [PRED] [PRED]...    │      │        │
│  │                                                                 │        │
│  │  [PRED] = To be predicted by model (autoregressive)            │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐        │
│  │                    TEMPORAL TRANSFORMER                         │        │
│  │                                                                 │        │
│  │  • Causal attention mask (모든 position)                       │        │
│  │  • Reference 구간: teacher forcing with ground truth           │        │
│  │  • Target 구간: autoregressive prediction                      │        │
│  │                                                                 │        │
│  │  Speaker style은 attention을 통해 reference에서 target으로 전달 │        │
│  └────────────────────────────────────────────────────────────────┘        │
│                                                                             │
│  Training:                                                                  │
│  • Loss는 target 구간에서만 계산                                           │
│  • Reference 구간은 context로만 사용 (no gradient)                         │
│                                                                             │
│  Inference:                                                                 │
│  • Reference audio → Mimi encode → prefix                                  │
│  • Autoregressive generation from target text                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 기법 비교 분석

| 속성 | Encoder-Based | Audio-Prompt |
|------|---------------|--------------|
| **메모리** | O(1) - 고정 192-dim | O(T_ref) - reference 길이 비례 |
| **Latency** | 낮음 (encoder 한 번) | 높음 (긴 context attention) |
| **Speaker Fidelity** | 중간 (global identity) | 높음 (fine-grained style) |
| **Prosody Control** | 제한적 | 우수 |
| **Cross-speaker Mix** | 가능 (embedding interpolation) | 어려움 |
| **Training Data** | 동일 화자 reference 필요 | 동일 세션 reference 필요 |
| **Inference Flexibility** | 높음 (다른 reference 사용 가능) | 중간 |

### 2.5 수학적 통합 프레임워크

두 방법을 통합하는 일반화된 수식:

$$
\mathbf{h}_t = \mathbf{E}_{text}(x_t^{text}) + \sum_{i=0}^{K-1} \mathbf{E}_{audio}^{(i)}(x_t^{audio,i}) + \underbrace{\mathbf{s}_{global}}_{\text{Encoder}} + \underbrace{\text{CrossAttn}(\mathbf{h}_t, \mathbf{H}_{ref})}_{\text{Audio-Prompt}}
$$

여기서:
- $\mathbf{s}_{global}$: Encoder 기반 global speaker identity
- $\text{CrossAttn}$: Reference sequence에 대한 cross-attention

---

## 3. Depth Transformer 구조 개선 Future Work

### 3.1 현재 Depth Transformer 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CURRENT DEPTH TRANSFORMER (Depformer)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: transformer_out [B, T, 4096] from Temporal Transformer              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Per-Timestep Processing                           │   │
│  │                                                                      │   │
│  │  For each t in T:                                                    │   │
│  │    ┌─────────────────────────────────────────────────────────────┐  │   │
│  │    │  Step 0: Input = transformer_out[t] + emb_audio[0]          │  │   │
│  │    │  Step 1: Input = depformer_out + emb_audio[1]               │  │   │
│  │    │  Step 2: Input = depformer_out + emb_audio[2]               │  │   │
│  │    │    ...                                                       │  │   │
│  │    │  Step 7: Input = depformer_out + emb_audio[7]               │  │   │
│  │    └─────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  │  Depformer: 6 transformer layers (non-causal within Ka dimension)   │   │
│  │  • Self-attention across dep_q positions                            │   │
│  │  • NO attention across time (temporal는 이미 처리됨)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Output: audio_logits [B, dep_q, T, 2048]                                  │
│                                                                             │
│  ❗ 현재 한계:                                                              │
│  • User audio (codebook 9-16)는 modeling 안 함 (dep_q=8)                   │
│  • Speaker conditioning이 Depformer에 직접 주입되지 않음                    │
│  • Cross-time attention 없음 (temporal 정보 제한적)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 개선 방향 1: Speaker-Conditioned Depformer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PROPOSED: SPEAKER-CONDITIONED DEPTH TRANSFORMER                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Current:  depformer_input = transformer_out + audio_emb                    │
│                                                                             │
│  Proposed: depformer_input = transformer_out + audio_emb + speaker_cond    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Modified Depformer Input                          │   │
│  │                                                                      │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │   │
│  │  │ transformer  │ + │  audio_emb   │ + │ speaker_cond │             │   │
│  │  │    _out[t]   │   │    [cb]      │   │   [4096]     │             │   │
│  │  └──────────────┘   └──────────────┘   └──────────────┘             │   │
│  │         ↓                                                            │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │                    Depformer Transformer                      │   │   │
│  │  │  • Speaker-aware acoustic generation                         │   │   │
│  │  │  • Each codebook prediction conditioned on speaker           │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  이점:                                                                      │
│  • Acoustic detail (높은 codebook)도 speaker에 맞춤                        │
│  • Voice quality, breathiness 등 fine-grained 특성 반영                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 개선 방향 2: Cross-Time Attention in Depformer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PROPOSED: TEMPORAL-AWARE DEPTH TRANSFORMER                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Current:  Each timestep processed independently                            │
│            depformer(t) doesn't see depformer(t-1, t-2, ...)               │
│                                                                             │
│  Proposed: Limited temporal context in Depformer                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Cross-Time Depformer                              │   │
│  │                                                                      │   │
│  │  Time:     t-2        t-1         t         t+1                     │   │
│  │            ↓          ↓          ↓          ↓                       │   │
│  │        ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐                    │   │
│  │  Ka=0  │ ○──○─┼───┼──○──○┼───┼──●──○┼───┼──○   │                    │   │
│  │  Ka=1  │ ○──○─┼───┼──○──○┼───┼──●──○┼───┼──○   │                    │   │
│  │  Ka=2  │ ○──○─┼───┼──○──○┼───┼──●──○┼───┼──○   │                    │   │
│  │   ...  │ ...  │   │ ...  │   │ ...  │   │ ...  │                    │   │
│  │  Ka=7  │ ○──○─┼───┼──○──○┼───┼──●──○┼───┼──○   │                    │   │
│  │        └──────┘   └──────┘   └──────┘   └──────┘                    │   │
│  │                                                                      │   │
│  │  ● = Current prediction                                             │   │
│  │  ○ = Context (attended via cross-time attention)                    │   │
│  │  ─ = Attention connections                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  이점:                                                                      │
│  • 시간적 연속성 개선 (smoother transitions)                               │
│  • Prosody patterns 더 잘 캡처                                             │
│                                                                             │
│  구현 고려사항:                                                             │
│  • Window size 제한 (e.g., ±2 frames)                                      │
│  • Causal constraint 유지 필요                                             │
│  • 메모리/계산 비용 증가                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 개선 방향 3: Adaptive Codebook Weighting

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PROPOSED: DYNAMIC CODEBOOK WEIGHTING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Current:  Fixed weights (codebook 0 = 100x, others = 1x)                   │
│                                                                             │
│  Proposed: Content-adaptive weighting based on phoneme/prosody              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Adaptive Weight Network                           │   │
│  │                                                                      │   │
│  │  Input:                                                              │   │
│  │  • Current text token (phoneme information)                          │   │
│  │  • Transformer hidden state (context)                                │   │
│  │  • Speaker embedding (speaker-specific weighting)                    │   │
│  │                                                                      │   │
│  │  ┌─────────────┐                                                     │   │
│  │  │  MLP        │ → [w0, w1, w2, ..., w7] (softmax normalized)       │   │
│  │  │  (context)  │                                                     │   │
│  │  └─────────────┘                                                     │   │
│  │                                                                      │   │
│  │  Loss = Σ wi × CE(logits[i], target[i])                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  직관:                                                                      │
│  • 자음: semantic codebook (CB0) 중요 → w0 높음                            │
│  • 모음: acoustic codebooks 중요 → w1-7 높음                               │
│  • 무성음: fine acoustic detail 필요 → higher codebooks 강조               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Future Work Roadmap

| Phase | Task | Priority | Complexity |
|-------|------|----------|------------|
| 3.1 | Speaker-Conditioned Depformer | High | Medium |
| 3.2 | Cross-Time Attention (limited window) | Medium | High |
| 3.3 | Adaptive Codebook Weighting | Low | Medium |
| 3.4 | User Audio Modeling (dep_q=16) | Medium | High |
| 3.5 | Multi-Scale Depformer | Research | Very High |

---

## 4. K-Moshi 아키텍처 다이어그램

### 4.1 전체 시스템 아키텍처 (현재 구현)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           K-MOSHI SYSTEM ARCHITECTURE v2.0                              │
│                        Zero-Shot Speaker Conditioning Enabled                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              INPUT PROCESSING LAYER                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐ │  │
│  │  │                          STEREO AUDIO INPUT                                  │ │  │
│  │  │                         [2, T_samples] @ 24kHz                               │ │  │
│  │  │                                                                              │ │  │
│  │  │    Channel 0 (LEFT)              Channel 1 (RIGHT)                          │ │  │
│  │  │    ┌─────────────┐               ┌─────────────┐                            │ │  │
│  │  │    │ MOSHI Voice │               │ USER Voice  │                            │ │  │
│  │  │    │ (AI Output) │               │ (Human)     │                            │ │  │
│  │  │    └──────┬──────┘               └──────┬──────┘                            │ │  │
│  │  │           │                             │                                    │ │  │
│  │  │           ▼                             ▼                                    │ │  │
│  │  │    ┌─────────────┐               ┌─────────────┐                            │ │  │
│  │  │    │ Mimi Encode │               │ Mimi Encode │                            │ │  │
│  │  │    │ [8, T_frames]│               │ [8, T_frames]│                           │ │  │
│  │  │    └──────┬──────┘               └──────┬──────┘                            │ │  │
│  │  │           │                             │                                    │ │  │
│  │  │           ▼                             ▼                                    │ │  │
│  │  │    ┌─────────────────────────────────────────────────────────────────┐      │ │  │
│  │  │    │              COMBINED CODES [17, T_frames]                       │      │ │  │
│  │  │    │  Index 0:     TEXT tokens (Inner Monologue)                     │      │ │  │
│  │  │    │  Index 1-8:   MOSHI audio codebooks                              │      │ │  │
│  │  │    │  Index 9-16:  USER audio codebooks (context only)               │      │ │  │
│  │  │    └─────────────────────────────────────────────────────────────────┘      │ │  │
│  │  └─────────────────────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                             │
│  ┌────────────────────────────────────────┼────────────────────────────────────────┐   │
│  │                            SPEAKER CONDITIONING MODULE                           │   │
│  │                                        │                                         │   │
│  │  Reference Audio (3-10s)               │                                         │   │
│  │  ┌─────────────┐                       │                                         │   │
│  │  │ [T_ref] @   │                       │                                         │   │
│  │  │   16kHz     │                       │                                         │   │
│  │  └──────┬──────┘                       │                                         │   │
│  │         │                              │                                         │   │
│  │         ▼                              │                                         │   │
│  │  ┌─────────────────┐                   │                                         │   │
│  │  │  ECAPA-TDNN     │                   │                                         │   │
│  │  │  Speaker Encoder│                   │                                         │   │
│  │  │  [192-dim]      │                   │                                         │   │
│  │  │  (FROZEN)       │                   │                                         │   │
│  │  └────────┬────────┘                   │                                         │   │
│  │           │                            │                                         │   │
│  │           ▼                            │                                         │   │
│  │  ┌─────────────────┐                   │                                         │   │
│  │  │ Speaker         │                   │                                         │   │
│  │  │ Conditioner     │────────────────────────────────────────┐                    │   │
│  │  │ (192→4096)      │                   │                    │                    │   │
│  │  │ + Scale (α=0.1) │                   │                    │                    │   │
│  │  │ (LEARNABLE)     │                   │                    │                    │   │
│  │  └─────────────────┘                   │                    │                    │   │
│  │                                        │                    │                    │   │
│  │  Output: speaker_condition             │                    │                    │   │
│  │          [B, 1, 4096]                  │                    │                    │   │
│  └────────────────────────────────────────┼────────────────────┼────────────────────┘   │
│                                           │                    │                        │
│  ┌────────────────────────────────────────┼────────────────────┼────────────────────┐   │
│  │                            EMBEDDING LAYER                   │                    │   │
│  │                                        │                    │                    │   │
│  │  codes[0] ───► text_emb ──────────────►│◄───────────────────┘                    │   │
│  │                [B, T, 4096]            │  (+ speaker_cond)                       │   │
│  │                                        │                                         │   │
│  │  codes[1:9] ──► Σ audio_emb[i] ───────►│                                         │   │
│  │                 [B, T, 4096]           │                                         │   │
│  │                                        ▼                                         │   │
│  │                          ┌─────────────────────────┐                             │   │
│  │                          │     combined_input      │                             │   │
│  │                          │     [B, T, 4096]        │                             │   │
│  │                          │ = text + audio + spk    │                             │   │
│  │                          └────────────┬────────────┘                             │   │
│  └───────────────────────────────────────┼──────────────────────────────────────────┘   │
│                                          │                                              │
│  ┌───────────────────────────────────────┼──────────────────────────────────────────┐   │
│  │                         TEMPORAL TRANSFORMER                                      │   │
│  │                          (Backbone: 32 layers)                                   │   │
│  │                                       │                                          │   │
│  │                                       ▼                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐│   │
│  │  │                                                                              ││   │
│  │  │   ┌─────────┐   ┌─────────┐   ┌─────────┐        ┌─────────┐               ││   │
│  │  │   │ Layer 0 │ → │ Layer 1 │ → │ Layer 2 │ → ... →│Layer 31 │               ││   │
│  │  │   │ (Attn+  │   │ (Attn+  │   │ (Attn+  │        │ (Attn+  │               ││   │
│  │  │   │  FFN)   │   │  FFN)   │   │  FFN)   │        │  FFN)   │               ││   │
│  │  │   └─────────┘   └─────────┘   └─────────┘        └─────────┘               ││   │
│  │  │                                                                              ││   │
│  │  │   Causal Attention: Each position attends only to previous positions        ││   │
│  │  │   Hidden dim: 4096, Heads: 32, KV Heads: 8 (GQA)                            ││   │
│  │  │                                                                              ││   │
│  │  └─────────────────────────────────────────────────────────────────────────────┘│   │
│  │                                       │                                          │   │
│  │                                       ▼                                          │   │
│  │                          transformer_out [B, T, 4096]                            │   │
│  │                                       │                                          │   │
│  │                          ┌────────────┴────────────┐                             │   │
│  │                          │                         │                             │   │
│  │                          ▼                         ▼                             │   │
│  │               ┌─────────────────┐       ┌─────────────────┐                      │   │
│  │               │   text_linear   │       │    Depformer    │                      │   │
│  │               │  (4096→32000)   │       │   (6 layers)    │                      │   │
│  │               └────────┬────────┘       └────────┬────────┘                      │   │
│  │                        │                         │                               │   │
│  │                        ▼                         ▼                               │   │
│  │               text_logits              audio_logits                              │   │
│  │               [B, T, 32000]            [B, 8, T, 2048]                           │   │
│  │                                                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              LOSS COMPUTATION                                     │   │
│  │                                                                                   │   │
│  │  text_loss = CrossEntropy(text_logits, codes[0])                                 │   │
│  │              × text_padding_weight (0.5 for padding tokens)                      │   │
│  │                                                                                   │   │
│  │  audio_loss = Σ wi × CrossEntropy(audio_logits[i], codes[1+i])                   │   │
│  │              w0 = 100 (semantic), w1-7 = 1 (acoustic)                            │   │
│  │                                                                                   │   │
│  │  total_loss = text_loss + audio_loss                                             │   │
│  └──────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Training Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           K-MOSHI TRAINING DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  JSONL Metadata                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ {"path": "dialogue_001.wav", "duration": 45.32}                                  │   │
│  │ {"path": "dialogue_002.wav", "duration": 38.15}                                  │   │
│  │ ...                                                                              │   │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                           │                                             │
│                                           ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                           ChunkedDataset                                         │   │
│  │                                                                                  │   │
│  │  • Load stereo WAV (24kHz)                                                       │   │
│  │  • Split into duration_sec chunks (e.g., 60s)                                    │   │
│  │  • Shuffle across chunks                                                         │   │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                           │                                             │
│                                           ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                     InterleavedTokenizer / StereoInterleavedTokenizer            │   │
│  │                                                                                  │   │
│  │  Input: wav [2, T_samples], start_sec, path                                      │   │
│  │                                                                                  │   │
│  │  Processing:                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │ 1. Load alignments from JSON                                             │    │   │
│  │  │    └─ {"alignments": [["안녕", [0.0, 0.5], "SPEAKER_MAIN"], ...]}        │    │   │
│  │  │                                                                          │    │   │
│  │  │ 2. Encode audio with Mimi                                                │    │   │
│  │  │    └─ moshi_tokens [8, T_frames], user_tokens [8, T_frames]              │    │   │
│  │  │                                                                          │    │   │
│  │  │ 3. Build text tokens (Interleaver)                                       │    │   │
│  │  │    └─ text_tokens [1, T_frames] aligned with audio                       │    │   │
│  │  │                                                                          │    │   │
│  │  │ 4. Sample speaker reference (NEW - Phase 2)                              │    │   │
│  │  │    └─ Find MOSHI speech outside target segment                           │    │   │
│  │  │    └─ Extract 3-10s, resample to 16kHz                                   │    │   │
│  │  │    └─ speaker_reference_audio [T_ref]                                    │    │   │
│  │  │                                                                          │    │   │
│  │  │ 5. Concatenate codes [17, T_frames]                                      │    │   │
│  │  │    └─ [text; moshi_audio; user_audio]                                    │    │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                  │   │
│  │  Output: Sample(codes, speaker_reference_audio, ...)                            │   │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                           │                                             │
│                                           ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              Batch Collation                                     │   │
│  │                                                                                  │   │
│  │  Batch:                                                                          │   │
│  │  • codes: [B, 17, T_frames]                                                      │   │
│  │  • speaker_reference_audios: list[Tensor] (variable length)                     │   │
│  │  • speaker_reference_texts: list[str]                                           │   │
│  │  • condition_attributes: Optional                                                │   │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                           │                                             │
│                                           ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              Training Loop                                       │   │
│  │                                                                                  │   │
│  │  for batch in data_loader:                                                       │   │
│  │      # 1. Extract speaker embeddings                                             │   │
│  │      speaker_emb = speaker_encoder(batch.speaker_reference_audios)  # [B, 192]  │   │
│  │                                                                                  │   │
│  │      # 2. Forward pass                                                           │   │
│  │      output = model(codes=batch.codes, speaker_embedding=speaker_emb)           │   │
│  │                                                                                  │   │
│  │      # 3. Compute loss                                                           │   │
│  │      loss = text_loss + audio_loss                                              │   │
│  │                                                                                  │   │
│  │      # 4. Backward and optimize                                                  │   │
│  │      loss.backward()                                                             │   │
│  │      optimizer.step()                                                            │   │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Inference Pipeline (Future)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           K-MOSHI INFERENCE PIPELINE (Future)                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              CLIENT REQUEST                                        │  │
│  │                                                                                    │  │
│  │  • Reference audio: speaker_ref.wav (3-10s, any sample rate)                      │  │
│  │  • Text prompt: "오늘 날씨가 참 좋네요. 산책하기 좋은 날이에요."                  │  │
│  │  • (Optional) Streaming user audio                                                 │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                             │
│                                           ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              PREPROCESSING                                         │  │
│  │                                                                                    │  │
│  │  1. Resample reference to 16kHz                                                    │  │
│  │  2. Extract speaker embedding: encoder(ref_audio) → [192]                         │  │
│  │  3. Project to sum_condition: conditioner(emb) → [1, 4096]                        │  │
│  │  4. Tokenize text: spm.encode(text) → token_ids                                   │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                             │
│                                           ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              AUTOREGRESSIVE GENERATION                             │  │
│  │                                                                                    │  │
│  │  for t in range(max_steps):                                                        │  │
│  │      # Current context                                                             │  │
│  │      context = [text_tokens[:t], audio_codes[:, :t], user_codes[:, :t]]           │  │
│  │                                                                                    │  │
│  │      # Forward with speaker conditioning                                           │  │
│  │      logits = model(context, sum_condition=speaker_cond)                          │  │
│  │                                                                                    │  │
│  │      # Sample next tokens                                                          │  │
│  │      text_token[t] = sample(logits.text)  # Or use prompt                         │  │
│  │      audio_codes[:, t] = sample(logits.audio, temperature=0.8)                    │  │
│  │                                                                                    │  │
│  │      # Decode audio (streaming)                                                    │  │
│  │      if t % decode_interval == 0:                                                 │  │
│  │          audio_chunk = mimi.decode(audio_codes[:, t-interval:t])                  │  │
│  │          yield audio_chunk                                                         │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                             │
│                                           ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              OUTPUT                                                │  │
│  │                                                                                    │  │
│  │  • Streamed audio: 24kHz waveform in speaker's voice                              │  │
│  │  • Full audio: complete synthesized speech                                        │  │
│  │  • (Optional) Text transcript                                                      │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 개발 히스토리 및 현재 상태

### 5.1 버전 히스토리

| Version | Date | Description |
|---------|------|-------------|
| V1 | 2026-01-10 | MONOLOGUE mode (9 codebooks) |
| V2 | 2026-01-12 | USER-STREAM mode (17 codebooks, dep_q=16) - deprecated |
| V3 | 2026-01-15 | FULL-DUPLEX mode (17 codebooks, dep_q=8) |
| **V4** | 2026-01-18 | **Modular Backbone + Custom Tokenizer** |
| **V4.1** | 2026-01-21 | **Zero-Shot Speaker Conditioning (Phase 1+2)** |

### 5.2 Phase 1: Speaker Encoder Integration (COMPLETED)

**Created Files:**
- `finetune/modules/__init__.py` (~30 lines)
- `finetune/modules/speaker_encoder.py` (~280 lines)
- `finetune/modules/speaker_conditioner.py` (~350 lines)
- `tests/__init__.py` (~2 lines)
- `tests/test_speaker_conditioning.py` (~350 lines)

**Modified Files:**
- `finetune/backbone/lm_model_wrapper.py` (+80 lines)
- `finetune/args.py` (+170 lines)

**Key Features:**
- ECAPA-TDNN speaker encoder (pretrained, frozen)
- Speaker conditioner with learnable projection and scale
- Integration via `sum_condition` mechanism in forward()

### 5.3 Phase 2: Reference Sampling Pipeline (COMPLETED)

**Modified Files:**
- `finetune/data/interleaver.py` (+350 lines)
  - Extended Sample/Batch classes
  - Added `_sample_speaker_reference()` method
  - Updated `get_interleaved_tokenizer()` factory
- `train.py` (+80 lines)
  - Speaker encoder/conditioner initialization
  - Training loop integration

**Key Features:**
- Automatic reference sampling from MOSHI channel
- Avoids target segment overlap
- Resampling 24kHz → 16kHz
- Configurable duration (3-10 seconds)

### 5.4 Current Configuration Example

```yaml
# example/korean_speaker_conditioning.yaml

# Speaker Conditioning (Phase 1+2)
speaker:
  enabled: true
  method: "encoder"  # "encoder" or "audio_prompt" (future)

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

# Korean Configuration
korean:
  enable_user_stream: false
  full_duplex_input: true

# Training
batch_size: 4
duration_sec: 60
max_steps: 10000
```

### 5.5 File Structure Summary

```
k-moshi-finetune/
├── finetune/
│   ├── backbone/
│   │   └── lm_model_wrapper.py    # ✅ Speaker conditioning integrated
│   ├── data/
│   │   ├── interleaver.py         # ✅ Reference sampling added
│   │   └── dataset.py
│   ├── modules/                   # ✅ NEW: Speaker conditioning modules
│   │   ├── __init__.py
│   │   ├── speaker_encoder.py
│   │   └── speaker_conditioner.py
│   └── args.py                    # ✅ Speaker conditioning args added
├── tests/
│   ├── __init__.py                # ✅ NEW
│   └── test_speaker_conditioning.py  # ✅ NEW
├── train.py                       # ✅ Speaker conditioning integrated
├── docs/
│   ├── SPEAKER_CONDITIONING_IMPLEMENTATION_LOG.md  # ✅ NEW
│   ├── K-MOSHI_SPEAKER_CONDITIONING_THEORY.md      # ✅ NEW (this file)
│   └── ...
└── example/
    └── korean_speaker_conditioning.yaml  # ✅ To be created
```

### 5.6 총 코드 변경량

| Category | Lines Added |
|----------|-------------|
| Phase 1: Speaker Modules | ~1,280 |
| Phase 2: Data Pipeline | ~550 |
| Documentation | ~800 |
| **Total** | **~2,630** |

### 5.7 Next Steps

1. **Integration Testing**: End-to-end test with speaker conditioning
2. **Phase 3**: VALL-E style audio prompt method
3. **Phase 4**: Training experiments with different configurations
4. **Phase 5**: Rust server integration for inference

---

## 참고 문헌

1. **Moshi Paper**: [arXiv:2410.00037](https://arxiv.org/abs/2410.00037)
2. **J-Moshi Paper**: [arXiv:2506.02979](https://arxiv.org/abs/2506.02979)
3. **ECAPA-TDNN**: [arXiv:2005.07143](https://arxiv.org/abs/2005.07143)
4. **VALL-E**: [arXiv:2301.02111](https://arxiv.org/abs/2301.02111)

---

*Document Version: 1.0*
*Created: 2026-01-21*
*Author: K-Moshi Development Team*
