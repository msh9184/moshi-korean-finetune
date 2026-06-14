# Moshi Dual-Transformer 아키텍처 정밀 분석

**작성일**: 2026-01-21
**목적**: Speaker Conditioning 연구 방향 구체화를 위한 Temporal/Depth Transformer 동작 원리 엄밀 분석

---

## 1. 개요 및 분석 목표

이 문서는 다음 핵심 질문들에 대한 **정확하고 엄밀한** 분석을 제공합니다:

1. **Temporal Transformer의 `sum_condition` 주입**: USER 오디오가 입력인 상황에서 MOSHI 출력 화자 임베딩을 모든 프레임에 더하는 것이 의미론적으로 올바른가?
2. **Depth Transformer의 정밀 동작**: timestep S에서 S+1로 넘어가는 과정의 변수 흐름과 각 변수가 모델링하는 바
3. **Speaker Conditioning 주입 가능 지점**: Temporal vs Depth Transformer에서의 역할 분리

---

## 2. Moshi 아키텍처 전체 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MOSHI DUAL-TRANSFORMER ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT (timestep t):                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  codes[B, 9, T] = [text_tokens, audio_cb0, audio_cb1, ..., audio_cb7]│   │
│  │                    ↑ MOSHI text   ↑ USER audio (8 Mimi codebooks)    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              TEMPORAL TRANSFORMER (7B params, ~32 layers)            │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  input_ = text_emb(codes[:,0])                                 │  │   │
│  │  │         + Σ audio_emb[i](codes[:,i+1]) for i in 0..7          │  │   │
│  │  │         + sum_condition  ← SPEAKER CONDITIONING HERE          │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  transformer_out = self.transformer(input_)                    │  │   │
│  │  │  Shape: [B, T, dim=4096]                                       │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                       │   │
│  │                              ├──────────────────┐                    │   │
│  │                              ▼                  ▼                    │   │
│  │  ┌──────────────────────┐   ┌──────────────────────────────────┐   │   │
│  │  │ text_logits =        │   │ depformer_in[k](transformer_out) │   │   │
│  │  │ text_linear(out_norm)│   │ Projects to depformer_dim=1024   │   │   │
│  │  │ → MOSHI TEXT OUTPUT  │   └──────────────────────────────────┘   │   │
│  │  └──────────────────────┘                   │                       │   │
│  └─────────────────────────────────────────────│───────────────────────┘   │
│                                                │                            │
│                                                ▼                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              DEPTH TRANSFORMER (Depformer, ~300M params, 6 layers)   │   │
│  │                                                                       │   │
│  │  FOR EACH timestep t in T (independently, batched as B*T):           │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  FOR k = 0 to dep_q-1 (sequentially, 8 codebooks):             │  │   │
│  │  │    if k == 0:                                                   │  │   │
│  │  │      token_in = depformer_text_emb(text_token[t])              │  │   │
│  │  │    else:                                                        │  │   │
│  │  │      token_in = depformer_emb[k-1](audio_token[k-1, t])        │  │   │
│  │  │                                                                 │  │   │
│  │  │    depformer_input = depformer_in[k](transformer_out[t])       │  │   │
│  │  │                    + token_in                                   │  │   │
│  │  │                                                                 │  │   │
│  │  │    ┌─ NO SPEAKER CONDITION INJECTION POINT ─┐                  │  │   │
│  │  │    │  dep_output = depformer(depformer_input)                  │  │   │
│  │  │    │  audio_logits[k] = linear[k](norm[k](dep_output))         │  │   │
│  │  │    └────────────────────────────────────────┘                  │  │   │
│  │  │                                                                 │  │   │
│  │  │    audio_token[k, t] = sample(audio_logits[k])  ← OUTPUT       │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OUTPUT:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  text_logits[B, 1, T, text_card]  → MOSHI가 말하는 텍스트            │   │
│  │  audio_logits[B, 8, T, card]      → MOSHI 음성 (8 Mimi codebooks)   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Temporal Transformer 상세 분석

### 3.1 `forward_text()` 함수 분석

**파일**: `moshi/moshi/models/lm.py:379-408`

```python
def forward_text(self, sequence, sum_condition=None, cross_attention_src=None):
    """
    Args:
        sequence: [B, K, S] where K = 1(text) + 8(audio) = 9
        sum_condition: [B, 1, D] speaker/condition embedding
    """
    B, K, S = sequence.shape
    input_sequence = _delay_sequence(sequence, self.delays, ...)

    # Step 1: Audio embedding summation (USER의 8개 Mimi codebook)
    input_ = None
    for cb_index in range(self.num_audio_codebooks):  # 0..7
        audio_emb = self.emb[cb_index](input_sequence[:, cb_index + self.audio_offset])
        input_ = audio_emb if input_ is None else input_ + audio_emb

    # Step 2: Text embedding addition (MOSHI의 inner monologue)
    text_emb = self.text_emb(input_sequence[:, 0])
    input_ = text_emb if input_ is None else input_ + text_emb

    # Step 3: Speaker condition addition  핵심 분석 지점
    if sum_condition is not None:
        input_ = input_ + sum_condition.to(input_)

    # Step 4: Transformer forward
    transformer_out = self.transformer(input_, cross_attention_src=cross_attention_src)
```

### 3.2 `sum_condition` 의미론 분석

#### 입력 데이터 구성 (Full-Duplex 모드)

| Channel | 내용 | 출처 |
|---------|------|------|
| `codes[:, 0, :]` | MOSHI text tokens | Ground Truth (학습) 또는 이전 예측 (추론) |
| `codes[:, 1:9, :]` | USER audio codes | Mimi로 인코딩된 사용자 음성 |

#### `sum_condition` 주입 시점의 텐서 상태

```
input_ = text_emb[MOSHI] + Σ audio_emb[USER]
       ↑                   ↑
       MOSHI가 말할 텍스트   USER가 말하는 음성

sum_condition → MOSHI 출력 화자의 speaker embedding
```

### 3.3 의미론적 올바름 분석

**질문**: USER 오디오 + MOSHI 텍스트가 합쳐진 `input_`에 MOSHI speaker embedding을 더하는 것이 올바른가?

**분석**:

1. **Temporal Transformer의 역할**:
   - 입력: USER가 말하는 것 (audio) + MOSHI가 말할 것 (text, teacher forcing)
   - 출력: 다음 timestep의 MOSHI text logits + Depformer 입력 (context)

2. **sum_condition의 의도된 역할**:
   - "이 context에서 MOSHI가 생성해야 할 출력의 화자 특성"
   - USER 입력을 이해하고, **그에 맞게 MOSHI 출력을 생성**하는 전체 과정을 조건화

3. **의미론적 해석**:
   ```
   Temporal Transformer 목적:
   "USER가 이렇게 말했고(audio_emb), MOSHI는 이런 내용을 말해야 하는데(text_emb),
    이 MOSHI는 [sum_condition] 화자 특성을 가진 존재로 말해야 한다"
   ```

4. **결론**: ✅ **의미론적으로 올바름**
   - `sum_condition`은 "USER 입력에 대한 조건"이 아니라 "MOSHI 출력에 대한 조건"
   - Transformer는 양방향 정보를 통합하여 MOSHI 출력을 생성
   - 화자 임베딩은 **출력 생성 과정 전체를 조건화**하는 것이 맞음

### 3.4 잠재적 문제점

**문제 1: 모든 timestep에 동일한 speaker embedding**
```
t=0: input_ + speaker_emb
t=1: input_ + speaker_emb  ← 동일한 값
t=2: input_ + speaker_emb  ← 동일한 값
...
```

- **이슈**: Prosody/style의 시간적 변화를 표현하지 못함
- **해결책**: Frame-level prosody 정보 추가 (Audio Prompt 방식)

**문제 2: 화자 특성이 USER 표현에도 영향**
- USER audio embedding에도 speaker_emb가 더해짐
- 이론적으로 USER 음성 이해에 편향 가능
- 실제로는 Transformer의 self-attention이 이를 분리하여 처리 가능

---

## 4. Depth Transformer (Depformer) 정밀 분석

### 4.1 핵심 코드 분석

#### 4.1.1 Training Mode: `forward_depformer_training()`

**파일**: `moshi/moshi/models/lm.py:410-448`

```python
def forward_depformer_training(self, sequence, transformer_out):
    """
    Training mode: 모든 timestep과 codebook을 병렬 처리

    Args:
        sequence: [B, K, T] - 전체 시퀀스
        transformer_out: [B, T, dim] - Temporal Transformer 출력
    """
    B, K, T = sequence.shape
    Ka = self.dep_q  # 8 (output codebooks)

    depformer_inputs = []
    for cb_index in range(Ka):
        # Temporal→Depformer 프로젝션 (codebook별 다른 linear 가능)
        if self.depformer_multi_linear:
            transformer_in = self.depformer_in[cb_index](transformer_out)
        else:
            transformer_in = self.depformer_in[0](transformer_out)

        # 이전 토큰 임베딩 (autoregressive)
        if cb_index == 0:
            # 첫 codebook: text token을 조건으로
            token_in = self.depformer_text_emb(sequence[:, 0])  # MOSHI text
        else:
            # 이후 codebook: 이전 audio codebook을 조건으로
            token_in = self.depformer_emb[cb_index - 1](
                sequence[:, cb_index + self.audio_offset - 1]
            )

        depformer_inputs.append(token_in + transformer_in)

    # [B, T, Ka, depformer_dim] → [B*T, Ka, depformer_dim]
    depformer_input = torch.stack(depformer_inputs, dim=2)
    depformer_input = depformer_input.view(B * T, Ka, -1)

    # 핵심: 각 timestep이 독립적으로 처리됨 (B*T batch)
    depformer_output = self.depformer(depformer_input)
```

#### 4.1.2 Inference Mode: `forward_depformer()` (Streaming)

**파일**: `moshi/moshi/models/lm.py:450-493`

```python
def forward_depformer(self, depformer_cb_index, sequence, transformer_out):
    """
    Streaming mode: 한 번에 1 timestep, 1 codebook만 처리

    Args:
        depformer_cb_index: 현재 처리할 codebook index (0~7)
        sequence: [B, 1, 1] - 이전 토큰
        transformer_out: [B, 1, dim] - 현재 timestep의 Temporal 출력
    """
    B, K, S = sequence.shape
    assert K == 1 and S == 1  # 1 codebook, 1 step

    # 프로젝션
    depformer_input = self.depformer_in[index](transformer_out)

    # 이전 토큰 임베딩
    if depformer_cb_index == 0:
        last_token_input = self.depformer_text_emb(sequence[:, 0])
    else:
        last_token_input = self.depformer_emb[depformer_cb_index - 1](sequence[:, 0])

    depformer_input = depformer_input + last_token_input

    # Streaming state 활용: 레이어별 순차 처리
    dep_output = self.depformer(depformer_input)

    logits = self.linears[depformer_cb_index](
        self.depformer_norms[depformer_cb_index](dep_output)
    )
    return logits
```

### 4.2 Timestep 간 변수 전달 분석

```
═══════════════════════════════════════════════════════════════════════════════
                    DEPFORMER EXECUTION FLOW (Timestep S → S+1)
═══════════════════════════════════════════════════════════════════════════════

TIMESTEP S:
┌────────────────────────────────────────────────────────────────────────────┐
│  transformer_out[S] ─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐      │
│  (from Temporal TF)      │     │     │     │     │     │     │     │      │
│                          ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼      │
│                    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐      │
│  depformer_in[k]:  │ k=0 │ k=1 │ k=2 │ k=3 │ k=4 │ k=5 │ k=6 │ k=7 │      │
│                    └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘      │
│                       │     │     │     │     │     │     │     │         │
│  Previous token:      │     │     │     │     │     │     │     │         │
│  ┌────────────────────┤     │     │     │     │     │     │     │         │
│  │ text_token[S]  ────┘     │     │     │     │     │     │     │         │
│  │ audio[0,S] ──────────────┘     │     │     │     │     │     │         │
│  │ audio[1,S] ────────────────────┘     │     │     │     │     │         │
│  │ audio[2,S] ──────────────────────────┘     │     │     │     │         │
│  │ audio[3,S] ────────────────────────────────┘     │     │     │         │
│  │ audio[4,S] ──────────────────────────────────────┘     │     │         │
│  │ audio[5,S] ────────────────────────────────────────────┘     │         │
│  │ audio[6,S] ──────────────────────────────────────────────────┘         │
│  └────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Depformer Input[k] = depformer_in[k](transformer_out[S]) + prev_token[k] │
│                                                                            │
│                    ┌─────────────────────────────┐                         │
│                    │      DEPFORMER (6 layers)    │                         │
│                    │   ┌───────────────────────┐ │                         │
│                    │   │  weights_per_step[k]  │ │ ← Codebook별 다른 가중치│
│                    │   │  Self-Attention       │ │   (선택적)              │
│                    │   │  Feed Forward         │ │                         │
│                    │   └───────────────────────┘ │                         │
│                    └─────────────────────────────┘                         │
│                                    │                                       │
│                                    ▼                                       │
│  Output:   ┌─────────────────────────────────────────────────────────┐    │
│            │ audio_logits[0,S], [1,S], [2,S], [3,S], [4,S], [5,S],   │    │
│            │ [6,S], [7,S] → MOSHI OUTPUT AUDIO TOKENS FOR TIMESTEP S │    │
│            └─────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ (timestep 경계)
                                    ▼
TIMESTEP S+1:
┌────────────────────────────────────────────────────────────────────────────┐
│  transformer_out[S+1] ← 새로운 Temporal TF 출력                            │
│                                                                            │
│  ★ 핵심: S에서 생성된 audio_logits는 S+1로 직접 전달되지 않음               │
│         S+1의 입력은 다시 sequence[:, :, S+1]에서 옴 (teacher forcing)     │
│         또는 inference 시 S에서 샘플링된 토큰이 sequence에 저장됨           │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 핵심 발견: Timestep 독립성

**Training에서**:
```python
depformer_input = depformer_input.view(B * T, Ka, -1)
# → 각 timestep이 완전히 독립적으로 처리됨
# → timestep S와 S+1 사이에 hidden state 공유 없음
```

**Inference에서**:
```python
with lm_model.depformer.streaming(B_cfg):
    for cb_index in range(lm_model.dep_q):
        # Streaming context 내에서 codebook별 순차 처리
        # 하지만 이 streaming은 "codebook 축"에서의 것
        # timestep 간 state는 별도로 관리
```

### 4.4 Depformer가 모델링하는 것

| 변수 | 역할 | 모델링 대상 |
|------|------|------------|
| `transformer_out[t]` | Temporal TF의 context | 현재 timestep의 대화 맥락, USER 입력 이해, MOSHI 응답 의도 |
| `token_in` (k=0) | MOSHI text token | 발화 내용의 semantic 정보 |
| `token_in` (k>0) | 이전 audio codebook | 이전 codebook의 acoustic 특성 (계층적 음성 생성) |
| `depformer_output` | 최종 표현 | 해당 timestep의 해당 codebook acoustic 특성 |
| `audio_logits[k]` | 출력 분포 | k번째 codebook의 토큰 확률 |

### 4.5 Depformer의 Speaker Conditioning 부재

**현재 상태**:
```python
# forward_depformer_training에서
depformer_input = token_in + transformer_in
# ← speaker_condition이 없음!

# forward_depformer에서도 마찬가지
depformer_input = depformer_input + last_token_input
# ← speaker_condition이 없음!
```

**의미**:
- Depformer는 **화자 독립적**으로 acoustic 토큰 생성
- 화자 특성은 오직 Temporal Transformer의 `transformer_out`을 통해 **간접적으로만** 전달
- **Fine acoustic detail** (prosody, speaker timbre)에 대한 직접적 조건화 불가

---

## 5. Speaker Conditioning 전략 분석

### 5.1 Option A: Temporal Transformer Only (현재 구조)

```
┌─────────────────────────────────────────────────────────────┐
│ Temporal Transformer                                        │
│   input_ = text_emb + Σaudio_emb + speaker_emb             │
│                                    ↑                        │
│                        [Global Speaker Identity]            │
│                                                             │
│   → transformer_out (화자 정보가 암묵적으로 인코딩됨)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Depformer                                                   │
│   depformer_input = project(transformer_out) + prev_token  │
│   (화자 정보는 transformer_out에 녹아있음, 직접 조건화 없음) │
└─────────────────────────────────────────────────────────────┘
```

**장점**:
- 구현 단순
- 기존 아키텍처 변경 최소화

**단점**:
- Depformer에서 화자 정보 희석
- Fine-grained acoustic control 불가
- Prosody/style 제어 어려움

### 5.2 Option B: Dual-Level Speaker Conditioning (제안)

```
┌─────────────────────────────────────────────────────────────┐
│ Temporal Transformer                                        │
│   input_ = text_emb + Σaudio_emb + speaker_emb_global      │
│                                    ↑                        │
│                        [Global Speaker Identity: 192D]      │
│                        "누구의 목소리인가"                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Depformer (Modified)                                        │
│   depformer_input = project(transformer_out)               │
│                   + prev_token                              │
│                   + speaker_emb_acoustic  ← NEW!           │
│                     ↑                                       │
│         [Acoustic Speaker Features: 1024D]                  │
│         "이 화자의 음색/prosody 특성"                        │
└─────────────────────────────────────────────────────────────┘
```

**구현 방안**:

```python
# lm.py 수정 (forward_depformer_training)
def forward_depformer_training(self, sequence, transformer_out,
                                speaker_condition_acoustic=None):  # NEW
    depformer_inputs = []
    for cb_index in range(Ka):
        transformer_in = self.depformer_in[cb_index](transformer_out)

        if cb_index == 0:
            token_in = self.depformer_text_emb(sequence[:, 0])
        else:
            token_in = self.depformer_emb[cb_index - 1](...)

        depformer_input = token_in + transformer_in

        # NEW: Acoustic speaker conditioning
        if speaker_condition_acoustic is not None:
            depformer_input = depformer_input + speaker_condition_acoustic

        depformer_inputs.append(depformer_input)
```

### 5.3 Option C: Hierarchical Audio Prompt (Zero-shot 최적)

```
┌─────────────────────────────────────────────────────────────┐
│ Speaker Encoder (ECAPA-TDNN)                                │
│   reference_audio → global_speaker_emb [1, 192]            │
│   "화자 정체성 요약"                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Temporal Transformer                                        │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ [REF_AUDIO_TOKENS] [SEP] [USER_AUDIO + MOSHI_TEXT]  │  │
│   │  ↑                        ↑                          │  │
│   │  Reference audio prompt   Actual dialogue           │  │
│   │  (Mimi encoded, ~3-5sec)                             │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   sum_condition = project(global_speaker_emb)              │
│   input_ = ... + sum_condition                             │
└─────────────────────────────────────────────────────────────┘
```

**Audio Prompt 역할**:
- Cross-attention 또는 Prefix로 제공
- Prosody, speaking rate, style 정보 포함
- Frame-level 정보로 시간적 특성 학습 가능

---

## 6. 각 변수가 모델링하는 바 (요약)

### 6.1 Temporal Transformer

| 변수 | Shape | 모델링 대상 |
|------|-------|------------|
| `text_emb` | [B, T, 4096] | MOSHI가 말할 내용의 semantic 의미 |
| `audio_emb[0..7]` | [B, T, 4096] | USER 음성의 acoustic/semantic 표현 |
| `sum_condition` | [B, 1, 4096] | MOSHI 출력의 global speaker identity |
| `input_` | [B, T, 4096] | 대화 맥락 + 출력 조건의 통합 표현 |
| `transformer_out` | [B, T, 4096] | 다음 토큰 생성을 위한 context (화자 정보 포함) |
| `text_logits` | [B, T, 32000] | MOSHI text 토큰 분포 |

### 6.2 Depformer

| 변수 | Shape | 모델링 대상 |
|------|-------|------------|
| `transformer_in` | [B, T, 1024] | Temporal context의 Depformer 차원 투영 |
| `token_in` (k=0) | [B, T, 1024] | MOSHI text의 semantic 정보 |
| `token_in` (k>0) | [B, T, 1024] | 이전 codebook의 acoustic 정보 |
| `depformer_input` | [B*T, K, 1024] | 현재 codebook 생성을 위한 조건 |
| `dep_output` | [B*T, K, 1024] | acoustic 특성 표현 |
| `audio_logits[k]` | [B, T, 1024] | k번째 codebook 토큰 분포 |

---

## 7. 연구 방향 제언

### 7.1 단기 목표 (Phase 1)

1. **Temporal Transformer sum_condition 활용** (현재 구조 유지)
   - ECAPA-TDNN speaker encoder 통합
   - `sum_condition = project(speaker_emb, dim=4096)`
   - 기존 `ConditionFuser` 메커니즘 활용

2. **학습 데이터 구성**
   - Reference audio: 동일 화자 5-30초
   - Target audio: 동일 화자의 다른 발화
   - Voice Conversion 불필요 (same speaker pair)

### 7.2 중기 목표 (Phase 2)

1. **Depformer Acoustic Conditioning 추가**
   - `speaker_condition_acoustic` 파라미터 추가
   - Codebook별 다른 conditioning 강도 (k=0 강, k=7 약)

2. **Audio Prompt 방식 도입**
   - Reference audio를 Mimi로 인코딩
   - Temporal Transformer 입력에 prefix로 추가
   - Cross-attention 또는 concat 방식 비교

### 7.3 장기 목표 (Phase 3)

1. **Hybrid Conditioning**
   - Global: Speaker encoder → Temporal TF sum_condition
   - Local: Audio prompt → Cross-attention
   - Acoustic: Acoustic features → Depformer conditioning

2. **Zero-shot Voice Cloning 평가**
   - Speaker similarity (cosine similarity on speaker embeddings)
   - MOS (naturalness)
   - WER (intelligibility)

---

## 8. Streaming/Inference 시나리오 분석

### 8.1 현재 Inference Flow

```python
# LMGen._init_streaming_state()
condition_sum = self.lm_model.fuser.get_sum(self.condition_tensors)
# → session 시작 시 한 번 계산되어 캐싱

# LMGen._step() 매 timestep마다
transformer_out = state.graphed_main(input_, state.condition_sum, ...)
# → 캐싱된 condition_sum 재사용
```

### 8.2 Speaker Conditioning 적용 방안

**Option 1: Session-level (현재 가능)**
```python
# Session 시작 시
speaker_emb = speaker_encoder(reference_audio)
condition_tensors = {"speaker": speaker_emb}
lm_gen = LMGen(lm_model, condition_tensors=condition_tensors)

# 모든 step에서 동일한 speaker_emb 사용
```

**Option 2: Turn-level (추가 구현 필요)**
```python
# 대화 턴마다 새로운 reference 가능
def update_speaker(self, new_reference_audio):
    new_emb = speaker_encoder(new_reference_audio)
    self.state.condition_sum = project(new_emb)
```

### 8.3 Audio Prompt as System Prompt

```
Streaming Context:
┌────────────────────────────────────────────────────────────────┐
│ [Audio Prompt: 3-5초 reference]                                │
│      ↓ (fixed, 처음에만 처리)                                  │
│ [SEP]                                                          │
│      ↓                                                         │
│ [Turn 1: USER audio | MOSHI response]                         │
│ [Turn 2: USER audio | MOSHI response]                         │
│ ...                                                            │
└────────────────────────────────────────────────────────────────┘

- Audio prompt은 KV cache에 저장
- 이후 턴에서 cross-attention으로 참조
- Memory efficient (prompt 반복 계산 불필요)
```

---

## 9. 결론

### 9.1 핵심 발견

1. **Temporal Transformer sum_condition**: USER audio + MOSHI text 통합 입력에 speaker embedding을 더하는 것은 **의미론적으로 올바름**. "출력 생성 과정 전체를 조건화"하는 역할.

2. **Depformer의 독립성**: 각 timestep이 완전히 독립적으로 처리되며, timestep 간 hidden state 공유 없음. **현재 speaker conditioning 없음**.

3. **Speaker 정보 흐름**:
   - Temporal TF: 직접 주입 (sum_condition)
   - Depformer: 간접 전달 (transformer_out에 인코딩)

### 9.2 권장 구현 순서

1. **즉시**: Temporal TF sum_condition으로 ECAPA-TDNN speaker encoder 통합
2. **다음**: Depformer acoustic conditioning 추가
3. **이후**: Audio prompt 방식 + Cross-attention 구현

### 9.3 예상 성능 향상

| 구성 | Speaker Similarity | 구현 난이도 |
|------|-------------------|------------|
| Temporal only | +5-8% | 낮음 |
| + Depformer conditioning | +3-5% | 중간 |
| + Audio prompt | +2-4% | 높음 |
| Full hybrid | +10-15% (Chroma 수준) | 높음 |

---

**문서 작성 완료**: 2026-01-21
**다음 단계**: Speaker Encoder 통합 구현 설계서 작성
