# K-Moshi Zero-shot Speaker Conditioning 연구 방향서

**작성일**: 2026-01-21
**버전**: 1.0
**목적**: Hierarchical Speaker Conditioning 연구의 구체적 방향 정립

---

## 1. 연구 배경 요약

### 1.1 핵심 질문과 답변

| 질문 | 답변 | 근거 |
|------|------|------|
| Temporal TF의 `sum_condition`에 speaker embedding 주입이 올바른가? | ✅ 올바름 | 출력 생성 과정 전체를 조건화하는 역할 |
| USER audio 입력에 speaker embedding이 더해지는 것이 문제인가? | ⚠️ 부분적 우려 | Transformer self-attention이 분리 처리 가능, 실험적 검증 필요 |
| Depformer에 speaker conditioning 추가가 필요한가? | ✅ 권장 | 현재 화자 정보가 간접적으로만 전달됨 |
| Audio Prompt 방식이 필요한가? | ✅ Zero-shot에 필수 | Prosody/style의 시간적 특성 학습에 효과적 |

### 1.2 현재 Moshi 아키텍처의 Speaker Conditioning

```
현재 상태:
┌─────────────────────────────────────────────────────────────────┐
│ Temporal Transformer                                            │
│   sum_condition → 지원 (ConditionFuser.get_sum)                │
│   cross_attention → 지원 (ConditionFuser.get_cross)            │
│   prepend → 코드 존재하나 현재 사용 안 함                        │
├─────────────────────────────────────────────────────────────────┤
│ Depformer                                                       │
│   speaker conditioning → ❌ 없음                                │
│   cross_attention → ❌ 없음                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Speaker Embedding vs Audio Prompt: 역할 분리

### 2.1 개념적 분리

| 측면 | Speaker Embedding | Audio Prompt |
|------|------------------|--------------|
| **정보 유형** | Global identity (누구인가) | Local style (어떻게 말하는가) |
| **시간적 특성** | Time-invariant | Time-variant |
| **추출 방법** | Speaker encoder (ECAPA-TDNN) | Mimi audio codec |
| **표현 형태** | Single vector [1, 192] | Sequence [T, 8×1024] |
| **주입 위치** | sum_condition (Temporal TF) | prepend/cross (Temporal TF) |
| **Zero-shot 역할** | 화자 정체성 고정 | Prosody/style 전달 |

### 2.2 정보 분해 모델

```
Reference Audio (5-30초)
         │
         ├──────────────────────────────────────────┐
         │                                          │
         ▼                                          ▼
┌─────────────────────┐                 ┌──────────────────────┐
│  Speaker Encoder    │                 │   Mimi Encoder       │
│  (ECAPA-TDNN)       │                 │   (8 RVQ codebooks)  │
└─────────────────────┘                 └──────────────────────┘
         │                                          │
         ▼                                          ▼
┌─────────────────────┐                 ┌──────────────────────┐
│ speaker_emb [1,192] │                 │ audio_tokens [T, 8]  │
│                     │                 │                      │
│ 포함 정보:          │                 │ 포함 정보:           │
│ - Voice timbre      │                 │ - Prosody patterns   │
│ - Pitch range       │                 │ - Speaking rate      │
│ - Formant structure │                 │ - Intonation         │
│ - Speaker identity  │                 │ - Emotional tone     │
│                     │                 │ - Rhythm             │
└─────────────────────┘                 └──────────────────────┘
         │                                          │
         ▼                                          ▼
   [Global Condition]                      [Local Context]
   "이 사람의 목소리로"                    "이렇게 말하면서"
```

### 2.3 상호 보완성

```
Speaker Embedding Only:
- 화자 정체성 ✅
- 평균적 특성만 반영
- Prosody 다양성 ❌
- 단조로운 출력 경향

Audio Prompt Only:
- 순간적 style ✅
- Reference 의존적
- Identity 일관성 ❌
- 긴 발화시 drift

Hybrid (둘 다):
- 화자 정체성 ✅
- Prosody 다양성 ✅
- Identity 일관성 ✅
- 자연스러운 변화 ✅
```

---

## 3. 주입 위치 분석

### 3.1 Moshi의 기존 Conditioning 메커니즘

**ConditionFuser (base.py:349-436)**:

```python
class ConditionFuser:
    FUSING_METHODS = ["sum", "prepend", "cross"]

    def get_sum(self, conditions):
        """Return tensor to be added to each step. Shape: [B, 1, D]"""
        # 모든 timestep에 동일한 값 더함

    def get_cross(self, conditions):
        """Return tensor for cross attention. Shape: [B, T_cond, D]"""
        # Cross-attention을 통해 조건 참조

    def get_prepend(self, conditions):
        """Return tensor to prepend to sequence. Shape: [B, T_prefix, D]"""
        # 시퀀스 앞에 붙임
```

### 3.2 각 방식의 적합성

| 방식 | Speaker Embedding | Audio Prompt | 이유 |
|------|------------------|--------------|------|
| **sum** | ✅ 최적 | ❌ 부적합 | Global → 모든 step 동일하게 적용 |
| **cross** | ❌ 비효율 | ✅ 최적 | Sequence → Attention으로 참조 |
| **prepend** | ❌ 부적합 | ✅ 가능 | Sequence → Context로 전달 |

### 3.3 제안 설계

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROPOSED ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Reference Audio ─────┬─────────────────────────────────────┐   │
│                       │                                     │   │
│                       ▼                                     ▼   │
│  ┌─────────────────────────┐         ┌───────────────────────┐ │
│  │ Speaker Encoder         │         │ Mimi Encoder          │ │
│  │ (ECAPA-TDNN, frozen)    │         │ (frozen)              │ │
│  └───────────┬─────────────┘         └───────────┬───────────┘ │
│              │                                   │             │
│              ▼                                   ▼             │
│  ┌─────────────────────────┐         ┌───────────────────────┐ │
│  │ speaker_emb [B, 1, 192] │         │ ref_codes [B, T, 8]   │ │
│  └───────────┬─────────────┘         └───────────┬───────────┘ │
│              │                                   │             │
│              ▼                                   ▼             │
│  ┌─────────────────────────┐         ┌───────────────────────┐ │
│  │ Linear(192 → 4096)      │         │ Audio Embeddings      │ │
│  │ (learnable)             │         │ (shared with main)    │ │
│  └───────────┬─────────────┘         └───────────┬───────────┘ │
│              │                                   │             │
│              │ sum_condition                     │ cross_src   │
│              │ [B, 1, 4096]                      │ [B, T, 4096]│
│              │                                   │             │
│              └────────────┬──────────────────────┘             │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              TEMPORAL TRANSFORMER                         │ │
│  │                                                           │ │
│  │  input_ = text_emb + Σaudio_emb + sum_condition          │ │
│  │                                                           │ │
│  │  transformer(input_, cross_attention_src=cross_src)      │ │
│  │                                                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              DEPFORMER (Enhanced)                         │ │
│  │                                                           │ │
│  │  Option A: transformer_out에서 간접 전달 (현재)           │ │
│  │  Option B: acoustic_condition 직접 주입 (제안)            │ │
│  │                                                           │ │
│  │  depformer_input = proj(transformer_out)                 │ │
│  │                   + prev_token                            │ │
│  │                   + acoustic_spk_emb  ← NEW              │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 구현 로드맵

### Phase 1: Temporal TF Speaker Embedding (2-3주)

**목표**: 기존 ConditionFuser 활용하여 speaker embedding 주입

**구현 내용**:
1. ECAPA-TDNN speaker encoder 통합
2. Speaker embedding → dim 4096 projection
3. ConditionProvider에 speaker conditioner 등록
4. ConditionFuser에 "sum" 방식으로 등록

**코드 변경 위치**:
- `moshi/conditioners/`: 새 SpeakerConditioner 클래스
- `train.py`: Speaker encoder 로드 및 condition 생성
- `finetune/data/`: Reference audio 로딩

**예상 결과**:
- Speaker similarity: +5-8%
- 구현 복잡도: 낮음

### Phase 2: Audio Prompt Cross-Attention (3-4주)

**목표**: Reference audio를 cross-attention으로 전달

**구현 내용**:
1. Reference audio → Mimi encoding
2. Audio embeddings 생성 (8 codebook sum)
3. ConditionFuser에 "cross" 방식으로 등록
4. Temporal Transformer cross-attention 활성화

**코드 변경 위치**:
- `moshi/conditioners/`: AudioPromptConditioner 클래스
- `moshi/models/lm.py`: cross_attention_src 연결
- `finetune/data/interleaver.py`: Reference 처리

**예상 결과**:
- Prosody similarity: +3-5%
- 구현 복잡도: 중간

### Phase 3: Depformer Acoustic Conditioning (2-3주)

**목표**: Depformer에 직접 speaker conditioning 추가

**구현 내용**:
1. Acoustic speaker embedding 생성 (별도 projection)
2. `forward_depformer_training()` 수정
3. `forward_depformer()` 수정
4. Codebook별 conditioning 강도 조절 (옵션)

**코드 변경 위치**:
- `moshi/models/lm.py`: forward_depformer* 함수들
- `finetune/wrapped_model.py`: 새 파라미터 전달

**예상 결과**:
- Fine acoustic detail: +2-4%
- 구현 복잡도: 중간

### Phase 4: 통합 및 최적화 (2-3주)

**목표**: 전체 시스템 통합 및 성능 최적화

**구현 내용**:
1. Phase 1-3 통합
2. Loss balancing 조정
3. Inference pipeline 최적화
4. Streaming support 검증

---

## 5. Streaming/Inference 시나리오 분석

### 5.1 Session 초기화 시점

```python
# Inference 시작 시 한 번만 실행
def initialize_session(reference_audio, speaker_encoder, mimi):
    # 1. Speaker embedding 추출
    speaker_emb = speaker_encoder(reference_audio)  # [1, 192]

    # 2. Audio prompt 생성
    ref_codes = mimi.encode(reference_audio)  # [1, T_ref, 8]

    # 3. ConditionTensors 구성
    condition_tensors = {
        "speaker": ConditionType(project(speaker_emb), mask),
        "audio_prompt": ConditionType(encode_prompt(ref_codes), mask),
    }

    # 4. LMGen 초기화
    lm_gen = LMGen(lm_model, condition_tensors=condition_tensors)
    return lm_gen
```

### 5.2 Streaming State 관리

```python
# LMGen._init_streaming_state()에서
def _init_streaming_state(self, batch_size):
    # sum_condition은 캐싱됨
    condition_sum = self.lm_model.fuser.get_sum(self.condition_tensors)

    # cross_attention_src도 캐싱
    condition_cross = self.lm_model.fuser.get_cross(self.condition_tensors)

    # KV cache에 cross-attention 결과 저장
    # → 이후 step에서 재계산 불필요
```

### 5.3 Turn-level Speaker Switch (고급)

```python
# 대화 중 화자 변경 시나리오 (미래 확장)
def switch_speaker(lm_gen, new_reference_audio):
    # 새 speaker embedding 계산
    new_speaker_emb = speaker_encoder(new_reference_audio)

    # State 업데이트 (KV cache는 유지하되 condition만 변경)
    lm_gen.state.condition_sum = project(new_speaker_emb)

    # Cross-attention cache는 선택적 invalidation
    # (전체 재계산 vs 부분 업데이트)
```

---

## 6. 학습 데이터 구성

### 6.1 데이터 형식

```
각 학습 샘플:
┌─────────────────────────────────────────────────────────────────┐
│ reference_audio: [24kHz, mono, 5-30초] - 화자 reference        │
│ dialogue_audio:  [24kHz, stereo, 10-100초] - 대화 음성         │
│                  L: MOSHI output (target speaker = reference)   │
│                  R: USER input (다른 화자)                      │
│ alignments:      [{word, (start, end), speaker}, ...]          │
└─────────────────────────────────────────────────────────────────┘

요구사항:
- reference_audio와 dialogue_audio의 MOSHI 채널은 동일 화자
- Voice Conversion 불필요 (same speaker training)
```

### 6.2 데이터 로딩 파이프라인

```python
# finetune/data/dataset.py 확장

class SpeakerConditionedDataset(Dataset):
    def __getitem__(self, idx):
        item = self.base_data[idx]

        # 기존: dialogue audio + alignments
        dialogue_audio = load_stereo(item['path'])
        alignments = load_alignments(item['path'])

        # 추가: reference audio (같은 화자의 다른 발화)
        reference_path = self.get_reference_for_speaker(item['speaker_id'])
        reference_audio = load_mono(reference_path)

        return {
            'dialogue_audio': dialogue_audio,
            'alignments': alignments,
            'reference_audio': reference_audio,
            'speaker_id': item['speaker_id'],
        }
```

### 6.3 Multi-speaker 데이터셋 구조

```
data/
├── speaker_001/
│   ├── dialogue_001.wav  # 대화 (stereo)
│   ├── dialogue_001.json # alignments
│   ├── reference_001.wav # reference (mono, same speaker)
│   ├── reference_002.wav
│   └── ...
├── speaker_002/
│   └── ...
└── manifest.jsonl
    # {"speaker_id": "001", "dialogue": "...", "references": ["..."]}
```

---

## 7. 평가 지표

### 7.1 객관적 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| **Speaker Similarity** | Reference와 생성 음성의 speaker embedding cosine similarity | > 0.85 |
| **WER** | ASR로 측정한 Word Error Rate | < 10% |
| **MCD** | Mel Cepstral Distortion | < 5.0 dB |
| **F0 RMSE** | Pitch contour 유사도 | < 20 Hz |

### 7.2 주관적 지표 (MOS)

| 지표 | 설명 | 목표 |
|------|------|------|
| **Naturalness MOS** | 자연스러움 (1-5) | > 4.0 |
| **Speaker Similarity MOS** | 화자 유사성 (1-5) | > 4.0 |
| **Overall MOS** | 전체 품질 | > 3.8 |

### 7.3 Zero-shot 평가 프로토콜

```
1. Seen speakers (학습 화자):
   - 학습 중 본 화자로 생성
   - Baseline 성능 측정

2. Unseen speakers (미학습 화자):
   - 학습에 없던 화자로 생성
   - Zero-shot 능력 측정

3. Cross-lingual (다국어):
   - 한국어 reference → 한국어 생성
   - 향후: 영어 reference → 한국어 생성
```

---

## 8. 위험 요소 및 완화 전략

### 8.1 기술적 위험

| 위험 | 가능성 | 영향 | 완화 전략 |
|------|--------|------|----------|
| Speaker embedding이 Depformer까지 전달 안 됨 | 중 | 높 | Phase 3에서 직접 주입 |
| Cross-attention 메모리 폭발 | 낮 | 중 | Reference 길이 제한 (5초) |
| Speaker-content entanglement | 중 | 중 | Disentanglement loss 추가 |
| 학습 불안정 | 낮 | 중 | 단계적 fine-tuning |

### 8.2 데이터 위험

| 위험 | 가능성 | 영향 | 완화 전략 |
|------|--------|------|----------|
| 화자 다양성 부족 | 중 | 높 | 다양한 데이터셋 수집 |
| Reference-target 불일치 | 낮 | 높 | 철저한 화자 검증 |
| 노이즈/품질 문제 | 중 | 중 | Audio preprocessing |

---

## 9. 결론 및 권장 사항

### 9.1 핵심 결론

1. **Temporal TF sum_condition**은 speaker embedding 주입에 **적합**
2. **Depformer conditioning 부재**는 fine acoustic detail 제어의 한계
3. **Hybrid approach** (Speaker Embedding + Audio Prompt)가 Zero-shot에 최적
4. **기존 ConditionFuser 활용** 가능하여 구현 효율적

### 9.2 권장 구현 순서

```
Week 1-3:  Phase 1 - Temporal TF Speaker Embedding
Week 4-7:  Phase 2 - Audio Prompt Cross-Attention
Week 8-10: Phase 3 - Depformer Acoustic Conditioning
Week 11-13: Phase 4 - 통합 및 최적화
```

### 9.3 성공 기준

| Milestone | 기준 | 시점 |
|-----------|------|------|
| Phase 1 완료 | Seen speaker similarity > 0.80 | Week 3 |
| Phase 2 완료 | Unseen speaker similarity > 0.75 | Week 7 |
| Phase 3 완료 | Naturalness MOS > 3.8 | Week 10 |
| 최종 목표 | Zero-shot speaker similarity > 0.85, MOS > 4.0 | Week 13 |

---

## 10. 참고 문헌

1. **Moshi Paper**: Défossez et al., "Moshi: A Speech-Text Foundation Model for Real-Time Dialogue" (2024)
2. **VALL-E**: Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (2023)
3. **FlashLabs Chroma 1.0**: arXiv:2601.11141 (2026) - Hybrid speaker conditioning approach
4. **ECAPA-TDNN**: Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention..." (2020)
5. **PersonaPlex**: NVIDIA's multi-speaker dialogue system

---

**문서 작성 완료**: 2026-01-21
**다음 단계**: Phase 1 구현 설계서 작성

---

## 부록: 코드 참조

### A. 핵심 파일 목록

| 파일 | 역할 | 수정 필요 여부 |
|------|------|---------------|
| `moshi/models/lm.py` | LMModel, Depformer | ✅ Phase 3 |
| `moshi/conditioners/base.py` | ConditionFuser | ✅ Phase 1-2 |
| `finetune/data/dataset.py` | 데이터 로딩 | ✅ 전체 |
| `finetune/wrapped_model.py` | 모델 래핑 | ✅ Phase 1-3 |
| `train.py` | 학습 루프 | ✅ Phase 1-3 |

### B. 주요 함수 시그니처

```python
# lm.py
def forward_text(self, sequence, sum_condition=None, cross_attention_src=None)
def forward_depformer_training(self, sequence, transformer_out)
def forward_depformer(self, depformer_cb_index, sequence, transformer_out)

# base.py
class ConditionFuser:
    def get_sum(self, conditions) -> torch.Tensor | None
    def get_cross(self, conditions) -> torch.Tensor | None
    def get_prepend(self, conditions) -> torch.Tensor | None
```
