# K-Moshi 아키텍처 제안서 v2.0

> **작성일**: 2026-01-20
> **목적**: K-Moshi 최적 아키텍처 선정 및 구현 방향 제안
> **기반 분석**: FULL_DUPLEX_ARCHITECTURE_ANALYSIS.md

---

## 1. 요약 (Executive Summary)

### 1.1 핵심 결론

| 항목 | Moshi 원본 | PersonaPlex | F-actor | **K-Moshi 제안** |
|------|-----------|-------------|---------|------------------|
| Voice Control | ❌ 없음 | Audio Token Cache | ECAPA-TDNN | **ECAPA-TDNN** |
| System Prompt | ❌ 없음 | Text Prompt | Instruction Prefix | **Instruction Prefix** |
| Full-Duplex | Dual-stream RVQ | Moshi 동일 | Dual-stream FSQ | **Dual-stream RVQ** (유지) |
| 학습 데이터 | 7M시간 | 3,467시간 | 2,000시간 | **1,000-2,000시간** |
| 학습 효율 | - | Fine-tuning | 48시간/4xA100 | **Phase별 점진적** |

### 1.2 권장 전략: **하이브리드 3-Phase 접근법**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    K-Moshi 하이브리드 전략                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1 (4주)         Phase 2 (4주)         Phase 3 (4주)                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐               │
│  │ PersonaPlex   │    │ F-actor       │    │ 최적화        │               │
│  │ 스타일        │───▶│ 스타일        │───▶│ (선택)        │               │
│  │               │    │               │    │               │               │
│  │ • Voice.pt    │    │ • ECAPA-TDNN  │    │ • FSQ 검토    │               │
│  │ • 기본 학습   │    │ • Instruction │    │ • MCTP 검토   │               │
│  │ • 한국어 검증 │    │ • 행동 제어   │    │ • 속도 최적화 │               │
│  └───────────────┘    └───────────────┘    └───────────────┘               │
│                                                                             │
│  목표: 빠른 검증       목표: 확장성 확보     목표: 성능 최적화               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 아키텍처 상세 설계

### 2.1 Phase 1: 기본 한국어 능력 (PersonaPlex 스타일)

#### 2.1.1 목표
- 기존 Moshi 아키텍처 유지
- 한국어 대화 기본 능력 확보
- Voice embedding으로 일관된 음성 품질

#### 2.1.2 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       K-Moshi Phase 1 Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Voice Prompt (Pre-computed)                       │    │
│  │                                                                     │    │
│  │  KMOSHI_V1.pt  ───────────────────────────────────────────────────▶ │    │
│  │                                                                     │    │
│  │  • 한국어 참조 음성 (20-30초)                                       │    │
│  │  • Mimi encode → audio token cache                                 │    │
│  │  • load_voice_prompt() → state.cache 설정                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Text Prompt (Runtime)                             │    │
│  │                                                                     │    │
│  │  "당신은 케이모시입니다. 한국어 AI 비서입니다.  │    │
│  │   친절하고 따뜻한 말투를 사용합니다. 사용자의 질문에 도움이 되는       │    │
│  │   답변을 제공합니다."                                                │    │
│  │                                                                     │    │
│  │  • SentencePiece tokenization                                       │    │
│  │  • Text embedding → model input                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Moshi 7B                                    │    │
│  │                                                                     │    │
│  │  • Transformer backbone (32 layers)                                │    │
│  │  • LoRA fine-tuning (rank=128)                                     │    │
│  │  • Depformer for audio generation                                  │    │
│  │  • 기존 Mimi codec 사용                                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.1.3 구현 체크리스트

```yaml
phase_1_implementation:
  voice_embedding:
    - [ ] 한국어 참조 음성 선정 (성우 또는 고품질 TTS)
    - [ ] Mimi encoder로 audio token 추출
    - [ ] .pt 파일로 저장 (KMOSHI_V1.pt)
    - [ ] PersonaPlex load_voice_prompt() 메서드 참조

  data_preparation:
    - [ ] KSponSpeech 또는 자체 데이터 확보
    - [ ] Stereo WAV 변환 (L=K-Moshi, R=User)
    - [ ] Whisper 한국어 전사 (word-level alignment)
    - [ ] JSONL 메타데이터 생성
    - [ ] 데이터 품질 검증

  training:
    - [ ] korean_phase1_fsdp.yaml 설정
    - [ ] LoRA 설정 (rank=128, ft_embed=True)
    - [ ] 학습 실행 (500-1000시간 데이터)
    - [ ] 체크포인트 저장 및 평가

  evaluation:
    - [ ] 한국어 WER 측정
    - [ ] 대화 품질 평가 (유창성, 관련성)
    - [ ] 음성 품질 평가 (MOS, Speaker Similarity)
```

### 2.2 Phase 2: Speaker & Behavior Control (F-actor 스타일)

#### 2.2.1 목표
- 동적 speaker embedding 지원
- Instruction-following 능력 획득
- 대화 행동 제어 (backchannel, interruption 등)

#### 2.2.2 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       K-Moshi Phase 2 Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   Speaker Embedding Module                           │    │
│  │                                                                     │    │
│  │  Reference Audio (5초)                                              │    │
│  │        │                                                            │    │
│  │        ▼                                                            │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │              ECAPA-TDNN (pretrained)                        │   │    │
│  │  │              • 한국어 화자 인식 모델                         │   │    │
│  │  │              • Output: 192-dim speaker vector               │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │        │                                                            │    │
│  │        ▼                                                            │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │         Speaker Projection Layer (trainable)                │   │    │
│  │  │         • Linear(192, 4096)                                 │   │    │
│  │  │         • Output: LLM token space embedding                 │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   Instruction Prefix                                 │    │
│  │                                                                     │    │
│  │  [Speaker_Emb] + [Instruction_Tokens]                               │    │
│  │                                                                     │    │
│  │  "conversation_narrative: 고객 서비스 상담                          │    │
│  │   initiation: system                                                │    │
│  │   backchannel_frequency: high                                       │    │
│  │   interruption_allowed: false                                       │    │
│  │   tone: friendly_professional"                                      │    │
│  │                                                                     │    │
│  │  → Concatenated and fed to LLM                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Modified Moshi 7B                                 │    │
│  │                                                                     │    │
│  │  Input Processing:                                                  │    │
│  │    original_input + speaker_emb + instruction_tokens                │    │
│  │                                                                     │    │
│  │  Training:                                                          │    │
│  │    • ECAPA-TDNN: Frozen                                             │    │
│  │    • Projection Layer: Trainable                                    │    │
│  │    • LLM Backbone: LoRA fine-tuning                                 │    │
│  │    • Depformer: Fine-tuning                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.2.3 Instruction-Following 데이터 형식

```yaml
# K-Moshi Instruction-Following 데이터 예시
instruction_data_format:
  sample_1:
    audio_path: "dialogue_001.wav"
    speaker_reference: "speaker_ref_001.wav"  # 5초 참조 음성
    instruction:
      narrative: "고객이 은행 계좌 잔액을 문의하는 상담"
      initiation: "system"  # K-Moshi가 먼저 인사
      backchannel: "medium"  # 적당한 맞장구
      interruption: "low"   # 끼어들기 자제
      tone: "professional"
    alignments:
      - ["안녕하세요", [0.0, 0.5], "SPEAKER_MAIN"]
      - ["○○은행입니다", [0.5, 1.2], "SPEAKER_MAIN"]
      - ...

  sample_2:
    audio_path: "dialogue_002.wav"
    speaker_reference: "speaker_ref_002.wav"
    instruction:
      narrative: "친구와 일상 대화"
      initiation: "user"    # 사용자가 먼저
      backchannel: "high"   # 활발한 맞장구
      interruption: "medium" # 자연스러운 끼어들기
      tone: "casual"
    alignments: [...]
```

#### 2.2.4 구현 체크리스트

```yaml
phase_2_implementation:
  speaker_embedding:
    - [ ] ECAPA-TDNN 모델 통합 (SpeechBrain 또는 PyTorch 구현)
    - [ ] Speaker projection layer 추가 (Linear 192→4096)
    - [ ] 참조 음성 처리 파이프라인 구현
    - [ ] Speaker embedding extraction 유틸리티

  instruction_system:
    - [ ] Instruction tokenizer 구현
    - [ ] Prefix concatenation 메커니즘
    - [ ] Instruction template 정의
    - [ ] Runtime instruction parsing

  data_generation:
    - [ ] Behavioral annotation 파이프라인
    - [ ] 다양한 시나리오 프롬프트 생성
    - [ ] LLM 기반 narrative rewriting (로컬 모델)
    - [ ] 품질 필터링 (text-speech alignment)

  training:
    - [ ] Phase 1 체크포인트에서 시작
    - [ ] Projection layer 추가 학습
    - [ ] Instruction-following 데이터로 fine-tuning
    - [ ] 행동 제어 평가

  evaluation:
    - [ ] Initiation accuracy 측정
    - [ ] Backchannel frequency correlation
    - [ ] Speaker similarity (reference vs generated)
    - [ ] Instruction adherence score
```

### 2.3 Phase 3: 최적화 (선택)

#### 2.3.1 Option A: FSQ 전환

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FSQ Optimization (F-actor 참고)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  현재 (RVQ + Depformer):                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Audio Token → Depformer → 순차적 8단계 예측                        │    │
│  │  장점: 높은 품질                                                    │    │
│  │  단점: 느린 추론 속도                                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  변경 후 (FSQ + Parallel Heads):                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Audio Token → 4개 Linear Head → 병렬 예측                          │    │
│  │  장점: 빠른 추론 (1단계)                                            │    │
│  │  단점: 품질 약간 저하 가능, 아키텍처 수정 필요                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  구현 난이도: ★★★★☆ (높음)                                              │
│  예상 효과: 추론 속도 3-4x 향상                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.3.2 Option B: MCTP 적용 (VITA-Audio 참고)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              Multiple Cross-modal Token Prediction (MCTP)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  기존:                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Forward Pass → 1개 audio token 생성 → 반복                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  MCTP 적용 후:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Forward Pass → N개 audio token 동시 생성 (N=4~8)                   │    │
│  │                                                                     │    │
│  │  Implementation:                                                    │    │
│  │  • N개의 prediction head 추가                                       │    │
│  │  • Attention mask 조정 (미래 토큰 참조 방지)                        │    │
│  │  • Progressive training (1→2→4→8)                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  구현 난이도: ★★★☆☆ (중간)                                              │
│  예상 효과: First-token latency 3-5x 감소 (236ms → ~50ms)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 데이터셋 전략

### 3.1 Phase별 데이터 구성

```yaml
phase_1_data:
  total_target: 500-1000시간
  composition:
    korean_dialogue:
      ratio: 60%
      sources: [KSponSpeech, AIHub자유대화]
      format: "Stereo WAV + Word alignments"
    identity_qa:
      ratio: 20%
      content: "K-Moshi 자기소개, 능력 설명"
      generation: "TTS + Script"
    customer_service:
      ratio: 20%
      scenarios: ["은행", "병원", "음식점", "쇼핑몰"]
      generation: "TTS + Script 또는 실제 데이터"

phase_2_data:
  total_target: 500-1000시간 추가
  composition:
    instruction_following:
      ratio: 50%
      format: "audio + speaker_ref + instruction + alignment"
      annotations:
        - narrative
        - initiation
        - backchannel_frequency
        - interruption_frequency
        - tone
    scenario_variations:
      ratio: 30%
      method: "Phase 1 데이터에 instruction 추가 (back-annotation)"
    speaker_diversity:
      ratio: 20%
      method: "다양한 speaker_ref 음성으로 합성"
```

### 3.2 Back-annotation 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Back-annotation Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 기존 대화 데이터                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  dialogue_001.wav + alignments                                      │    │
│  │  (instruction 없음)                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  Step 2: 대화 분석 (Local LLM)                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Input: Full transcript + timing information                        │    │
│  │                                                                     │    │
│  │  Local LLM 프롬프트:                                                 │    │
│  │  "다음 대화를 분석하고 instruction을 생성하세요:                     │    │
│  │   1. 대화 시나리오 요약 (narrative)                                 │    │
│  │   2. 누가 먼저 말했는지 (initiation)                                 │    │
│  │   3. 맞장구 빈도 추정 (backchannel)                                 │    │
│  │   4. 끼어들기 빈도 추정 (interruption)                              │    │
│  │   5. 전반적인 톤 (tone)"                                            │    │
│  │                                                                     │    │
│  │  Output: Structured instruction                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  Step 3: Instruction 통합                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  dialogue_001.wav + alignments + instruction                        │    │
│  │  → Phase 2 학습 데이터                                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 구현 일정

### 4.1 Gantt Chart

```
Week:    1    2    3    4    5    6    7    8    9   10   11   12
        ├────┴────┴────┴────┼────┴────┴────┴────┼────┴────┴────┴────┤

Phase 1: PersonaPlex Style
├─────────────────────────────┤
│ Voice Embedding [Week 1]    │
│ Data Preparation [Week 1-2] │
│ Training [Week 2-3]         │
│ Evaluation [Week 4]         │
└─────────────────────────────┘

Phase 2: F-actor Style
                              ├─────────────────────────────┤
                              │ ECAPA-TDNN Integration [5]  │
                              │ Instruction System [5-6]    │
                              │ Data Generation [6-7]       │
                              │ Training [7-8]              │
                              │ Evaluation [8]              │
                              └─────────────────────────────┘

Phase 3: Optimization (Optional)
                                                            ├─────────────────────┤
                                                            │ Architecture [9-10] │
                                                            │ Training [10-11]    │
                                                            │ Evaluation [11-12]  │
                                                            └─────────────────────┘
```

### 4.2 리소스 요구사항

```yaml
compute_resources:
  phase_1:
    gpu: "8x A100 80GB (FSDP)"
    training_time: "~48-72 hours"
    data_storage: "~1TB (audio + processed)"

  phase_2:
    gpu: "8x A100 80GB (FSDP)"
    training_time: "~48-72 hours"
    additional: "ECAPA-TDNN inference GPU"

  phase_3:
    gpu: "8x A100 80GB"
    training_time: "~96+ hours (architecture change)"

human_resources:
  - ML Engineer: 1-2명 (모델 학습)
  - Data Engineer: 1명 (데이터 파이프라인)
  - Audio Engineer: 1명 (음성 품질 평가)
```

---

## 5. 위험 요소 및 대응

### 5.1 기술적 위험

| 위험 요소 | 확률 | 영향 | 대응 방안 |
|----------|------|------|----------|
| 한국어 음성 품질 저하 | 중 | 높음 | Phase 1에서 품질 기준 미달 시 데이터/설정 조정 |
| ECAPA-TDNN 통합 복잡성 | 낮 | 중간 | SpeechBrain 라이브러리 활용, 사전 테스트 |
| Instruction-following 학습 어려움 | 중 | 높음 | F-actor 논문 참고, 점진적 학습 |
| FSQ 전환 시 품질 저하 | 높음 | 높음 | Phase 3 optional, 품질 모니터링 후 결정 |

### 5.2 데이터 위험

| 위험 요소 | 확률 | 영향 | 대응 방안 |
|----------|------|------|----------|
| 데이터 부족 | 중 | 높음 | TTS 합성 데이터로 보강 |
| 데이터 품질 문제 | 중 | 중간 | 자동 필터링 + 수동 검수 |
| 저작권 이슈 | 낮 | 높음 | 공개 데이터셋 우선 사용, 라이선스 확인 |

---

## 6. 평가 지표

### 6.1 Phase별 평가 기준

```yaml
phase_1_metrics:
  speech_quality:
    - korean_wer: "< 15%"
    - mos_score: "> 3.5"
    - speaker_similarity: "> 0.7"
  conversation_quality:
    - response_relevance: "Human evaluation"
    - fluency: "Human evaluation"
    - latency: "< 500ms first token"

phase_2_metrics:
  instruction_following:
    - initiation_accuracy: "> 90%"
    - backchannel_correlation: "> 0.4"
    - interruption_correlation: "> 0.2"
  speaker_control:
    - speaker_similarity_ref: "> 0.5"
    - voice_consistency: "Cross-utterance similarity"

phase_3_metrics:
  performance:
    - inference_speedup: "> 2x"
    - first_token_latency: "< 200ms"
    - quality_degradation: "< 5% WER increase"
```

---

## 7. 결론 및 권장사항

### 7.1 최종 권장 아키텍처

**K-Moshi v2.0 = Moshi 기반 + ECAPA-TDNN Speaker Embedding + Instruction Prefix**

이 접근법은:
1. **검증된 Moshi 아키텍처** 활용으로 안정성 확보
2. **F-actor의 효율적인 speaker embedding** 방식 채택
3. **점진적 개발**로 각 단계별 품질 검증 가능
4. **확장 가능한 구조**로 향후 최적화 여지 확보

### 7.2 즉시 시작 가능한 작업

```yaml
immediate_actions:
  week_1:
    - "한국어 참조 음성 선정 및 녹음"
    - "KMOSHI_V1.pt voice embedding 생성"
    - "korean_phase1_fsdp.yaml 설정 완료"
    - "KSponSpeech 데이터 전처리 시작"

  week_2:
    - "Phase 1 학습 시작"
    - "ECAPA-TDNN 모델 테스트 (SpeechBrain)"
    - "Instruction template 설계"

  parallel_research:
    - "F-actor 코드 상세 분석 (공개 시)"
    - "PersonaPlex voice prompt 메커니즘 상세 분석"
    - "FSQ vs RVQ 품질 비교 논문 검토"
```

---

## 8. 참고 자료

### 8.1 핵심 논문
1. [Moshi](https://arxiv.org/abs/2410.00037) - 기본 아키텍처
2. [F-actor](https://arxiv.org/html/2601.11329) - Speaker Embedding + Instruction
3. [PersonaPlex](https://github.com/NVIDIA/personaplex) - Voice Prompt 방식

### 8.2 관련 문서
- `docs/FULL_DUPLEX_ARCHITECTURE_ANALYSIS.md` - 상세 아키텍처 분석
- `docs/TRAINING_RECIPE_ANALYSIS_KO.md` - 학습 레시피 분석
- `CLAUDE.md` - 프로젝트 가이드

### 8.3 코드 참조
- `moshi/moshi/models/lm.py` - Moshi LM 구현
- PersonaPlex `lm.py` - Voice prompt 메서드
- `finetune/data/interleaver.py` - 데이터 인터리빙

---

*Last Updated: 2026-01-20*
*Document: K-Moshi Architecture Proposal v2.0*
*Author: K-Moshi Research Team*
