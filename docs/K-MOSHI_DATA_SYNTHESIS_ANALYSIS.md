# K-Moshi Data Synthesis Analysis Report

> **작성일**: 2026-01-13
> **목적**: K-Moshi 한국어 대화 데이터 합성을 위한 종합 분석
> **전략**: Hybrid Strategy (Option C) - Bootstrap with External TTS → Self-Generation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Post-Training Methods Analysis](#2-post-training-methods-analysis)
3. [Text Corpus Analysis](#3-text-corpus-analysis)
4. [Conversational TTS Models Analysis](#4-conversational-tts-models-analysis)
5. [TTS Model Ratio Strategy](#5-tts-model-ratio-strategy)
6. [Recommendations](#6-recommendations)
7. [References](#7-references)

---

## 1. Executive Summary

### 1.1 핵심 질문 및 답변 요약

| 질문 | 핵심 답변 |
|------|-----------|
| **Post-training 방식** | Moshi는 명시적 RLHF 미사용, J-Moshi도 SFT/DPO 미사용. Speech 모델에서의 RLHF는 아직 연구 초기 단계 |
| **Text Corpus 선택** | Moshi: Fisher corpus + synthetic scripts, J-Moshi: JapanesePersonaChat 등 4개 대화 데이터셋 |
| **Conversational TTS** | CSM-1B/Nari Labs Dia는 영어 중심. 한국어에는 OpenAudio S1 Mini 권장 |
| **TTS 비율 전략** | **OpenAudio S1 Mini 70% + Supertonic-2 30%** 권장 (품질 vs 속도 트레이드오프) |

### 1.2 핵심 인사이트

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    K-MOSHI HYBRID STRATEGY OVERVIEW                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Bootstrap (External TTS)                                       │
│  ├─ OpenAudio S1 Mini: 고품질 대화 생성 (70%)                            │
│  ├─ Supertonic-2: 빠른 대량 생성 (30%)                                   │
│  └─ 목표: ~500-1000시간 초기 학습 데이터                                  │
│                                                                          │
│  Phase 2: Self-Generation                                                │
│  ├─ 학습된 K-Moshi 모델로 추가 데이터 생성                               │
│  ├─ WER 기반 샘플 선택 (J-Moshi 방식)                                    │
│  └─ 목표: ~500-1000시간 추가 합성                                        │
│                                                                          │
│  Total Target: ~1000-2000시간 대화 데이터                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Post-Training Methods Analysis

### 2.1 Post-Training 개요

Post-training은 사전학습된 모델을 특정 작업이나 사용자 선호도에 맞게 조정하는 단계입니다.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     POST-TRAINING METHODS TAXONOMY                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Supervised Fine-Tuning (SFT)                                    │    │
│  │ • 레이블된 데이터로 직접 학습                                    │    │
│  │ • Input → Target 매핑 학습                                      │    │
│  │ • 가장 단순하고 안정적인 방법                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                           ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ RLHF (Reinforcement Learning from Human Feedback)               │    │
│  │ • Step 1: Reward Model 학습 (선호도 데이터)                      │    │
│  │ • Step 2: PPO로 정책 최적화                                     │    │
│  │ • 복잡하고 불안정, 높은 연산 비용                                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                           ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ DPO (Direct Preference Optimization)                            │    │
│  │ • RLHF의 단순화 버전                                            │    │
│  │ • Reward Model 없이 직접 선호도 학습                            │    │
│  │ • Classification loss로 변환                                     │    │
│  │ • 훨씬 안정적이고 효율적                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Moshi의 Post-Training 접근법

Moshi는 4단계 학습 파이프라인을 사용합니다:

| 단계 | 설명 | 데이터 |
|------|------|--------|
| **Pre-training** | 비지도 음성 데이터 학습 | 대규모 음성 데이터 |
| **Post-training** | Speaker diarization 기반 멀티스트림 시뮬레이션 | Diarized audio |
| **Fine-tuning** | Fisher corpus로 Full-duplex 학습 | 2000시간 전화 대화 |
| **Instruct-FT** | 합성 대화 스크립트로 instruction 학습 | 20,000시간 TTS 합성 |

**핵심 발견:**
- Moshi는 **명시적 RLHF/DPO를 사용하지 않음**
- 대신 대규모 합성 데이터(20K hours)로 품질 향상
- "Inner Monologue" 기법으로 텍스트-음성 정렬

### 2.3 J-Moshi의 Post-Training 접근법

J-Moshi도 RLHF/DPO를 사용하지 않습니다:

| 단계 | 데이터량 | 방법 |
|------|---------|------|
| Pre-training | 60,000시간 | J-CHAT corpus (ZeRO-3, 128 V100) |
| Fine-tuning | 344시간 | 고품질 스테레오 대화 |
| Data Augmentation | 602시간 | Multi-stream TTS 합성 |

**J-Moshi의 합성 데이터 전략:**
```python
# J-Moshi 합성 데이터 생성 파이프라인
def j_moshi_synthesis_pipeline(text_dialogues):
    # 1. LLM으로 대화체 변환
    spoken_dialogues = llm_rewrite(text_dialogues, model="Gemma-2-27b")

    # 2. Multi-stream TTS로 10개 샘플 생성
    for dialogue in spoken_dialogues:
        samples = []
        for seed in range(10):
            audio = multi_stream_tts(dialogue, seed=seed)
            wer = compute_wer(audio, dialogue.text)
            samples.append((audio, wer))

        # 3. WER 기준 최적 샘플 선택
        best_sample = min(samples, key=lambda x: x[1])
        yield best_sample
```

### 2.4 Speech 모델에서의 RLHF/DPO 현황

#### 2.4.1 현재 연구 상태

Speech 모델에서의 RLHF/DPO는 아직 **초기 연구 단계**입니다:

| 연구 | 방법 | 대상 | 결과 |
|------|------|------|------|
| UC Berkeley (2024) | RLAIF | TTS 감정 표현 | 감정 표현 향상 |
| OSU (2025) | DLPO | Diffusion TTS | UTMOS 3.65 달성 |
| ICLR 2025 | UNO | Zero-shot TTS | MOS, WER 개선 |
| Fish Audio | Online RLHF | OpenAudio S1 | TTS-Arena2 1위 |

#### 2.4.2 Speech 모델 RLHF의 도전 과제

```
┌─────────────────────────────────────────────────────────────────────────┐
│              SPEECH MODEL RLHF CHALLENGES                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. 평가 지표의 주관성                                                   │
│     • MOS (Mean Opinion Score)는 사람마다 다름                          │
│     • 자동 평가 지표(UTMOS, NISQA)는 실제 품질과 괴리                   │
│                                                                          │
│  2. Reward Model 설계의 어려움                                           │
│     • 텍스트와 달리 음성의 "좋음"을 정의하기 어려움                      │
│     • 발음 정확도, 자연스러움, 감정, 운율 등 다차원적                    │
│                                                                          │
│  3. 선호도 데이터 수집 비용                                              │
│     • 음성 쌍 비교는 텍스트보다 시간 소요 큼                             │
│     • 대규모 데이터셋 구축 어려움                                        │
│                                                                          │
│  4. Full-duplex 대화 모델의 복잡성                                       │
│     • 단순 TTS보다 평가 차원이 많음                                      │
│     • Turn-taking, backchannel 등 대화적 요소 평가 필요                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.5 K-Moshi 권장 Post-Training 전략

**권장: SFT + 대규모 합성 데이터 (Moshi/J-Moshi 방식)**

| 단계 | 방법 | 근거 |
|------|------|------|
| 1단계 | External TTS로 대규모 SFT 데이터 생성 | 검증된 Moshi 방식 |
| 2단계 | Self-Generation으로 데이터 증강 | J-Moshi 602시간 추가 효과 확인 |
| (선택) | Online RLHF 실험 | OpenAudio S1 성공 사례 참고 |

**RLHF/DPO 미권장 이유:**
1. Full-duplex 대화 모델에 대한 검증된 사례 없음
2. 선호도 데이터 구축 비용 높음
3. Moshi/J-Moshi 모두 RLHF 없이 성공

---

## 3. Text Corpus Analysis

### 3.1 Moshi의 Text/Dialogue Corpus

#### 3.1.1 Pre-training 텍스트 데이터

Moshi의 텍스트 백본(Helium)은 다음 데이터로 학습:

| 데이터 소스 | 용도 |
|-------------|------|
| Wikipedia | 일반 지식 |
| Stack Exchange | 기술 지식 |
| 과학 논문 컬렉션 | 학술 지식 |

#### 3.1.2 Fine-tuning 대화 데이터

**Fisher Corpus (핵심!):**
- 2,000시간의 전화 대화
- 랜덤 페어링된 참가자 간 주어진 주제 토론
- **각 화자가 별도 채널에 녹음** (스테레오)
- 원본 8kHz → AudioSR로 24kHz 업샘플링

```
Fisher Corpus 특징:
┌─────────────────────────────────────────────────────────────────────────┐
│  Channel L (Moshi)  │  Channel R (User)                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  Speaker A의 음성   │  Speaker B의 음성                                  │
│  + A의 transcription│  + B의 transcription                              │
│                                                                          │
│  → Full-duplex 학습의 Ground Truth로 활용                               │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 3.1.3 Instruct Fine-tuning 데이터

Moshi는 170시간의 스크립트된 대화를 수집하여:
1. Multi-speaker TTS 시스템 학습
2. 합성 대화 스크립트 기반 **20,000시간** TTS 데이터 생성

### 3.2 J-Moshi의 Text Corpus

#### 3.2.1 Pre-training 데이터

**J-CHAT Corpus:**
- 69,000시간 일본어 대화 (YouTube, Podcasts)
- 모노 음성 → Speaker Diarization으로 스테레오화

#### 3.2.2 Fine-tuning 대화 데이터 (344시간)

| 데이터셋 | 시간 | 특징 |
|----------|------|------|
| Japanese Callhome | 16h | 전화 대화 |
| CSJ (Corpus of Spontaneous Japanese) | 12h | 2인 대화 |
| Travel Agency Dialogue | 115h | Zoom 녹음 |
| Casual Dialogue (in-house) | 148h | 32명 화자 |
| Consultation Dialogue (in-house) | 53h | 32명 화자 |

#### 3.2.3 합성 데이터용 Text Corpus (602시간 TTS 생성)

| Text Corpus | 대화 수 | 특징 |
|-------------|---------|------|
| JapanesePersonaChat | - | 페르소나 기반 대화 |
| JapaneseEmpatheticDialogues | - | 공감 대화 |
| Japanese Daily Dialogue Corpus | - | 일상 대화 |
| RealPersonaChat | - | 실제 페르소나 대화 |
| **Total** | **43,739** | Gemma-2-27b로 구어체 변환 |

### 3.3 한국어 대화 Corpus 옵션

#### 3.3.1 권장 Text Corpus

| 데이터셋 | 규모 | 특징 | 접근성 |
|----------|------|------|--------|
| **AI Hub 감성 대화** | 대규모 | 60가지 감정, 세대별 분리 | 신청 필요 |
| **모두의 말뭉치 구어** | 100만 어절 | 국립국어원, 고품질 | 공개 |
| **모두의 말뭉치 메신저** | 92만 어절 | 일상 대화체 | 공개 |
| **AI Hub 대화체** | 10만 문장 | AI 번역용 | 신청 필요 |
| **KorQuAD 대화** | - | QA 형식 대화 | 공개 |

#### 3.3.2 권장 음성 대화 Corpus

| 데이터셋 | 규모 | 형식 | 우선순위 |
|----------|------|------|----------|
| **KsponSpeech** | 969h | 2인 대화, 스테레오 | ⭐⭐⭐⭐⭐ |
| **AI Hub 자유대화** | 1000h+ | 2인 대화 | ⭐⭐⭐⭐ |
| **AI Hub 감성대화 음성** | - | 감정 레이블 포함 | ⭐⭐⭐⭐ |

#### 3.3.3 Text → TTS 합성용 권장 데이터셋

J-Moshi 방식을 한국어에 적용할 때 권장하는 텍스트 대화 데이터:

```
┌─────────────────────────────────────────────────────────────────────────┐
│            KOREAN TEXT DIALOGUE CORPUS FOR TTS SYNTHESIS                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1차 권장 (대화 형식 최적화)                                             │
│  ├─ AI Hub 감성 대화 말뭉치                                             │
│  ├─ 모두의 말뭉치 - 메신저 대화                                          │
│  └─ 한국어 일상 대화 코퍼스                                              │
│                                                                          │
│  2차 권장 (LLM 변환 필요)                                                │
│  ├─ KorQuAD → 대화체 변환                                               │
│  ├─ 영화 리뷰 → 토론 대화 생성                                   │
│  └─ 위키피디아 → QA 대화 생성                                           │
│                                                                          │
│  변환 파이프라인:                                                        │
│  Text Corpus → LLM (구어체 변환) → TTS → WER 필터링 → 학습 데이터       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Text → Spoken Dialogue 변환 전략

J-Moshi의 검증된 방식을 한국어에 적용:

```python
# 한국어 대화 변환 파이프라인 예시
KOREAN_CONVERSION_PROMPT = """
다음 텍스트 대화를 자연스러운 한국어 구어체로 변환해주세요.

원칙:
1. 문어체 → 구어체 변환 (예: "것입니다" → "거야", "하였다" → "했어")
2. 자연스러운 추임새 추가 (예: "음...", "그래서...", "아~")
3. 적절한 존대어/반말 유지
4. 실제 대화처럼 짧은 문장으로 분할

원본 대화:
{original_dialogue}

변환된 구어체 대화:
"""

def convert_to_spoken_korean(text_dialogue, llm_model="gemma-2-27b"):
    """텍스트 대화를 구어체로 변환"""
    prompt = KOREAN_CONVERSION_PROMPT.format(original_dialogue=text_dialogue)
    return llm_model.generate(prompt)
```

---

## 4. Conversational TTS Models Analysis

### 4.1 CSM-1B (Sesame) 분석

#### 4.1.1 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CSM-1B ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │ Text Tokenizer  │     │ Mimi Codec      │     │ Backbone (1B)   │   │
│  │ (Llama)         │     │ (Kyutai)        │     │ (Llama variant) │   │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘   │
│           │                       │                       │             │
│           └───────────┬───────────┘                       │             │
│                       ↓                                   │             │
│              ┌─────────────────┐                          │             │
│              │ Interleaved     │ ─────────────────────────┘             │
│              │ T, A, T, A, ... │                                        │
│              └────────┬────────┘                                        │
│                       ↓                                                 │
│              ┌─────────────────┐                                        │
│              │ Decoder (100M)  │ → Codebook 1~N-1                       │
│              └────────┬────────┘                                        │
│                       ↓                                                 │
│              ┌─────────────────┐                                        │
│              │ Audio Output    │                                        │
│              └─────────────────┘                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.1.2 특징 및 한계

| 항목 | 내용 |
|------|------|
| **장점** | 대화 컨텍스트 인식, 자연스러운 턴테이킹 |
| **한계** | 영어 전용, 텍스트 생성 불가 (오디오만) |
| **한국어** | ❌ 미지원 (영어 중심 학습) |

#### 4.1.3 한국어 적용 가능성

```
CSM-1B 한국어 Fine-tuning 가능성:

✅ 가능한 부분:
- Mimi 코덱은 언어 무관 (오디오 인코딩)
- Backbone 구조는 다국어 적용 가능

❌ 제약 사항:
- 공식 한국어 fine-tuning 가이드 없음
- 텍스트 토크나이저가 영어 기반
- 학습 데이터 및 프롬프트가 영어 최적화

권장: K-Moshi에서는 사용하지 않음 (비용 대비 효과 불확실)
```

### 4.2 Nari Labs Dia 분석

#### 4.2.1 모델 정보

| 항목 | 내용 |
|------|------|
| **파라미터** | 1.6B |
| **특징** | Ultra-realistic dialogue 생성, 단일 패스 |
| **경쟁** | NotebookLM, ElevenLabs 수준 품질 |
| **출시** | 2025년 4월 (Dia2: 2025년 11월) |

#### 4.2.2 언어 지원 현황

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NARI LABS DIA LANGUAGE SUPPORT                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  현재 지원:                                                              │
│  ├─ 영어: ✅ 완전 지원 (최적화)                                          │
│  └─ 기타: ⚠️ 처리 가능하나 성능 저하                                     │
│                                                                          │
│  커뮤니티 Fine-tune:                                                     │
│  ├─ 중국어: ✅ 커뮤니티 파인튠 존재                                       │
│  ├─ 일본어: ✅ 커뮤니티 파인튠 존재                                       │
│  └─ 한국어: ❓ 명시적 언급 없음                                          │
│                                                                          │
│  로드맵 (향후):                                                          │
│  ├─ 4-5B 파라미터로 확장                                                 │
│  └─ 영어 외 언어 네이티브 지원 예정                                       │
│                                                                          │
│  특이사항:                                                               │
│  └─ Nari Labs는 한국 기반 회사 (Nari = 한국어 "나리"/백합)               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.2.3 한국어 적용 가능성

| 평가 | 내용 |
|------|------|
| **가능성** | 중간 (한국 회사이나 영어 우선 개발) |
| **권장** | 공식 한국어 지원 발표 대기 |
| **대안** | OpenAudio S1 Mini 사용 권장 |

### 4.3 한국어 지원 Conversational TTS 옵션

#### 4.3.1 비교 분석

| 모델 | 한국어 | 대화 인식 | 음성 복제 | 속도 | 품질 |
|------|--------|----------|----------|------|------|
| **OpenAudio S1 Mini** | ✅ | ⚠️ 제한적 | ✅ | 중간 | ⭐⭐⭐⭐⭐ |
| **Supertonic-2** | ✅ | ❌ | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **CSM-1B** | ❌ | ✅ | ⚠️ | 중간 | ⭐⭐⭐⭐ |
| **Nari Labs Dia** | ⚠️ | ✅ | ✅ | 중간 | ⭐⭐⭐⭐⭐ |
| **XTTS-v2** | ✅ | ❌ | ✅ | 느림 | ⭐⭐⭐ |

#### 4.3.2 권장 선택

```
K-Moshi용 TTS 선택 결론:

┌─────────────────────────────────────────────────────────────────────────┐
│  PRIMARY: OpenAudio S1 Mini                                             │
│  ├─ 한국어 네이티브 지원 (13개 언어 중 하나)                            │
│  ├─ 10-30초 오디오로 음성 복제 가능                                      │
│  ├─ RLHF 적용으로 자연스러움 극대화                                      │
│  ├─ TTS-Arena2 1위 모델의 경량화 버전                                   │
│  └─ 500M 파라미터로 효율적                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  SECONDARY: Supertonic-2                                                │
│  ├─ 한국어 전용 최적화                                                  │
│  ├─ 167x 실시간 속도 (가장 빠름)                                        │
│  ├─ 66M 파라미터 (매우 가벼움)                                          │
│  └─ 대량 생성 시 비용 효율적                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. TTS Model Ratio Strategy

### 5.1 제안된 옵션 분석

사용자 제안: **TTS 통일 사용 (OpenAudio S1 Mini OR Supertonic-2)**

#### 5.1.1 옵션 비교

| 옵션 | 장점 | 단점 |
|------|------|------|
| **100% OpenAudio S1 Mini** | 최고 품질, 음성 다양성 | 느린 속도, 높은 연산 비용 |
| **100% Supertonic-2** | 최고 속도, 낮은 비용 | 제한된 음성, 낮은 품질 |
| **혼합 (제안)** | 품질과 속도 균형 | 구현 복잡성 |

#### 5.1.2 비율별 트레이드오프

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TTS RATIO TRADE-OFF ANALYSIS                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  100% S1 Mini    ████████████████████ 품질: ⭐⭐⭐⭐⭐  속도: ⭐⭐       │
│   90% S1 Mini    ██████████████████▒▒ 품질: ⭐⭐⭐⭐⭐  속도: ⭐⭐⭐     │
│   80% S1 Mini    ████████████████▒▒▒▒ 품질: ⭐⭐⭐⭐   속도: ⭐⭐⭐     │
│   70% S1 Mini    ██████████████▒▒▒▒▒▒ 품질: ⭐⭐⭐⭐   속도: ⭐⭐⭐⭐   │ ← 권장
│   60% S1 Mini    ████████████▒▒▒▒▒▒▒▒ 품질: ⭐⭐⭐     속도: ⭐⭐⭐⭐   │
│   50% S1 Mini    ██████████▒▒▒▒▒▒▒▒▒▒ 품질: ⭐⭐⭐     속도: ⭐⭐⭐⭐⭐ │
│   30% S1 Mini    ██████▒▒▒▒▒▒▒▒▒▒▒▒▒▒ 품질: ⭐⭐      속도: ⭐⭐⭐⭐⭐ │
│  100% Supertonic ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ 품질: ⭐⭐      속도: ⭐⭐⭐⭐⭐ │
│                                                                          │
│  █ = OpenAudio S1 Mini    ▒ = Supertonic-2                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 권장 전략: 70:30 (OpenAudio S1 Mini : Supertonic-2)

#### 5.2.1 근거

| 요소 | 분석 |
|------|------|
| **품질 우선** | Full-duplex 대화 모델은 TTS 품질에 민감 |
| **다양성** | S1 Mini의 음성 복제로 화자 다양성 확보 |
| **효율성** | 30% Supertonic으로 대량 생성 가속 |
| **J-Moshi 참고** | 602시간 합성 데이터가 효과적이었음 |

#### 5.2.2 구현 전략

```python
# 권장 TTS 비율 구현

class HybridTTSPipeline:
    def __init__(self):
        self.s1_mini = OpenAudioS1Mini()
        self.supertonic = Supertonic2()
        self.ratio = 0.70  # S1 Mini 비율

    def synthesize_dialogue(self, dialogue, speaker_moshi, speaker_user):
        """
        대화 합성 전략:
        - Moshi 발화: 주로 S1 Mini (음성 일관성)
        - User 발화: 혼합 (다양성)
        """
        results = []

        for turn in dialogue.turns:
            if turn.speaker == "MOSHI":
                # Moshi: 항상 S1 Mini (일관된 음성)
                audio = self.s1_mini.synthesize(
                    turn.text,
                    voice=speaker_moshi
                )
            else:
                # User: 70% S1 Mini, 30% Supertonic
                if random.random() < self.ratio:
                    audio = self.s1_mini.synthesize(
                        turn.text,
                        voice=speaker_user
                    )
                else:
                    audio = self.supertonic.synthesize(turn.text)

            results.append(audio)

        return self.merge_stereo(results)
```

#### 5.2.3 대안 전략: 역할별 분리

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ALTERNATIVE: ROLE-BASED TTS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Strategy A: Quality-First (권장)                                        │
│  ├─ Moshi (AI): 100% OpenAudio S1 Mini                                  │
│  │   └─ 이유: AI 음성 품질이 모델 학습에 직접적 영향                     │
│  └─ User: 50% S1 Mini + 50% Supertonic                                  │
│       └─ 이유: 사용자 음성 다양성이 일반화에 도움                        │
│                                                                          │
│  Strategy B: Speed-First                                                │
│  ├─ Moshi (AI): 80% S1 Mini + 20% Supertonic                            │
│  └─ User: 30% S1 Mini + 70% Supertonic                                  │
│                                                                          │
│  Strategy C: Diversity-First                                             │
│  ├─ Moshi (AI): 100% S1 Mini (5개 이상 음성 클론)                        │
│  └─ User: 100% S1 Mini (10개 이상 음성 클론)                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 음성 다양성 전략

#### 5.3.1 권장 음성 수

| 역할 | 최소 | 권장 | 비고 |
|------|------|------|------|
| Moshi (AI) | 2 | 3-5 | 남/여 다양성 |
| User | 5 | 10-20 | 연령/성별 다양성 |

#### 5.3.2 음성 소스 옵션

```
음성 클론 소스 (OpenAudio S1 Mini용):

1. 공개 TTS 음성 샘플
   - KsponSpeech 화자 샘플
   - AI Hub 음성 데이터 샘플

2. 전문 성우 녹음 (권장)
   - 10-30초 고품질 샘플
   - 다양한 감정/톤 포함

3. 합성 음성 기반 클론 (비권장)
   - 품질 저하 가능성
```

---

## 6. Recommendations

### 6.1 최종 권장 사항 요약

#### 6.1.1 Post-Training 전략

| 항목 | 권장 | 근거 |
|------|------|------|
| **방법** | SFT + 대규모 합성 데이터 | Moshi/J-Moshi 검증 |
| **RLHF/DPO** | ❌ 미적용 | 검증 사례 부재, 비용 높음 |
| **합성 데이터 규모** | 500-1000시간 | J-Moshi 602시간 효과 확인 |

#### 6.1.2 Text Corpus 선택

| 우선순위 | 데이터셋 | 용도 |
|----------|----------|------|
| 1 | AI Hub 감성 대화 | 직접 TTS 합성 |
| 2 | 모두의 말뭉치 (구어/메신저) | 직접 TTS 합성 |
| 3 | KorQuAD + LLM 변환 | 대화 생성 |

#### 6.1.3 TTS 전략

| 항목 | 권장 |
|------|------|
| **Primary TTS** | OpenAudio S1 Mini (70%) |
| **Secondary TTS** | Supertonic-2 (30%) |
| **Moshi 음성** | 3-5개 클론 음성 |
| **User 음성** | 10-20개 클론 음성 |

### 6.2 Implementation Roadmap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    K-MOSHI DATA SYNTHESIS ROADMAP                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Preparation                                                    │
│  ├─ [1.1] Text Corpus 수집 및 정제                                      │
│  │   ├─ AI Hub 감성 대화 다운로드                                       │
│  │   ├─ 모두의 말뭉치 구어/메신저 다운로드                              │
│  │   └─ LLM 구어체 변환 파이프라인 구축                                 │
│  ├─ [1.2] TTS 환경 구축                                                 │
│  │   ├─ OpenAudio S1 Mini 설치                                         │
│  │   ├─ Supertonic-2 설치                                              │
│  │   └─ Voice cloning 샘플 준비 (Moshi 3개, User 10개)                 │
│  └─ [1.3] 파이프라인 검증                                               │
│       └─ 소규모 테스트 (10시간 합성)                                    │
│                                                                          │
│  Phase 2: Bootstrap TTS Generation                                       │
│  ├─ [2.1] OpenAudio S1 Mini로 고품질 합성 (~350시간)                    │
│  ├─ [2.2] Supertonic-2로 대량 합성 (~150시간)                           │
│  └─ [2.3] WER 기반 품질 필터링                                          │
│                                                                          │
│  Phase 3: Initial K-Moshi Training                                       │
│  ├─ [3.1] Bootstrap 데이터로 초기 학습                                   │
│  └─ [3.2] 품질 평가 및 검증                                              │
│                                                                          │
│  Phase 4: Self-Generation (Optional)                                     │
│  ├─ [4.1] 학습된 K-Moshi로 추가 대화 생성                               │
│  ├─ [4.2] WER 기반 샘플 선택 (J-Moshi 방식)                             │
│  └─ [4.3] 추가 학습 및 반복                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 예상 비용/시간

| 단계 | 예상 시간 | 예상 연산 비용 |
|------|----------|---------------|
| Phase 1 | 1주 | 낮음 |
| Phase 2 | 2-3주 | 높음 (GPU 필요) |
| Phase 3 | 1-2주 | 매우 높음 (8x A100) |
| Phase 4 | 1-2주 | 높음 |

---

## 7. References

### 7.1 핵심 논문

- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/html/2410.00037v2)
- [J-Moshi: Towards a Japanese Full-duplex Spoken Dialogue System](https://arxiv.org/html/2506.02979)
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)

### 7.2 모델 및 도구

- [OpenAudio S1 Mini (Fish Audio)](https://github.com/fishaudio/fish-speech)
- [Sesame CSM-1B](https://github.com/SesameAILabs/csm)
- [Nari Labs Dia](https://github.com/nari-labs/dia)
- [Supertonic-2](https://github.com/supertone-inc/supertonic-2)

### 7.3 한국어 데이터셋

- [모두의 말뭉치](https://corpus.korean.go.kr/)
- [AI Hub](https://aihub.or.kr/)
- [Korpora](https://github.com/ko-nlp/Korpora)
- [AwesomeKorean_Data](https://github.com/songys/AwesomeKorean_Data)

### 7.4 RLHF/TTS 연구

- [Reinforcement Learning for Fine-tuning TTS Diffusion Models](https://arxiv.org/html/2405.14632v1)
- [Enhancing Emotional TTS through RLAIF (UC Berkeley)](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2024/EECS-2024-23.html)
- [Zero-shot TTS with Human Feedback (ICLR 2025)](https://openreview.net/forum?id=bAdSmSR10C)

---

*Last Updated: 2026-01-13*
*Document Version: 1.0*
*Author: K-Moshi Development Team*
