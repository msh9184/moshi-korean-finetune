# Training Paradigm Taxonomy: A Comprehensive Guide

> **작성일**: 2026-01-13
> **목적**: Pre-training, Fine-tuning, SFT, Instruction Tuning, Post-training 등 학습 패러다임의 명확한 정의와 상관관계 정리
> **대상**: LLM/NLP 및 Speech/Audio 분야

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Training Lifecycle Overview](#2-training-lifecycle-overview)
3. [Detailed Definitions](#3-detailed-definitions)
4. [Relationship Hierarchy](#4-relationship-hierarchy)
5. [LLM/NLP Domain Examples](#5-llmnlp-domain-examples)
6. [Speech/Audio Domain Examples](#6-speechaudio-domain-examples)
7. [Moshi Training Pipeline Analysis](#7-moshi-training-pipeline-analysis)
8. [Data Composition Guide](#8-data-composition-guide)
9. [Practical Decision Framework](#9-practical-decision-framework)
10. [References](#10-references)

---

## 1. Executive Summary

### 1.1 핵심 용어 관계도

```mermaid
flowchart TD
    PRE["PRE-TRAINING<br/>대규모 비지도/자기지도 학습<br/>일반적인 언어/음성 표현 학습<br/>예: Next Token Prediction, Masked LM, Contrastive Learning"]
    PRE --> POST
    subgraph POST["POST-TRAINING (광의): Pre-training 이후 모든 학습 단계의 총칭"]
        subgraph FT["FINE-TUNING (광의): 특정 태스크/도메인에 적응시키는 모든 학습"]
            TASK["Task-Specific Fine-tuning<br/>(태스크 FT)"]
            DOMAIN["Domain-Specific Fine-tuning<br/>(도메인 FT)"]
            BEHAVIOR["Behavior Fine-tuning<br/>(행동 FT)"]
            TASK --> SFT
            DOMAIN --> SFT
            BEHAVIOR --> SFT
            subgraph SFT["SUPERVISED FINE-TUNING (SFT): 레이블된 (input, output) 쌍으로 지도 학습"]
                IT["INSTRUCTION TUNING<br/>자연어 명령어 형식의 SFT<br/>(Instruction, Input, Output) 형식<br/>다양한 태스크를 명령어로 통합"]
            end
        end
        subgraph ALIGN["ALIGNMENT (협의의 Post-training): 인간 선호도/가치에 맞게 조정"]
            RLHF["RLHF<br/>(PPO 기반)"]
            DPO["DPO<br/>(직접 최적화)"]
            RLAIF["RLAIF<br/>(AI 피드백)"]
        end
    end
```

### 1.2 Quick Reference Table

| 용어 | 정의 | 데이터 | 목적 | 상위 개념 |
|------|------|--------|------|-----------|
| **Pre-training** | 대규모 비지도 학습 | 비레이블, 대용량 | 일반 표현 학습 | - |
| **Post-training** | Pre-training 이후 모든 학습 | 다양함 | 특화/정렬 | - |
| **Fine-tuning** | 특정 태스크/도메인 적응 | 소규모, 태스크별 | 태스크 성능 향상 | Post-training |
| **SFT** | 레이블 데이터로 지도 학습 | (input, output) 쌍 | 원하는 출력 학습 | Fine-tuning |
| **Instruction Tuning** | 명령어 형식의 SFT | (instruction, input, output) | 명령 수행 능력 | SFT |
| **Alignment** | 인간 선호도 정렬 | 선호도 데이터 | 안전성/유용성 | Post-training |
| **RLHF** | 강화학습 기반 정렬 | 인간 피드백 | 선호도 최적화 | Alignment |
| **DPO** | 직접 선호도 최적화 | 선호도 쌍 | RLHF 단순화 | Alignment |

---

## 2. Training Lifecycle Overview

### 2.1 LLM Training Lifecycle

```mermaid
flowchart TD
    S1["Stage 1: PRE-TRAINING<br/>Input: 대규모 텍스트 코퍼스 (Wikipedia, Books, Web 등)<br/>Method: Next Token Prediction (Autoregressive)<br/>Output: Base Model (예: GPT-3, LLaMA)<br/>Scale: 수조 토큰, 수천 GPU-hours"]
    S2["Stage 2: SUPERVISED FINE-TUNING (SFT)<br/>Input: 고품질 (prompt, response) 쌍 데이터<br/>Method: Cross-Entropy Loss on Target Tokens<br/>Output: SFT Model (예: InstructGPT SFT)<br/>Scale: 10K-100K 샘플"]
    S3["Stage 3: ALIGNMENT (RLHF/DPO)<br/>Input: 인간 선호도 데이터 (response A 선호 response B)<br/>Method: Reward Model + PPO (RLHF) 또는 Direct Optimization (DPO)<br/>Output: Aligned Model (예: ChatGPT)<br/>Scale: 50K-500K 비교 쌍"]
    S1 --> S2 --> S3
    NOTE["※ 참고: OpenAI는 Stage 2+3을 합쳐 Post-training 이라 칭함"]
    S3 -.-> NOTE
```

### 2.2 Speech/Audio Training Lifecycle

```mermaid
flowchart TD
    S1["Stage 1: PRE-TRAINING (Self-Supervised)<br/>방법 A: Contrastive Learning (wav2vec 2.0) - 마스킹된 위치의 양자화된 표현 예측, 대조 손실로 긍정/부정 샘플 구분<br/>방법 B: Masked Prediction (HuBERT) - 이산 타겟으로 마스크된 영역 예측<br/>방법 C: Weakly Supervised (Whisper) - 대규모 (audio, transcript) 쌍으로 직접 학습, 680K 시간 웹 크롤링 데이터"]
    S2["Stage 2: TASK-SPECIFIC FINE-TUNING<br/>ASR: CTC/Seq2Seq Loss on (audio, transcript)<br/>TTS: Reconstruction Loss on (text, audio)<br/>SLU: Classification Loss on (audio, intent/slot)"]
    S3["Stage 3: INSTRUCTION/ALIGNMENT (Optional, Emerging)<br/>Audio SFT: 음성 프롬프트로 SFT (Qwen-Audio)<br/>Audio DPO: 음성 품질 선호도 최적화 (연구 단계)"]
    S1 --> S2 --> S3
```

---

## 3. Detailed Definitions

### 3.1 Pre-training (사전 학습)

**정의**: 대규모 데이터에서 **비지도 또는 자기지도 학습**을 통해 **일반적인 표현(representation)**을 학습하는 단계

#### 3.1.1 LLM에서의 Pre-training

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         LLM PRE-TRAINING METHODS                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. Autoregressive LM (GPT 계열)                                                │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 목표: P(w_t | w_1, ..., w_{t-1}) 예측                                       │ │
│  │ 입력: "The cat sat on the"                                                 │ │
│  │ 타겟: "mat"                                                                 │ │
│  │ 손실: Cross-Entropy on next token                                          │ │
│  │                                                                             │ │
│  │ 예시 모델: GPT-2, GPT-3, LLaMA, Mistral                                     │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  2. Masked Language Model (BERT 계열)                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 목표: P(w_i | context) 예측 (양방향)                                        │ │
│  │ 입력: "The [MASK] sat on the mat"                                          │ │
│  │ 타겟: "cat"                                                                 │ │
│  │ 손실: Cross-Entropy on masked tokens                                       │ │
│  │                                                                             │ │
│  │ 예시 모델: BERT, RoBERTa, ALBERT                                            │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  3. Encoder-Decoder (T5 계열)                                                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 목표: Span Corruption 복원                                                  │ │
│  │ 입력: "The <extra_id_0> sat on <extra_id_1>"                                │ │
│  │ 타겟: "<extra_id_0> cat <extra_id_1> the mat"                               │ │
│  │                                                                             │ │
│  │ 예시 모델: T5, FLAN-T5, UL2                                                 │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 Speech에서의 Pre-training

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       SPEECH PRE-TRAINING METHODS                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. Contrastive Learning (wav2vec 2.0)                                          │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 방법:                                                                       │ │
│  │   1) CNN으로 raw audio → latent representations                             │ │
│  │   2) 일부 프레임 마스킹                                                     │ │
│  │   3) Transformer로 context 학습                                             │ │
│  │   4) 마스크 위치에서 true vs distractors 구분                               │ │
│  │                                                                             │ │
│  │ 손실: InfoNCE (Contrastive Loss)                                            │ │
│  │                                                                             │ │
│  │         exp(sim(c_t, q_t) / τ)                                              │ │
│  │  L = - log ─────────────────────────────                                    │ │
│  │         exp(sim(c_t, q_t)) + Σ exp(sim(c_t, q_n))                           │ │
│  │                                                                             │ │
│  │ c_t = context representation, q_t = true target, q_n = negatives            │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  2. Masked Prediction (HuBERT)                                                  │ │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 방법:                                                                       │ │
│  │   1) K-means로 오디오를 이산 단위로 클러스터링                               │ │
│  │   2) 마스킹된 영역의 클러스터 ID 예측                                       │ │
│  │                                                                             │ │
│  │ 손실: Cross-Entropy on discrete units                                       │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  3. Weakly Supervised (Whisper)                                                 │ │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 특징: 순수 self-supervised가 아닌 (audio, text) 쌍 사용                     │ │
│  │ 데이터: 680K 시간의 웹 크롤링 (audio, transcript)                           │ │
│  │ 손실: Seq2Seq Cross-Entropy                                                 │ │
│  │                                                                             │ │
│  │ ※ "weakly" = 웹에서 자동 수집된 노이즈 있는 레이블                          │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Fine-tuning (미세 조정)

**정의**: Pre-trained 모델을 **특정 태스크나 도메인에 적응**시키는 학습

#### 3.2.1 Fine-tuning의 종류

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          FINE-TUNING TAXONOMY                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. Task-Specific Fine-tuning (태스크별 미세 조정)                              │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 목적: 특정 태스크 성능 최적화                                               │ │
│  │ 예시:                                                                       │ │
│  │   • BERT → Sentiment Classification                                        │ │
│  │   • wav2vec 2.0 → ASR with CTC loss                                        │ │
│  │   • GPT → Summarization                                                    │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  2. Domain-Specific Fine-tuning (도메인별 미세 조정)                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 목적: 특정 도메인 지식/어휘 습득                                            │ │
│  │ 예시:                                                                       │ │
│  │   • LLaMA → Medical LLM (의료 텍스트)                                       │ │
│  │   • Whisper → Medical ASR (의료 음성)                                       │ │
│  │   • GPT → Legal Document Generator                                         │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  3. Behavior Fine-tuning (행동 미세 조정)                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 목적: 모델의 응답 스타일/행동 조정                                          │ │
│  │ 예시:                                                                       │ │
│  │   • Base LLM → Chat Model (대화형)                                          │ │
│  │   • TTS Model → Emotional TTS (감정 표현)                                   │ │
│  │   • ASR → Conversational ASR (대화체 인식)                                  │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  4. Parameter-Efficient Fine-tuning (파라미터 효율적 미세 조정)                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 방법:                                                                       │ │
│  │   • LoRA: 저랭크 어댑터 추가                                                │ │
│  │   • Adapter: 중간 레이어 삽입                                               │ │
│  │   • Prefix Tuning: 소프트 프롬프트 학습                                     │ │
│  │   • QLoRA: 양자화 + LoRA                                                    │ │
│  │                                                                             │ │
│  │ 장점: 메모리 효율적, 원본 가중치 보존                                        │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Supervised Fine-tuning (SFT)

**정의**: **레이블된 (input, output) 쌍**으로 지도 학습하는 Fine-tuning

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SUPERVISED FINE-TUNING (SFT) DETAIL                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  핵심 특징:                                                                     │
│  • "Supervised" = 입력과 출력이 모두 주어진 데이터                              │
│  • 모델이 주어진 입력에 대해 정확한 출력을 생성하도록 학습                       │
│  • Pre-training과 달리 명시적인 정답(target)이 존재                              │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                     LLM SFT 데이터 예시                                     │ │
│  ├───────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                             │ │
│  │  Input (Prompt):                                                            │ │
│  │  "다음 문장을 영어로 번역하세요: 오늘 날씨가 좋네요."                        │ │
│  │                                                                             │ │
│  │  Output (Target):                                                           │ │
│  │  "The weather is nice today."                                               │ │
│  │                                                                             │ │
│  │  손실 계산:                                                                  │ │
│  │  L = -Σ log P(output_token_i | input, output_tokens_<i)                     │ │
│  │                                                                             │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                    Speech SFT 데이터 예시                                   │ │
│  ├───────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                             │ │
│  │  ASR SFT:                                                                   │ │
│  │    Input: [오디오 파형]                                                     │ │
│  │    Output: "안녕하세요 오늘 날씨가 좋네요"                                   │ │
│  │                                                                             │ │
│  │  TTS SFT:                                                                   │ │
│  │    Input: "안녕하세요"                                                       │ │
│  │    Output: [오디오 코드/파형]                                                │ │
│  │                                                                             │ │
│  │  Audio QA SFT (Qwen-Audio):                                                 │ │
│  │    Input: [오디오] + "이 오디오에서 어떤 감정이 느껴지나요?"                 │ │
│  │    Output: "화자는 기쁜 감정을 표현하고 있습니다."                           │ │
│  │                                                                             │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Instruction Tuning (명령어 튜닝)

**정의**: **자연어 명령어(instruction) 형식**의 데이터로 SFT하는 것

#### 3.4.1 Instruction Tuning vs 일반 SFT

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               INSTRUCTION TUNING vs GENERAL SFT                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────┬─────────────────────────────────────┐ │
│  │         일반 SFT (Task-Specific)    │        Instruction Tuning           │ │
│  ├─────────────────────────────────────┼─────────────────────────────────────┤ │
│  │                                     │                                     │ │
│  │  데이터 형식:                       │  데이터 형식:                       │ │
│  │  (input, output)                    │  (instruction, input, output)       │ │
│  │                                     │                                     │ │
│  │  예시:                              │  예시:                              │ │
│  │  Input: "I love this movie"         │  Instruction: "Classify the         │ │
│  │  Output: "positive"                 │   sentiment of the following text." │ │
│  │                                     │  Input: "I love this movie"         │ │
│  │                                     │  Output: "positive"                 │ │
│  │                                     │                                     │ │
│  │  특징:                              │  특징:                              │ │
│  │  • 단일 태스크 최적화               │  • 다양한 태스크 통합               │ │
│  │  • 태스크별 모델 필요               │  • 하나의 모델로 여러 태스크        │ │
│  │  • Zero-shot 능력 제한적            │  • Zero-shot 능력 향상              │ │
│  │                                     │                                     │ │
│  │  모델 예시:                         │  모델 예시:                         │ │
│  │  • BERT + Classification Head       │  • FLAN-T5                          │ │
│  │  • GPT-2 for Summarization          │  • InstructGPT                      │ │
│  │                                     │  • Alpaca, Vicuna                   │ │
│  │                                     │                                     │ │
│  └─────────────────────────────────────┴─────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### 3.4.2 Instruction Tuning 데이터 형식

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    INSTRUCTION TUNING DATA FORMATS                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Format 1: Standard Instruction Format                                          │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ {                                                                           │ │
│  │   "instruction": "다음 텍스트를 요약하세요.",                               │ │
│  │   "input": "인공지능은 컴퓨터 과학의 한 분야로...(긴 텍스트)...",           │ │
│  │   "output": "인공지능은 인간의 지능을 모방하는 컴퓨터 기술입니다."          │ │
│  │ }                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Format 2: Chat/Conversation Format                                             │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ {                                                                           │ │
│  │   "messages": [                                                             │ │
│  │     {"role": "system", "content": "You are a helpful assistant."},          │ │
│  │     {"role": "user", "content": "Python으로 피보나치 수열을 구현해줘"},      │ │
│  │     {"role": "assistant", "content": "def fib(n):\n    if n <= 1:..."}      │ │
│  │   ]                                                                         │ │
│  │ }                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Format 3: Few-shot Format (FLAN)                                               │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ "Classify the sentiment. Examples:                                          │ │
│  │  'Great movie!' -> positive                                                 │ │
│  │  'Terrible acting' -> negative                                              │ │
│  │                                                                             │ │
│  │  Now classify: 'I enjoyed every minute of it'"                              │ │
│  │                                                                             │ │
│  │  -> "positive"                                                              │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Post-training (후학습)

**정의**: Pre-training 이후 수행되는 **모든 추가 학습 단계의 총칭**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         POST-TRAINING SCOPE                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                    POST-TRAINING (광의)                                    │ │
│  │                                                                             │ │
│  │  = Pre-training 이후의 모든 학습                                            │ │
│  │                                                                             │ │
│  │  포함 범위:                                                                 │ │
│  │  ├─ Fine-tuning (모든 유형)                                                 │ │
│  │  ├─ SFT                                                                     │ │
│  │  ├─ Instruction Tuning                                                      │ │
│  │  ├─ Alignment (RLHF, DPO, RLAIF)                                            │ │
│  │  └─ Continued Pre-training                                                  │ │
│  │                                                                             │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                    POST-TRAINING (협의 - OpenAI/Anthropic 용례)            │ │
│  │                                                                             │ │
│  │  = SFT + Alignment                                                          │ │
│  │                                                                             │ │
│  │  "Post-training transforms a capable but potentially problematic            │ │
│  │   base model into a helpful, harmless, and honest assistant."               │ │
│  │                                                                             │ │
│  │  주요 목적:                                                                 │ │
│  │  1. Instruction Following (명령 수행 능력)                                  │ │
│  │  2. Helpfulness (유용성)                                                    │ │
│  │  3. Safety (안전성)                                                         │ │
│  │  4. Honesty (정직성)                                                        │ │
│  │                                                                             │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ※ 용어 사용 주의:                                                              │
│  • Google: "Instruction Fine-tuning"                                           │
│  • OpenAI: "Post-training" (SFT + RLHF)                                        │
│  • Meta: "Fine-tuning" + "Alignment"                                           │
│  • 학계: "Alignment" 또는 "Preference Learning"                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.6 Alignment (정렬)

**정의**: 모델을 **인간의 선호도, 가치, 의도**에 맞게 조정하는 학습

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ALIGNMENT METHODS                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. RLHF (Reinforcement Learning from Human Feedback)                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  Step 1: Reward Model 학습                                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │  Prompt: "Explain quantum physics"                                   │   │ │
│  │  │  Response A: "Quantum physics is the study of..."  (선호됨)          │   │ │
│  │  │  Response B: "I don't know about that..."  (비선호)                  │   │ │
│  │  │                                                                      │   │ │
│  │  │  → Reward Model: R(prompt, response) → scalar score                  │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                             │ │
│  │  Step 2: PPO로 정책 최적화                                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │  │  max E[R(response)] - β * KL(π || π_ref)                             │   │ │
│  │  │                                                                      │   │ │
│  │  │  - Reward 최대화하면서                                                │   │ │
│  │  │  - 원본 모델(π_ref)에서 너무 벗어나지 않도록 제약                     │   │ │
│  │  └─────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                             │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  2. DPO (Direct Preference Optimization)                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  핵심 아이디어: Reward Model 없이 직접 선호도 학습                          │ │
│  │                                                                             │ │
│  │  손실 함수:                                                                 │ │
│  │  L = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x)                               │ │
│  │                    - log π(y_l|x)/π_ref(y_l|x)))]                           │ │
│  │                                                                             │ │
│  │  y_w = preferred (winner), y_l = dispreferred (loser)                       │ │
│  │                                                                             │ │
│  │  장점:                                                                      │ │
│  │  - Reward Model 학습 불필요                                                 │ │
│  │  - PPO 불필요 (복잡한 RL 제거)                                              │ │
│  │  - 안정적인 학습                                                            │ │
│  │  - 구현이 단순                                                              │ │
│  │                                                                             │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  3. RLAIF (RL from AI Feedback)                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  = RLHF와 동일한 프레임워크                                                 │ │
│  │  - 단, Human labeler 대신 AI (강력한 LLM)이 피드백 제공                     │ │
│  │                                                                             │ │
│  │  예: Constitutional AI (Anthropic)                                          │ │
│  │  - Claude가 헌법(principles)에 따라 자기 출력을 평가                        │ │
│  │                                                                             │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Relationship Hierarchy

### 4.1 용어 포함 관계

```mermaid
flowchart TD
    POST["Post-training (광의)"]
    POST --> CPT["Continued Pre-training<br/>추가 비지도 데이터로 사전학습 연장"]
    POST --> FT["Fine-tuning"]
    POST --> ALIGN["Alignment / Preference Learning"]

    FT --> UFT["Unsupervised Fine-tuning<br/>Domain-adaptive Pre-training"]
    FT --> SFT["Supervised Fine-tuning (SFT) ◄ 가장 일반적"]

    SFT --> TSFT["Task-Specific SFT<br/>단일 태스크 최적화"]
    SFT --> IT["Instruction Tuning ◄ SFT의 특수한 형태"]

    IT --> ZS["Zero-shot Instruction Tuning"]
    IT --> FS["Few-shot Instruction Tuning"]
    IT --> COT["Chain-of-Thought Instruction Tuning"]

    ALIGN --> RLHF["RLHF (PPO-based)"]
    ALIGN --> DPO["DPO (Direct Optimization)"]
    ALIGN --> RLAIF["RLAIF (AI Feedback)"]
    ALIGN --> IPO["IPO (Identity Preference Optimization)"]
    ALIGN --> KTO["KTO (Kahneman-Tversky Optimization)"]

    KEY["※ 핵심 관계: Instruction Tuning ⊂ SFT ⊂ Fine-tuning ⊂ Post-training. Alignment ⊂ Post-training (but ⊄ Fine-tuning in strict sense)"]
```

### 4.2 시간순 Training Pipeline

```mermaid
flowchart TD
    P0["PHASE 0: Data Collection &amp; Preparation<br/>대규모 텍스트/오디오 수집<br/>전처리, 필터링, 품질 관리"]
    P1["PHASE 1: Pre-training<br/>비지도/자기지도 학습<br/>수조 토큰 / 수만 시간 오디오<br/>Output: Base Model"]
    P2["PHASE 2: Supervised Fine-tuning (SFT)<br/>고품질 (input, output) 쌍 학습<br/>10K-100K 샘플<br/>Output: SFT Model<br/>└ Instruction Tuning (optional, 포함 가능)"]
    P3["PHASE 3: Alignment (Optional but common for chat models)<br/>선호도 데이터 학습<br/>RLHF, DPO, RLAIF 등<br/>Output: Aligned/Chat Model"]
    P4["PHASE 4: Deployment Optimization (Optional)<br/>Quantization (INT8, INT4)<br/>Distillation<br/>Pruning"]
    P0 --> P1 --> P2 --> P3 --> P4
```

---

## 5. LLM/NLP Domain Examples

### 5.1 실제 모델별 학습 파이프라인

```mermaid
flowchart TD
    subgraph G1["GPT-3 → InstructGPT → ChatGPT (OpenAI)"]
        direction TB
        A1["GPT-3 (Base)<br/>Pre-training: 300B tokens, Next Token Prediction<br/>결과: 강력하지만 명령 수행에 불안정"]
        A2["InstructGPT (SFT)<br/>SFT: 13K 고품질 demonstration 데이터<br/>데이터: (prompt, ideal_response) 쌍<br/>결과: 명령 수행 능력 향상"]
        A3["InstructGPT (RLHF)<br/>RLHF: 33K comparison 데이터로 Reward Model 학습 → PPO로 정책 최적화<br/>결과: 인간 선호도에 더 부합"]
        A4["ChatGPT<br/>추가 SFT + RLHF 반복<br/>대화 형식 최적화"]
        A1 --> A2 --> A3 --> A4
    end
    subgraph G2["LLaMA → Alpaca → Vicuna (Open Source)"]
        direction TB
        B1["LLaMA (Meta)<br/>Pre-training: 1.4T tokens<br/>공개 데이터만 사용"]
        B2["Alpaca (Stanford)<br/>Instruction Tuning: 52K instruction 데이터<br/>데이터 출처: Self-Instruct (GPT-3.5로 생성)<br/>방법: Full Fine-tuning"]
        B3["Vicuna (LMSYS)<br/>Instruction Tuning: 70K ShareGPT 대화 데이터<br/>실제 사용자-ChatGPT 대화"]
        B1 --> B2 --> B3
    end
    subgraph G3["T5 → FLAN-T5 (Google)"]
        direction TB
        C1["T5<br/>Pre-training: Span Corruption (C4 데이터)"]
        C2["FLAN-T5<br/>Instruction Tuning: 1,836 tasks (Flan Collection)<br/>Zero-shot, Few-shot, CoT 템플릿 혼합<br/>60+ NLP 데이터셋을 instruction 형식으로 변환"]
        C1 --> C2
    end
```

### 5.2 Instruction Tuning 데이터셋 예시

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                 INSTRUCTION TUNING DATASET EXAMPLES                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. FLAN Collection (Google)                                                    │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 규모: 1,836 tasks, 15M+ examples                                           │ │
│  │ 형식: Zero-shot, Few-shot, Chain-of-Thought 혼합                           │ │
│  │                                                                             │ │
│  │ 예시 (감정 분류):                                                           │ │
│  │ ┌─────────────────────────────────────────────────────────────────────┐   │ │
│  │ │ Zero-shot:                                                          │   │ │
│  │ │ "Is the following sentence positive or negative? 'I love this!'"   │   │ │
│  │ │ → "positive"                                                        │   │ │
│  │ │                                                                     │   │ │
│  │ │ Few-shot:                                                           │   │ │
│  │ │ "Classify sentiment:                                                │   │ │
│  │ │  'Great movie!' → positive                                          │   │ │
│  │ │  'Terrible!' → negative                                             │   │ │
│  │ │  'I love this!' → "                                                 │   │ │
│  │ │ → "positive"                                                        │   │ │
│  │ └─────────────────────────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  2. Self-Instruct / Alpaca (Stanford)                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 규모: 52K instructions                                                      │ │
│  │ 생성: GPT-3.5로 자동 생성                                                   │ │
│  │                                                                             │ │
│  │ 예시:                                                                       │ │
│  │ {                                                                           │ │
│  │   "instruction": "Give three tips for staying healthy.",                   │ │
│  │   "input": "",                                                              │ │
│  │   "output": "1. Eat a balanced diet...\n2. Exercise...\n3. Get sleep..."   │ │
│  │ }                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  3. ShareGPT / OpenAssistant (Community)                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │ 규모: 70K-160K 대화                                                         │ │
│  │ 특징: 실제 사용자 대화 (Multi-turn)                                         │ │
│  │                                                                             │ │
│  │ 예시:                                                                       │ │
│  │ {                                                                           │ │
│  │   "conversations": [                                                        │ │
│  │     {"from": "human", "value": "Python으로 정렬 알고리즘 구현해줘"},        │ │
│  │     {"from": "gpt", "value": "여러 정렬 알고리즘을 구현해드리겠습니다..."}  │ │
│  │   ]                                                                         │ │
│  │ }                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Speech/Audio Domain Examples

### 6.1 음성 모델 학습 파이프라인

```mermaid
flowchart TD
    subgraph W2V["wav2vec 2.0 (Meta) - ASR"]
        direction TB
        W1["Stage 1: Self-Supervised Pre-training<br/>데이터: 53K hours unlabeled audio (Libri-Light)<br/>방법: Contrastive Learning (마스킹 + InfoNCE loss)<br/>결과: 범용 음성 표현 학습"]
        W2["Stage 2: Supervised Fine-tuning (ASR)<br/>데이터: 10분 ~ 960시간 (audio, transcript) 쌍<br/>방법: CTC Loss 또는 Seq2Seq Loss<br/>결과: 10분 데이터로도 WER 4.8% 달성<br/>손실 함수 (CTC): L_CTC = -log P(y given x) = -log Σ P(π given x) (all valid alignments π)"]
        W1 --> W2
    end
    subgraph WH["Whisper (OpenAI) - Multilingual ASR"]
        direction TB
        WH1["Pre-training (Weakly Supervised)<br/>데이터: 680K hours (audio, noisy transcript) from web<br/>방법: Encoder-Decoder, Seq2Seq<br/>특징: 순수 self-supervised가 아님 (text 레이블 있음), but weakly = 노이즈 있는 웹 크롤링 데이터<br/>멀티태스크 학습: Transcription (영어), Translation (X→영어), Language ID, Timestamp prediction"]
        WH2["Fine-tuning (Domain-specific)<br/>예: 의료 ASR, 콜센터 ASR<br/>데이터: 도메인별 (audio, transcript) 쌍"]
        WH1 --> WH2
    end
    subgraph QA["Qwen-Audio (Alibaba) - Audio Understanding LLM"]
        direction TB
        Q1["Stage 1: Audio Encoder Pre-training<br/>Whisper Encoder 활용"]
        Q2["Stage 2: Multi-task Pre-training<br/>데이터: 30+ datasets (ASR, Audio QA, Sound Classification, etc.)<br/>방법: Audio-Text 매핑 학습"]
        Q3["Stage 3: Instruction Fine-tuning (Audio SFT)<br/>데이터: Instruction 형식의 Audio QA<br/>예시: Input [Audio clip] + 이 오디오에서 들리는 악기는? → Output 피아노와 바이올린이 들립니다."]
        Q4["Stage 4: DPO Alignment (Qwen2-Audio)<br/>방법: 음성 응답 선호도 데이터로 DPO<br/>목적: 더 자연스럽고 정확한 응답"]
        Q1 --> Q2 --> Q3 --> Q4
    end
```

### 6.2 TTS 모델 학습 파이프라인

```mermaid
flowchart TD
    subgraph VALLE["VALL-E / XTTS (Zero-shot TTS)"]
        direction TB
        V1["Stage 1: Pre-training<br/>데이터: 60K+ hours (text, audio) 쌍<br/>방법: Neural Codec Language Model - Audio → Codec tokens (EnCodec/Mimi), Text + Audio prompt → Audio tokens 생성"]
        V2["Stage 2: Fine-tuning (Voice Cloning)<br/>데이터: Target speaker의 10-30초 샘플<br/>방법: Few-shot adaptation"]
        V1 --> V2
    end
    subgraph OA["OpenAudio S1 (Fish Audio) - RLHF TTS"]
        direction TB
        O1["Stage 1: Pre-training<br/>데이터: 2M hours audio<br/>방법: Autoregressive Codec LM"]
        O2["Stage 2: SFT<br/>데이터: 고품질 (text, audio) 쌍<br/>방법: Standard SFT"]
        O3["Stage 3: Online RLHF<br/>방법: 1) TTS 출력 생성, 2) MOS 예측 모델로 품질 점수화, 3) PPO로 높은 MOS 점수 받도록 최적화<br/>특징: Speech 도메인에서 RLHF 성공 사례"]
        O1 --> O2 --> O3
    end
```

---

## 7. Moshi Training Pipeline Analysis

### 7.1 Moshi의 4단계 학습 파이프라인

```mermaid
flowchart TD
    S1["Stage 1: UNSUPERVISED PRE-TRAINING<br/>목적: 일반적인 언어/음성 표현 학습<br/>Text Backbone (Helium): 데이터 Wikipedia, Stack Exchange, 과학 논문 / 방법 Next Token Prediction / 크기 7B parameters<br/>Audio Codec (Mimi): 데이터 대규모 오디오 / 방법 RVQ-based Neural Codec / 출력 8 codebooks @ 12.5Hz<br/>분류: PRE-TRAINING"]
    S2["Stage 2: POST-TRAINING (Multi-stream Simulation)<br/>목적: 다중 스트림 처리 능력 학습<br/>데이터: 모노 오디오 + Speaker Diarization, 가상의 멀티 스트림 시뮬레이션<br/>방법: Diarization으로 화자 분리, 랜덤하게 Moshi/User 채널 할당, Inner Monologue 추가<br/>분류: POST-TRAINING (Simulated Multi-stream SFT)"]
    S3["Stage 3: FINE-TUNING (Fisher Corpus)<br/>목적: Full-duplex 대화 능력 학습<br/>데이터: Fisher Corpus (2,000 hours) - 전화 대화, 각 화자가 별도 채널에 녹음 (Ground Truth 스테레오), 8kHz → 24kHz 업샘플링 (AudioSR)<br/>방법: whisper-timestamped로 전사, (codes, text) → next codes 예측<br/>분류: SUPERVISED FINE-TUNING (Real Stereo Data)"]
    S4["Stage 4: INSTRUCT FINE-TUNING (Synthetic Data)<br/>목적: 대화 품질 향상, 다양한 시나리오 학습<br/>데이터 생성: 1) 170시간 scripted 대화 수집, 2) Multi-speaker TTS 시스템 학습, 3) 합성 대화 스크립트 생성, 4) TTS로 20,000시간 합성 데이터 생성<br/>분류: INSTRUCTION TUNING (Synthetic SFT)<br/>※ 주의: Moshi에서 Instruct Fine-tuning은 일반적인 NLP의 instruction tuning과 다름 → 합성 대화 데이터로의 추가 SFT를 의미"]
    S1 --> S2 --> S3 --> S4
    NOTE["※ ALIGNMENT (RLHF/DPO): Moshi는 명시적으로 사용하지 않음"]
    S4 -.-> NOTE
```

### 7.2 Moshi 학습 단계 분류 정리

| Moshi 용어 | 일반적 분류 | 데이터 | 목적 |
|-----------|-----------|--------|------|
| Unsupervised Pre-training | **Pre-training** | 비레이블 | 일반 표현 학습 |
| Post-training | **SFT** (Simulated) | 시뮬레이션 스테레오 | 멀티스트림 학습 |
| Fine-tuning | **SFT** (Real Data) | Fisher (실제 스테레오) | Full-duplex 학습 |
| Instruct Fine-tuning | **SFT** (Synthetic) | TTS 합성 대화 | 품질 향상 |

### 7.3 J-Moshi의 학습 파이프라인 비교

```mermaid
flowchart TD
    S1["Stage 1: Pre-training (J-CHAT)<br/>데이터: 60,000 hours (from J-CHAT 69K hours)<br/>처리: 모노 → Speaker Diarization → 스테레오<br/>방법: Simulated multi-stream training<br/>분류: Continued Pre-training + Simulated SFT 혼합 (Moshi의 Stage 1+2에 해당)"]
    S2["Stage 2: Fine-tuning (Real Stereo)<br/>데이터: 344 hours 고품질 스테레오 대화 - Japanese Callhome 16h, CSJ 12h, Travel Agency 115h, In-house 201h<br/>분류: SUPERVISED FINE-TUNING (Moshi의 Stage 3에 해당)"]
    S3["Stage 3: Data Augmentation (Multi-stream TTS)<br/>데이터: 602 hours TTS 합성<br/>소스 텍스트: JapanesePersonaChat, JapaneseEmpatheticDialogues 등<br/>방법: 1) LLM (Gemma-2-27b)로 구어체 변환, 2) Multi-stream TTS로 10개 샘플 생성, 3) WER 기반 최적 샘플 선택<br/>분류: SYNTHETIC DATA SFT (Moshi의 Stage 4에 해당)"]
    S1 --> S2 --> S3
    NOTE["※ RLHF/DPO: 사용하지 않음 (Moshi와 동일)"]
    S3 -.-> NOTE
```

---

## 8. Data Composition Guide

### 8.1 학습 단계별 데이터 특성

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DATA CHARACTERISTICS BY TRAINING STAGE                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        PRE-TRAINING DATA                                 │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                          │   │
│  │  특징:                                                                   │   │
│  │  • 대규모 (TB 단위)                                                      │   │
│  │  • 비레이블 또는 약한 레이블                                             │   │
│  │  • 다양성 중요 (도메인, 스타일, 언어)                                    │   │
│  │  • 품질보다 양이 중요                                                    │   │
│  │                                                                          │   │
│  │  LLM 예시:                                                               │   │
│  │  • Common Crawl, Wikipedia, Books, Code                                  │   │
│  │  • 수조 토큰                                                             │   │
│  │                                                                          │   │
│  │  Speech 예시:                                                            │   │
│  │  • Libri-Light (60K hours), J-CHAT (69K hours)                           │   │
│  │  • YouTube/Podcast 크롤링                                                │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        SFT DATA                                          │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                          │   │
│  │  특징:                                                                   │   │
│  │  • 중간 규모 (10K-100K 샘플)                                             │   │
│  │  • 고품질 레이블 필수                                                    │   │
│  │  • (input, output) 쌍 형식                                               │   │
│  │  • 양보다 품질이 중요                                                    │   │
│  │                                                                          │   │
│  │  LLM 예시:                                                               │   │
│  │  • Human demonstrations (prompt, ideal_response)                         │   │
│  │  • 인간 작성자 또는 강력한 LLM 생성                                      │   │
│  │                                                                          │   │
│  │  Speech 예시:                                                            │   │
│  │  • Fisher Corpus (2K hours stereo)                                       │   │
│  │  • 전문 녹음 스튜디오 데이터                                             │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    INSTRUCTION TUNING DATA                               │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                          │   │
│  │  특징:                                                                   │   │
│  │  • (instruction, input, output) 3-tuple 형식                             │   │
│  │  • 다양한 태스크/명령어 포함                                             │   │
│  │  • Zero-shot, Few-shot, CoT 템플릿 혼합                                  │   │
│  │                                                                          │   │
│  │  LLM 예시:                                                               │   │
│  │  • FLAN Collection (1,836 tasks)                                         │   │
│  │  • Alpaca (52K instructions)                                             │   │
│  │  • ShareGPT (70K conversations)                                          │   │
│  │                                                                          │   │
│  │  Speech 예시 (emerging):                                                 │   │
│  │  • Audio QA: [audio] + "질문" → "답변"                                   │   │
│  │  • Audio Instruction: "이 오디오를 요약해줘" → "요약"                    │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    ALIGNMENT (PREFERENCE) DATA                           │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │                                                                          │   │
│  │  특징:                                                                   │   │
│  │  • (prompt, chosen, rejected) 3-tuple 형식                               │   │
│  │  • 인간/AI 평가자의 선호도                                               │   │
│  │  • 상대적 비교 (A > B)                                                   │   │
│  │                                                                          │   │
│  │  LLM 예시:                                                               │   │
│  │  • Anthropic HH-RLHF (170K comparisons)                                  │   │
│  │  • OpenAI comparison data                                                │   │
│  │  • UltraFeedback                                                         │   │
│  │                                                                          │   │
│  │  Speech 예시 (연구 단계):                                                │   │
│  │  • TTS 품질 선호도: Audio A > Audio B                                    │   │
│  │  • MOS 점수 기반 자동 레이블링                                           │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 K-Moshi를 위한 데이터 구성 권장

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    K-MOSHI DATA COMPOSITION RECOMMENDATION                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Phase 1: Bootstrap with External TTS (SFT)                                     │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  목표: ~500-1000시간 합성 데이터                                           │ │
│  │                                                                             │ │
│  │  텍스트 소스:                                                               │ │
│  │  ├─ AI Hub 감성 대화 → TTS 합성                                            │ │
│  │  ├─ 모두의 말뭉치 구어/메신저 → TTS 합성                                   │ │
│  │  └─ LLM으로 대화 생성 → TTS 합성                                           │ │
│  │                                                                             │ │
│  │  데이터 형식:                                                               │ │
│  │  {                                                                          │ │
│  │    "path": "stereo_audio.wav",  // L=Moshi, R=User                         │ │
│  │    "duration": 45.2,                                                        │ │
│  │    "alignments": [["안녕하세요", [0.0, 0.8], "SPEAKER_MAIN"], ...]         │ │
│  │  }                                                                          │ │
│  │                                                                             │ │
│  │  분류: SUPERVISED FINE-TUNING (Synthetic Data)                              │ │
│  │                                                                             │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Phase 2: Self-Generation (Data Augmentation)                                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  목표: ~500-1000시간 추가 데이터                                           │ │
│  │                                                                             │ │
│  │  방법 (J-Moshi 방식):                                                       │ │
│  │  1) Phase 1으로 학습된 K-Moshi 사용                                         │ │
│  │  2) 새로운 텍스트 대화 입력                                                 │ │
│  │  3) Multi-stream TTS로 여러 샘플 생성                                       │ │
│  │  4) WER 기반 최적 샘플 선택                                                 │ │
│  │                                                                             │ │
│  │  분류: SELF-TRAINING / DATA AUGMENTATION                                    │ │
│  │                                                                             │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Phase 3 (Optional): Real Stereo Fine-tuning                                    │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  목표: 실제 한국어 스테레오 대화 데이터로 Fine-tuning                       │ │
│  │                                                                             │ │
│  │  데이터 소스:                                                               │ │
│  │  ├─ KsponSpeech (스테레오 가능 부분)                                        │ │
│  │  └─ 자체 녹음 데이터                                                        │ │
│  │                                                                             │ │
│  │  분류: SUPERVISED FINE-TUNING (Real Data)                                   │ │
│  │                                                                             │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Practical Decision Framework

### 9.1 학습 방법 선택 가이드

```mermaid
flowchart TD
    Q1{"Q1: 모델을 처음부터 학습하나요?"}
    Q1 -->|YES| PRE["PRE-TRAINING 필요<br/>대규모 비지도 데이터 필요<br/>수천 GPU-hours 필요"]
    Q1 -->|NO| BASE["기존 Pre-trained 모델 사용"]
    BASE --> Q2{"Q2: 특정 태스크에 최적화하나요?"}
    Q2 -->|YES| TSFT["TASK-SPECIFIC FINE-TUNING<br/>해당 태스크의 (input, output) 데이터 필요"]
    Q2 -->|NO| Q3{"Q3: 다양한 명령어를 수행하게 하나요?"}
    Q3 -->|YES| IT["INSTRUCTION TUNING<br/>(instruction, input, output) 형식 데이터<br/>다양한 태스크 템플릿"]
    Q3 -->|NO| Q4{"Q4: 인간 선호도에 맞추나요?"}
    Q4 -->|YES| ALIGN["ALIGNMENT (RLHF/DPO)<br/>선호도 비교 데이터 필요<br/>(prompt, chosen, rejected) 형식"]
    Q4 -->|NO| GSFT["General SFT<br/>고품질 (input, output) 쌍"]
```

### 9.2 K-Moshi 학습 권장 전략

| 단계 | 방법 | 데이터 | 목표 |
|------|------|--------|------|
| 1 | **Synthetic SFT** | TTS 합성 대화 500-1000h | Full-duplex 기본 능력 |
| 2 | **Self-Generation** | K-Moshi 자체 생성 500-1000h | 데이터 증강 |
| 3 | **Real Data SFT** (Optional) | 실제 스테레오 대화 | 품질 향상 |
| - | ~~RLHF/DPO~~ | - | 미권장 (검증 사례 없음) |

---

## 10. References

### 10.1 핵심 논문

**LLM Training:**
- [Training language models to follow instructions (InstructGPT)](https://openai.com/index/instruction-following/)
- [FLAN: Introducing instruction fine-tuning](https://research.google/blog/introducing-flan-more-generalizable-language-models-with-instruction-fine-tuning/)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

**Speech Training:**
- [wav2vec 2.0: Self-Supervised Learning of Speech](https://arxiv.org/abs/2006.11477)
- [Whisper: Robust Speech Recognition](https://openai.com/research/whisper)
- [Moshi: Speech-Text Foundation Model](https://arxiv.org/abs/2410.00037)
- [J-Moshi: Japanese Full-duplex Dialogue](https://arxiv.org/abs/2506.02979)

### 10.2 참고 자료

- [Instruction Tuning Survey](https://arxiv.org/abs/2308.10792)
- [Understanding SFT (Cameron R. Wolfe)](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised)
- [LLM Fine-tuning Guide (IBM)](https://www.ibm.com/think/topics/instruction-tuning)
- [Post-training Methods (Red Hat)](https://developers.redhat.com/articles/2025/11/04/post-training-methods-language-models)

---

*Last Updated: 2026-01-13*
*Document Version: 1.0*
*Author: K-Moshi Development Team*
