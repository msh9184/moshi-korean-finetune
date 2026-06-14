# K-Moshi Full Finetuning 하이퍼파라미터 가이드

> **Version**: 1.0
> **Date**: 2026-01-13
> **Purpose**: J-Moshi 및 Moshi 벤치마킹 기반 최적 하이퍼파라미터 추천

---

## 목차

1. [개요](#1-개요)
2. [학습 환경 및 구성](#2-학습-환경-및-구성)
3. [Stage 1: Pre-training 하이퍼파라미터](#3-stage-1-pre-training-하이퍼파라미터)
4. [Stage 2: Fine-tuning 하이퍼파라미터](#4-stage-2-fine-tuning-하이퍼파라미터)
5. [컴포넌트별 학습 전략](#5-컴포넌트별-학습-전략)
6. [추천 설정 파일](#6-추천-설정-파일)
7. [학습 모니터링 체크리스트](#7-학습-모니터링-체크리스트)
8. [참조 논문 및 코드](#8-참조-논문-및-코드)

---

## 1. 개요

### 1.1 학습 목표

K-Moshi는 한국어 Full-Duplex Spoken Dialogue 모델로, 다음 목표를 가집니다:

- **Stage 1 (Pre-training)**: 13,000시간 한국어 대화 데이터로 기본 한국어 음성 대화 능력 습득
- **Stage 2 (Fine-tuning)**: 고품질 합성 Q&A 데이터로 대화 능력 향상

### 1.2 모델 구성

```
┌─────────────────────────────────────────────────────────────────┐
│                     K-Moshi Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│  Component              │ Source      │ Training Status          │
├─────────────────────────┼─────────────┼──────────────────────────┤
│  Mimi Encoder/Decoder   │ Moshi       │ ❄️ FROZEN                │
│  Temporal Transformer   │ Moshi 7B    │ 🔥 Full Finetune         │
│  Depth Transformer      │ Scratch     │ 🆕 From Scratch          │
│  Text Embedding         │ Scratch     │ 🆕 From Scratch (32k)    │
│  Text Linear            │ Scratch     │ 🆕 From Scratch          │
│  Audio Embeddings       │ Moshi       │ 🔥 Full Finetune         │
│  Audio Linears          │ Moshi       │ 🔥 Full Finetune         │
└─────────────────────────┴─────────────┴──────────────────────────┘
```

### 1.3 참조 모델 비교

| 항목 | Moshi (Original) | J-Moshi | K-Moshi (추천) |
|------|------------------|---------|----------------|
| **언어** | English | Japanese | Korean |
| **Pre-train 데이터** | 7M hours | ~60K hours | **13K hours** |
| **Fine-tune 데이터** | 2K + 20K TTS | 344h + 602h TTS | TBD |
| **GPU** | - | 128x V100 32GB | **32x A100 80GB** |
| **Backbone** | Helium 7B | Moshi 7B | Moshi 7B |
| **Mimi Training** | Full | Frozen | **Frozen** |

---

## 2. 학습 환경 및 구성

### 2.1 하드웨어 환경

```yaml
Hardware:
  nodes: 4
  gpus_per_node: 8
  total_gpus: 32
  gpu_type: "NVIDIA A100 80GB"
  total_vram: "2,560 GB"
  interconnect: "NVLink + InfiniBand"
```

### 2.2 J-Moshi vs K-Moshi GPU 비교

| Metric | J-Moshi (V100 32GB) | K-Moshi (A100 80GB) | 비율 |
|--------|---------------------|---------------------|------|
| **Pre-train GPUs** | 128 | 32 | 0.25x |
| **총 VRAM** | 4,096 GB | 2,560 GB | 0.63x |
| **GPU 성능** | 1x (baseline) | ~2.5x (estimated) | 2.5x |
| **실효 처리량** | 1x | ~1.6x | ≈1.6x |

> **결론**: 32x A100은 128x V100 대비 약 **60-70%** 처리량. 학습 시간 약 1.5-1.7배 예상.

---

## 3. Stage 1: Pre-training 하이퍼파라미터

### 3.1 J-Moshi Pre-training 설정 (Reference)

```yaml
# J-Moshi Pre-training on J-CHAT (~60,000 hours)
Hardware:
  gpus: 128 x V100 32GB
  parallelism: ZeRO-3 (DeepSpeed)
  mixed_precision: float16
  activation_checkpointing: true

Data:
  corpus: "J-CHAT"
  hours: 60000
  max_sequence_length: "2.7 minutes (2,048 tokens)"

Batch:
  total_batch_size: 512
  # per_gpu: 512 / 128 = 4 samples

Optimizer:
  type: "AdamW"
  lr: 3e-5
  betas: [0.9, 0.95]  # Llama 2 style
  eps: 1e-5
  weight_decay: 0.1

Scheduler:
  type: "linear_warmup"
  warmup_steps: 500
  # After warmup: constant LR

Loss:
  text_padding_weight: 0.5  # PAD tokens 50% weight
  first_codebook_weight: 100  # semantic:acoustic = 100:1

Training:
  epochs: 1
  total_steps: 8880
  training_time: "36 hours"
```

### 3.2 K-Moshi Pre-training 추천 설정

```yaml
# K-Moshi Pre-training on Korean Dialogue (~13,000 hours)
# =============================================================================

Hardware:
  gpus: 32 x A100 80GB (4 nodes x 8 GPUs)
  parallelism: FSDP  # PyTorch native, simpler than ZeRO-3
  mixed_precision: bfloat16  # A100 native support
  activation_checkpointing: true

Data:
  corpus: "Korean Dialogue"
  hours: 13000
  max_sequence_length: "2.7 minutes (2,048 tokens)"
  # At 12.5Hz: 2,048 tokens = 163.84 seconds ≈ 2.7분

# ─────────────────────────────────────────────────────────────────────────────
# BATCH SIZE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
#
# J-Moshi: 512 total batch / 128 GPUs = 4 per GPU
#
# K-Moshi 목표: 비슷한 effective batch size 유지
#   - 32 GPUs × 4 per GPU = 128 (너무 작음)
#   - Gradient Accumulation으로 보상
#
# 권장: batch_size=4, num_microbatches=4 (grad accum)
#   → Effective batch = 32 × 4 × 4 = 512 (J-Moshi와 동일!)
#
# A100 80GB 메모리 예상:
#   - 7B model (bf16): ~14GB
#   - Optimizer states (Adam): ~28GB
#   - Activations (with checkpointing): ~20GB per sample
#   - 4 samples: ~80GB (꽉 찰 수 있음)
#
# 안전한 설정: batch_size=2, num_microbatches=8
#   → Effective batch = 32 × 2 × 8 = 512
# ─────────────────────────────────────────────────────────────────────────────

Batch:
  batch_size: 2  # per GPU
  num_microbatches: 8  # gradient accumulation steps
  effective_batch_size: 512  # 32 × 2 × 8

# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE LENGTH (duration_sec)
# ─────────────────────────────────────────────────────────────────────────────
#
# J-Moshi: 2.7분 = 162초 (2,048 tokens at 12.5Hz)
#
# 권장: duration_sec=160 (≈2,048 tokens)
#   - J-Moshi와 동일한 컨텍스트 길이
#   - 긴 대화 패턴 학습 가능
#
# 메모리 부족 시: duration_sec=80 (≈1,024 tokens)
# ─────────────────────────────────────────────────────────────────────────────

Sequence:
  duration_sec: 160  # 2.7분 = J-Moshi와 동일
  # Alternative if OOM: 80 (1.3분)

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
#
# J-Moshi는 Llama 2 스타일 AdamW 사용:
#   - β1=0.9, β2=0.95 (standard: 0.9, 0.999)
#   - 낮은 β2는 gradient의 빠른 적응에 유리
#   - 음성 모델에서 검증된 설정
# ─────────────────────────────────────────────────────────────────────────────

Optimizer:
  type: "AdamW"
  lr: 3e-5  # J-Moshi Pre-training과 동일
  betas: [0.9, 0.95]  # Llama 2 style (중요!)
  eps: 1e-5
  weight_decay: 0.1
  foreach: false  # FSDP resume compatibility (CRITICAL!)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
#
# J-Moshi: Linear warmup → Constant LR
#
# 권장: Linear warmup → Cosine decay (더 안정적)
#   - warmup_steps: 500 (J-Moshi와 동일)
#   - min_lr: 1e-7 (최종 학습률)
# ─────────────────────────────────────────────────────────────────────────────

Scheduler:
  type: "cosine_warmup"  # or "warmup_linear" for J-Moshi style
  warmup_steps: 500
  min_lr: 1e-7

# ─────────────────────────────────────────────────────────────────────────────
# LOSS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
#
# Moshi/J-Moshi 공통:
#   - PAD token: 50% weight (너무 많은 PAD 학습 방지)
#   - Semantic:Acoustic = 100:1 (첫 번째 codebook 강조)
#
# J-Moshi 발견: 일본어에서 PAD 비율 88% (영어 65%)
#   → 한국어도 비슷할 것으로 예상
#   → PAD weight 조정 가능성 있음
# ─────────────────────────────────────────────────────────────────────────────

Loss:
  first_codebook_weight_multiplier: 100.0
  text_padding_weight: 0.5
  # 한국어 특성에 따라 0.3~0.5 조정 가능

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING STEPS CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
#
# 데이터셋 크기: 13,000 hours
# 샘플 길이: 2.7분 = 0.045 hours
# 총 샘플 수: 13,000 / 0.045 ≈ 288,889 samples
# Effective batch size: 512
# Steps per epoch: 288,889 / 512 ≈ 564 steps
#
# J-Moshi: 1 epoch on 60,000h → 8,880 steps
#   → 60,000 / 0.045 / 512 ≈ 2,604 steps (이론값)
#   → 실제 8,880 steps = ~3.4 epochs 상당
#
# K-Moshi 권장:
#   - 1 epoch: ~565 steps
#   - 3 epochs: ~1,695 steps
#   - 5 epochs: ~2,825 steps (권장)
#
# 더 적은 데이터(13K vs 60K)이므로 더 많은 epoch 필요
# ─────────────────────────────────────────────────────────────────────────────

Training:
  max_steps: 3000  # ~5 epochs on 13K hours
  # Alternative: 5000 steps for more thorough training

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING TIME ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────
#
# J-Moshi: 8,880 steps in 36 hours on 128x V100
#   → ~245 steps/hour
#   → ~14.7 seconds/step
#
# K-Moshi on 32x A100 (estimated):
#   - A100 vs V100: ~2.5x speedup
#   - 32 vs 128 GPUs: 0.25x throughput
#   - Net factor: ~0.6x
#   → ~147 steps/hour
#   → ~24.5 seconds/step
#
# 3,000 steps: ~20 hours
# 5,000 steps: ~34 hours
# ─────────────────────────────────────────────────────────────────────────────

Checkpointing:
  save_freq: 100  # Every 100 steps
  max_keep: 10
  save_optimizer: true
```

### 3.3 Pre-training 요약 테이블

| Parameter | J-Moshi | K-Moshi (추천) | 비고 |
|-----------|---------|----------------|------|
| **데이터** | 60K hours | 13K hours | ~5x 적음 |
| **GPU** | 128x V100 | 32x A100 | 성능 보상 |
| **Parallelism** | ZeRO-3 | FSDP | PyTorch native |
| **Precision** | fp16 | bf16 | A100 최적화 |
| **Duration** | 162s | 160s | 동일 |
| **Batch (per GPU)** | 4 | 2 | 메모리 고려 |
| **Grad Accum** | - | 8 | Effective 512 유지 |
| **LR** | 3e-5 | 3e-5 | 동일 |
| **Warmup** | 500 steps | 500 steps | 동일 |
| **Scheduler** | Linear→Const | Cosine | 약간 개선 |
| **Epochs** | 1 (effective ~3.4) | 5 | 적은 데이터 보상 |
| **Steps** | 8,880 | 3,000-5,000 | 데이터 비례 |
| **Time** | 36h | ~20-34h | 추정 |

---

## 4. Stage 2: Fine-tuning 하이퍼파라미터

### 4.1 J-Moshi Fine-tuning 설정 (Reference)

```yaml
# J-Moshi Fine-tuning on Stereo Dialogue (344 hours)
Hardware:
  gpus: 16 x V100 32GB

Data:
  corpus: "Stereo Spoken Dialogue"
  hours: 344
  split: "94:3:3 (train:valid:test)"

Batch:
  total_batch_size: 16
  # per_gpu: 16 / 16 = 1 sample

# ─────────────────────────────────────────────────────────────────────────────
# KEY INSIGHT: Two-Rate Learning
# ─────────────────────────────────────────────────────────────────────────────
#
# J-Moshi는 TempFormer와 DepFormer에 다른 학습률 사용:
#   - TempFormer (main transformer): 2e-6
#   - DepFormer (depth transformer): 4e-6 (2배)
#
# 이유:
#   - TempFormer: Pre-trained weights 보존 필요 → 낮은 LR
#   - DepFormer: 더 적극적인 adaptation 필요 → 높은 LR
# ─────────────────────────────────────────────────────────────────────────────

Optimizer:
  type: "AdamW"
  tempformer_lr: 2e-6  # TempFormer (main)
  depformer_lr: 4e-6   # DepFormer (2x)
  betas: [0.9, 0.95]
  eps: 1e-5
  weight_decay: 0.1

Training:
  epochs: 3
  total_steps: 1423
  training_time: "2 hours"
```

### 4.2 K-Moshi Fine-tuning 추천 설정

```yaml
# K-Moshi Fine-tuning on High-Quality Q&A Data
# =============================================================================
#
# 가정: 고품질 합성 Q&A 데이터 (예: 500-1000시간)
#
# ─────────────────────────────────────────────────────────────────────────────

Hardware:
  gpus: 32 x A100 80GB
  parallelism: FSDP
  mixed_precision: bfloat16

Data:
  corpus: "Korean Synthetic Q&A Dialogue"
  hours: 500  # TBD
  format: "stereo (L=Moshi, R=User)"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH SIZE (Fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────
#
# J-Moshi: batch=16 on 16 GPUs = 1 per GPU
#
# K-Moshi: 32 GPUs, 더 큰 effective batch 가능
#   - batch_size=2, num_microbatches=4
#   - Effective: 32 × 2 × 4 = 256
#
# 또는 J-Moshi와 유사하게:
#   - batch_size=1, num_microbatches=1
#   - Effective: 32 × 1 × 1 = 32
# ─────────────────────────────────────────────────────────────────────────────

Batch:
  batch_size: 2
  num_microbatches: 4
  effective_batch_size: 256

Sequence:
  duration_sec: 160  # Pre-training과 동일

# ─────────────────────────────────────────────────────────────────────────────
# TWO-RATE LEARNING (핵심!)
# ─────────────────────────────────────────────────────────────────────────────
#
# J-Moshi의 핵심 발견: TempFormer와 DepFormer에 다른 LR 사용
#
# K-Moshi 상황:
#   - TempFormer: Pre-trained (Moshi 7B) → 보존 필요
#   - DepFormer: From scratch → 더 적극적 학습
#   - Text Embedding/Linear: From scratch → 더 적극적 학습
#
# 권장 학습률 비율:
#   - TempFormer: 1x (base)
#   - DepFormer: 2x (J-Moshi와 동일)
#   - Text components: 2-4x (scratch이므로)
# ─────────────────────────────────────────────────────────────────────────────

Optimizer:
  type: "AdamW"

  # Three-Rate Learning 권장
  tempformer_lr: 2e-6    # TempFormer (Pre-trained 보존)
  depformer_lr: 4e-6     # DepFormer (2x, J-Moshi와 동일)
  text_lr: 8e-6          # Text embedding/linear (4x, scratch)

  # 또는 Two-Rate (간단한 버전)
  # lr: 2e-6              # TempFormer + Text
  # depformer_lr: 4e-6    # DepFormer

  betas: [0.9, 0.95]
  eps: 1e-5
  weight_decay: 0.1

Scheduler:
  type: "cosine_warmup"
  warmup_steps: 100  # Fine-tuning은 짧은 warmup
  min_lr: 1e-8

Loss:
  first_codebook_weight_multiplier: 100.0
  text_padding_weight: 0.5

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING STEPS (Fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────
#
# J-Moshi: 344h → 1,423 steps (3 epochs)
#   → 344h / 0.045h per sample = ~7,644 samples
#   → 7,644 / 16 batch = ~478 steps/epoch
#   → 3 epochs × 478 = 1,434 steps
#
# K-Moshi (500h 가정):
#   → 500h / 0.045h = ~11,111 samples
#   → 11,111 / 256 batch = ~43 steps/epoch
#   → 5 epochs × 43 = ~217 steps
#
# 더 많은 학습 권장: 500-1000 steps
# ─────────────────────────────────────────────────────────────────────────────

Training:
  max_steps: 500
  epochs: 5  # 또는 step 기반

Checkpointing:
  save_freq: 50
  max_keep: 10
```

### 4.3 Fine-tuning 요약 테이블

| Parameter | J-Moshi | K-Moshi (추천) | 비고 |
|-----------|---------|----------------|------|
| **데이터** | 344h | 500-1000h | TBD |
| **GPU** | 16x V100 | 32x A100 | - |
| **Batch** | 16 total | 256 total | 더 큰 batch |
| **TempFormer LR** | 2e-6 | 2e-6 | 동일 |
| **DepFormer LR** | 4e-6 | 4e-6 | 동일 (2x) |
| **Text LR** | - | 8e-6 | 신규 (scratch) |
| **Warmup** | - | 100 steps | 짧은 warmup |
| **Epochs** | 3 | 5 | - |
| **Steps** | 1,423 | 500-1000 | - |
| **Time** | 2h | ~1-2h | 추정 |

---

## 5. 컴포넌트별 학습 전략

### 5.1 컴포넌트 분류 및 학습 전략

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    K-Moshi Component Training Strategy                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ FROZEN COMPONENTS (No Training)                                      │   │
│  │                                                                      │   │
│  │  • Mimi Encoder      - 24kHz → 12.5Hz audio tokens                  │   │
│  │  • Mimi Decoder      - Audio tokens → 24kHz waveform                │   │
│  │  • Mimi RVQ          - 8-level residual vector quantization         │   │
│  │                                                                      │   │
│  │  이유: J-Moshi 논문에서 Mimi가 일본어에서도 잘 동작함을 확인.        │   │
│  │        한국어도 동일하게 적용 가능할 것으로 기대.                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PRE-TRAINED + FINE-TUNED (LR: 2e-6 ~ 3e-5)                          │   │
│  │                                                                      │   │
│  │  • Temporal Transformer (7B)                                         │   │
│  │      - 32 layers, 32 heads, 4096 hidden                             │   │
│  │      - Pre-trained on English (Moshi)                                │   │
│  │      - 한국어 adaptation 필요                                        │   │
│  │      - LR: 3e-5 (pre-train) → 2e-6 (fine-tune)                      │   │
│  │                                                                      │   │
│  │  • Audio Embeddings (8 codebooks)                                    │   │
│  │      - 동일 vocab size 유지 (2048 per codebook)                      │   │
│  │      - Pre-trained weights 활용                                      │   │
│  │                                                                      │   │
│  │  • Audio Linears (8 output heads)                                    │   │
│  │      - Pre-trained weights 활용                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ FROM SCRATCH (LR: 4e-6 ~ 8e-6)                                       │   │
│  │                                                                      │   │
│  │  • Depth Transformer (DepFormer)                                     │   │
│  │      - 6 layers (smaller than TempFormer)                           │   │
│  │      - Audio token generation along depth axis                       │   │
│  │      - 완전히 새로 학습 필요                                         │   │
│  │      - LR: 4e-6 (2x TempFormer)                                     │   │
│  │                                                                      │   │
│  │  • Text Embedding (32k Korean vocab)                                 │   │
│  │      - 새로운 한국어 토크나이저 (32k vocab)                          │   │
│  │      - Random initialization                                         │   │
│  │      - LR: 4e-6 ~ 8e-6                                              │   │
│  │                                                                      │   │
│  │  • Text Linear (output projection)                                   │   │
│  │      - 새로운 vocab에 맞춤                                           │   │
│  │      - Random initialization                                         │   │
│  │      - LR: 4e-6 ~ 8e-6                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Parameter Group 설정

```python
# 권장 param_groups 설정 (finetune/scheduler.py에 구현 가능)

def get_multi_rate_optimizer(
    model: torch.nn.Module,
    tempformer_lr: float = 3e-5,
    depformer_lr: float = 6e-5,  # 2x
    text_lr: float = 1.2e-4,     # 4x
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-5,
) -> torch.optim.AdamW:
    """
    Multi-rate optimizer for K-Moshi full finetuning.

    Learning rate strategy:
    - TempFormer (pre-trained): 1x base LR
    - DepFormer (from scratch): 2x base LR
    - Text components (from scratch): 4x base LR
    """
    tempformer_params = []
    depformer_params = []
    text_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "depformer" in name.lower():
            depformer_params.append(param)
        elif "text_emb" in name.lower() or "text_linear" in name.lower():
            text_params.append(param)
        else:
            tempformer_params.append(param)

    param_groups = [
        {"params": tempformer_params, "lr": tempformer_lr, "name": "tempformer"},
        {"params": depformer_params, "lr": depformer_lr, "name": "depformer"},
        {"params": text_params, "lr": text_lr, "name": "text"},
    ]

    return torch.optim.AdamW(
        param_groups,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        foreach=False,  # CRITICAL for FSDP!
    )
```

---

## 6. 추천 설정 파일

### 6.1 Stage 1: Pre-training Config

```yaml
# example/korean_pretrain_stage1.yaml
# =============================================================================
# K-Moshi Stage 1: Pre-training on 13K hours Korean Dialogue
# Hardware: 4 nodes x 8 GPUs (32x A100 80GB)
# =============================================================================

# Data
data:
  train_data: './data/korean_pretrain_train.jsonl'
  eval_data: './data/korean_pretrain_valid.jsonl'
  shuffle: true

# Model
moshi_paths:
  hf_repo_id: null
  moshi_path: '/path/to/moshi/model.safetensors'
  mimi_path: '/path/to/mimi/tokenizer.safetensors'
  tokenizer_path: '/path/to/korean_tokenizer_32k.model'
  config_path: '/path/to/config.json'

# Backbone
backbone:
  type: "moshi"
  moshi:
    hidden_dim: 4096
    num_layers: 32
    num_heads: 32
    gradient_checkpointing: true

# Full Finetuning
full_finetuning: true
lora:
  enable: false

# Korean settings
korean:
  enable_user_stream: false
  full_duplex_input: true

# Distributed
distributed_backend: fsdp

# Loss
first_codebook_weight_multiplier: 100.0
text_padding_weight: 0.5

# Batch & Sequence
duration_sec: 160          # 2.7분 (J-Moshi와 동일)
batch_size: 2              # per GPU
num_microbatches: 8        # grad accum (effective: 512)
max_steps: 3000            # ~5 epochs on 13K hours
max_norm: 1.0
gradient_checkpointing: true

# Optimizer (J-Moshi Pre-training style)
optim:
  lr: 3.0e-5               # Base LR
  depformer_lr: 6.0e-5     # 2x for DepFormer (scratch)
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95              # Llama 2 style
  eps: 1.0e-5
  pct_start: 0.05

# Scheduler
scheduler:
  type: 'cosine_warmup'
  warmup_steps: 500        # J-Moshi와 동일
  min_lr: 1.0e-7

param_dtype: bfloat16

# Checkpointing
checkpoint:
  enabled: true
  save_freq: 100
  max_keep: 10
  resume_if_exist: true

# Logging
seed: 42
log_freq: 10
eval_freq: 100
do_eval: true
eval_samples: 50

run_dir: './runs/korean_pretrain_stage1'
```

### 6.2 Stage 2: Fine-tuning Config

```yaml
# example/korean_finetune_stage2.yaml
# =============================================================================
# K-Moshi Stage 2: Fine-tuning on High-Quality Q&A Data
# Hardware: 4 nodes x 8 GPUs (32x A100 80GB)
# =============================================================================

# Data
data:
  train_data: './data/korean_qa_train.jsonl'
  eval_data: './data/korean_qa_valid.jsonl'
  shuffle: true

# Model (Stage 1 checkpoint)
moshi_paths:
  hf_repo_id: null
  moshi_path: './runs/korean_pretrain_stage1/checkpoint_3000/model.safetensors'
  mimi_path: '/path/to/mimi/tokenizer.safetensors'
  tokenizer_path: '/path/to/korean_tokenizer_32k.model'
  config_path: '/path/to/config.json'

# Backbone
backbone:
  type: "moshi"
  moshi:
    hidden_dim: 4096
    num_layers: 32
    num_heads: 32
    gradient_checkpointing: true

# Full Finetuning
full_finetuning: true
lora:
  enable: false

# Korean settings
korean:
  enable_user_stream: false
  full_duplex_input: true

# Distributed
distributed_backend: fsdp

# Loss
first_codebook_weight_multiplier: 100.0
text_padding_weight: 0.5

# Batch & Sequence (Fine-tuning)
duration_sec: 160
batch_size: 2
num_microbatches: 4        # Smaller grad accum (effective: 256)
max_steps: 500
max_norm: 1.0
gradient_checkpointing: true

# Optimizer (J-Moshi Fine-tuning style - Two/Three Rate)
optim:
  lr: 2.0e-6               # TempFormer (conservative)
  depformer_lr: 4.0e-6     # DepFormer (2x, J-Moshi와 동일)
  # text_lr: 8.0e-6        # Optional: Text components (4x)
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  eps: 1.0e-5
  pct_start: 0.2           # Shorter warmup ratio

# Scheduler
scheduler:
  type: 'cosine_warmup'
  warmup_steps: 100        # Shorter warmup for fine-tuning
  min_lr: 1.0e-8

param_dtype: bfloat16

# Checkpointing
checkpoint:
  enabled: true
  save_freq: 50
  max_keep: 10
  resume_if_exist: true

# Logging
seed: 42
log_freq: 5
eval_freq: 50
do_eval: true
eval_samples: 30

run_dir: './runs/korean_finetune_stage2'
```

---

## 7. 학습 모니터링 체크리스트

### 7.1 Pre-training 모니터링

```markdown
## Pre-training Health Checks

### Step 0-100 (Initial)
- [ ] Loss 급격히 감소하는지 확인 (10+ → 5-6)
- [ ] Text token loss vs PAD token loss 비율 확인
- [ ] Semantic token loss >> Acoustic token loss 확인 (100:1 weight 적용됨)
- [ ] GPU 메모리 안정적인지 확인 (~70-80GB per GPU)
- [ ] Gradient norm 안정적인지 확인 (< 10.0)

### Step 100-500 (Warmup)
- [ ] Learning rate가 warmup 완료 후 최대값 도달
- [ ] Loss 지속적으로 감소
- [ ] 체크포인트 저장 정상 동작

### Step 500-3000 (Main Training)
- [ ] Loss plateau 없이 지속 감소
- [ ] Eval loss도 감소 추세 (overfitting 체크)
- [ ] Text accuracy 향상
- [ ] 주기적 샘플 생성으로 품질 확인

### Expected Loss Curves (J-Moshi Reference)
# 시작: Text ~8.0, PAD ~6.0, Semantic ~4.0, Acoustic ~2.0
# 종료: Text ~2.5, PAD ~1.5, Semantic ~1.0, Acoustic ~0.5
```

### 7.2 Fine-tuning 모니터링

```markdown
## Fine-tuning Health Checks

### Step 0-50 (Initial)
- [ ] Pre-trained checkpoint 정상 로드 확인
- [ ] Loss 시작값이 pre-training 종료값보다 약간 높음 (정상)
- [ ] 새 데이터에 빠르게 적응하는지 확인

### Step 50-500 (Main Training)
- [ ] Loss 급격히 감소 후 안정화
- [ ] Eval loss 안정적 (overfitting 주의)
- [ ] 생성 샘플 품질 향상

### Quality Metrics
- [ ] Perplexity (PPL) 감소
- [ ] Turn-taking 자연스러움
- [ ] Speech overlap/backchannel 적절함
```

---

## 8. 참조 논문 및 코드

### 8.1 핵심 참조 문헌

| 문헌 | 핵심 기여 | 링크 |
|------|----------|------|
| **Moshi Paper** | 원본 아키텍처, 학습 방법론 | [arXiv:2410.00037](https://arxiv.org/abs/2410.00037) |
| **J-Moshi Paper** | 일본어 적응, Two-rate learning | [arXiv:2506.02979](https://arxiv.org/abs/2506.02979) |
| **Llama 2** | AdamW β 설정 (0.9, 0.95) | [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) |
| **DeepSpeed ZeRO** | 대규모 분산 학습 | [Paper](https://arxiv.org/abs/1910.02054) |

### 8.2 코드 참조

```
moshi-finetune (Official)
├── example/moshi_7B.yaml      # 공식 설정 템플릿
├── finetune/args.py           # 하이퍼파라미터 정의
└── train.py                   # 학습 루프

moshi-korean-finetune (This Project)
├── finetune/scheduler.py      # Two-rate optimizer 구현
├── finetune/args.py           # 확장된 설정 옵션
└── example/korean_v4_*.yaml   # 한국어 설정 파일들

j-moshi (Reference)
└── configs/                   # DeepSpeed 설정
```

### 8.3 핵심 하이퍼파라미터 요약

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                K-MOSHI HYPERPARAMETER QUICK REFERENCE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OPTIMIZER (AdamW - Llama 2 Style)                                          │
│  ─────────────────────────────────                                          │
│  β1 = 0.9, β2 = 0.95, ε = 1e-5, weight_decay = 0.1                         │
│  foreach = False  # CRITICAL for FSDP!                                      │
│                                                                             │
│  LEARNING RATES                                                             │
│  ─────────────────                                                          │
│  Pre-training:   TempFormer 3e-5,  DepFormer 6e-5  (2x)                     │
│  Fine-tuning:    TempFormer 2e-6,  DepFormer 4e-6  (2x)                     │
│                                                                             │
│  SCHEDULER                                                                  │
│  ─────────                                                                  │
│  Type: Cosine warmup (or linear warmup → constant)                          │
│  Warmup: 500 steps (pre-train), 100 steps (fine-tune)                       │
│  Min LR: 1e-7 (pre-train), 1e-8 (fine-tune)                                │
│                                                                             │
│  BATCH SIZE                                                                 │
│  ──────────                                                                 │
│  Pre-training: 512 effective (2 per GPU × 8 accum × 32 GPUs)               │
│  Fine-tuning:  256 effective (2 per GPU × 4 accum × 32 GPUs)               │
│                                                                             │
│  SEQUENCE LENGTH                                                            │
│  ───────────────                                                            │
│  Duration: 160 seconds (2,048 tokens at 12.5Hz)                             │
│  = 2.7 minutes (J-Moshi와 동일)                                             │
│                                                                             │
│  LOSS WEIGHTS                                                               │
│  ────────────                                                               │
│  PAD token: 0.5 (50% weight reduction)                                      │
│  Semantic:Acoustic = 100:1 (first_codebook_weight=100)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: FSDP vs ZeRO-3 비교

| Feature | FSDP (PyTorch) | ZeRO-3 (DeepSpeed) |
|---------|----------------|---------------------|
| **구현** | PyTorch native | DeepSpeed library |
| **설정** | 간단 | 복잡한 config 필요 |
| **호환성** | PyTorch 생태계 | DeepSpeed 의존 |
| **성능** | 비슷 | 약간 최적화 |
| **디버깅** | 쉬움 | 어려움 |
| **권장** | ✅ K-Moshi | J-Moshi |

## Appendix B: 메모리 계산

```python
# A100 80GB 메모리 예산

# 모델 파라미터 (7B, bf16)
model_size = 7e9 * 2 / 1e9  # ~14GB

# Optimizer states (Adam: 2x model for m, v)
optimizer_size = 14 * 2  # ~28GB

# Activations (with gradient checkpointing)
# ~5GB per layer × 32 layers / checkpointing factor(~4) ≈ 40GB for batch=2
activations = 40  # GB for batch_size=2

# 여유 공간
buffer = 80 - 14 - 28 - 40  # ≈ -2GB (꽉 찰 수 있음!)

# → batch_size=2가 안전한 최대값
# → batch_size=1이면 더 안전 (gradient accumulation으로 보상)
```

---

*Document created: 2026-01-13*
*Based on: J-Moshi Paper (arXiv:2506.02979), Moshi Official Code*
*For: K-Moshi Full Finetuning Project*
