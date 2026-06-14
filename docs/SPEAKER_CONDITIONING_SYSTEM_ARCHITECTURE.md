# K-Moshi Speaker Conditioning System Architecture

## 목차
1. [시스템 개요](#1-시스템-개요)
2. [Training Pipeline](#2-training-pipeline)
3. [Validation/Evaluation Pipeline](#3-validationevaluation-pipeline)
4. [Sample Saver Pipeline](#4-sample-saver-pipeline)
5. [Rust Backend Serving](#5-rust-backend-serving)
6. [데이터 흐름 다이어그램](#6-데이터-흐름-다이어그램)
7. [모듈 상호작용](#7-모듈-상호작용)

---

## 1. 시스템 개요

K-Moshi Speaker Conditioning System은 zero-shot speaker adaptation을 위한 종합 시스템입니다.

### 1.1 핵심 컴포넌트

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     K-MOSHI SPEAKER CONDITIONING SYSTEM                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐  ┌───────────────────┐  ┌──────────────────────────┐ │
│  │ Speaker Encoder  │  │  Audio Prompt     │  │  Speaker Conditioner     │ │
│  │ (W2v-BERT 2.0)   │  │  Module           │  │  (Projection Layer)      │ │
│  │                  │  │                   │  │                          │ │
│  │ Input: Audio     │  │ Input: Codes      │  │ Input: Embedding [256]   │ │
│  │ Output: [256]    │  │ Output: Prompted  │  │ Output: [4096]           │ │
│  └────────┬─────────┘  │         Codes     │  │ → Temporal TF hidden     │ │
│           │            └─────────┬─────────┘  └────────────┬─────────────┘ │
│           │                      │                          │              │
│           └──────────────────────┴──────────────────────────┘              │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        LMModel (Moshi 7B)                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Temporal Transformer: hidden_states + speaker_condition        │ │  │
│  │  │ DepFormer: 8 depth codebook prediction (dep_q=8)               │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 두 가지 컨디셔닝 방식

| 방식 | 설명 | 적용 위치 |
|------|------|-----------|
| **Global Condition** | Speaker embedding → Temporal TF에 sum_condition으로 추가 | 전체 시퀀스 |
| **Local Condition** | Reference audio/text를 시퀀스 앞에 prepend (PersonaPlex) | Prompt 구간만 |

**권장 설정**: `method: "both"` - 두 방식 모두 사용

---

## 2. Training Pipeline

### 2.1 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐                                                                │
│  │ JSONL Data  │ train_data: './data/korean_v4_train.jsonl'                     │
│  │ (paths)     │                                                                │
│  └──────┬──────┘                                                                │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      InterleavedTokenizer                                │   │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────┐ │   │
│  │  │ Stereo WAV  │────▶│ Mimi Encode │────▶│ Text-Audio Interleave      │ │   │
│  │  │ (L=Moshi,   │     │ [8, T]      │     │ codes: [B, 17, T]          │ │   │
│  │  │  R=User)    │     └─────────────┘     │ ├─ [0]: text tokens        │ │   │
│  │  └─────────────┘                         │ ├─ [1-8]: moshi audio      │ │   │
│  │                                          │ └─ [9-16]: user audio      │ │   │
│  │  ┌─────────────┐                         └─────────────────────────────┘ │   │
│  │  │ JSON 전사   │ alignments: [("안녕", (0.0, 0.5), "SPEAKER_MAIN"), ...]  │   │
│  │  └─────────────┘                                                         │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         │ batch.codes: [B, 17, T]                                               │
│         │ batch.audio_paths: [path1, path2, ...]                                │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      AudioPromptModule                                   │   │
│  │                                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │   │
│  │  │  AudioPromptSampler.sample_single_word_count()                    │   │   │
│  │  │                                                                   │   │   │
│  │  │  Input: codes [17, T], exclude_start/end                          │   │   │
│  │  │                                                                   │   │   │
│  │  │  Process:                                                         │   │   │
│  │  │    1. Extract text_tokens = codes[0, :]                           │   │   │
│  │  │    2. Count valid tokens (exclude PAD=0, EOS=3, 32000)            │   │   │
│  │  │    3. Find segment with min_words ~ max_words tokens              │   │   │
│  │  │    4. Random start position (avoid overlap with training segment) │   │   │
│  │  │                                                                   │   │   │
│  │  │  Output: AudioPromptSample                                        │   │   │
│  │  │    - audio_codes: [8, T_prompt] (moshi)                           │   │   │
│  │  │    - text_tokens: [T_prompt]                                      │   │   │
│  │  │    - user_audio_codes: [8, T_prompt] (user, if 17-codebook)       │   │   │
│  │  │    - start_idx, end_idx                                           │   │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │   │
│  │  │  Prepend Prompt to Codes                                          │   │   │
│  │  │                                                                   │   │   │
│  │  │  prompted_codes = [prompt_codes | original_codes]                 │   │   │
│  │  │  prompt_mask = [True, True, ... | False, False, ...]              │   │   │
│  │  │              └── prompt 구간 ─┘ └─ training 구간 ─┘               │   │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         │ prompted_codes: [B, 17, T+P]  (P = prompt length)                     │
│         │ prompt_mask: [B, T+P]                                                 │
│         │ prompt_samples: List[AudioPromptSample]                               │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      Speaker Encoder                                     │   │
│  │                                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │   │
│  │  │  Reference Audio Extraction                                       │   │   │
│  │  │                                                                   │   │   │
│  │  │  1. Mimi.decode(prompt_samples[i].audio_codes)                    │   │   │
│  │  │     → reference_audio: [B, T_audio] @ 24kHz                       │   │   │
│  │  │                                                                   │   │   │
│  │  │  2. Resample 24kHz → 16kHz (W2v-BERT 2.0 input rate)              │   │   │
│  │  │     → reference_audio_16k: [B, T_audio_16k]                       │   │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │   │
│  │  │  W2v-BERT 2.0 Speaker Verification                                │   │   │
│  │  │                                                                   │   │   │
│  │  │  Model: w2v-bert-2.0_SV (0.14% EER SOTA on VoxCeleb1-O)           │   │   │
│  │  │  Input: reference_audio_16k [B, T]                                │   │   │
│  │  │  Process:                                                         │   │   │
│  │  │    - 25 Transformer layers with MFA                               │   │   │
│  │  │    - Attentive Statistics Pooling (ASP)                           │   │   │
│  │  │  Output: speaker_embedding [B, 256]                               │   │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         │ speaker_embedding: [B, 256]                                           │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      Speaker Conditioner                                 │   │
│  │                                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │   │
│  │  │  Projection: [256] → [4096]                                       │   │   │
│  │  │                                                                   │   │   │
│  │  │  speaker_condition = Linear(speaker_embedding)  # [B, 4096]       │   │   │
│  │  │  speaker_condition = LayerNorm(speaker_condition)                 │   │   │
│  │  │  speaker_condition = speaker_condition * scale  # initial 0.1     │   │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         │ speaker_condition: [B, 4096]                                          │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      LMModel Forward                                     │   │
│  │                                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │   │
│  │  │  Input Embedding                                                  │   │   │
│  │  │                                                                   │   │   │
│  │  │  text_embed = embed_text(prompted_codes[:, 0, :])                 │   │   │
│  │  │  audio_embed = sum(embed_audio[k](prompted_codes[:, 1+k, :])      │   │   │
│  │  │                    for k in range(dep_q))                         │   │   │
│  │  │  input_embed = text_embed + audio_embed  # [B, T+P, 4096]         │   │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │   │
│  │  │  Temporal Transformer (32 layers)                                 │   │   │
│  │  │                                                                   │   │   │
│  │  │  for layer in transformer_layers:                                 │   │   │
│  │  │      hidden = layer(hidden)                                       │   │   │
│  │  │      if speaker_condition is not None:                            │   │   │
│  │  │          hidden = hidden + speaker_condition.unsqueeze(1)         │   │   │
│  │  │          # ↑ Global conditioning via sum_condition                │   │   │
│  │  │                                                                   │   │   │
│  │  │  Output: hidden_states [B, T+P, 4096]                             │   │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │   │
│  │  │  Output Heads                                                     │   │   │
│  │  │                                                                   │   │   │
│  │  │  text_logits = text_head(hidden)     # [B, T+P, vocab_size]       │   │   │
│  │  │  audio_logits = depformer(hidden)    # [B, dep_q, T+P, 2048]      │   │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      Loss Computation                                    │   │
│  │                                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │   │
│  │  │  CRITICAL: prompt_mask 적용                                       │   │   │
│  │  │                                                                   │   │   │
│  │  │  # Prompt 구간은 loss에서 제외                                    │   │   │
│  │  │  text_loss = CE(text_logits, codes[:, 0])                         │   │   │
│  │  │  text_loss = text_loss.masked_select(~prompt_mask).mean()         │   │   │
│  │  │                                                                   │   │   │
│  │  │  audio_loss = CE(audio_logits, codes[:, 1:9])                     │   │   │
│  │  │  audio_loss[0] *= 100.0  # first_codebook_weight_multiplier       │   │   │
│  │  │  audio_loss = audio_loss.masked_select(~prompt_mask).mean()       │   │   │
│  │  │                                                                   │   │   │
│  │  │  total_loss = text_loss + audio_loss                              │   │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      Backward & Optimizer Step                           │   │
│  │                                                                          │   │
│  │  total_loss.backward()                                                   │   │
│  │  optimizer.step()  # AdamW                                               │   │
│  │  scheduler.step()  # CosineWarmup                                        │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 주요 모듈 입출력

| 모듈 | 입력 | 출력 |
|------|------|------|
| **InterleavedTokenizer** | WAV (24kHz, stereo), JSON alignments | codes [B, 17, T] |
| **AudioPromptModule** | codes [B, 17, T] | prompted_codes [B, 17, T+P], prompt_mask [B, T+P], prompt_samples |
| **Speaker Encoder** | reference_audio [B, T] @ 16kHz | speaker_embedding [B, 256] |
| **Speaker Conditioner** | speaker_embedding [B, 256] | speaker_condition [B, 4096] |
| **LMModel** | prompted_codes, speaker_condition | text_logits, audio_logits |

### 2.3 Batch-Level Metadata 저장 (Fix 적용)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BATCH-LEVEL METADATA STORAGE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  prompt_samples: List[AudioPromptSample]                                    │
│      │                                                                      │
│      ├── [0] → reference_texts[0], reference_start_secs[0], ...             │
│      ├── [1] → reference_texts[1], reference_start_secs[1], ...             │
│      ├── [2] → reference_texts[2], reference_start_secs[2], ...             │
│      └── [B-1] → reference_texts[B-1], reference_start_secs[B-1], ...       │
│                                                                             │
│  EvalSpeakerConditioningInfo:                                               │
│      reference_texts: List[str]         # 각 배치 항목의 reference text     │
│      reference_start_secs: List[float]  # 각 배치 항목의 시작 시간          │
│      reference_end_secs: List[float]    # 각 배치 항목의 끝 시간            │
│      source_files: List[str]            # 각 배치 항목의 원본 파일 경로     │
│                                                                             │
│  sample_saver.save_sample(batch_idx=i):                                     │
│      → speaker_metadata.json 에 정확한 per-sample 정보 저장                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Validation/Evaluation Pipeline

### 3.1 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         VALIDATION PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐                                                                │
│  │ JSONL Data  │ eval_data: './data/korean_v4_valid.jsonl'                      │
│  └──────┬──────┘                                                                │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      DETERMINISTIC SAMPLING                              │   │
│  │                                                                          │   │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │   │
│  │  │  Key Differences from Training:                                   │   │   │
│  │  │                                                                   │   │   │
│  │  │  1. deterministic=True                                            │   │   │
│  │  │     - NO torch.randint() calls                                    │   │   │
│  │  │     - Same input → Same reference selection                       │   │   │
│  │  │                                                                   │   │   │
│  │  │  2. sample_strategy="start" (recommended)                         │   │   │
│  │  │     - Always select from beginning of audio                       │   │   │
│  │  │     - Reproducible results across runs                            │   │   │
│  │  │                                                                   │   │   │
│  │  │  3. fixed_word_count=25 (or fixed_duration_sec=10.0)              │   │   │
│  │  │     - Fixed reference length for consistency                      │   │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      AudioPromptModule (deterministic=True)              │   │
│  │                                                                          │   │
│  │  prompted_codes, prompt_mask, prompt_samples = audio_prompt_module(      │   │
│  │      codes,                                                              │   │
│  │      exclude_start=None,  # No exclusion for eval                        │   │
│  │      exclude_end=None,                                                   │   │
│  │      deterministic=True,  # ← CRITICAL for reproducibility               │   │
│  │  )                                                                       │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      BATCH-LEVEL INFO COLLECTION                         │   │
│  │                                                                          │   │
│  │  # prompt_samples 전체를 순회하여 배치 레벨 정보 수집                    │   │
│  │  reference_start_secs = []                                               │   │
│  │  reference_end_secs = []                                                 │   │
│  │  reference_texts = []                                                    │   │
│  │                                                                          │   │
│  │  for ps in prompt_samples:                                               │   │
│  │      reference_start_secs.append(ps.start_idx / frame_rate)              │   │
│  │      reference_end_secs.append(ps.end_idx / frame_rate)                  │   │
│  │      reference_texts.append(decode_text(ps.text_tokens))                 │   │
│  │                                                                          │   │
│  │  # EvalSpeakerConditioningInfo에 저장                                    │   │
│  │  info.reference_texts = reference_texts                                  │   │
│  │  info.reference_start_secs = reference_start_secs                        │   │
│  │  info.reference_end_secs = reference_end_secs                            │   │
│  │  info.source_files = batch.audio_paths                                   │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      Model Forward (Same as Training)                    │   │
│  │                                                                          │   │
│  │  output = model(                                                         │   │
│  │      codes=prompted_codes,                                               │   │
│  │      condition_tensors=condition_tensors,                                │   │
│  │      speaker_embedding=speaker_embedding,                                │   │
│  │  )                                                                       │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      Loss Computation & Metrics                          │   │
│  │                                                                          │   │
│  │  eval_loss = text_loss + audio_loss                                      │   │
│  │  eval_perplexity = 2^eval_loss                                           │   │
│  │                                                                          │   │
│  │  state.this_eval_loss = eval_loss.item()                                 │   │
│  │  state.this_eval_perplexity = eval_perplexity.item()                     │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      Return EvalReturnData                               │   │
│  │                                                                          │   │
│  │  return EvalReturnData(                                                  │   │
│  │      original_codes=codes,                # GT audio 생성용              │   │
│  │      prompted_codes=prompted_codes,       # 메트릭 계산용                │   │
│  │      output=output,                                                      │   │
│  │      speaker_conditioning_info=info,      # Sample saver용 메타데이터    │   │
│  │      audio_paths=batch.audio_paths,                                      │   │
│  │  )                                                                       │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 FSDP 동기화 프로토콜

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       FSDP SYNCHRONIZED EVALUATION                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  문제: 각 rank가 서로 다른 양의 eval 데이터를 가질 수 있음                      │
│  해결: "ALL ranks must have data" 전략                                          │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  while True:                                                           │    │
│  │      # 각 rank에서 데이터 유무 확인                                    │    │
│  │      try:                                                              │    │
│  │          batch = next(eval_iterator)                                   │    │
│  │          has_data = torch.tensor([1])                                  │    │
│  │      except StopIteration:                                             │    │
│  │          has_data = torch.tensor([0])                                  │    │
│  │                                                                        │    │
│  │      # 모든 rank에서 has_data 수집                                     │    │
│  │      all_has_data = [...]                                              │    │
│  │      dist.all_gather(all_has_data, has_data)                           │    │
│  │                                                                        │    │
│  │      # 어느 한 rank라도 데이터가 없으면 전체 종료                      │    │
│  │      if not all(t.item() == 1 for t in all_has_data):                  │    │
│  │          break  # 모든 rank가 함께 종료                                │    │
│  │                                                                        │    │
│  │      # 모든 rank가 데이터 있음 → 함께 model() 호출                     │    │
│  │      output = model(codes=prompted_codes)  # FSDP: 모든 rank 참여      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Sample Saver Pipeline

### 4.1 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SAMPLE SAVER PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Input: EvalReturnData from evaluate()                                          │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      sample_saver.save_samples()                         │   │
│  │                                                                          │   │
│  │  for batch_idx in range(batch_size):                                     │   │
│  │      ┌───────────────────────────────────────────────────────────────┐   │   │
│  │      │  save_sample(batch_idx)                                       │   │   │
│  │      │                                                               │   │   │
│  │      │  1. Get per-sample metadata (CRITICAL FIX)                    │   │   │
│  │      │     ┌─────────────────────────────────────────────────────┐   │   │   │
│  │      │     │  # 배치 레벨 리스트에서 batch_idx로 추출             │   │   │   │
│  │      │     │  ref_start = info.reference_start_secs[batch_idx]   │   │   │   │
│  │      │     │  ref_end = info.reference_end_secs[batch_idx]       │   │   │   │
│  │      │     │  ref_text = info.reference_texts[batch_idx]         │   │   │   │
│  │      │     │  source_file = info.source_files[batch_idx]         │   │   │   │
│  │      │     └─────────────────────────────────────────────────────┘   │   │   │
│  │      │                                                               │   │   │
│  │      │  2. Save Reference Audio (reference.wav)                      │   │   │
│  │      │     ┌─────────────────────────────────────────────────────┐   │   │   │
│  │      │     │  reference_audio = info.reference_audio[batch_idx]  │   │   │   │
│  │      │     │  sf.write(                                          │   │   │   │
│  │      │     │      sample_dir / "reference.wav",                  │   │   │   │
│  │      │     │      reference_audio,                               │   │   │   │
│  │      │     │      samplerate=24000,  # Mimi output               │   │   │   │
│  │      │     │  )                                                  │   │   │   │
│  │      │     └─────────────────────────────────────────────────────┘   │   │   │
│  │      │                                                               │   │   │
│  │      │  3. Save Speaker Metadata (speaker_metadata.json)             │   │   │
│  │      │     ┌─────────────────────────────────────────────────────┐   │   │   │
│  │      │     │  {                                                  │   │   │   │
│  │      │     │    "enabled": true,                                 │   │   │   │
│  │      │     │    "method": "both",                                │   │   │   │
│  │      │     │    "reference_start_sec": 5.32,    # per-sample     │   │   │   │
│  │      │     │    "reference_end_sec": 15.32,     # per-sample     │   │   │   │
│  │      │     │    "reference_text": "안녕하세요", # per-sample     │   │   │   │
│  │      │     │    "source_file": "dialog_123.wav",# per-sample     │   │   │   │
│  │      │     │    "batch_idx": 0                  # debugging      │   │   │   │
│  │      │     │  }                                                  │   │   │   │
│  │      │     └─────────────────────────────────────────────────────┘   │   │   │
│  │      │                                                               │   │   │
│  │      │  4. Save GT Audio (gt_audio.wav)                              │   │   │
│  │      │     ┌─────────────────────────────────────────────────────┐   │   │   │
│  │      │     │  codes = original_codes[batch_idx]  # NOT prompted  │   │   │   │
│  │      │     │  gt_audio = mimi.decode(codes[1:9])                 │   │   │   │
│  │      │     │  sf.write(sample_dir / "gt_audio.wav", gt_audio)    │   │   │   │
│  │      │     └─────────────────────────────────────────────────────┘   │   │   │
│  │      │                                                               │   │   │
│  │      │  5. Save Predicted Audio (pred_audio.wav)                     │   │   │
│  │      │     ┌─────────────────────────────────────────────────────┐   │   │   │
│  │      │     │  pred_codes = output.logits.argmax(-1)[batch_idx]   │   │   │   │
│  │      │     │  pred_audio = mimi.decode(pred_codes)               │   │   │   │
│  │      │     │  sf.write(sample_dir / "pred_audio.wav", pred_audio)│   │   │   │
│  │      │     └─────────────────────────────────────────────────────┘   │   │   │
│  │      │                                                               │   │   │
│  │      │  6. Save Text Comparison (text_comparison.json)               │   │   │
│  │      │     ┌─────────────────────────────────────────────────────┐   │   │   │
│  │      │     │  {                                                  │   │   │   │
│  │      │     │    "gt_text": "오늘 날씨가 좋네요",                 │   │   │   │
│  │      │     │    "pred_text": "오늘 날씨가 좋네요",               │   │   │   │
│  │      │     │    "reference_text": "안녕하세요"                   │   │   │   │
│  │      │     │  }                                                  │   │   │   │
│  │      │     └─────────────────────────────────────────────────────┘   │   │   │
│  │      └───────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  Output Directory Structure:                                                    │
│                                                                                 │
│  runs/korean_moshi_stage2_speaker/samples/                                      │
│  ├── train/                                                                     │
│  │   ├── step_00100/                                                            │
│  │   │   ├── sample_00/                                                         │
│  │   │   │   ├── reference.wav                                                  │
│  │   │   │   ├── gt_audio.wav                                                   │
│  │   │   │   ├── pred_audio.wav                                                 │
│  │   │   │   ├── speaker_metadata.json   ← 각 샘플별 고유 metadata              │
│  │   │   │   └── text_comparison.json                                           │
│  │   │   ├── sample_01/                                                         │
│  │   │   │   └── ...  (다른 reference, 다른 source_file)                        │
│  │   │   └── sample_02/                                                         │
│  │   └── step_00200/                                                            │
│  └── valid/                                                                     │
│      └── step_00100/                                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Per-Sample Metadata 저장 로직 (Fix 적용 후)

```python
# sample_saver.py (수정 후)

def save_sample(self, batch_idx: int, ...):
    # CRITICAL: 배치 레벨 리스트에서 batch_idx로 정확한 값 추출

    # Timing info
    ref_start_secs = getattr(speaker_conditioning_info, 'reference_start_secs', None)
    if ref_start_secs is not None and batch_idx < len(ref_start_secs):
        sample_start_sec = ref_start_secs[batch_idx]
        sample_end_sec = ref_end_secs[batch_idx]
    else:
        # Legacy fallback
        sample_start_sec = speaker_conditioning_info.reference_start_sec
        sample_end_sec = speaker_conditioning_info.reference_end_sec

    # Reference text
    ref_texts = getattr(speaker_conditioning_info, 'reference_texts', None)
    if ref_texts is not None and batch_idx < len(ref_texts):
        sample_ref_text = ref_texts[batch_idx]
    else:
        sample_ref_text = speaker_conditioning_info.reference_text

    # Source file
    source_files = getattr(speaker_conditioning_info, 'source_files', None)
    if source_files is not None and batch_idx < len(source_files):
        sample_source_file = source_files[batch_idx]
    else:
        sample_source_file = speaker_conditioning_info.source_file

    # Save metadata with per-sample values
    speaker_metadata = {
        "reference_start_sec": sample_start_sec,
        "reference_end_sec": sample_end_sec,
        "reference_text": sample_ref_text,
        "source_file": sample_source_file,
        "batch_idx": batch_idx,  # For debugging
    }
```

---

## 5. Rust Backend Serving

### 5.1 서빙 시나리오 개요

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        RUST BACKEND SERVING SCENARIO                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      CLIENT (Web Browser)                               │    │
│  │                                                                         │    │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │    │
│  │  │ User Microphone  │  │ Speaker Reference│  │ Real-time Display   │   │    │
│  │  │ (Audio Input)    │  │ (3-10s Audio)    │  │ (Text + Audio Out)  │   │    │
│  │  └────────┬─────────┘  └────────┬─────────┘  └──────────▲───────────┘   │    │
│  │           │                     │                       │               │    │
│  │           │    WebSocket (Opus encoded)                 │               │    │
│  │           └─────────────────────┴───────────────────────┘               │    │
│  └──────────────────────────────────────│───────────────────────────────────┘    │
│                                         │                                        │
│                                         ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      RUST MOSHI BACKEND                                 │    │
│  │                                                                         │    │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │    │
│  │  │  1. WebSocket Handler                                             │  │    │
│  │  │     - Receive: Opus encoded user audio (streaming)                │  │    │
│  │  │     - Receive: Speaker reference audio (initial)                  │  │    │
│  │  │     - Send: Moshi response audio (streaming)                      │  │    │
│  │  │     - Send: Text tokens (streaming)                               │  │    │
│  │  └───────────────────────────────────────────────────────────────────┘  │    │
│  │                            │                                            │    │
│  │                            ▼                                            │    │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │    │
│  │  │  2. Audio Preprocessing                                           │  │    │
│  │  │     - Opus decode → PCM 24kHz                                     │  │    │
│  │  │     - Resample 24kHz → 16kHz (for speaker encoder)                │  │    │
│  │  └───────────────────────────────────────────────────────────────────┘  │    │
│  │                            │                                            │    │
│  │                            ▼                                            │    │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │    │
│  │  │  3. Speaker Encoder (W2v-BERT 2.0)                                │  │    │
│  │  │     - Input: reference_audio [1, T] @ 16kHz                       │  │    │
│  │  │     - Output: speaker_embedding [1, 256]                          │  │    │
│  │  │     - Cached: 세션 시작 시 한 번만 계산                           │  │    │
│  │  └───────────────────────────────────────────────────────────────────┘  │    │
│  │                            │                                            │    │
│  │                            ▼                                            │    │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │    │
│  │  │  4. Mimi Encoder (Real-time)                                      │  │    │
│  │  │     - Input: user_audio [1, chunk_samples] @ 24kHz                │  │    │
│  │  │     - Output: user_codes [1, 8, chunk_frames]                     │  │    │
│  │  │     - Streaming: 80ms chunks (1 frame = 1920 samples)             │  │    │
│  │  └───────────────────────────────────────────────────────────────────┘  │    │
│  │                            │                                            │    │
│  │                            ▼                                            │    │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │    │
│  │  │  5. LMModel Inference (Streaming)                                 │  │    │
│  │  │                                                                   │  │    │
│  │  │  ┌─────────────────────────────────────────────────────────────┐  │  │    │
│  │  │  │  Input Construction                                         │  │  │    │
│  │  │  │                                                             │  │  │    │
│  │  │  │  codes = [                                                  │  │  │    │
│  │  │  │      text_tokens,      # [1, T] - autoregressive 생성       │  │  │    │
│  │  │  │      moshi_audio,      # [8, T] - autoregressive 생성       │  │  │    │
│  │  │  │      user_audio,       # [8, T] - 실시간 입력               │  │  │    │
│  │  │  │  ]                                                          │  │  │    │
│  │  │  │                                                             │  │  │    │
│  │  │  │  speaker_condition = project(speaker_embedding)  # [1, 4096]│  │  │    │
│  │  │  └─────────────────────────────────────────────────────────────┘  │  │    │
│  │  │                                                                   │  │    │
│  │  │  ┌─────────────────────────────────────────────────────────────┐  │  │    │
│  │  │  │  Temporal Transformer (Streaming Mode)                      │  │  │    │
│  │  │  │                                                             │  │  │    │
│  │  │  │  - KV Cache 사용 (past_key_values)                          │  │  │    │
│  │  │  │  - 한 프레임씩 처리 (80ms latency)                          │  │  │    │
│  │  │  │  - speaker_condition 추가 (sum_condition)                   │  │  │    │
│  │  │  └─────────────────────────────────────────────────────────────┘  │  │    │
│  │  │                                                                   │  │    │
│  │  │  ┌─────────────────────────────────────────────────────────────┐  │  │    │
│  │  │  │  DepFormer (Streaming Mode)                                 │  │  │    │
│  │  │  │                                                             │  │  │    │
│  │  │  │  - 8 depth steps per frame                                  │  │  │    │
│  │  │  │  - Delay pattern: d=0 → d=1 → ... → d=7                     │  │  │    │
│  │  │  │  - Output: moshi_codes [1, 8, 1]                            │  │  │    │
│  │  │  └─────────────────────────────────────────────────────────────┘  │  │    │
│  │  │                                                                   │  │    │
│  │  │  Output: text_token [1], moshi_codes [8]                          │  │    │
│  │  └───────────────────────────────────────────────────────────────────┘  │    │
│  │                            │                                            │    │
│  │                            ▼                                            │    │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │    │
│  │  │  6. Mimi Decoder (Real-time)                                      │  │    │
│  │  │     - Input: moshi_codes [1, 8, 1]                                │  │    │
│  │  │     - Output: audio_samples [1, 1920] @ 24kHz                     │  │    │
│  │  │     - Streaming: 80ms audio per frame                             │  │    │
│  │  └───────────────────────────────────────────────────────────────────┘  │    │
│  │                            │                                            │    │
│  │                            ▼                                            │    │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │    │
│  │  │  7. Output Encoding & Streaming                                   │  │    │
│  │  │     - Opus encode audio                                           │  │    │
│  │  │     - Package text tokens                                         │  │    │
│  │  │     - WebSocket send to client                                    │  │    │
│  │  └───────────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Speaker Conditioning 통합 포인트

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  SPEAKER CONDITIONING INTEGRATION POINTS                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Session Initialization (Once per session)                              │    │
│  │                                                                         │    │
│  │  1. Client uploads reference audio (3-10 seconds)                       │    │
│  │     ↓                                                                   │    │
│  │  2. Rust backend receives and preprocesses                              │    │
│  │     ↓                                                                   │    │
│  │  3. Speaker Encoder extracts embedding                                  │    │
│  │     speaker_embedding = w2v_bert2(reference_audio)  # [1, 256]          │    │
│  │     ↓                                                                   │    │
│  │  4. Speaker Conditioner projects embedding                              │    │
│  │     speaker_condition = project(speaker_embedding)  # [1, 4096]         │    │
│  │     ↓                                                                   │    │
│  │  5. Cache for session lifetime                                          │    │
│  │     session.speaker_condition = speaker_condition                       │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  Streaming Inference (Per frame, 80ms)                                  │    │
│  │                                                                         │    │
│  │  for each user_audio_chunk:                                             │    │
│  │      # 1. Encode user audio                                             │    │
│  │      user_codes = mimi.encode(user_audio_chunk)                         │    │
│  │                                                                         │    │
│  │      # 2. Prepare input codes                                           │    │
│  │      input_codes = cat([text_tokens, moshi_audio, user_audio])          │    │
│  │                                                                         │    │
│  │      # 3. LM forward with speaker conditioning                          │    │
│  │      output = lm_model(                                                 │    │
│  │          codes=input_codes,                                             │    │
│  │          speaker_embedding=session.speaker_condition,  # ← Cached       │    │
│  │          past_key_values=session.kv_cache,                              │    │
│  │      )                                                                  │    │
│  │                                                                         │    │
│  │      # 4. Update KV cache                                               │    │
│  │      session.kv_cache = output.past_key_values                          │    │
│  │                                                                         │    │
│  │      # 5. Decode audio                                                  │    │
│  │      moshi_audio = mimi.decode(output.moshi_codes)                      │    │
│  │                                                                         │    │
│  │      # 6. Send to client                                                │    │
│  │      websocket.send(moshi_audio, text_token)                            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Rust 코드 수정 포인트 (TODO)

```
moshi/rust/
├── moshi-backend/
│   ├── src/
│   │   ├── main.rs           # WebSocket 핸들러
│   │   ├── session.rs        # 세션 관리 + speaker_condition 캐시
│   │   ├── speaker.rs        # [NEW] Speaker encoder 통합
│   │   └── inference.rs      # LM 추론 + speaker_embedding 전달
│   └── Cargo.toml
│
└── moshi-core/
    ├── src/
    │   ├── lm_model.rs       # speaker_embedding 파라미터 추가
    │   ├── transformer.rs    # sum_condition 적용
    │   └── lib.rs
    └── Cargo.toml
```

---

## 6. 데이터 흐름 다이어그램

### 6.1 End-to-End 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           END-TO-END DATA FLOW                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              TRAINING DATA FLOW                                   │  │
│  │                                                                                   │  │
│  │  WAV (stereo) + JSON     InterleavedTokenizer      codes [B, 17, T]               │  │
│  │       │                        │                        │                         │  │
│  │       ├───────────────────────►│                        │                         │  │
│  │       │                        ▼                        │                         │  │
│  │       │                 ┌─────────────┐                 │                         │  │
│  │       │                 │ Mimi Encode │                 │                         │  │
│  │       │                 │ Text Align  │                 │                         │  │
│  │       │                 └──────┬──────┘                 │                         │  │
│  │       │                        │                        │                         │  │
│  │       │                        ▼                        │                         │  │
│  │       │                 ┌─────────────────────┐         │                         │  │
│  │       │                 │ AudioPromptModule   │────────►│                         │  │
│  │       │                 │ (Word-count based)  │         │                         │  │
│  │       │                 └──────┬──────────────┘         │                         │  │
│  │       │                        │                        │                         │  │
│  │       │               prompt_samples                    │                         │  │
│  │       │                        │                        │                         │  │
│  │       │                        ▼                        │                         │  │
│  │       │                 ┌─────────────────────┐         │                         │  │
│  │       │                 │ Mimi Decode         │         │                         │  │
│  │       │                 │ (reference audio)   │         │                         │  │
│  │       │                 └──────┬──────────────┘         │                         │  │
│  │       │                        │                        │                         │  │
│  │       │            reference_audio [B, T] @ 24kHz       │                         │  │
│  │       │                        │                        │                         │  │
│  │       │                        ▼                        │                         │  │
│  │       │                 ┌─────────────────────┐         │                         │  │
│  │       │                 │ Resample 24k → 16k  │         │                         │  │
│  │       │                 └──────┬──────────────┘         │                         │  │
│  │       │                        │                        │                         │  │
│  │       │            reference_audio [B, T'] @ 16kHz      │                         │  │
│  │       │                        │                        │                         │  │
│  │       │                        ▼                        │                         │  │
│  │       │                 ┌─────────────────────┐         │                         │  │
│  │       │                 │ W2v-BERT 2.0 SV     │         │                         │  │
│  │       │                 │ (Speaker Encoder)   │         │                         │  │
│  │       │                 └──────┬──────────────┘         │                         │  │
│  │       │                        │                        │                         │  │
│  │       │            speaker_embedding [B, 256]           │                         │  │
│  │       │                        │                        │                         │  │
│  │       │                        ▼                        │                         │  │
│  │       │                 ┌─────────────────────┐         │                         │  │
│  │       │                 │ Speaker Conditioner │         │                         │  │
│  │       │                 │ Linear + LayerNorm  │         │                         │  │
│  │       │                 └──────┬──────────────┘         │                         │  │
│  │       │                        │                        │                         │  │
│  │       │            speaker_condition [B, 4096]          │                         │  │
│  │       │                        │                        │                         │  │
│  │       │                        └────────────────────────┼──────────────────────┐  │  │
│  │       │                                                 │                      │  │  │
│  │       │                                                 ▼                      ▼  │  │
│  │       │                                          ┌─────────────────────────────┐  │  │
│  │       │                                          │ LMModel (Moshi 7B)          │  │  │
│  │       │                                          │                             │  │  │
│  │       │                                          │ prompted_codes [B, 17, T+P] │  │  │
│  │       │                                          │ + speaker_condition [B,4096]│  │  │
│  │       │                                          │                             │  │  │
│  │       │                                          │ → text_logits, audio_logits │  │  │
│  │       │                                          └──────────────┬──────────────┘  │  │
│  │       │                                                         │                 │  │
│  │       │                                                         ▼                 │  │
│  │       │                                          ┌─────────────────────────────┐  │  │
│  │       │                                          │ Loss Computation            │  │  │
│  │       │                                          │ (prompt_mask 적용)          │  │  │
│  │       │                                          └──────────────┬──────────────┘  │  │
│  │       │                                                         │                 │  │
│  │       │                                                         ▼                 │  │
│  │       │                                          ┌─────────────────────────────┐  │  │
│  │       │                                          │ Backward & Optimizer        │  │  │
│  │       │                                          └─────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 텐서 Shape 요약

| 단계 | 텐서 | Shape | 설명 |
|------|------|-------|------|
| Input | codes | [B, 17, T] | Full-duplex mode (1 text + 8 moshi + 8 user) |
| AudioPrompt | prompted_codes | [B, 17, T+P] | Prompt prepended |
| AudioPrompt | prompt_mask | [B, T+P] | True for prompt positions |
| AudioPrompt | prompt_samples | List[AudioPromptSample] | B samples, each with audio_codes [8, P], text_tokens [P] |
| Mimi Decode | reference_audio | [B, T_audio] | @ 24kHz |
| Resample | reference_audio | [B, T_audio'] | @ 16kHz (for speaker encoder) |
| Speaker Encoder | speaker_embedding | [B, 256] | W2v-BERT 2.0 output |
| Speaker Conditioner | speaker_condition | [B, 4096] | Matches Temporal TF hidden dim |
| LMModel | text_logits | [B, T+P, vocab_size] | Text prediction |
| LMModel | audio_logits | [B, dep_q, T+P, 2048] | Audio codebook prediction |

---

## 7. 모듈 상호작용

### 7.1 모듈 의존성 그래프

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           MODULE DEPENDENCY GRAPH                                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│                              ┌───────────────┐                                          │
│                              │   train.py    │                                          │
│                              │  (Entry Point)│                                          │
│                              └───────┬───────┘                                          │
│                                      │                                                  │
│              ┌───────────────────────┼───────────────────────┐                          │
│              │                       │                       │                          │
│              ▼                       ▼                       ▼                          │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐                  │
│  │   data/           │   │   modules/        │   │   monitoring/     │                  │
│  │   ├─ dataset.py   │   │   ├─ audio_prompt │   │   ├─ sample_saver │                  │
│  │   ├─ data_loader  │   │   └─ speaker_*    │   │   ├─ metrics_log  │                  │
│  │   └─ interleaver  │   │                   │   │   └─ tensorboard  │                  │
│  └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘                  │
│            │                       │                       │                            │
│            │                       │                       │                            │
│            ▼                       ▼                       ▼                            │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              eval.py                                              │  │
│  │                                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  EvalSpeakerConditioningInfo                                                │  │  │
│  │  │  ├─ Single fields (legacy): reference_text, reference_start_sec, ...       │  │  │
│  │  │  └─ Batch fields (new): reference_texts[], reference_start_secs[], ...     │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  EvalReturnData                                                             │  │  │
│  │  │  ├─ original_codes: for sample saving (GT audio)                            │  │  │
│  │  │  ├─ prompted_codes: for metrics (matches output.mask)                       │  │  │
│  │  │  └─ speaker_conditioning_info: metadata for sample_saver                    │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│            │                       │                       │                            │
│            │                       │                       │                            │
│            └───────────────────────┼───────────────────────┘                            │
│                                    │                                                    │
│                                    ▼                                                    │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                         External Dependencies                                     │  │
│  │                                                                                   │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────────┐    │  │
│  │  │ moshi (LMModel) │  │ mimi (Codec)    │  │ W2v-BERT 2.0 (Speaker Encoder)  │    │  │
│  │  │                 │  │                 │  │                                 │    │  │
│  │  │ - 7B Transformer│  │ - 8 codebooks   │  │ - 25 layers + MFA               │    │  │
│  │  │ - dep_q=8       │  │ - 24kHz         │  │ - ASP pooling                   │    │  │
│  │  │ - sum_condition │  │ - 12.5Hz frames │  │ - 256-dim embedding             │    │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 주요 인터페이스

```python
# audio_prompt.py
class AudioPromptModule:
    def __call__(
        self,
        codes: torch.Tensor,         # [B, 17, T]
        exclude_start: int = None,   # Training segment start
        exclude_end: int = None,     # Training segment end
        deterministic: bool = False, # True for eval
    ) -> Tuple[torch.Tensor, torch.Tensor, List[AudioPromptSample]]:
        # Returns: prompted_codes, prompt_mask, prompt_samples

# speaker_encoder.py
class SpeakerEncoder:
    def forward(
        self,
        audio: torch.Tensor,         # [B, T] @ 16kHz
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        # Returns: speaker_embedding [B, 256]

# speaker_conditioner.py
class SpeakerConditioner:
    def forward(
        self,
        speaker_embedding: torch.Tensor,  # [B, 256]
    ) -> torch.Tensor:
        # Returns: speaker_condition [B, 4096]

# eval.py
def evaluate(...) -> EvalReturnData:
    # Returns: codes, output, speaker_conditioning_info

# sample_saver.py
class SampleSaver:
    def save_sample(
        self,
        batch_idx: int,
        codes: torch.Tensor,
        output: Any,
        speaker_conditioning_info: EvalSpeakerConditioningInfo,
        ...
    ) -> None:
        # Saves: reference.wav, gt_audio.wav, pred_audio.wav, speaker_metadata.json
```

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-01-24 | 1.0 | 초기 문서 작성 |
|  |  | - Training, Validation, Sample Saver, Rust Backend 흐름도 |
|  |  | - Batch-level metadata 저장 수정 반영 |
|  |  | - Word-count 기반 reference 선택 추가 |

---

*Document: K-Moshi Speaker Conditioning System Architecture*
*Version: 1.0*
*Last Updated: 2026-01-24*
