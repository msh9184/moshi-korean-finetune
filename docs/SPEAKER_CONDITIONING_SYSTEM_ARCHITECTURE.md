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

```mermaid
flowchart TD
    subgraph SYS["K-MOSHI SPEAKER CONDITIONING SYSTEM"]
        SE["Speaker Encoder (W2v-BERT 2.0)<br/>Input: Audio<br/>Output: [256]"]
        AP["Audio Prompt Module<br/>Input: Codes<br/>Output: Prompted Codes"]
        SC["Speaker Conditioner (Projection Layer)<br/>Input: Embedding [256]<br/>Output: [4096]<br/>to Temporal TF hidden"]
        LM["LMModel (Moshi 7B)"]
        LMI["Temporal Transformer: hidden_states + speaker_condition<br/>DepFormer: 8 depth codebook prediction (dep_q=8)"]
        SE --> MERGE[ ]
        AP --> MERGE
        SC --> MERGE
        MERGE --> LM
        LM --> LMI
    end
    style MERGE height:0px,width:0px
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

```mermaid
flowchart TD
    JSONL["JSONL Data (paths)<br/>train_data: './data/korean_v4_train.jsonl'"]
    TOK["InterleavedTokenizer<br/>Stereo WAV (L=Moshi, R=User) to Mimi Encode [8, T]<br/>to Text-Audio Interleave codes: [B, 17, T]<br/>[0]: text tokens, [1-8]: moshi audio, [9-16]: user audio<br/>JSON 전사 alignments"]
    APM["AudioPromptModule<br/>AudioPromptSampler.sample_single_word_count()<br/>1. Extract text_tokens, 2. Count valid tokens (exclude PAD/EOS/32000)<br/>3. Find segment min_words ~ max_words, 4. Random start (avoid overlap)<br/>Output: AudioPromptSample (audio_codes, text_tokens, user_audio_codes, start_idx, end_idx)<br/>Prepend Prompt to Codes: prompted_codes = [prompt_codes | original_codes]<br/>prompt_mask = [True ... | False ...]"]
    SENC["Speaker Encoder<br/>Reference Audio Extraction: Mimi.decode to reference_audio @ 24kHz<br/>Resample 24kHz to 16kHz to reference_audio_16k<br/>W2v-BERT 2.0 SV (0.14% EER SOTA on VoxCeleb1-O)<br/>25 Transformer layers with MFA, Attentive Statistics Pooling (ASP)<br/>Output: speaker_embedding [B, 256]"]
    SCOND["Speaker Conditioner<br/>Projection: [256] to [4096]<br/>Linear then LayerNorm then * scale (initial 0.1)<br/>speaker_condition [B, 4096]"]
    FWD["LMModel Forward<br/>Input Embedding: text_embed + audio_embed [B, T+P, 4096]<br/>Temporal Transformer (32 layers): hidden + speaker_condition (Global conditioning via sum_condition)<br/>Output Heads: text_logits [B, T+P, vocab_size], audio_logits [B, dep_q, T+P, 2048]"]
    LOSS["Loss Computation (prompt_mask 적용)<br/>Prompt 구간은 loss에서 제외<br/>text_loss = CE masked_select(~prompt_mask).mean()<br/>audio_loss = CE, audio_loss[0] *= 100.0 (first_codebook_weight_multiplier)<br/>total_loss = text_loss + audio_loss"]
    BWD["Backward & Optimizer Step<br/>total_loss.backward()<br/>optimizer.step() (AdamW)<br/>scheduler.step() (CosineWarmup)"]
    JSONL --> TOK
    TOK -->|"batch.codes [B, 17, T], batch.audio_paths"| APM
    APM -->|"prompted_codes [B, 17, T+P], prompt_mask [B, T+P], prompt_samples"| SENC
    SENC -->|"speaker_embedding [B, 256]"| SCOND
    SCOND -->|"speaker_condition [B, 4096]"| FWD
    FWD --> LOSS
    LOSS --> BWD
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

```mermaid
flowchart TD
    JSONL["JSONL Data<br/>eval_data: './data/korean_v4_valid.jsonl'"]
    DET["DETERMINISTIC SAMPLING<br/>Key Differences from Training:<br/>1. deterministic=True (NO torch.randint(), Same input to Same reference selection)<br/>2. sample_strategy='start' (recommended, always from beginning, reproducible)<br/>3. fixed_word_count=25 (or fixed_duration_sec=10.0)"]
    APM["AudioPromptModule (deterministic=True)<br/>prompted_codes, prompt_mask, prompt_samples = audio_prompt_module(codes, exclude_start=None, exclude_end=None, deterministic=True)"]
    INFO["BATCH-LEVEL INFO COLLECTION<br/>prompt_samples 전체를 순회하여 배치 레벨 정보 수집<br/>for ps in prompt_samples: append start/end secs and decode_text<br/>info.reference_texts / reference_start_secs / reference_end_secs / source_files"]
    FWD["Model Forward (Same as Training)<br/>output = model(codes=prompted_codes, condition_tensors=condition_tensors, speaker_embedding=speaker_embedding)"]
    LOSS["Loss Computation & Metrics<br/>eval_loss = text_loss + audio_loss<br/>eval_perplexity = 2^eval_loss<br/>state.this_eval_loss / this_eval_perplexity"]
    RET["Return EvalReturnData<br/>original_codes (GT audio 생성용), prompted_codes (메트릭 계산용)<br/>output, speaker_conditioning_info (Sample saver용 메타데이터), audio_paths"]
    JSONL --> DET
    DET --> APM
    APM --> INFO
    INFO --> FWD
    FWD --> LOSS
    LOSS --> RET
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

```mermaid
flowchart TD
    IN["Input: EvalReturnData from evaluate()"]
    LOOP["sample_saver.save_samples()<br/>for batch_idx in range(batch_size): save_sample(batch_idx)"]
    S1["1. Get per-sample metadata (CRITICAL FIX)<br/>ref_start = info.reference_start_secs[batch_idx]<br/>ref_end = info.reference_end_secs[batch_idx]<br/>ref_text = info.reference_texts[batch_idx]<br/>source_file = info.source_files[batch_idx]"]
    S2["2. Save Reference Audio (reference.wav)<br/>reference_audio = info.reference_audio[batch_idx]<br/>sf.write(sample_dir / 'reference.wav', reference_audio, samplerate=24000)"]
    S3["3. Save Speaker Metadata (speaker_metadata.json)<br/>enabled: true, method: 'both'<br/>reference_start_sec: 5.32, reference_end_sec: 15.32 (per-sample)<br/>reference_text: '안녕하세요', source_file: 'dialog_123.wav' (per-sample)<br/>batch_idx: 0 (debugging)"]
    S4["4. Save GT Audio (gt_audio.wav)<br/>codes = original_codes[batch_idx] (NOT prompted)<br/>gt_audio = mimi.decode(codes[1:9])"]
    S5["5. Save Predicted Audio (pred_audio.wav)<br/>pred_codes = output.logits.argmax(-1)[batch_idx]<br/>pred_audio = mimi.decode(pred_codes)"]
    S6["6. Save Text Comparison (text_comparison.json)<br/>gt_text: '오늘 날씨가 좋네요', pred_text: '오늘 날씨가 좋네요'<br/>reference_text: '안녕하세요'"]
    IN --> LOOP
    LOOP --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5
    S5 --> S6
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

```mermaid
flowchart TD
    subgraph CLIENT["CLIENT (Web Browser)"]
        MIC["User Microphone (Audio Input)"]
        REF["Speaker Reference (3-10s Audio)"]
        DISP["Real-time Display (Text + Audio Out)"]
    end
    subgraph BACKEND["RUST MOSHI BACKEND"]
        WS["1. WebSocket Handler<br/>Receive: Opus encoded user audio (streaming)<br/>Receive: Speaker reference audio (initial)<br/>Send: Moshi response audio (streaming), Text tokens (streaming)"]
        PRE["2. Audio Preprocessing<br/>Opus decode to PCM 24kHz<br/>Resample 24kHz to 16kHz (for speaker encoder)"]
        SENC["3. Speaker Encoder (W2v-BERT 2.0)<br/>Input: reference_audio [1, T] @ 16kHz<br/>Output: speaker_embedding [1, 256]<br/>Cached: 세션 시작 시 한 번만 계산"]
        MENC["4. Mimi Encoder (Real-time)<br/>Input: user_audio [1, chunk_samples] @ 24kHz<br/>Output: user_codes [1, 8, chunk_frames]<br/>Streaming: 80ms chunks (1 frame = 1920 samples)"]
        LM["5. LMModel Inference (Streaming)<br/>Input Construction: codes = [text_tokens [1,T], moshi_audio [8,T] (autoregressive), user_audio [8,T] (실시간 입력)]<br/>speaker_condition = project(speaker_embedding) [1, 4096]<br/>Temporal Transformer (Streaming): KV Cache, 한 프레임씩 처리 (80ms latency), speaker_condition 추가 (sum_condition)<br/>DepFormer (Streaming): 8 depth steps per frame, Delay pattern d=0 to d=7, Output moshi_codes [1, 8, 1]<br/>Output: text_token [1], moshi_codes [8]"]
        MDEC["6. Mimi Decoder (Real-time)<br/>Input: moshi_codes [1, 8, 1]<br/>Output: audio_samples [1, 1920] @ 24kHz<br/>Streaming: 80ms audio per frame"]
        OUT["7. Output Encoding & Streaming<br/>Opus encode audio, Package text tokens<br/>WebSocket send to client"]
        WS --> PRE --> SENC --> MENC --> LM --> MDEC --> OUT
    end
    MIC -->|"WebSocket (Opus encoded)"| WS
    REF -->|"WebSocket (Opus encoded)"| WS
    OUT --> DISP
```

### 5.2 Speaker Conditioning 통합 포인트

```mermaid
flowchart TD
    subgraph INIT["Session Initialization (Once per session)"]
        I1["1. Client uploads reference audio (3-10 seconds)"]
        I2["2. Rust backend receives and preprocesses"]
        I3["3. Speaker Encoder extracts embedding<br/>speaker_embedding = w2v_bert2(reference_audio) [1, 256]"]
        I4["4. Speaker Conditioner projects embedding<br/>speaker_condition = project(speaker_embedding) [1, 4096]"]
        I5["5. Cache for session lifetime<br/>session.speaker_condition = speaker_condition"]
        I1 --> I2 --> I3 --> I4 --> I5
    end
    subgraph STREAM["Streaming Inference (Per frame, 80ms)"]
        T1["for each user_audio_chunk"]
        T2["1. Encode user audio<br/>user_codes = mimi.encode(user_audio_chunk)"]
        T3["2. Prepare input codes<br/>input_codes = cat([text_tokens, moshi_audio, user_audio])"]
        T4["3. LM forward with speaker conditioning<br/>output = lm_model(codes=input_codes, speaker_embedding=session.speaker_condition (Cached), past_key_values=session.kv_cache)"]
        T5["4. Update KV cache<br/>session.kv_cache = output.past_key_values"]
        T6["5. Decode audio<br/>moshi_audio = mimi.decode(output.moshi_codes)"]
        T7["6. Send to client<br/>websocket.send(moshi_audio, text_token)"]
        T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7
    end
    INIT --> STREAM
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

```mermaid
flowchart TD
    SRC["WAV (stereo) + JSON"]
    TOK["InterleavedTokenizer<br/>Mimi Encode, Text Align"]
    APM["AudioPromptModule (Word-count based)"]
    MDEC["Mimi Decode (reference audio)"]
    RS["Resample 24k to 16k"]
    SV["W2v-BERT 2.0 SV (Speaker Encoder)"]
    SCOND["Speaker Conditioner<br/>Linear + LayerNorm"]
    LM["LMModel (Moshi 7B)<br/>prompted_codes [B, 17, T+P]<br/>+ speaker_condition [B, 4096]<br/>to text_logits, audio_logits"]
    LOSS["Loss Computation (prompt_mask 적용)"]
    BWD["Backward & Optimizer"]
    SRC --> TOK
    TOK -->|"codes [B, 17, T]"| APM
    TOK --> MDEC
    APM -->|"prompt_samples"| MDEC
    APM -->|"prompted_codes [B, 17, T+P]"| LM
    MDEC -->|"reference_audio [B, T] @ 24kHz"| RS
    RS -->|"reference_audio [B, T'] @ 16kHz"| SV
    SV -->|"speaker_embedding [B, 256]"| SCOND
    SCOND -->|"speaker_condition [B, 4096]"| LM
    LM --> LOSS
    LOSS --> BWD
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

```mermaid
flowchart TD
    ENTRY["train.py (Entry Point)"]
    DATA["data/<br/>dataset.py, data_loader, interleaver"]
    MODULES["modules/<br/>audio_prompt, speaker_*"]
    MON["monitoring/<br/>sample_saver, metrics_log, tensorboard"]
    EVAL["eval.py<br/>EvalSpeakerConditioningInfo (Single legacy fields + Batch new fields: reference_texts[], reference_start_secs[], ...)<br/>EvalReturnData (original_codes for GT audio, prompted_codes for metrics, speaker_conditioning_info for sample_saver)"]
    EXT["External Dependencies"]
    MOSHI["moshi (LMModel)<br/>7B Transformer, dep_q=8, sum_condition"]
    MIMI["mimi (Codec)<br/>8 codebooks, 24kHz, 12.5Hz frames"]
    W2V["W2v-BERT 2.0 (Speaker Encoder)<br/>25 layers + MFA, ASP pooling, 256-dim embedding"]
    ENTRY --> DATA
    ENTRY --> MODULES
    ENTRY --> MON
    DATA --> EVAL
    MODULES --> EVAL
    MON --> EVAL
    EVAL --> EXT
    EXT --> MOSHI
    EXT --> MIMI
    EXT --> W2V
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
