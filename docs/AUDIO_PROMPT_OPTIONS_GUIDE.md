# Audio Prompt Options Guide for K-Moshi

**작성일**: 2026-01-22
**버전**: 2.0 (PersonaPlex-Only)

---

## 1. 개요

Audio Prompt는 **PersonaPlex 스타일**의 Reference Audio/Text Prompting을 구현합니다.
학습 중 Moshi 스트림에서 참조 세그먼트를 샘플링하여 메인 시퀀스 앞에 prepend합니다.

> **중요**: V2.0부터 VALL-E 스타일(audio_only)은 제거되었습니다.
> PersonaPlex 스타일(audio_text)만 지원하여 **항상 Audio + Text 모두** 프롬프트에 포함됩니다.

### 1.1 Prompting 모드

| 모드 | 설명 | 프롬프트 내용 | 권장 상황 |
|------|------|--------------|----------|
| `speaker_embedding` | 기존 방식 | 프롬프트 없음, sum_condition만 | 안정적인 학습, 초기 실험 |
| **`audio_text`** | PersonaPlex 스타일 | **Audio + Text 모두 prepend** | **권장 (최고 성능)** |

### 1.2 PersonaPlex vs VALL-E

| 특성 | VALL-E (미지원) | PersonaPlex (지원) |
|------|----------------|-------------------|
| 프롬프트 내용 | Audio codes만 | **Audio codes + Text tokens** |
| 화자 정보 | 음향 정보만 | 음향 + 언어적 패턴 |
| 성능 | 기본 | **향상된 화자 적응** |

### 1.3 동작 방식

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                       PersonaPlex Style Prompting Flow                         │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   Original Moshi Stream [9, T_full]                                          │
│   ┌─────────────────────────────────────────────────────────────────────────┐ │
│   │ Text:  [안녕][하세요][PAD][오늘][날씨][좋네요][PAD][네][감사]           │ │
│   │ Audio: [C0-7][C0-7][C0-7][C0-7][C0-7][C0-7][C0-7][C0-7][C0-7]...        │ │
│   └─────────────────────────────────────────────────────────────────────────┘ │
│                    ↓ Random Sampling (10-15초)                                │
│   ┌─────────────────────────┐                                                │
│   │ Reference Prompt        │ (exclude_start~exclude_end 회피)               │
│   │ Text:  [안녕][하세요]   │ ← Text tokens (Moshi text stream)              │
│   │ Audio: [C0-7][C0-7]     │ ← Audio codes (8 Mimi codebooks)               │
│   └─────────────────────────┘                                                │
│                    ↓ Prepend to Main Sequence                                 │
│   ┌─────────────────────────┬────────────────────────────────────────────────┐│
│   │ PROMPT (Loss 제외)      │ MAIN (Loss 계산)                               ││
│   │ Text:  [안녕][하세요]   │ [오늘][날씨][좋네요][PAD][네][감사]            ││
│   │ Audio: [C0-7][C0-7]     │ [C0-7][C0-7][C0-7][C0-7][C0-7][C0-7]          ││
│   └─────────────────────────┴────────────────────────────────────────────────┘│
│           prompt_mask=True           prompt_mask=False                        │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 설정 옵션 상세

### 2.1 AudioPromptConfig 파라미터

```python
@dataclass
class AudioPromptConfig:
    # 기본 설정
    enable: bool = False               # Audio prompting 활성화
    mode: str = "speaker_embedding"    # "audio_text" 또는 "speaker_embedding"

    # 지속 시간 설정 (10-15초 권장)
    min_duration_sec: float = 10.0     # 최소 프롬프트 길이 (초)
    max_duration_sec: float = 15.0     # 최대 프롬프트 길이 (초)

    # 샘플링 전략
    sample_strategy: str = "random"    # "random", "start", "end", "voiced"
    avoid_overlap: bool = True         # 학습 세그먼트와 겹침 방지

    # 특수 토큰
    include_special_tokens: bool = True   # BOS/EOS 마커 포함
    prompt_position: str = "prefix"       # "prefix" 또는 "interleaved"

    # 기술적 설정
    audio_sample_rate: int = 24000     # Mimi 샘플레이트
    text_frame_rate: float = 12.5      # Moshi 프레임 레이트
```

### 2.2 샘플링 전략 (sample_strategy)

| 전략 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **`random`** | 무작위 위치 선택 | 다양한 화자 특성 학습 | 품질 편차 |
| `start` | 시작 부분 사용 | 일관성 | 항상 같은 영역 |
| `end` | 끝 부분 사용 | 최근 컨텍스트 | 항상 같은 영역 |
| `voiced` | VAD 기반 발화 구간 | 고품질 | 추가 처리 필요 |

### 2.3 권장 설정

**PersonaPlex Style (권장)**:

```yaml
speaker:
  enabled: true
  method: "both"  # encoder + audio prompt 결합

  audio_prompt:
    enable: true
    mode: "audio_text"         # Audio + Text 모두 포함 (PersonaPlex)
    min_duration_sec: 10.0     # 10초 이상
    max_duration_sec: 15.0     # 15초 이하
    sample_strategy: "random"   # 랜덤 샘플링
    avoid_overlap: true         # 학습 세그먼트와 겹침 방지
```

---

## 3. 프레임 계산

### 3.1 Duration → Frames 변환

```python
frame_rate = 12.5  # Hz (80ms per frame)

# 초 → 프레임
min_frames = int(min_duration_sec * frame_rate)
max_frames = int(max_duration_sec * frame_rate)

# 예시: 10-15초
# min_frames = 10 * 12.5 = 125 frames
# max_frames = 15 * 12.5 = 187.5 → 187 frames
```

### 3.2 프레임 → 오디오 샘플 변환

```python
audio_sample_rate = 24000  # Mimi
frame_rate = 12.5

samples_per_frame = audio_sample_rate / frame_rate  # 1920 samples

# 프레임 인덱스 → 오디오 샘플 인덱스
start_sample = start_frame * 1920
end_sample = end_frame * 1920
```

---

## 4. 로직 검증

### 4.1 AudioPromptSampler 핵심 로직

```python
class AudioPromptSampler:
    def sample_single(self, codes, exclude_start, exclude_end):
        """
        PersonaPlex Style: 항상 Audio + Text 모두 추출

        1. 유효 샘플링 영역 계산 (avoid_overlap 고려)
        2. duration_frames 랜덤 선택 (min_frames ~ max_frames)
        3. 유효 영역에서 start_frame 선택
        4. audio_codes (8 codebooks)와 text_tokens 모두 추출
        5. AudioPromptSample 반환
        """

    def _get_valid_regions(self, total_frames, exclude_start, exclude_end):
        """
        avoid_overlap=True일 때:
        - Region 1: [0, exclude_start) - 학습 세그먼트 앞
        - Region 2: [exclude_end, total_frames) - 학습 세그먼트 뒤

        각 영역이 min_frames 이상일 때만 유효
        """

    def apply_prompts(self, codes, prompts, pad_token_id):
        """
        PersonaPlex Style: Audio + Text 모두 prepend

        1. 배치 내 최대 프롬프트 길이 계산
        2. prompted_codes 텐서 생성 [B, 9, T_prompt + T]
        3. prompt_mask 텐서 생성 [B, T_prompt + T]
        4. 각 배치 아이템에:
           - text_tokens → prompted_codes[b, 0, :prompt_len]
           - audio_codes → prompted_codes[b, 1:9, :prompt_len]
        5. prompt_mask에서 프롬프트 영역 True 마킹
        """
```

### 4.2 검증 포인트

| 검증 항목 | 상태 | 설명 |
|-----------|------|------|
| 유효 영역 계산 | ✅ | `avoid_overlap` 적용됨 |
| duration 랜덤 샘플링 | ✅ | `torch.randint` 사용 |
| audio_codes 추출 | ✅ | `codes[1:9, start:end]` (8 codebooks) |
| text_tokens 추출 | ✅ | `codes[0, start:end]` (항상 포함) |
| prompt_mask 생성 | ✅ | Loss 계산 시 제외 가능 |
| 배치 패딩 처리 | ✅ | `max_prompt_len`으로 정렬 |

---

## 5. 통합 지점

### 5.1 train.py 통합

```python
# train.py에서 AudioPromptModule 초기화
if args.speaker and args.speaker.enabled:
    audio_prompt_config = AudioPromptConfig(
        enable=args.speaker.audio_prompt.enable,
        mode=args.speaker.audio_prompt.mode,  # "audio_text" (PersonaPlex)
        min_duration_sec=args.speaker.audio_prompt.min_duration_sec,
        max_duration_sec=args.speaker.audio_prompt.max_duration_sec,
        sample_strategy=args.speaker.audio_prompt.sample_strategy,
    )
    audio_prompt_module = AudioPromptModule(audio_prompt_config)

# 학습 루프에서 적용
for batch in data_loader:
    codes = batch.codes  # [B, 9, T]

    if audio_prompt_module is not None and audio_prompt_module.config.enable:
        # PersonaPlex: Audio + Text 모두 prepend
        prompted_codes, prompt_mask, prompt_samples = audio_prompt_module(
            codes,
            exclude_start=0,      # 또는 실제 학습 세그먼트 시작
            exclude_end=codes.shape[2],
        )

        # 모델 forward (prompted_codes 사용)
        output = model(codes=prompted_codes, ...)

        # Loss 계산 시 prompt_mask 활용 (프롬프트 영역 제외)
        loss = compute_loss(..., prompt_mask=prompt_mask)
    else:
        output = model(codes=codes, ...)
        loss = compute_loss(...)
```

### 5.2 Loss 계산 시 prompt_mask 사용

```python
def compute_loss_with_mask(logits, targets, prompt_mask):
    """
    prompt_mask가 True인 위치는 loss 계산에서 제외
    """
    if prompt_mask is not None:
        # prompt 영역 마스킹
        valid_mask = ~prompt_mask  # True → loss 계산

        # 유효 위치만 선택
        logits = logits[:, :, valid_mask]
        targets = targets[:, :, valid_mask]

    loss = F.cross_entropy(logits, targets)
    return loss
```

---

## 6. Speaker Embedding과의 결합

### 6.1 method="both" 동작 (권장)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Combined Speaker Conditioning                             │
│              (PersonaPlex Audio+Text Prompt + Speaker Encoder)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Reference Audio (10-15초)                                                 │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │ [Raw Audio Waveform]                                                  │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│                    ↓                              ↓                         │
│   ┌─────────────────────────┐    ┌─────────────────────────────────────┐   │
│   │ Speaker Encoder         │    │ Audio Prompt Sampler                 │   │
│   │ (W2v-BERT 2.0 SV)       │    │ (Mimi Codes + Text Tokens)          │   │
│   │                         │    │                                      │   │
│   │ Output: [B, 256]        │    │ Output: [B, 9, T_prompt]            │   │
│   │ (Global Embedding)      │    │ (Audio[8] + Text[1])                │   │
│   └───────────┬─────────────┘    └──────────────┬──────────────────────┘   │
│               ↓                                  ↓                          │
│   ┌─────────────────────────┐    ┌─────────────────────────────────────┐   │
│   │ Speaker Conditioner     │    │ Prepend to Sequence                 │   │
│   │ Linear(256 → 4096)      │    │ [Prompt | Main Sequence]            │   │
│   │                         │    │                                      │   │
│   │ Output: sum_condition   │    │ Output: prompted_codes              │   │
│   └───────────┬─────────────┘    └──────────────┬──────────────────────┘   │
│               ↓                                  ↓                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       Temporal Transformer                           │   │
│   │                                                                      │   │
│   │   hidden_states = hidden_states + sum_condition  (Global Condition)  │   │
│   │   attention([Prompt | Main])                     (Local Condition)   │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 장점

- **Global Condition (sum_condition)**: 전체 시퀀스에 화자 정보 주입 (Speaker Encoder)
- **Local Condition (prompt)**: Attention을 통한 세밀한 스타일 전이 (PersonaPlex)
- **Redundancy**: 두 경로가 상호 보완하여 안정적인 화자 적응
- **Richer Information**: Text tokens가 언어적 패턴 정보 추가 제공

---

## 7. YAML 설정 템플릿

### 7.1 권장 설정 (PersonaPlex + W2v-BERT 2.0 SV)

```yaml
# example/korean_moshi_stage1_pretrain_spk.yaml

speaker:
  enabled: true
  method: "both"  # encoder + prompt 결합

  # W2v-BERT 2.0 SV Encoder (SOTA)
  encoder:
    encoder_type: "w2v_bert2"
    pretrained_path: "/path/to/model"
    output_dim: 256
    freeze: true
    sample_rate: 16000
    normalize_embedding: true

  # Speaker Conditioner
  conditioner:
    output_dim: 4096          # Moshi hidden dim
    initial_scale: 0.1
    use_layernorm: true
    dropout: 0.0
    learnable_scale: true
    scale_mode: "multiply"

  # Reference Sampler (for raw audio → speaker encoder)
  reference_sampler:
    min_duration_sec: 10.0    # 10초
    max_duration_sec: 15.0    # 15초
    sample_rate: 24000
    target_sample_rate: 16000

  # PersonaPlex Style Audio Prompt (Audio + Text)
  audio_prompt:
    enable: true
    mode: "audio_text"        # 항상 Audio + Text 모두 포함
    min_duration_sec: 10.0    # 10초
    max_duration_sec: 15.0    # 15초
    sample_strategy: "random" # 랜덤 샘플링
    avoid_overlap: true       # 학습 세그먼트와 겹침 방지
```

---

## 8. 주의사항

### 8.1 메모리 고려

- Audio prompt가 활성화되면 시퀀스 길이가 증가
- `max_duration_sec=15`일 때 최대 187 프레임 추가
- batch_size 조정 필요할 수 있음

### 8.2 학습 안정성

- 초기에는 `method="encoder"`로 시작
- 안정화 후 `method="both"`로 전환 권장
- `initial_scale=0.1`로 시작하여 점진적 증가

### 8.3 데이터 요구사항

- 각 샘플이 `min_duration_sec` 이상이어야 함
- 샘플이 너무 짧으면 fallback 로직 적용

### 8.4 이전 버전에서 마이그레이션

V1에서 `mode: "audio_only"` 사용 시:
```yaml
# V1 (deprecated)
audio_prompt:
  mode: "audio_only"   # ❌ 더 이상 지원하지 않음

# V2 (권장)
audio_prompt:
  mode: "audio_text"   # ✅ PersonaPlex 스타일
```

---

## 9. 버전 히스토리

| 버전 | 날짜 | 변경 사항 |
|------|------|----------|
| 2.0 | 2026-01-22 | PersonaPlex-only로 단순화 (`audio_only` 모드 제거) |
| 1.0 | 2026-01-22 | 초기 버전 (VALL-E + PersonaPlex 지원) |

---

*Last Updated: 2026-01-22*
*Author: K-Moshi Development Team*
