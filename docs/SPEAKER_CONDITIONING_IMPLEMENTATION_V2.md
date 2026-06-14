# K-Moshi Speaker Conditioning Implementation V2

**작성일**: 2026-01-22
**상태**: ✅ 구현 완료 (Linux 서버 동기화 대기)
**버전**: V2.1 - W2v-BERT 2.0 SV + VALL-E Style Audio/Text Prompting

---

## 1. 개요

이 문서는 K-Moshi의 Zero-Shot Speaker Conditioning 구현에 대한 최신 업데이트를 정리합니다.

### 1.1 주요 업데이트 (V2.1)

| 항목 | V1 (이전) | V2.1 (현재) |
|------|-----------|-----------|
| **Speaker Encoder** | ECAPA-TDNN only (192-dim) | ECAPA-TDNN + **W2v-BERT 2.0 SV** (256-dim SOTA) |
| **Audio Prompting** | 미지원 | **VALL-E Style** (audio_only, audio_text) |
| **Conditioning Method** | encoder only | encoder, prompt, **both** (권장) |
| **Reference Duration** | 3-10초 | **10-15초** (사용자 설정) |
| **Reference Sampler** | Basic | Enhanced with overlap avoidance |
| **Documentation** | 기본 | 다운로드 가이드, 아키텍처 다이어그램, 옵션 가이드 추가 |

### 1.2 권장 설정 요약

```yaml
speaker:
  enabled: true
  method: "both"  # encoder + prompt 결합 (최고 성능)

  encoder:
    encoder_type: "w2v_bert2"  # SOTA 0.14% EER
    pretrained_path: "/path/to/model"
    output_dim: 256
    freeze: true

  audio_prompt:
    enable: true
    mode: "audio_text"        # PersonaPlex 스타일
    min_duration_sec: 10.0    # 10초
    max_duration_sec: 15.0    # 15초
    sample_strategy: "random"
    avoid_overlap: true
```

---

## 2. 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          K-Moshi Speaker Conditioning V2.1                               │
│                    W2v-BERT 2.0 SV (256-dim) + VALL-E Style Prompting                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   Reference Audio (10-15초)                                                             │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                    [Raw Audio Waveform @ 24kHz]                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                           │                                  │                          │
│                           ↓                                  ↓                          │
│   ┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐ │
│   │      PATH 1: Speaker Encoder          │   │      PATH 2: Audio Prompt             │ │
│   │         (Global Condition)            │   │         (Local Condition)             │ │
│   ├───────────────────────────────────────┤   ├───────────────────────────────────────┤ │
│   │                                       │   │                                       │ │
│   │  ┌─────────────────────────────────┐  │   │  Reference Codes [9, T_ref]           │ │
│   │  │ Resample 24kHz → 16kHz          │  │   │  ┌─────────────────────────────────┐  │ │
│   │  └──────────────┬──────────────────┘  │   │  │ Text:  [안녕][하세요][PAD]...   │  │ │
│   │                 ↓                     │   │  │ Audio: [C0-7][C0-7][C0-7]...    │  │ │
│   │  ┌─────────────────────────────────┐  │   │  └─────────────────────────────────┘  │ │
│   │  │ W2v-BERT 2.0 SV Encoder         │  │   │                  ↓                    │ │
│   │  │  ├─ MFA (25 layers concat)      │  │   │  Prepend to Main Sequence            │ │
│   │  │  ├─ ASP (Attentive Stats Pool)  │  │   │  ┌─────────────────────────────────┐  │ │
│   │  │  └─ Bottleneck → 256-dim        │  │   │  │ [PROMPT | MAIN SEQUENCE]       │  │ │
│   │  └──────────────┬──────────────────┘  │   │  │ prompt_mask: [True...|False...] │  │ │
│   │                 ↓                     │   │  └─────────────────────────────────┘  │ │
│   │  speaker_embedding: [B, 256]          │   │                                       │ │
│   │                 ↓                     │   │  prompted_codes: [B, 9, T_ref+T_main] │ │
│   │  ┌─────────────────────────────────┐  │   │                                       │ │
│   │  │ Speaker Conditioner             │  │   └───────────────────────────────────────┘ │
│   │  │  Linear(256→4096) + LN + Scale  │  │                      │                      │
│   │  └──────────────┬──────────────────┘  │                      │                      │
│   │                 ↓                     │                      │                      │
│   │  sum_condition: [B, 4096]             │                      │                      │
│   └───────────────────────────────────────┘                      │                      │
│                           │                                      │                      │
│                           └──────────────────┬───────────────────┘                      │
│                                              ↓                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         Temporal Transformer (7B)                                │   │
│   │                                                                                  │   │
│   │   hidden = embed(prompted_codes) + sum_condition.unsqueeze(1)                   │   │
│   │             └─ Local Condition via Attention ─┘    └─ Global Condition ─┘       │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                              ↓                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  Loss Computation (prompt_mask=True 영역 제외)                                   │   │
│   │                                                                                  │   │
│   │  text_loss = CE(text_logits[:, ~prompt_mask], targets[:, ~prompt_mask])         │   │
│   │  audio_loss = CE(audio_logits[:, ~prompt_mask], targets[:, ~prompt_mask])       │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Speaker Encoder 옵션

### 3.1 W2v-BERT 2.0 SV  권장

- **출력 차원**: 256
- **성능**: VoxCeleb1-O에서 **0.14% EER** (State-of-the-Art)
- **논문**: https://arxiv.org/abs/2510.04213
- **모델**: https://huggingface.co/zl389/w2v-bert-2.0_SV
- **의존성**: `pip install transformers>=4.35.0`
- **다운로드 가이드**: `docs/W2V_BERT2_DOWNLOAD_GUIDE.md`

```yaml
encoder:
  encoder_type: "w2v_bert2"
  pretrained_path: "/path/to/model"
  output_dim: 256
  freeze: true
  w2v_bert2_n_mfa_layers: -1  # 모든 레이어 사용
  w2v_bert2_pooling: "ASP"
```

### 3.2 ECAPA-TDNN (대안)

- **출력 차원**: 192
- **모델**: SpeechBrain `speechbrain/spkrec-ecapa-voxceleb`
- **의존성**: `pip install speechbrain`
- **특징**: 경량, 빠른 추론

### 3.3 Dummy Encoder (테스트용)

- **용도**: 의존성 없이 파이프라인 테스트
- **출력**: 설정 가능한 차원의 랜덤 임베딩

---

## 4. Audio Prompting (VALL-E Style)

### 4.1 프롬프팅 모드

| 모드 | 설명 | 프롬프트 내용 |
|------|------|--------------|
| `speaker_embedding` | 기존 방식 | Audio prompt 없음, sum_condition만 |
| `audio_only` | VALL-E 스타일 | Audio codes만 prepend |
| **`audio_text`** | PersonaPlex 스타일 | Audio + Text 모두 prepend ⭐ |

### 4.2 권장 설정 (10-15초)

```yaml
audio_prompt:
  enable: true
  mode: "audio_text"        # Audio + Text 모두 포함
  min_duration_sec: 10.0    # 10초
  max_duration_sec: 15.0    # 15초
  sample_strategy: "random" # 랜덤 샘플링
  avoid_overlap: true       # 학습 세그먼트와 겹침 방지
```

**참조**: `docs/AUDIO_PROMPT_OPTIONS_GUIDE.md`

---

## 5. 생성/수정된 파일 목록

### 5.1 새로 생성된 파일

| 파일 | 설명 |
|------|------|
| `finetune/modules/speaker_encoder.py` | Speaker Encoder (ECAPA-TDNN, W2v-BERT 2.0, Dummy) |
| `finetune/modules/speaker_conditioner.py` | Speaker Conditioner 및 ReferenceSampler |
| `finetune/modules/audio_prompt.py` | VALL-E 스타일 Audio/Text Prompting |
| `finetune/modules/__init__.py` | 모듈 exports |
| `docs/W2V_BERT2_DOWNLOAD_GUIDE.md` | W2v-BERT 2.0 SV 모델 다운로드 가이드 |
| `docs/AUDIO_PROMPT_OPTIONS_GUIDE.md` | Audio Prompt 옵션 가이드 |
| `docs/SPEAKER_CONDITIONING_ARCHITECTURE_DIAGRAM.md` | 전체 아키텍처 다이어그램 |

### 5.2 수정된 파일

| 파일 | 수정 내용 |
|------|----------|
| `finetune/args.py` | `SpeakerEncoderArgs`, `AudioPromptArgs`, `SpeakerConditioningArgs` 추가 |
| `train.py` | speaker_embedding 파라미터 호환성 수정 |
| `finetune/backbone/lm_model_wrapper.py` | speaker conditioning 메서드 추가 |
| `example/korean_moshi_stage1_pretrain_spk.yaml` | 권장 speaker 설정 업데이트 |

### 5.3 버그 수정

| 파일 | 수정 내용 |
|------|----------|
| `finetune/modules/speaker_encoder.py` | `os` import 추가, 중복 함수 제거 |

---

## 6. 전체 YAML 설정 예시

### 6.1 korean_moshi_stage1_pretrain_spk.yaml

```yaml
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
    w2v_bert2_n_mfa_layers: -1
    w2v_bert2_pooling: "ASP"

  # Speaker Conditioner
  conditioner:
    output_dim: 4096
    initial_scale: 0.1
    use_layernorm: true
    dropout: 0.0
    learnable_scale: true
    scale_mode: "multiply"

  # Reference Sampler
  reference_sampler:
    min_duration_sec: 10.0
    max_duration_sec: 15.0
    sample_rate: 24000
    target_sample_rate: 16000

  # VALL-E Style Audio Prompt
  audio_prompt:
    enable: true
    mode: "audio_text"
    min_duration_sec: 10.0
    max_duration_sec: 15.0
    sample_strategy: "random"
    avoid_overlap: true
```

---

## 7. Linux 서버 동기화 체크리스트

### 7.1 파일 동기화

```bash
# WSL → GPU 서버 동기화 (rsync 또는 scp)
rsync -avz --exclude='*.pyc' --exclude='__pycache__' \
    /path/to/workspace \
    user@gpu-server:/path/to/workspace
```

### 7.2 의존성 설치

```bash
# GPU 서버에서 실행
pip install transformers>=4.35.0 speechbrain huggingface_hub
```

### 7.3 W2v-BERT 2.0 SV 모델 다운로드

```bash
# 가이드 참조: docs/W2V_BERT2_DOWNLOAD_GUIDE.md
mkdir -p /path/to/model
huggingface-cli download zl389/w2v-bert-2.0_SV \
    --local-dir /path/to/model
```

### 7.4 파이프라인 테스트

```bash
# Dummy encoder로 먼저 테스트
# korean_moshi_stage1_pretrain_spk.yaml에서 encoder_type: "dummy" 설정 후 실행
torchrun --nproc-per-node 1 -m train example/korean_moshi_stage1_pretrain_spk.yaml
```

---

## 8. 다음 단계

### 8.1 즉시 실행 가능

1. ✅ 코드 동기화: WSL → Linux 서버
2. ✅ 의존성 설치: transformers, speechbrain
3. ✅ 모델 다운로드: W2v-BERT 2.0 SV
4. ✅ 테스트 실행: encoder_type: "dummy"로 파이프라인 검증

### 8.2 통합 작업 필요

1. **train.py 통합**: AudioPromptModule을 학습 루프에 완전 통합
2. **Loss Masking**: prompt_mask를 사용한 loss 계산 최적화
3. **Inference 지원**: 추론 시 Reference Audio 로딩 로직

### 8.3 추후 개선

1. **Custom Encoder**: 팀 자체 개발 모델 통합
2. **VAD 기반 Sampling**: 발화 구간 선호 샘플링
3. **Cross-Attention Prompting**: 고급 프롬프트 인코딩

---

## 9. 참조 문서

| 문서 | 설명 |
|------|------|
| `docs/W2V_BERT2_DOWNLOAD_GUIDE.md` | W2v-BERT 2.0 SV 다운로드 가이드 |
| `docs/AUDIO_PROMPT_OPTIONS_GUIDE.md` | Audio Prompt 옵션 상세 설명 |
| `docs/SPEAKER_CONDITIONING_ARCHITECTURE_DIAGRAM.md` | 전체 아키텍처 다이어그램 |

### 외부 참조

- **W2v-BERT 2.0 SV Paper**: https://arxiv.org/abs/2510.04213
- **W2v-BERT 2.0 SV Model**: https://huggingface.co/zl389/w2v-bert-2.0_SV
- **VALL-E Paper**: https://arxiv.org/abs/2301.02111
- **NVIDIA PersonaPlex**: Reference audio/text conditioning approach

---

*Last Updated: 2026-01-22*
*Author: K-Moshi Development Team*
