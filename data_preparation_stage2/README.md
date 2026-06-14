# Data Preparation Stage 2: Synthetic Dialogue Generation

> K-Moshi 한국어 합성 대화 데이터 생성 파이프라인

## Overview

Stage 2는 External TTS를 사용하여 한국어 Full-Duplex 대화 데이터를 합성하는 파이프라인입니다.

### 핵심 기능

1. **Text Corpus 처리**: AI Hub, 모두의 말뭉치 등 텍스트 대화 데이터 로딩 및 구어체 변환
2. **K-Moshi Identity**: 일관된 자기소개 및 정체성 Q&A 데이터 생성
3. **Full-Duplex Timing**: 오버랩, 백채널, 바지인 등 자연스러운 대화 타이밍
4. **TTS 합성**: OpenAudio S1 Mini + Supertonic-2 하이브리드 전략
5. **Voice 관리**: Moshi 단일 음성 + User 다양한 음성
6. **품질 관리**: WER 필터링, 오디오 품질 검증

## Directory Structure

```
data_preparation_stage2/
├── corpus/                 # 텍스트 코퍼스 처리
├── identity/               # K-Moshi 정체성 시스템
│   └── data/               # Q&A 템플릿
├── timing/                 # Full-Duplex 타이밍
├── tts/                    # TTS 통합
├── voice/                  # 음성 관리
│   └── samples/            # 참조 음성 샘플
├── synthesis/              # 대화 합성 엔진
├── writers/                # 출력 형식
├── quality/                # 품질 관리
├── orchestrators/          # 파이프라인 오케스트레이터
├── scripts/                # 실행 스크립트
├── tests/                  # 테스트
└── configs/                # 설정 파일
```

## Quick Start

### 1. 설정 파일 준비

```bash
# 기본 설정 복사 및 수정
cp configs/default.yaml configs/my_config.yaml
# 경로, GPU 설정 등 수정
```

### 2. 음성 샘플 준비

```bash
# Moshi 참조 음성 (10-30초, 고품질)
voice/samples/moshi/moshi_reference.wav

# User 참조 음성들 (10개 이상 권장)
voice/samples/users/user_01.wav
voice/samples/users/user_02.wav
...
```

### 3. 합성 실행

```bash
# 단일 GPU
python -m data_preparation_stage2.scripts.run_synthesis configs/my_config.yaml

# 분산 처리 (8 GPU)
python -m data_preparation_stage2.scripts.run_distributed configs/my_config.yaml
```

## Configuration

### TTS Strategy

```yaml
tts:
  strategy:
    moshi_ratio: 1.0    # Moshi: 100% OpenAudio S1 Mini (일관성)
    user_ratio: 0.7     # User: 70% S1 Mini + 30% Supertonic (다양성)
```

### Full-Duplex Timing

```yaml
timing:
  overlap:
    probability: 0.35   # ~35% 확률로 오버랩
  backchannel:
    probability: 0.25   # ~25% 확률로 백채널 삽입
```

### Quality Filtering

```yaml
quality:
  wer:
    max_wer_moshi: 0.15  # Moshi WER 임계값
    max_wer_user: 0.25   # User WER 임계값
    samples_per_dialogue: 10  # 대화당 10개 샘플 생성 후 최적 선택
```

## Output Format

### Stereo WAV
- Sample Rate: 24kHz
- Channels: 2 (L=Moshi, R=User)
- Format: WAV or FLAC

### Alignment JSON
```json
{
  "alignments": [
    ["안녕하세요", [0.0, 0.85], "SPEAKER_MAIN"],
    ["네", [1.2, 1.4], "SPEAKER_USER"]
  ]
}
```

### Manifest JSONL
```jsonl
{"path": "audio/001.wav", "duration": 45.32, "wer_moshi": 0.08}
```

## Dependencies

```bash
pip install -e ".[stage2]"
```

### Required Models
- OpenAudio S1 Mini
- Supertonic-2
- Whisper Large-v3 (alignment)

## References

- [K-Moshi Data Synthesis Analysis](../docs/K-MOSHI_DATA_SYNTHESIS_ANALYSIS.md)
- [Implementation Plan](../docs/DATA_PREPARATION_STAGE2_IMPLEMENTATION.md)
- [J-Moshi Paper](https://arxiv.org/abs/2506.02979)
