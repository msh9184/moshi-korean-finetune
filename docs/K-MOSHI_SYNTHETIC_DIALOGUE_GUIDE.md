# K-Moshi 한국어 Full-Duplex Dialogue 합성 데이터 생성 가이드

## 개요

이 문서는 K-Moshi 학습을 위한 한국어 Full-Duplex (동시 양방향) 대화 스테레오 음성 데이터를 생성하는 구체적인 방법론을 제시합니다.

### 핵심 요구사항

| 요구사항 | 설명 |
|----------|------|
| **스테레오 포맷** | L 채널: Moshi (AI), R 채널: User |
| **자연스러운 중첩** | 대화 중 겹침(overlap), 끼어들기(barge-in) 표현 |
| **일관된 화자 음성** | Moshi 화자는 항상 동일한 음색 유지 |
| **단어 수준 타임스탬프** | 텍스트-음성 정렬을 위한 정밀 타이밍 |
| **고품질 한국어** | 자연스러운 구어체 한국어 표현 |

---

## 1. 접근 방식 비교

### 1.1 세 가지 주요 접근법

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Full-Duplex Dialogue 데이터 생성 방법                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Approach A     │  │  Approach B     │  │  Approach C                 │  │
│  │  J-Moshi Style  │  │  External TTS   │  │  Hybrid                     │  │
│  │                 │  │                 │  │                             │  │
│  │  K-Moshi 모델을  │  │  외부 한국어 TTS │  │  초기: External TTS         │  │
│  │  TTS로 활용     │  │  모델 활용       │  │  후기: K-Moshi Self-Gen     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                             │
│  장점:                 장점:                 장점:                           │
│  - 도메인 일치         - 즉시 시작 가능       - 점진적 품질 향상              │
│  - Self-improvement   - 고품질 한국어 TTS    - 리스크 분산                   │
│  - J-Moshi 검증됨      - 다양한 음색 선택                                    │
│                                                                             │
│  단점:                 단점:                 단점:                           │
│  - 초기 모델 필요      - 도메인 불일치        - 복잡한 파이프라인             │
│  - Bootstrap 필요     - Overlap 처리 어려움  - 전환 시점 결정 필요           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 방법별 상세 비교

| 항목 | Approach A (J-Moshi Style) | Approach B (External TTS) | Approach C (Hybrid) |
|------|---------------------------|--------------------------|---------------------|
| **초기 비용** | 높음 (모델 학습 필요) | 낮음 | 중간 |
| **데이터 품질** | 높음 (도메인 일치) | 중간 | 점진적 향상 |
| **Overlap 자연성** | 매우 높음 | 낮음 | 향상됨 |
| **확장성** | 무제한 | TTS 비용 의존 | 무제한 |
| **구현 복잡도** | 높음 | 낮음 | 중간 |
| **J-Moshi 성과** | 602시간, WER 24.6% | N/A | N/A |

---

## 2. Approach A: J-Moshi Style (Multi-stream TTS)

### 2.1 핵심 원리

J-Moshi가 검증한 방법으로, 학습된 Moshi 모델 자체를 TTS로 활용하여 대화 데이터를 생성합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    J-Moshi Multi-stream TTS Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [텍스트 대화 코퍼스]                                                         │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────┐                                                        │
│  │ LLM Rewriting   │  "안녕하세요" → "아~ 안녕하세요~"                         │
│  │ (Gemma-2-27B)   │  문어체 → 구어체 변환                                    │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │ K-Moshi Model   │  Multi-stream Generation                               │
│  │ (as TTS)        │  - Semantic Delay: 25                                  │
│  └────────┬────────┘  - Acoustic Delay: 27                                  │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │ Sample Selection│  10개 샘플 생성 → 최저 WER 선택                          │
│  │ (WER-based)     │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  [Stereo WAV + Word Timestamps]                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 J-Moshi의 실제 구현 (참조)

```python
# j-moshi-finetune/utils/data.py - Delay & Padding 메커니즘

def delay_and_pad_streams(
    list_of_streams: list[np.ndarray],
    delays: list[int],              # 각 스트림별 지연
    initial_token_ids: list[int],   # 시작 토큰
    padding_token_ids: list[int],   # 패딩 토큰
) -> list[np.array]:
    """
    Autoregressive 인과성을 위한 스트림 지연 적용

    delays 예시:
    - Text (inner monologue): delay=0
    - Audio Semantic (codebook 0): delay=25
    - Audio Acoustic (codebook 1-7): delay=27, 29, 31, ...
    """
    delayed = []
    for i, stream in enumerate(list_of_streams):
        d = delays[i]
        padded = np.concatenate([
            np.array([initial_token_ids[i]] * d),  # 앞에 초기 토큰
            stream,
            np.array([padding_token_ids[i]] * d),  # 뒤에 패딩 토큰
        ])
        delayed.append(padded)
    return delayed
```

### 2.3 한국어 적용 시 필요 사항

#### A. 초기 K-Moshi 모델 학습 (Bootstrap)

Multi-stream TTS를 사용하려면 먼저 기본적인 한국어 음성 합성이 가능한 모델이 필요합니다.

```yaml
# Bootstrap Phase 1: 기초 한국어 음성 학습
data:
  train_data: './data/korean_monologue_100h.jsonl'  # 단일 화자 한국어 음성

korean:
  enable_user_stream: false  # Monologue 모드로 시작
  full_duplex_input: false
```

#### B. LLM을 활용한 구어체 변환

```python
# scripts/convert_to_spoken_korean.py

import json
from openai import OpenAI  # 또는 Korean LLM API

SPOKEN_CONVERSION_PROMPT = """
다음 문어체 한국어 대화를 자연스러운 구어체로 변환해주세요.
- 추임새 추가 (음, 아, 네, 그래서)
- 축약형 사용 (그것은 → 그건, 하였습니다 → 했어요)
- 반복, 머뭇거림 포함 가능
- 원래 의미는 유지

입력 대화:
{dialogue}

구어체 변환:
"""

def convert_to_spoken(dialogue: list[dict]) -> list[dict]:
    """문어체 대화를 구어체로 변환"""
    client = OpenAI()

    dialogue_text = "\n".join([
        f"{turn['speaker']}: {turn['text']}"
        for turn in dialogue
    ])

    response = client.chat.completions.create(
        model="gpt-4",  # 또는 한국어 LLM
        messages=[{
            "role": "user",
            "content": SPOKEN_CONVERSION_PROMPT.format(dialogue=dialogue_text)
        }],
        temperature=0.7
    )

    # 파싱 및 반환
    return parse_spoken_dialogue(response.choices[0].message.content)
```

#### C. Multi-stream 생성 코드

```python
# scripts/generate_multistream_dialogue.py

import torch
from moshi.models import LMModel
from moshi.tokenizers import MimiTokenizer

class KoreanMultistreamGenerator:
    """K-Moshi 모델을 활용한 Multi-stream 대화 생성"""

    def __init__(
        self,
        model: LMModel,
        mimi: MimiTokenizer,
        semantic_delay: int = 25,
        acoustic_delay: int = 27,
    ):
        self.model = model
        self.mimi = mimi
        self.semantic_delay = semantic_delay
        self.acoustic_delay = acoustic_delay

    def generate_dialogue(
        self,
        text_turns: list[dict],  # [{"speaker": "A/B", "text": "..."}, ...]
        num_samples: int = 10,
        seed_base: int = 42,
    ) -> tuple[torch.Tensor, list[dict]]:
        """
        텍스트 대화를 스테레오 오디오로 변환

        Returns:
            audio: [2, samples] 스테레오 오디오
            timestamps: 단어별 타임스탬프
        """
        best_audio = None
        best_wer = float('inf')
        best_timestamps = None

        for i in range(num_samples):
            torch.manual_seed(seed_base + i)

            # Multi-stream 생성
            audio, timestamps = self._generate_single(text_turns)

            # WER 계산 (WhisperX 활용)
            wer = self._compute_wer(audio, text_turns)

            if wer < best_wer:
                best_wer = wer
                best_audio = audio
                best_timestamps = timestamps

        return best_audio, best_timestamps

    def _generate_single(self, text_turns: list[dict]):
        """단일 샘플 생성"""
        # 1. 텍스트 토큰화 with 시간 정보
        text_tokens, timing = self._tokenize_dialogue(text_turns)

        # 2. Delay 적용
        delayed_text = self._apply_delay(text_tokens, delay=0)

        # 3. Autoregressive 생성
        with torch.no_grad():
            output = self.model.generate(
                text_tokens=delayed_text,
                max_new_tokens=text_tokens.shape[-1] * 2,
                temperature=0.8,
                top_k=250,
            )

        # 4. 오디오 디코딩
        audio = self.mimi.decode(output.audio_codes)

        # 5. 타임스탬프 추출
        timestamps = self._extract_timestamps(output, timing)

        return audio, timestamps
```

### 2.4 장단점 분석

**장점:**
- J-Moshi에서 602시간 생성으로 검증됨
- 도메인 일치 (같은 모델로 학습/생성)
- Overlap이 자연스럽게 생성됨
- Self-improvement 가능

**단점:**
- 초기 Bootstrap 모델 필요 (최소 100시간 학습)
- 구현 복잡도 높음
- WER 24.6%로 완벽하지 않음

---

## 3. Approach B: External Korean TTS

### 3.1 사용 가능한 한국어 TTS 모델

| TTS 모델 | 특징 | Zero-Shot | 품질 | 권장도 |
|----------|------|-----------|------|--------|
| **Nari Labs Dia** | 한국어 특화, 대화 최적화 | O | 높음 | ⭐⭐⭐⭐⭐ |
| **XTTS v2** | 다국어, 음색 복제 | O | 높음 | ⭐⭐⭐⭐ |
| **StyleTTS2-Ko** | 한국어 미세조정 버전 | X | 높음 | ⭐⭐⭐⭐ |
| **Parler-TTS** | 설명 기반 생성 | O | 중간 | ⭐⭐⭐ |
| **MeloTTS** | 빠른 추론 | O | 중간 | ⭐⭐⭐ |

### 3.2 Nari Labs Dia 활용 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    External TTS Pipeline (Nari Labs Dia)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [한국어 텍스트 대화]                                                         │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Speaker A (Moshi)          │ Speaker B (User)                       │   │
│  │ - 고정 Reference Audio     │ - 다양한 Reference Audio               │   │
│  │ - Nari Labs Dia TTS        │ - XTTS v2 또는 다른 TTS               │   │
│  └──────────────┬─────────────┴──────────────────┬─────────────────────┘   │
│                 │                                 │                         │
│                 ▼                                 ▼                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Timing Scheduler                                 │   │
│  │  - 턴 기반 타이밍 계산                                               │   │
│  │  - Overlap 구간 삽입                                                 │   │
│  │  - 자연스러운 간격 추가                                               │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Stereo Mixer                                     │   │
│  │  L Channel: Speaker A (Moshi)                                       │   │
│  │  R Channel: Speaker B (User)                                        │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    WhisperX Alignment                               │   │
│  │  - 각 채널별 word-level timestamp 추출                               │   │
│  │  - JSON 형식 저장                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 구현 코드

#### A. TTS 기반 대화 생성기

```python
# data_preparation/external_tts_generator.py

import torch
import torchaudio
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class DialogueTurn:
    speaker: str          # "moshi" or "user"
    text: str
    start_time: float     # 초 단위
    end_time: float
    overlap_with_prev: float = 0.0  # 이전 턴과의 중첩 시간

@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    speaker: str

class ExternalTTSDialogueGenerator:
    """외부 TTS를 활용한 대화 데이터 생성기"""

    def __init__(
        self,
        moshi_tts,           # Nari Labs Dia 또는 고정 화자 TTS
        user_tts,            # XTTS v2 또는 다양한 화자 TTS
        moshi_reference: str, # Moshi 화자 참조 오디오 경로
        sample_rate: int = 24000,
    ):
        self.moshi_tts = moshi_tts
        self.user_tts = user_tts
        self.moshi_reference = moshi_reference
        self.sample_rate = sample_rate

    def generate_dialogue(
        self,
        turns: List[DialogueTurn],
        user_reference: Optional[str] = None,
    ) -> Tuple[torch.Tensor, List[WordTimestamp]]:
        """
        대화 턴들을 스테레오 오디오로 변환

        Args:
            turns: 대화 턴 리스트
            user_reference: User 화자 참조 오디오 (None이면 랜덤)

        Returns:
            stereo_audio: [2, samples] 텐서
            timestamps: 단어별 타임스탬프 리스트
        """
        # 전체 길이 계산
        total_duration = max(t.end_time for t in turns)
        total_samples = int(total_duration * self.sample_rate)

        # 스테레오 버퍼 초기화
        stereo = torch.zeros(2, total_samples)
        timestamps = []

        for turn in turns:
            # TTS 선택
            if turn.speaker == "moshi":
                tts = self.moshi_tts
                reference = self.moshi_reference
                channel = 0  # Left
            else:
                tts = self.user_tts
                reference = user_reference
                channel = 1  # Right

            # 음성 합성
            audio, word_times = self._synthesize_turn(
                tts, turn.text, reference
            )

            # 타이밍 조정 및 배치
            start_sample = int(turn.start_time * self.sample_rate)
            end_sample = start_sample + audio.shape[-1]

            # 버퍼 확장 (필요시)
            if end_sample > stereo.shape[-1]:
                padding = torch.zeros(2, end_sample - stereo.shape[-1])
                stereo = torch.cat([stereo, padding], dim=-1)

            # 오디오 배치 (크로스페이드로 부드럽게)
            stereo[channel, start_sample:end_sample] = self._crossfade_mix(
                stereo[channel, start_sample:end_sample],
                audio
            )

            # 타임스탬프 조정
            for wt in word_times:
                timestamps.append(WordTimestamp(
                    word=wt['word'],
                    start=turn.start_time + wt['start'],
                    end=turn.start_time + wt['end'],
                    speaker=turn.speaker,
                ))

        return stereo, timestamps

    def _synthesize_turn(
        self,
        tts,
        text: str,
        reference: Optional[str]
    ) -> Tuple[torch.Tensor, List[dict]]:
        """단일 턴 음성 합성"""
        # TTS 합성
        audio = tts.synthesize(
            text=text,
            reference_audio=reference,
            sample_rate=self.sample_rate,
        )

        # WhisperX로 단어 타임스탬프 추출
        word_times = self._extract_word_timestamps(audio, text)

        return audio, word_times

    def _extract_word_timestamps(
        self,
        audio: torch.Tensor,
        text: str
    ) -> List[dict]:
        """WhisperX를 사용한 단어 타임스탬프 추출"""
        import whisperx

        # 임시 파일로 저장
        temp_path = "/tmp/temp_tts_output.wav"
        torchaudio.save(temp_path, audio.unsqueeze(0), self.sample_rate)

        # WhisperX 처리
        model = whisperx.load_model("large-v3", device="cuda", language="ko")
        result = model.transcribe(temp_path)

        # 정렬 모델 로드
        align_model, metadata = whisperx.load_align_model(
            language_code="ko", device="cuda"
        )
        result = whisperx.align(
            result["segments"], align_model, metadata,
            temp_path, device="cuda"
        )

        # 단어 타임스탬프 추출
        word_times = []
        for segment in result["segments"]:
            for word in segment.get("words", []):
                word_times.append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                })

        return word_times

    def _crossfade_mix(
        self,
        existing: torch.Tensor,
        new: torch.Tensor,
        fade_samples: int = 480  # 20ms at 24kHz
    ) -> torch.Tensor:
        """부드러운 크로스페이드 믹싱"""
        if existing.abs().max() < 0.01:
            return new

        # 페이드 커브 생성
        fade_in = torch.linspace(0, 1, fade_samples)
        fade_out = torch.linspace(1, 0, fade_samples)

        # 크로스페이드 적용
        if len(new) > fade_samples:
            new[:fade_samples] *= fade_in
        if len(existing) > fade_samples:
            existing[-fade_samples:] *= fade_out

        return existing + new
```

#### B. Overlap 스케줄러

```python
# data_preparation/overlap_scheduler.py

import random
from dataclasses import dataclass
from typing import List

@dataclass
class OverlapConfig:
    """대화 중첩 설정"""
    # 중첩 확률
    overlap_probability: float = 0.3

    # 중첩 길이 (초)
    min_overlap: float = 0.1
    max_overlap: float = 0.5

    # 턴 간 간격 (초)
    min_gap: float = 0.1
    max_gap: float = 0.8

    # 끼어들기 확률
    bargein_probability: float = 0.1

class DialogueTimingScheduler:
    """자연스러운 대화 타이밍 생성"""

    def __init__(self, config: OverlapConfig = None):
        self.config = config or OverlapConfig()

    def schedule_turns(
        self,
        turns: List[dict],  # [{"speaker": "A/B", "text": "...", "duration": 2.5}, ...]
    ) -> List[DialogueTurn]:
        """
        대화 턴들에 자연스러운 타이밍 할당

        Returns:
            타이밍이 할당된 DialogueTurn 리스트
        """
        scheduled = []
        current_time = 0.0

        for i, turn in enumerate(turns):
            # 이전 턴과의 관계 결정
            overlap = 0.0
            if i > 0:
                if random.random() < self.config.overlap_probability:
                    # 중첩 발생
                    overlap = random.uniform(
                        self.config.min_overlap,
                        min(self.config.max_overlap, turns[i-1]['duration'] * 0.3)
                    )
                    current_time -= overlap
                else:
                    # 일반 간격
                    gap = random.uniform(
                        self.config.min_gap,
                        self.config.max_gap
                    )
                    current_time += gap

            # 끼어들기 처리
            if i > 0 and random.random() < self.config.bargein_probability:
                # 이전 화자 발화 중간에 시작
                prev_turn = scheduled[-1]
                current_time = prev_turn.start_time + prev_turn.end_time * 0.7
                overlap = prev_turn.end_time - current_time

            # DialogueTurn 생성
            scheduled.append(DialogueTurn(
                speaker="moshi" if turn['speaker'] == 'A' else "user",
                text=turn['text'],
                start_time=max(0, current_time),
                end_time=current_time + turn['duration'],
                overlap_with_prev=overlap,
            ))

            current_time = scheduled[-1].end_time

        return scheduled
```

### 3.4 Nari Labs Dia 통합

```python
# data_preparation/nari_dia_wrapper.py

import torch
from dia.model import Dia

class NariDiaWrapper:
    """Nari Labs Dia TTS 래퍼"""

    def __init__(self, model_path: str = "nari-labs/Dia-1.6B"):
        self.model = Dia.from_pretrained(model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def synthesize(
        self,
        text: str,
        reference_audio: str = None,
        sample_rate: int = 24000,
        speed: float = 1.0,
    ) -> torch.Tensor:
        """
        텍스트를 음성으로 변환

        Args:
            text: 합성할 텍스트
            reference_audio: 화자 참조 오디오 경로 (optional)
            sample_rate: 출력 샘플레이트
            speed: 말하기 속도 (1.0 = 정상)

        Returns:
            audio: [samples] 오디오 텐서
        """
        # Dia는 대화 형식 지원
        # [S1] 텍스트 [S2] 형식으로 단일 화자 지정 가능
        formatted_text = f"[S1]{text}[S1]"

        with torch.no_grad():
            output = self.model.generate(
                text=formatted_text,
                audio_prompt=reference_audio,
                max_tokens=2048,
                cfg_scale=3.0,
            )

        # 리샘플링 (필요시)
        if output.sample_rate != sample_rate:
            output.audio = torchaudio.functional.resample(
                output.audio, output.sample_rate, sample_rate
            )

        return output.audio
```

### 3.5 장단점 분석

**장점:**
- 즉시 시작 가능 (Bootstrap 불필요)
- 고품질 한국어 TTS 활용
- 다양한 화자 음색 선택 가능
- 구현 상대적으로 간단

**단점:**
- Overlap 처리가 자연스럽지 않을 수 있음
- TTS 모델과 Moshi 학습 도메인 불일치
- 대규모 생성 시 TTS 비용 발생
- 실시간 대화의 미묘한 특성 표현 어려움

---

## 4. Approach C: Hybrid (권장)

### 4.1 2단계 전략

가장 실용적인 접근법으로, External TTS로 시작하여 점진적으로 Self-generation으로 전환합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Hybrid Approach Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Bootstrap (External TTS)                                          │
│  ──────────────────────────────────                                         │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐   │
│  │ 한국어 텍스트    │ → │ Nari Labs Dia   │ → │ 초기 학습 데이터        │   │
│  │ 대화 코퍼스      │   │ + XTTS v2       │   │ (~100시간)              │   │
│  └─────────────────┘   └─────────────────┘   └─────────────────────────┘   │
│                                                       │                     │
│                                                       ▼                     │
│  Phase 2: Initial Training                                                  │
│  ─────────────────────────                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ K-Moshi V4 Training (Full-Duplex Mode)                              │   │
│  │ - 100시간 External TTS 데이터로 학습                                  │   │
│  │ - Backbone: Moshi 7B 또는 HFLM 3B                                 │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼                                          │
│  Phase 3: Self-Generation (Multi-stream TTS)                               │
│  ────────────────────────────────────────────                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 학습된 K-Moshi를 TTS로 활용                                          │   │
│  │ - J-Moshi 방식의 Multi-stream 생성                                   │   │
│  │ - WER 기반 품질 선택                                                 │   │
│  │ - 600+ 시간 추가 생성                                                │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼                                          │
│  Phase 4: Iterative Improvement                                            │
│  ──────────────────────────────                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 생성 데이터 + 실제 데이터 혼합 학습                                    │   │
│  │ → 품질 향상 → 더 나은 생성 → 반복                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Phase별 상세 구현

#### Phase 1: Bootstrap 데이터 생성 (External TTS)

```python
# data_preparation/phase1_bootstrap.py

from external_tts_generator import ExternalTTSDialogueGenerator
from overlap_scheduler import DialogueTimingScheduler, OverlapConfig
from nari_dia_wrapper import NariDiaWrapper
import json
from pathlib import Path
from tqdm import tqdm
import torchaudio

def generate_bootstrap_dataset(
    dialogue_corpus_path: str,      # 한국어 텍스트 대화 코퍼스
    output_dir: str,
    moshi_reference_audio: str,     # Moshi 화자 참조 오디오
    target_hours: int = 100,
):
    """
    Phase 1: External TTS로 Bootstrap 데이터 생성
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TTS 초기화
    moshi_tts = NariDiaWrapper("nari-labs/Dia-1.6B")
    user_tts = XTTSWrapper("coqui/XTTS-v2")  # 별도 구현 필요

    generator = ExternalTTSDialogueGenerator(
        moshi_tts=moshi_tts,
        user_tts=user_tts,
        moshi_reference=moshi_reference_audio,
    )

    scheduler = DialogueTimingScheduler(OverlapConfig(
        overlap_probability=0.25,
        min_overlap=0.1,
        max_overlap=0.4,
    ))

    # 대화 코퍼스 로드
    with open(dialogue_corpus_path) as f:
        dialogues = json.load(f)

    # JSONL 메타데이터 파일
    jsonl_path = output_dir / "train.jsonl"

    total_duration = 0
    processed = 0

    with open(jsonl_path, 'w') as jsonl_file:
        for idx, dialogue in enumerate(tqdm(dialogues)):
            if total_duration >= target_hours * 3600:
                break

            try:
                # 턴 정보 준비
                turns = prepare_turns_from_dialogue(dialogue)

                # 타이밍 스케줄링
                scheduled_turns = scheduler.schedule_turns(turns)

                # 오디오 생성
                stereo_audio, timestamps = generator.generate_dialogue(
                    scheduled_turns
                )

                # 파일 저장
                wav_filename = f"dialogue_{idx:06d}.wav"
                wav_path = output_dir / wav_filename
                json_path = output_dir / f"dialogue_{idx:06d}.json"

                # 스테레오 WAV 저장 (24kHz, 16-bit)
                torchaudio.save(
                    wav_path,
                    stereo_audio,
                    24000,
                    encoding="PCM_S",
                    bits_per_sample=16
                )

                # 타임스탬프 JSON 저장
                alignments = [
                    [ts.word, [ts.start, ts.end],
                     "SPEAKER_MAIN" if ts.speaker == "moshi" else "SPEAKER_USER"]
                    for ts in timestamps
                ]
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump({"alignments": alignments}, jf, ensure_ascii=False)

                # JSONL 항목 추가
                duration = stereo_audio.shape[-1] / 24000
                jsonl_file.write(json.dumps({
                    "path": str(wav_path),
                    "duration": duration
                }) + '\n')

                total_duration += duration
                processed += 1

            except Exception as e:
                print(f"Error processing dialogue {idx}: {e}")
                continue

    print(f"Generated {processed} dialogues, total {total_duration/3600:.1f} hours")
    return jsonl_path

def prepare_turns_from_dialogue(dialogue: dict) -> list:
    """대화 데이터를 턴 리스트로 변환"""
    turns = []
    for utterance in dialogue['utterances']:
        # TTS로 예상 길이 추정 (대략 초당 4음절)
        estimated_duration = len(utterance['text']) / 4.0

        turns.append({
            'speaker': 'A' if utterance['speaker'] == 'assistant' else 'B',
            'text': utterance['text'],
            'duration': estimated_duration,
        })
    return turns
```

#### Phase 2: 초기 K-Moshi 학습

```yaml
# example/korean_phase1_bootstrap.yaml

data:
  train_data: './data/bootstrap_100h/train.jsonl'
  eval_data: './data/bootstrap_100h/valid.jsonl'
  shuffle: true

backbone:
  type: "moshi"  # 또는 "hf_lm"
  moshi:
    hidden_dim: 4096
    num_layers: 32
    gradient_checkpointing: true

korean:
  enable_user_stream: false
  full_duplex_input: true  # 스테레오 입력

# 보수적 학습률 (Bootstrap 데이터)
optim:
  lr: 2.0e-5
  depformer_lr: 2.0e-5
  weight_decay: 0.1

scheduler:
  type: 'cosine_warmup'
  warmup_steps: 500
  min_lr: 1.0e-7

# 학습 설정
duration_sec: 40
batch_size: 64
num_microbatches: 8
max_steps: 5000
```

#### Phase 3: Self-Generation (Multi-stream)

```python
# data_preparation/phase3_self_generation.py

import torch
from pathlib import Path
from typing import List, Dict
import json
from tqdm import tqdm

class KMoshiSelfGenerator:
    """학습된 K-Moshi 모델을 TTS로 활용한 Self-Generation"""

    def __init__(
        self,
        model_checkpoint: str,
        mimi_path: str,
        tokenizer_path: str,
        semantic_delay: int = 25,
        acoustic_delay: int = 27,
    ):
        # 체크포인트에서 모델 로드
        self.model = self._load_model(model_checkpoint)
        self.mimi = self._load_mimi(mimi_path)
        self.tokenizer = self._load_tokenizer(tokenizer_path)

        self.semantic_delay = semantic_delay
        self.acoustic_delay = acoustic_delay

    def generate_dataset(
        self,
        text_dialogues: List[Dict],
        output_dir: str,
        samples_per_dialogue: int = 10,
        target_hours: int = 600,
    ):
        """
        텍스트 대화에서 스테레오 오디오 데이터셋 생성

        J-Moshi 논문 방식:
        - 각 대화당 10개 샘플 생성
        - WER 기반 최적 샘플 선택
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = output_dir / "train.jsonl"
        total_duration = 0

        with open(jsonl_path, 'w') as jsonl_file:
            for idx, dialogue in enumerate(tqdm(text_dialogues)):
                if total_duration >= target_hours * 3600:
                    break

                # 다중 샘플 생성 및 최적 선택
                best_audio, best_timestamps, best_wer = self._generate_best_sample(
                    dialogue,
                    num_samples=samples_per_dialogue,
                )

                if best_wer > 0.5:  # WER 50% 초과 시 스킵
                    continue

                # 파일 저장
                wav_path = output_dir / f"dialogue_{idx:06d}.wav"
                json_path = output_dir / f"dialogue_{idx:06d}.json"

                self._save_outputs(
                    wav_path, json_path,
                    best_audio, best_timestamps
                )

                duration = best_audio.shape[-1] / 24000
                jsonl_file.write(json.dumps({
                    "path": str(wav_path),
                    "duration": duration,
                    "wer": best_wer,
                }) + '\n')

                total_duration += duration

        print(f"Generated {total_duration/3600:.1f} hours of self-generated data")

    def _generate_best_sample(
        self,
        dialogue: Dict,
        num_samples: int,
    ) -> tuple:
        """다중 샘플 생성 및 WER 기반 선택"""
        best_audio = None
        best_timestamps = None
        best_wer = float('inf')

        for seed in range(num_samples):
            torch.manual_seed(42 + seed)

            # Multi-stream 생성
            audio, timestamps = self._generate_multistream(dialogue)

            # WER 계산
            wer = self._compute_wer(audio, dialogue)

            if wer < best_wer:
                best_wer = wer
                best_audio = audio
                best_timestamps = timestamps

        return best_audio, best_timestamps, best_wer

    def _generate_multistream(self, dialogue: Dict) -> tuple:
        """Multi-stream 방식 오디오 생성"""
        # 텍스트 토큰화
        text_tokens = self._tokenize_dialogue_text(dialogue)

        # Delay 적용
        delayed_tokens = self._apply_delays(text_tokens)

        # Autoregressive 생성
        with torch.no_grad():
            output = self.model.generate(
                text_tokens=delayed_tokens,
                max_new_tokens=3000,  # ~240초 at 12.5Hz
                temperature=0.8,
                top_k=250,
                semantic_delay=self.semantic_delay,
                acoustic_delay=self.acoustic_delay,
            )

        # Undelay 및 디코딩
        audio_codes = self._undelay_codes(output.audio_codes)
        stereo_audio = self.mimi.decode(audio_codes)

        # 타임스탬프 추출
        timestamps = self._extract_timestamps(output.text_tokens, dialogue)

        return stereo_audio, timestamps

    def _compute_wer(self, audio: torch.Tensor, dialogue: Dict) -> float:
        """WhisperX를 사용한 WER 계산"""
        import whisperx
        import tempfile
        import torchaudio

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            # 모노로 변환 (두 채널 믹스)
            mono = audio.mean(dim=0, keepdim=True)
            torchaudio.save(temp_path, mono, 24000)

        # Whisper 전사
        model = whisperx.load_model("large-v3", device="cuda", language="ko")
        result = model.transcribe(temp_path)

        # 원본 텍스트와 비교
        reference = " ".join([u['text'] for u in dialogue['utterances']])
        hypothesis = " ".join([s['text'] for s in result['segments']])

        # WER 계산
        from jiwer import wer
        return wer(reference, hypothesis)
```

### 4.3 장단점 분석

**장점:**
- 리스크 분산 (External TTS 실패 시에도 진행 가능)
- 점진적 품질 향상
- J-Moshi의 검증된 방법 활용
- 대규모 확장 가능 (Self-generation)

**단점:**
- 전체 파이프라인 복잡
- Phase 전환 시점 결정 필요
- 초기 투자 비용 (External TTS)

---

## 5. 한국어 텍스트 대화 소스

### 5.1 공개 데이터셋

| 데이터셋 | 크기 | 특징 | 링크 |
|----------|------|------|------|
| **AI Hub 자유대화** | 10만+ 대화 | 일상 대화, 다양한 주제 | [AI Hub](https://aihub.or.kr) |
| **KorQuAD 대화** | 7만+ 대화 | 질의응답 형식 | [KorQuAD](https://korquad.github.io/) |
| **Korean Daily Dialogue** | 1만+ 대화 | 일상 대화 | GitHub |
| **KoDialogues** | 5천+ 대화 | 멀티턴 대화 | HuggingFace |

### 5.2 LLM 생성 대화

```python
# data_preparation/generate_dialogues_llm.py

from openai import OpenAI
import json

DIALOGUE_GENERATION_PROMPT = """
자연스러운 한국어 일상 대화를 생성해주세요.

요구사항:
1. 두 명의 화자 (A: AI 어시스턴트, B: 사용자)
2. 5-10 턴의 대화
3. 구어체 사용 (추임새, 축약형 포함)
4. 주제: {topic}

출력 형식 (JSON):
{{
  "utterances": [
    {{"speaker": "B", "text": "사용자 발화"}},
    {{"speaker": "A", "text": "AI 응답"}},
    ...
  ]
}}
"""

def generate_dialogues(
    topics: list,
    dialogues_per_topic: int = 100,
    output_path: str = "dialogues.json",
):
    """LLM을 사용한 대화 생성"""
    client = OpenAI()
    all_dialogues = []

    for topic in topics:
        for _ in range(dialogues_per_topic):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": DIALOGUE_GENERATION_PROMPT.format(topic=topic)
                }],
                temperature=0.9,
            )

            try:
                dialogue = json.loads(response.choices[0].message.content)
                dialogue['topic'] = topic
                all_dialogues.append(dialogue)
            except:
                continue

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_dialogues, f, ensure_ascii=False, indent=2)

    return all_dialogues

# 사용 예
topics = [
    "날씨와 계절",
    "음식과 요리",
    "여행과 관광",
    "업무와 직장",
    "취미와 여가",
    "건강과 운동",
    "쇼핑과 패션",
    "기술과 IT",
]
```

---

## 6. 단어 수준 타임스탬프 추출

### 6.1 WhisperX 한국어 설정

```python
# data_preparation/whisperx_korean.py

import whisperx
import torch
import torchaudio

class KoreanWhisperXAligner:
    """WhisperX를 활용한 한국어 단어 정렬"""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
    ):
        self.device = device

        # Whisper 모델 로드
        self.model = whisperx.load_model(
            model_size,
            device=device,
            language="ko",
            compute_type="float16",
        )

        # 한국어 정렬 모델 로드
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="ko",
            device=device,
        )

    def extract_word_timestamps(
        self,
        audio_path: str,
        channel: int = None,  # None: 모노/스테레오 전체, 0: Left, 1: Right
    ) -> list:
        """
        오디오에서 단어 수준 타임스탬프 추출

        Args:
            audio_path: 오디오 파일 경로
            channel: 처리할 채널 (스테레오의 경우)

        Returns:
            [{"word": "안녕", "start": 0.0, "end": 0.3}, ...]
        """
        # 오디오 로드
        audio, sr = torchaudio.load(audio_path)

        # 채널 선택 (스테레오의 경우)
        if channel is not None and audio.shape[0] > 1:
            audio = audio[channel:channel+1]

        # 모노로 변환
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # 16kHz로 리샘플링 (Whisper 요구사항)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        # 임시 파일 저장
        temp_path = "/tmp/whisperx_temp.wav"
        torchaudio.save(temp_path, audio, 16000)

        # 전사
        result = self.model.transcribe(temp_path, batch_size=16)

        # 단어 정렬
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            temp_path,
            self.device,
        )

        # 단어 타임스탬프 추출
        word_timestamps = []
        for segment in result["segments"]:
            for word_info in segment.get("words", []):
                word_timestamps.append({
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"],
                })

        return word_timestamps

    def align_stereo_dialogue(
        self,
        audio_path: str,
    ) -> dict:
        """
        스테레오 대화에서 양쪽 채널의 타임스탬프 추출

        Returns:
            {
                "moshi": [{"word": ..., "start": ..., "end": ...}, ...],
                "user": [{"word": ..., "start": ..., "end": ...}, ...]
            }
        """
        # Left 채널 (Moshi)
        moshi_timestamps = self.extract_word_timestamps(audio_path, channel=0)

        # Right 채널 (User)
        user_timestamps = self.extract_word_timestamps(audio_path, channel=1)

        return {
            "moshi": moshi_timestamps,
            "user": user_timestamps,
        }
```

### 6.2 K-Moshi 학습용 JSON 형식 변환

```python
# data_preparation/convert_to_kmoshi_format.py

def convert_to_kmoshi_alignments(
    moshi_timestamps: list,
    user_timestamps: list,
) -> dict:
    """
    WhisperX 출력을 K-Moshi 학습용 JSON 형식으로 변환

    K-Moshi 형식:
    {
        "alignments": [
            ["단어", [시작, 끝], "SPEAKER_MAIN/SPEAKER_USER"],
            ...
        ]
    }
    """
    alignments = []

    # Moshi 타임스탬프 추가
    for ts in moshi_timestamps:
        alignments.append([
            ts["word"],
            [ts["start"], ts["end"]],
            "SPEAKER_MAIN"
        ])

    # User 타임스탬프 추가
    for ts in user_timestamps:
        alignments.append([
            ts["word"],
            [ts["start"], ts["end"]],
            "SPEAKER_USER"
        ])

    # 시간순 정렬
    alignments.sort(key=lambda x: x[1][0])

    return {"alignments": alignments}
```

---

## 7. 품질 검증 및 필터링

### 7.1 자동 품질 검증

```python
# data_preparation/quality_validation.py

from dataclasses import dataclass
from typing import List, Dict
import torch
import torchaudio

@dataclass
class QualityMetrics:
    wer: float              # Word Error Rate
    cer: float              # Character Error Rate
    duration_ratio: float   # 실제/예상 길이 비율
    snr: float              # Signal-to-Noise Ratio
    overlap_ratio: float    # 중첩 구간 비율
    silence_ratio: float    # 무음 비율

class DialogueQualityValidator:
    """대화 데이터 품질 검증"""

    def __init__(
        self,
        whisperx_aligner,
        min_wer: float = 0.5,
        min_snr: float = 15.0,
        max_silence_ratio: float = 0.4,
    ):
        self.aligner = whisperx_aligner
        self.min_wer = min_wer
        self.min_snr = min_snr
        self.max_silence_ratio = max_silence_ratio

    def validate(
        self,
        audio_path: str,
        reference_text: str,
    ) -> tuple[bool, QualityMetrics]:
        """
        오디오 품질 검증

        Returns:
            (is_valid, metrics)
        """
        # 오디오 로드
        audio, sr = torchaudio.load(audio_path)

        # WER/CER 계산
        transcription = self._transcribe(audio_path)
        wer, cer = self._compute_wer_cer(transcription, reference_text)

        # SNR 계산
        snr = self._compute_snr(audio)

        # Overlap 비율 계산
        overlap_ratio = self._compute_overlap_ratio(audio)

        # Silence 비율 계산
        silence_ratio = self._compute_silence_ratio(audio)

        # Duration 비율 계산
        duration_ratio = audio.shape[-1] / sr / self._estimate_duration(reference_text)

        metrics = QualityMetrics(
            wer=wer,
            cer=cer,
            duration_ratio=duration_ratio,
            snr=snr,
            overlap_ratio=overlap_ratio,
            silence_ratio=silence_ratio,
        )

        # 유효성 판단
        is_valid = (
            wer <= self.min_wer and
            snr >= self.min_snr and
            silence_ratio <= self.max_silence_ratio and
            0.7 <= duration_ratio <= 1.5
        )

        return is_valid, metrics

    def _compute_snr(self, audio: torch.Tensor) -> float:
        """Signal-to-Noise Ratio 계산"""
        # 간단한 SNR 추정 (RMS 기반)
        rms = audio.pow(2).mean().sqrt()
        noise_floor = audio.abs().quantile(0.1)

        if noise_floor > 0:
            snr = 20 * torch.log10(rms / noise_floor)
            return snr.item()
        return 30.0  # 기본값

    def _compute_overlap_ratio(self, audio: torch.Tensor) -> float:
        """스테레오에서 중첩 구간 비율 계산"""
        if audio.shape[0] < 2:
            return 0.0

        # 에너지 기반 활성화 감지
        frame_size = 480  # 20ms at 24kHz
        left_energy = audio[0].unfold(0, frame_size, frame_size).pow(2).mean(1)
        right_energy = audio[1].unfold(0, frame_size, frame_size).pow(2).mean(1)

        threshold = 0.01
        left_active = left_energy > threshold
        right_active = right_energy > threshold

        overlap_frames = (left_active & right_active).sum()
        total_frames = max(left_active.sum(), right_active.sum())

        return (overlap_frames / total_frames).item() if total_frames > 0 else 0.0

    def _compute_silence_ratio(self, audio: torch.Tensor) -> float:
        """무음 구간 비율 계산"""
        frame_size = 480
        energy = audio.pow(2).mean(0).unfold(0, frame_size, frame_size).mean(1)

        threshold = 0.001
        silence_frames = (energy < threshold).sum()
        total_frames = energy.shape[0]

        return (silence_frames / total_frames).item()
```

### 7.2 필터링 파이프라인

```python
# data_preparation/filter_dataset.py

def filter_dataset(
    input_jsonl: str,
    output_jsonl: str,
    min_wer: float = 0.5,
    min_duration: float = 5.0,
    max_duration: float = 120.0,
):
    """품질 기준에 따라 데이터셋 필터링"""
    validator = DialogueQualityValidator(
        whisperx_aligner=KoreanWhisperXAligner(),
        min_wer=min_wer,
    )

    valid_count = 0
    total_count = 0

    with open(input_jsonl) as f_in, open(output_jsonl, 'w') as f_out:
        for line in tqdm(f_in):
            entry = json.loads(line)
            total_count += 1

            # 길이 필터
            if not (min_duration <= entry['duration'] <= max_duration):
                continue

            # 품질 검증
            is_valid, metrics = validator.validate(
                entry['path'],
                entry.get('reference_text', '')
            )

            if is_valid:
                entry['quality_metrics'] = {
                    'wer': metrics.wer,
                    'snr': metrics.snr,
                    'overlap_ratio': metrics.overlap_ratio,
                }
                f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                valid_count += 1

    print(f"Filtered: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
```

---

## 8. 전체 파이프라인 실행

### 8.1 권장 실행 순서

```bash
# Phase 1: Bootstrap 데이터 생성 (External TTS)
# ─────────────────────────────────────────────

# 1.1 한국어 대화 코퍼스 준비
python data_preparation/generate_dialogues_llm.py \
    --output dialogues_10k.json \
    --count 10000

# 1.2 External TTS로 오디오 생성
python data_preparation/phase1_bootstrap.py \
    --dialogue_corpus dialogues_10k.json \
    --output_dir ./data/bootstrap_100h \
    --moshi_reference ./reference/moshi_voice.wav \
    --target_hours 100

# 1.3 품질 필터링
python data_preparation/filter_dataset.py \
    --input ./data/bootstrap_100h/train.jsonl \
    --output ./data/bootstrap_100h/train_filtered.jsonl \
    --min_wer 0.4

# Phase 2: 초기 K-Moshi 학습
# ──────────────────────────

# 2.1 학습 실행
torchrun --nproc-per-node 8 -m train example/korean_phase1_bootstrap.yaml

# Phase 3: Self-Generation (Multi-stream)
# ────────────────────────────────────────

# 3.1 추가 대화 텍스트 준비
python data_preparation/generate_dialogues_llm.py \
    --output dialogues_100k.json \
    --count 100000

# 3.2 Self-Generation
python data_preparation/phase3_self_generation.py \
    --model_checkpoint ./runs/phase1/checkpoint_final.pt \
    --dialogue_corpus dialogues_100k.json \
    --output_dir ./data/selfgen_600h \
    --target_hours 600

# 3.3 품질 필터링
python data_preparation/filter_dataset.py \
    --input ./data/selfgen_600h/train.jsonl \
    --output ./data/selfgen_600h/train_filtered.jsonl

# Phase 4: 최종 학습
# ─────────────────

# 4.1 Bootstrap + Self-Gen 데이터 결합
cat ./data/bootstrap_100h/train_filtered.jsonl \
    ./data/selfgen_600h/train_filtered.jsonl \
    > ./data/combined_700h/train.jsonl

# 4.2 최종 학습
torchrun --nproc-per-node 8 -m train example/korean_v4_fsdp_moshi.yaml \
    --data.train_data ./data/combined_700h/train.jsonl
```

### 8.2 예상 리소스 및 시간

| Phase | 데이터 크기 | GPU 시간 | 비용 예상 |
|-------|------------|----------|-----------|
| Phase 1 (External TTS) | 100시간 | TTS API 비용 | ~$500-1000 |
| Phase 2 (Initial Train) | 100시간 | ~40시간 (8xA100) | ~$400 |
| Phase 3 (Self-Gen) | 600시간 | ~100시간 (8xA100) | ~$1000 |
| Phase 4 (Final Train) | 700시간 | ~80시간 (8xA100) | ~$800 |

**총 예상: ~$2,500-3,000 (클라우드 GPU 기준)**

---

## 9. 참고 문헌

1. **J-Moshi Paper** (arXiv:2506.02979) - Multi-stream TTS 방법론
2. **DialoSpeech** (arXiv:2510.08373) - Dual-speaker dialogue synthesis
3. **ConversaSynth** (arXiv:2409.00946) - LLM + multi-speaker TTS framework
4. **Open-Source Full-Duplex Datasets** (arXiv:2509.04093) - 데이터셋 구축 가이드
5. **Nari Labs Dia** - 한국어 대화 TTS 모델
6. **WhisperX** - 단어 수준 타임스탬프 추출

---

## 10. 체크리스트

### 데이터 준비
- [ ] 한국어 텍스트 대화 코퍼스 확보
- [ ] Moshi 화자 참조 오디오 준비
- [ ] External TTS 모델 설정 (Nari Labs Dia / XTTS)
- [ ] WhisperX 한국어 정렬 모델 준비

### Phase 1: Bootstrap
- [ ] Phase 1 Bootstrap 데이터 생성 (100시간)
- [ ] 품질 검증 및 필터링
- [ ] JSONL 메타데이터 생성

### Phase 2: Initial Training
- [ ] K-Moshi Phase 1 학습 완료
- [ ] 체크포인트 검증

### Phase 3: Self-Generation
- [ ] Multi-stream 생성 코드 구현
- [ ] 600+ 시간 Self-Generation
- [ ] WER 기반 품질 선택

### Phase 4: Final Training
- [ ] 데이터 결합 (Bootstrap + Self-Gen)
- [ ] 최종 K-Moshi 학습
- [ ] 품질 평가 및 서빙 테스트

---

*Last Updated: 2026-01-13*
*Document: K-Moshi Synthetic Dialogue Generation Guide*
