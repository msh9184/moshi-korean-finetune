# K-Moshi Text Post-Processing: Future Work Design Document

> **Status**: 🔮 Future Work (Not Yet Implemented)
> **Priority**: Medium
> **Created**: 2025-01-01
> **Last Updated**: 2025-01-01
> **Author**: K-Moshi Development Team

---

## 1. Executive Summary

### 1.1 배경 (Background)

K-Moshi의 Inner Monologue 텍스트 출력에서 띄어쓰기, 구두점, 숫자/약어 표현이 자연스럽지 않은 문제가 발견되었습니다. 분석 결과, 이는 학습 파이프라인의 `character_level_interpolation` 설정과 관련된 구조적 문제로 확인되었습니다.

### 1.2 현재 결정 (Current Decision)

- **즉시 적용**: `character_level_interpolation: false` 설정 변경
- **Future Work**: 텍스트 후처리(Post-Processing) 파이프라인은 안정성을 위해 추후 구현으로 연기

### 1.3 연기 사유 (Rationale for Deferral)

1. **복잡성 증가 우려**: 현재 학습 파이프라인에 추가 기능 도입 시 안정성 저하 가능
2. **검증 필요**: 후처리 효과와 지연시간 트레이드오프에 대한 충분한 테스트 필요
3. **다국어 지원 고려**: 한국어뿐 아니라 영어 등 다국어 지원 설계 필요

---

## 2. 문제 분석 (Problem Analysis)

### 2.1 현상 (Symptoms)

Step 75 학습 시점의 출력 예시:
```
REF: 노사노사부를수는없지만왜부를수없죠?이게근데입에짝짝붙는것이...
HYP: 그실래노노.어요.냐면수는도.제아데그장에서장에서이수짝이것것처럼...

WER: 77.2%  |  CER: 44.4%
```

**문제점**:
- 띄어쓰기 없음 (공백 누락)
- 구두점 불규칙
- 숫자/약어 미변환 (ITN 미적용)

### 2.2 근본 원인 (Root Cause)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ROOT CAUSE ANALYSIS                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [character_level_interpolation=True 설정 시]                           │
│                                                                          │
│  입력 alignments:                                                        │
│    ("안녕", (0.0, 0.5), "SPEAKER_MAIN")                                 │
│    ("하세요", (0.6, 1.0), "SPEAKER_MAIN")                               │
│                        ↓                                                │
│  _word_to_character_alignments():                                       │
│    [("안",0.0,0.25), ("녕",0.25,0.5), ("하",0.6,0.73), ...]             │
│                        ↓                                                │
│  _tokenize_with_character_interpolation():                              │
│    full_text = "".join(char for char, _, _ in char_alignments)         │
│    결과: "안녕하세요" ← 단어 사이 공백 없음!                            │
│                        ↓                                                │
│  tokenize():                                                            │
│    SentencePiece가 공백 없는 텍스트를 토큰화                            │
│    → ▁ 마커가 단어 경계에서 생성되지 않음                              │
│                        ↓                                                │
│  [모델 학습]                                                            │
│    모델이 띄어쓰기 없는 텍스트를 "정답"으로 학습                        │
│    → 추론 시에도 띄어쓰기 없이 출력                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 해결책 (Solution)

**즉시 적용 (Implemented)**:
```yaml
# example/korean_v4_fsdp_moshi.yaml
korean:
  interleaver:
    character_level_interpolation: false  # ← 변경
```

**추가 개선 (Future Work)**:
- 텍스트 후처리 파이프라인 구현

---

## 3. 텍스트 후처리 설계 (Post-Processing Design)

### 3.1 설계 원칙 (Design Principles)

1. **선택적 활성화**: 후처리를 켜고 끌 수 있는 토글 옵션 필수
2. **최소 지연**: 실시간 대화 시 체감되지 않는 수준의 처리 시간 (< 20ms)
3. **다국어 지원**: 한국어뿐 아니라 영어 등 다국어 확장 가능한 구조
4. **단일 툴킷 집중**: 복잡성 최소화를 위해 검증된 하나의 툴킷만 사용

### 3.2 활성화 옵션 구조 (Toggle Options)

```yaml
# 학습 파이프라인 설정 (train config)
text_postprocessing:
  enabled: true                    # 전체 켜기/끄기
  stages:
    noise_normalization: true      # Stage 1: 노이즈 정규화
    spacing_correction: true       # Stage 2: 띄어쓰기 교정
    punctuation_restoration: true  # Stage 3: 구두점 복원
    inverse_text_norm: false       # Stage 4: ITN (선택적)
  language: "auto"                 # "ko", "en", "auto" (자동 감지)
```

```rust
// Rust 백엔드 설정 (serving config)
{
  "text_postprocessing": {
    "enabled": true,
    "language": "ko",
    "stages": ["noise", "spacing", "punctuation"]
  }
}
```

```
[API/WebSocket 요청 시 런타임 토글]

Client → Server (session config):
{
  "text_postprocessing": true,  // 세션별 활성화
  "language_hint": "ko"
}
```

### 3.3 후처리 파이프라인 (Processing Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TEXT POST-PROCESSING PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [모델 출력 텍스트]                                                      │
│          │                                                              │
│          ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Stage 1: 노이즈 정규화 (Noise Normalization)                     │   │
│  │ ────────────────────────────────────────────────────────────── │   │
│  │ • SentencePiece 마커 제거: "▁" → " "                            │   │
│  │ • 특수문자 정규화: "\", "/", "_" 제거                           │   │
│  │ • 반복 문자 정리: "ㅋㅋㅋㅋㅋ" → "ㅋㅋ"                         │   │
│  │ • 인코딩 깨짐 복구: "?" → ""                                    │   │
│  │ • 지연시간: < 1ms                                               │   │
│  │ • 구현: 정규표현식 기반 (모든 언어 공통)                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│          │                                                              │
│          ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Stage 2: 띄어쓰기/토큰화 교정 (Spacing/Tokenization)             │   │
│  │ ────────────────────────────────────────────────────────────── │   │
│  │ • 한국어: 형태소 분석 기반 띄어쓰기 복원                         │   │
│  │ • 영어: 단어 경계 정규화 (이미 양호한 경우 스킵)                 │   │
│  │ • 지연시간: 5-15ms                                              │   │
│  │ • 도구: Lindera (Rust) - 한국어/일본어/중국어 지원              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│          │                                                              │
│          ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Stage 3: 구두점 복원 (Punctuation Restoration)                   │   │
│  │ ────────────────────────────────────────────────────────────── │   │
│  │ • 문장 끝 마침표/물음표/느낌표 추가                              │   │
│  │ • 의문문 패턴 감지: "~까", "~니", "~죠" → "?"                   │   │
│  │ • 지연시간: < 1ms                                               │   │
│  │ • 구현: 규칙 기반 (언어별 패턴)                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│          │                                                              │
│          ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Stage 4: ITN - Inverse Text Normalization (선택적)               │   │
│  │ ────────────────────────────────────────────────────────────── │   │
│  │ • 숫자 변환: "천구백구십일" → "1991"                             │   │
│  │ • 약어 변환: "에이아이" → "AI", "지피티" → "GPT"                │   │
│  │ • 단위 변환: "퍼센트" → "%", "달러" → "$"                       │   │
│  │ • 지연시간: 1-3ms                                               │   │
│  │ • 구현: 규칙 기반 매핑 테이블 (언어별)                           │   │
│  │ • 참고: NVIDIA NeMo ITN은 한국어 미지원 (2025.01 기준)          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│          │                                                              │
│          ▼                                                              │
│  [최종 출력 텍스트]                                                      │
│                                                                          │
│  총 예상 지연시간: 7-20ms (Moshi 프레임 간격 80ms 대비 허용 가능)       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 툴킷 선정 (Toolkit Selection)

### 4.1 비교 분석 (Comparative Analysis)

| 도구 | 언어 | Rust 지원 | 속도 | 정확도 | 오프라인 | 다국어 |
|------|------|-----------|------|--------|----------|--------|
| **Lindera** | Rust | ✅ 네이티브 | ⚡⚡⚡ | 87% | ✅ | ko, ja, zh |
| Kiwipiepy | Python | ❌ FFI 필요 | ⚡⚡ | 94% | ✅ | ko only |
| PyKoSpacing | Python | ❌ FFI 필요 | ⚡⚡ | 99% | ✅ | ko only |
| py-hanspell | Python | ❌ 불가 | ❌ 느림 | 높음 | ❌ API | ko only |
| NVIDIA NeMo | Python | ❌ FFI 필요 | ⚡⚡ | 높음 | ✅ | ❌ ko 미지원 |

### 4.2 권장 툴킷: Lindera

**선정 사유**:
1. **Rust 네이티브**: Moshi Rust 백엔드와 직접 통합 가능
2. **다국어 지원**: 한국어(ko-dic), 일본어(ipadic), 중국어(cc-cedict) 지원
3. **빠른 속도**: 네이티브 Rust 구현으로 최소 지연
4. **오프라인 동작**: 외부 API 의존 없음 (사내 네트워크 제한 대응)
5. **활발한 유지보수**: crates.io에서 지속 업데이트 중

**한계점 및 대응**:
- 정확도가 Kiwipiepy(94%) 대비 낮음(87%)
- 대응: Stage 1(노이즈 정규화)과 Stage 3(구두점)으로 보완

### 4.3 Lindera 통합 방법

```toml
# moshi/rust/moshi-backend/Cargo.toml
[dependencies]
lindera = { version = "0.28", features = ["ko-dic"] }
```

```rust
// moshi/rust/moshi-backend/src/korean_postprocess.rs
use lindera::tokenizer::Tokenizer;
use lindera::mode::Mode;
use lindera::DictionaryKind;

pub struct TextPostProcessor {
    enabled: bool,
    ko_tokenizer: Option<Tokenizer>,
    // 향후 확장: ja_tokenizer, zh_tokenizer
}

impl TextPostProcessor {
    pub fn new(enabled: bool, language: &str) -> Result<Self> {
        let ko_tokenizer = if enabled && language == "ko" {
            Some(Tokenizer::new(Mode::Normal, DictionaryKind::KoDic)?)
        } else {
            None
        };

        Ok(Self { enabled, ko_tokenizer })
    }

    pub fn process(&self, text: &str) -> String {
        if !self.enabled {
            return text.to_string();  // 비활성화 시 원본 반환
        }
        // ... 후처리 로직
    }
}
```

---

## 5. 다국어 지원 설계 (Multilingual Support)

### 5.1 언어별 처리 전략

| 언어 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|------|---------|---------|---------|---------|
| **한국어 (ko)** | 공통 | Lindera ko-dic | 한국어 패턴 | 한국어 ITN |
| **영어 (en)** | 공통 | 스킵 (이미 양호) | 영어 패턴 | 영어 ITN |
| **일본어 (ja)** | 공통 | Lindera ipadic | 일본어 패턴 | 일본어 ITN |
| **중국어 (zh)** | 공통 | Lindera cc-cedict | 중국어 패턴 | 중국어 ITN |

### 5.2 언어 감지 (Language Detection)

```rust
fn detect_language(text: &str) -> &'static str {
    let has_hangul = text.chars().any(|c| ('\u{AC00}'..='\u{D7AF}').contains(&c));
    let has_hiragana = text.chars().any(|c| ('\u{3040}'..='\u{309F}').contains(&c));
    let has_chinese = text.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c));

    if has_hangul { "ko" }
    else if has_hiragana { "ja" }
    else if has_chinese { "zh" }
    else { "en" }
}
```

### 5.3 영어 특별 처리

영어는 SentencePiece 토큰화가 이미 단어 경계를 잘 보존하므로:

```rust
fn process_english(text: &str) -> String {
    // Stage 2 스킵 - 띄어쓰기 교정 불필요
    let text = normalize_noise(text);       // Stage 1
    let text = restore_punctuation_en(text); // Stage 3
    text
}
```

---

## 6. 통합 아키텍처 (Integration Architecture)

### 6.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        K-MOSHI ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [학습 파이프라인 - Python]                                              │
│  ─────────────────────────                                               │
│  train.py                                                                │
│    └─ sample_saver.py                                                    │
│         └─ KoreanTextPostProcessor (선택적)                              │
│              • 학습 로그 가독성 향상용                                    │
│              • 설정: text_postprocessing.enabled                         │
│                                                                          │
│  [서빙 파이프라인 - Rust]                                                │
│  ──────────────────────                                                  │
│  moshi-backend/                                                          │
│    └─ stream_both.rs                                                     │
│         └─ text() 메소드                                                 │
│              └─ TextPostProcessor::process() (선택적)                    │
│                   • 런타임 토글 가능                                      │
│                   • 설정: config.text_postprocessing.enabled             │
│                                                                          │
│  [클라이언트 - JavaScript]                                               │
│  ────────────────────────                                                │
│  Web Client                                                              │
│    └─ 백업 후처리 (Rust에서 처리 안 된 경우)                             │
│         • 최소한의 정규화만 수행                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Rust 백엔드 통합 위치

```rust
// moshi/rust/moshi-backend/src/stream_both.rs

impl AppStateInner {
    fn text(
        &self,
        prev_text_token: u32,
        text_token: u32,
        config: &Config,
        post_processor: Option<&TextPostProcessor>,  // 추가
    ) -> Option<String> {
        // 기존 SentencePiece 디코딩
        let raw_text = self.text_tokenizer.decode_piece_ids(&[text_token]).ok()?;

        // 후처리 적용 (활성화된 경우)
        match post_processor {
            Some(pp) if pp.enabled => Some(pp.process(&raw_text)),
            _ => Some(raw_text),
        }
    }
}
```

---

## 7. 성능 예측 (Performance Estimation)

### 7.1 지연시간 분석

| 구성 | 총 지연시간 | Moshi 프레임 대비 | 권장 상황 |
|------|-------------|-------------------|-----------|
| 후처리 OFF | 0ms | 0% | 최저 지연 필요 시 |
| Stage 1만 | < 1ms | ~1% | 최소 정규화 |
| Stage 1+2 | 5-15ms | 6-19% | 일반 사용 |
| 전체 (1+2+3+4) | 7-20ms | 9-25% | 최고 품질 |

**Moshi 프레임 간격**: 80ms (12.5Hz)
→ 20ms 지연은 전체 지연의 25%이지만, 사용자 체감상 수용 가능 범위

### 7.2 메모리 사용량

| 구성요소 | 메모리 |
|----------|--------|
| Lindera ko-dic | ~50MB |
| ITN 매핑 테이블 | ~1MB |
| 정규표현식 캐시 | < 1MB |
| **총계** | ~52MB |

---

## 8. 구현 로드맵 (Implementation Roadmap)

### Phase 0: 즉시 적용 (Completed)
- [x] `character_level_interpolation: false` 설정
- [x] 문제 분석 및 문서화

### Phase 1: 학습 파이프라인 (Future)
- [ ] `sample_saver.py`에 `KoreanTextPostProcessor` 추가
- [ ] 설정 파일에 `text_postprocessing` 섹션 추가
- [ ] 단위 테스트 작성

### Phase 2: Rust 백엔드 (Future)
- [ ] `Cargo.toml`에 Lindera 의존성 추가
- [ ] `korean_postprocess.rs` 모듈 구현
- [ ] `stream_both.rs`에 통합
- [ ] 설정 파일 파싱 추가

### Phase 3: 다국어 확장 (Future)
- [ ] 영어 후처리 규칙 추가
- [ ] 일본어/중국어 지원 (Lindera 활용)
- [ ] 언어 자동 감지 구현

### Phase 4: 최적화 (Future)
- [ ] 벤치마크 및 프로파일링
- [ ] 캐싱 전략 구현
- [ ] 지연시간 최적화

---

## 9. 리스크 및 대응 방안 (Risks & Mitigations)

| 리스크 | 영향 | 대응 방안 |
|--------|------|-----------|
| Lindera 정확도 부족 | 중 | Stage 1/3로 보완, 사용자 피드백 수집 |
| 지연시간 초과 | 고 | 토글 옵션으로 비활성화 가능 |
| Rust 빌드 복잡성 증가 | 중 | Feature flag로 선택적 빌드 |
| 다국어 확장 시 복잡도 | 중 | 언어별 모듈 분리 |

---

## 10. 참고 자료 (References)

### 10.1 도구 및 라이브러리

- [Lindera - Rust Morphological Analyzer](https://github.com/lindera/lindera)
- [Kiwipiepy - Korean Morphological Analyzer](https://github.com/bab2min/kiwipiepy)
- [NVIDIA NeMo Text Processing](https://github.com/NVIDIA/NeMo-text-processing)
- [KoLM - Korean Text Normalization](https://github.com/scarletcho/KoLM)

### 10.2 관련 문서

- K-Moshi Korean Tokenizer Guide: `docs/KOREAN_TOKENIZER_GUIDE.md`
- Training Recipe Analysis: `docs/TRAINING_RECIPE_ANALYSIS.md`
- Training Recipe Analysis (Korean): `docs/TRAINING_RECIPE_ANALYSIS_KO.md`

### 10.3 관련 코드

- Interleaver: `finetune/data/interleaver.py`
- Sample Saver: `finetune/monitoring/sample_saver.py`
- Rust Backend: `moshi/rust/moshi-backend/src/stream_both.rs`

---

## 11. 결론 (Conclusion)

텍스트 후처리 기능은 K-Moshi의 Inner Monologue 가독성을 크게 향상시킬 수 있는 중요한 기능입니다. 그러나 현재 단계에서는 학습 파이프라인의 안정성을 우선시하여 Future Work로 분류합니다.

**핵심 권장사항**:
1. **즉시**: `character_level_interpolation: false`로 설정하여 근본 문제 해결
2. **추후**: Lindera 기반 후처리 파이프라인 구현 (토글 옵션 필수)
3. **확장**: 다국어 지원을 고려한 모듈화 설계

---

*Document Version: 1.0*
*Project: K-Moshi (Korean Moshi Finetuning)*
