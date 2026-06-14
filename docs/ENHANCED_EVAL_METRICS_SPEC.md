# K-Moshi Enhanced Evaluation Metrics 구현 명세서

## 1. 개요

### 1.1 목적
기존 평가 시스템(Perplexity, WER/CER)을 확장하여 Moshi 모델의 다차원적 성능을 평가할 수 있는 종합 평가 프레임워크 구축

### 1.2 현재 구현 상태

| 카테고리 | 메트릭 | 상태 | 파일 위치 |
|---------|--------|------|----------|
| 기본 손실 | Perplexity | ✅ 구현됨 | `eval.py` |
| 기본 손실 | Text/Audio Loss | ✅ 구현됨 | `eval.py`, `loss.py` |
| 텍스트 품질 | WER (Inner Monologue) | ✅ 구현됨 | `advanced_monitor.py` |
| 텍스트 품질 | CER (Inner Monologue) | ✅ 구현됨 | `advanced_monitor.py` |
| 코드북 분석 | Per-codebook Loss | ✅ 구현됨 | `advanced_monitor.py` |
| 코드북 분석 | Codebook Entropy | ✅ 구현됨 | `advanced_monitor.py` |
| 학습 건강 | Gradient Norm | ✅ 구현됨 | `advanced_monitor.py` |
| 학습 건강 | NaN/Inf Detection | ✅ 구현됨 | `advanced_monitor.py` |
| 오디오 품질 | PESQ | ❌ 미구현 | - |
| 오디오 품질 | STOI | ❌ 미구현 | - |
| 정렬 품질 | Text-Audio Alignment | ❌ 미구현 | - |
| 의미 품질 | BLEU Score | ❌ 미구현 | - |
| 의미 품질 | Semantic Similarity | ❌ 미구현 | - |
| 대화 품질 | Turn-Taking Analysis | ❌ 미구현 | - |

### 1.3 핵심 개념 정리

**WER/CER가 측정하는 것:**
- ❌ ASR(음성인식) 성능이 아님
- ✅ **Inner Monologue 예측 정확도**: Moshi가 생성하는 텍스트 스트림의 예측 품질
- Ground Truth: 학습 데이터의 text alignment
- Prediction: 모델의 text_logits에서 argmax로 추출한 토큰

**Perplexity가 측정하는 것:**
- `2^(text_loss + audio_loss)`: 전체 예측 불확실성
- 낮을수록 모델이 다음 토큰을 더 확신있게 예측

---

## 2. 아키텍처 설계

### 2.1 모듈 구조

```
finetune/monitoring/
├── __init__.py                    # 기존 + 새 모듈 export
├── metrics_logger.py              # TensorBoard/W&B 로깅 (기존)
├── advanced_monitor.py            # WER/CER, Codebook, Gradient (기존)
├── sample_saver.py                # 오디오/텍스트 샘플 저장 (기존)
├── research_logger.py             # 연구용 로깅 (기존)
│
├── audio_quality_monitor.py       # 🆕 PESQ, STOI, MCD
├── semantic_monitor.py            # 🆕 BLEU, Semantic Similarity
├── alignment_monitor.py           # 🆕 Text-Audio Alignment
├── dialogue_monitor.py            # 🆕 Turn-Taking, Response Quality
└── enhanced_evaluation.py         # 🆕 통합 평가 오케스트레이터
```

### 2.2 클래스 계층 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     EnhancedEvaluationOrchestrator                       │
│  - Coordinates all monitors                                              │
│  - Manages evaluation lifecycle                                          │
│  - Aggregates metrics for logging                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ AdvancedTraining │  │ AudioQuality     │  │ SemanticQuality  │       │
│  │ Monitor (기존)    │  │ Monitor (신규)   │  │ Monitor (신규)   │       │
│  │                  │  │                  │  │                  │       │
│  │ - WER/CER        │  │ - PESQ           │  │ - BLEU           │       │
│  │ - Codebook Loss  │  │ - STOI           │  │ - Semantic Sim   │       │
│  │ - Gradient       │  │ - MCD            │  │ - Consistency    │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐                              │
│  │ Alignment        │  │ Dialogue         │                              │
│  │ Monitor (신규)   │  │ Monitor (신규)   │                              │
│  │                  │  │                  │                              │
│  │ - Timing Acc     │  │ - Turn-Taking    │                              │
│  │ - Boundary Prec  │  │ - Response Rel.  │                              │
│  │ - Sync Score     │  │ - Flow Quality   │                              │
│  └──────────────────┘  └──────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 상세 구현 명세

### 3.1 Audio Quality Monitor

**파일:** `finetune/monitoring/audio_quality_monitor.py`

**의존성:**
```python
# requirements.txt 추가 필요
pesq>=0.0.4          # PESQ 계산
pystoi>=0.3.3        # STOI 계산
# 또는 torchaudio (일부 메트릭 지원)
```

**핵심 구현:**

```python
@dataclass
class AudioQualityResult:
    """Audio quality metrics result."""
    pesq_score: Optional[float] = None      # -0.5 ~ 4.5 (높을수록 좋음)
    stoi_score: Optional[float] = None      # 0 ~ 1 (높을수록 좋음)
    mcd_score: Optional[float] = None       # Mel Cepstral Distortion (낮을수록 좋음)
    snr_db: Optional[float] = None          # Signal-to-Noise Ratio
    sample_count: int = 0
    errors: List[str] = field(default_factory=list)


class AudioQualityMonitor:
    """
    오디오 품질 평가 모니터.

    Mimi 디코더로 복원된 오디오와 원본 오디오를 비교하여
    재합성 품질을 측정합니다.

    주의: PESQ/STOI는 계산 비용이 높으므로 eval_freq 간격으로만 실행
    """

    def __init__(
        self,
        mimi_model,                    # Mimi 디코더 (sample_saver에서 공유)
        sample_rate: int = 24000,
        enabled: bool = True,
        compute_pesq: bool = True,     # PESQ 계산 여부
        compute_stoi: bool = True,     # STOI 계산 여부
        compute_mcd: bool = True,      # MCD 계산 여부
        max_samples: int = 10,         # 평가할 최대 샘플 수
    ):
        self.mimi = mimi_model
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.compute_pesq = compute_pesq
        self.compute_stoi = compute_stoi
        self.compute_mcd = compute_mcd
        self.max_samples = max_samples

    def evaluate_batch(
        self,
        audio_codes_gt: torch.Tensor,      # [B, 8, T] Ground Truth 코드
        audio_codes_pred: torch.Tensor,    # [B, 8, T] 예측 코드 (logits에서 argmax)
        audio_mask: torch.Tensor,          # [B, 8, T] 유효 위치 마스크
    ) -> AudioQualityResult:
        """
        배치 오디오 품질 평가.

        과정:
        1. Mimi 디코더로 audio_codes → waveform 변환
        2. GT waveform vs Pred waveform 비교
        3. PESQ, STOI, MCD 계산
        """
        # 구현 세부사항 생략
        pass

    def _decode_to_waveform(self, codes: torch.Tensor) -> torch.Tensor:
        """Mimi 디코더로 코드를 waveform으로 변환."""
        with torch.no_grad():
            return self.mimi.decode(codes)

    def _compute_pesq(self, ref: np.ndarray, deg: np.ndarray) -> float:
        """PESQ 점수 계산 (16kHz 리샘플링 필요)."""
        from pesq import pesq
        # PESQ는 8kHz 또는 16kHz만 지원
        ref_16k = librosa.resample(ref, orig_sr=self.sample_rate, target_sr=16000)
        deg_16k = librosa.resample(deg, orig_sr=self.sample_rate, target_sr=16000)
        return pesq(16000, ref_16k, deg_16k, 'wb')  # wideband

    def _compute_stoi(self, ref: np.ndarray, deg: np.ndarray) -> float:
        """STOI 점수 계산."""
        from pystoi import stoi
        return stoi(ref, deg, self.sample_rate, extended=False)

    def _compute_mcd(self, ref: np.ndarray, deg: np.ndarray) -> float:
        """Mel Cepstral Distortion 계산."""
        # 구현: librosa로 MFCC 추출 후 유클리드 거리
        pass
```

**TensorBoard 메트릭:**
```yaml
eval.audio_quality/pesq: "PESQ 점수 (음질)"
eval.audio_quality/stoi: "STOI 점수 (명료도)"
eval.audio_quality/mcd: "MCD 점수 (스펙트럼 왜곡)"
eval.audio_quality/snr: "Signal-to-Noise Ratio (dB)"
```

---

### 3.2 Semantic Quality Monitor

**파일:** `finetune/monitoring/semantic_monitor.py`

**의존성:**
```python
# requirements.txt 추가 필요
sacrebleu>=2.3.1     # BLEU 계산
sentence-transformers>=2.2.0  # Semantic Similarity (선택적)
```

**핵심 구현:**

```python
@dataclass
class SemanticQualityResult:
    """Semantic quality metrics result."""
    bleu_score: float = 0.0           # 0 ~ 100 (높을수록 좋음)
    bleu_1: float = 0.0               # 1-gram precision
    bleu_2: float = 0.0               # 2-gram precision
    bleu_3: float = 0.0               # 3-gram precision
    bleu_4: float = 0.0               # 4-gram precision
    semantic_similarity: Optional[float] = None  # 0 ~ 1 (코사인 유사도)
    semantic_consistency: float = 0.0  # 문맥 일관성 점수
    sample_count: int = 0


class SemanticQualityMonitor:
    """
    의미적 품질 평가 모니터.

    Inner Monologue 텍스트의 의미적 품질을 평가합니다:
    - BLEU: n-gram 기반 텍스트 유사도
    - Semantic Similarity: 임베딩 기반 의미 유사도
    - Consistency: 문맥 일관성 (이전 발화와의 연속성)

    주의: WER/CER와 다른 관점
    - WER/CER: 토큰 레벨 정확도 (순서 중요)
    - BLEU: n-gram 겹침 비율 (순서 유연)
    - Semantic: 의미 보존 여부 (표현 유연)
    """

    def __init__(
        self,
        tokenizer,                     # SentencePiece 토크나이저
        enabled: bool = True,
        compute_bleu: bool = True,
        compute_semantic: bool = False,  # 계산 비용 높음
        semantic_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_samples: int = 50,
    ):
        self.tokenizer = tokenizer
        self.enabled = enabled
        self.compute_bleu = compute_bleu
        self.compute_semantic = compute_semantic

        # Semantic similarity model (lazy load)
        self._semantic_model = None
        self._semantic_model_name = semantic_model_name

    def evaluate_batch(
        self,
        references: List[str],    # Ground truth 텍스트
        hypotheses: List[str],    # 예측 텍스트
    ) -> SemanticQualityResult:
        """배치 의미 품질 평가."""
        pass

    def _compute_bleu(self, references: List[str], hypotheses: List[str]) -> dict:
        """BLEU 점수 계산."""
        from sacrebleu import corpus_bleu
        # sacrebleu는 reference를 리스트의 리스트로 받음
        refs = [[r] for r in references]
        result = corpus_bleu(hypotheses, refs)
        return {
            "bleu": result.score,
            "bleu_1": result.precisions[0],
            "bleu_2": result.precisions[1],
            "bleu_3": result.precisions[2],
            "bleu_4": result.precisions[3],
        }

    def _compute_semantic_similarity(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> float:
        """임베딩 기반 의미 유사도 계산."""
        if self._semantic_model is None:
            from sentence_transformers import SentenceTransformer
            self._semantic_model = SentenceTransformer(self._semantic_model_name)

        ref_embeddings = self._semantic_model.encode(references)
        hyp_embeddings = self._semantic_model.encode(hypotheses)

        # 코사인 유사도 평균
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = [
            cosine_similarity([r], [h])[0][0]
            for r, h in zip(ref_embeddings, hyp_embeddings)
        ]
        return np.mean(similarities)
```

**TensorBoard 메트릭:**
```yaml
eval.semantic/bleu: "BLEU 종합 점수"
eval.semantic/bleu_1: "Unigram Precision"
eval.semantic/bleu_2: "Bigram Precision"
eval.semantic/bleu_3: "Trigram Precision"
eval.semantic/bleu_4: "4-gram Precision"
eval.semantic/similarity: "Semantic Similarity (Embedding)"
```

---

### 3.3 Alignment Quality Monitor

**파일:** `finetune/monitoring/alignment_monitor.py`

**핵심 구현:**

```python
@dataclass
class AlignmentQualityResult:
    """Text-Audio alignment quality metrics."""
    timing_accuracy: float = 0.0      # 타이밍 정확도 (0~1)
    boundary_precision: float = 0.0   # 단어 경계 정밀도
    boundary_recall: float = 0.0      # 단어 경계 재현율
    sync_score: float = 0.0           # 동기화 점수
    avg_timing_error_ms: float = 0.0  # 평균 타이밍 오차 (ms)
    sample_count: int = 0


class AlignmentQualityMonitor:
    """
    Text-Audio 정렬 품질 평가 모니터.

    Inner Monologue 텍스트가 오디오와 얼마나 잘 동기화되어 있는지 평가합니다.

    평가 대상:
    1. Timing Accuracy: 텍스트 토큰이 올바른 시간 프레임에 배치되었는지
    2. Boundary Precision: 단어 시작/끝 경계의 정확도
    3. Sync Score: 전체적인 텍스트-오디오 동기화 품질

    활용:
    - Interleaver 품질 검증
    - alignment JSON의 품질 평가
    - 학습 데이터 품질 모니터링
    """

    def __init__(
        self,
        frame_rate: float = 12.5,      # Moshi 프레임 레이트 (12.5Hz)
        tolerance_frames: int = 2,      # 허용 오차 프레임 수
        enabled: bool = True,
    ):
        self.frame_rate = frame_rate
        self.frame_duration_ms = 1000.0 / frame_rate  # 80ms
        self.tolerance_frames = tolerance_frames
        self.enabled = enabled

    def evaluate_alignment(
        self,
        text_codes: torch.Tensor,           # [B, 1, T] 텍스트 토큰
        text_padding_id: int,               # 패딩 토큰 ID
        end_of_text_padding_id: int,        # EOP 토큰 ID
        alignments: List[List[tuple]],      # [("단어", (시작, 끝), "화자"), ...]
    ) -> AlignmentQualityResult:
        """
        정렬 품질 평가.

        과정:
        1. text_codes에서 실제 텍스트 토큰 위치 추출
        2. alignments에서 예상 텍스트 위치 계산
        3. 예측 vs 실제 위치 비교
        """
        pass

    def _extract_text_boundaries(
        self,
        text_codes: torch.Tensor,
        padding_ids: set,
    ) -> List[Tuple[int, int]]:
        """텍스트 토큰의 시작/끝 프레임 인덱스 추출."""
        pass

    def _alignments_to_frames(
        self,
        alignments: List[tuple],
    ) -> List[Tuple[int, int]]:
        """alignment 정보를 프레임 인덱스로 변환."""
        boundaries = []
        for word, (start_sec, end_sec), speaker in alignments:
            if speaker == "SPEAKER_MAIN":
                start_frame = int(start_sec * self.frame_rate)
                end_frame = int(end_sec * self.frame_rate)
                boundaries.append((start_frame, end_frame))
        return boundaries

    def _compute_timing_accuracy(
        self,
        predicted_boundaries: List[Tuple[int, int]],
        expected_boundaries: List[Tuple[int, int]],
    ) -> float:
        """타이밍 정확도 계산."""
        pass
```

**TensorBoard 메트릭:**
```yaml
eval.alignment/timing_accuracy: "타이밍 정확도"
eval.alignment/boundary_precision: "경계 정밀도"
eval.alignment/boundary_recall: "경계 재현율"
eval.alignment/boundary_f1: "경계 F1 점수"
eval.alignment/sync_score: "동기화 점수"
eval.alignment/avg_error_ms: "평균 타이밍 오차 (ms)"
```

---

### 3.4 Dialogue Quality Monitor

**파일:** `finetune/monitoring/dialogue_monitor.py`

**핵심 구현:**

```python
@dataclass
class DialogueQualityResult:
    """Dialogue quality metrics for full-duplex mode."""
    turn_taking_score: float = 0.0     # 턴테이킹 자연스러움 (0~1)
    overlap_ratio: float = 0.0         # 발화 중첩 비율
    response_latency_ms: float = 0.0   # 평균 응답 지연시간
    silence_ratio: float = 0.0         # 침묵 비율
    interruption_count: int = 0        # 끼어들기 횟수
    backchannels_detected: int = 0     # 맞장구 감지 횟수
    sample_count: int = 0


class DialogueQualityMonitor:
    """
    대화 품질 평가 모니터 (Full-Duplex 모드용).

    Moshi의 실시간 대화 능력을 평가합니다:
    - Turn-Taking: 적절한 발화 교대
    - Response Latency: 사용자 발화 후 Moshi 응답 시간
    - Overlap Handling: 동시 발화 처리 품질
    - Backchannels: 맞장구/상호작용 감지

    주의: Full-Duplex 모드에서만 의미있는 메트릭
    """

    def __init__(
        self,
        frame_rate: float = 12.5,
        enabled: bool = True,
        overlap_threshold_frames: int = 3,   # 중첩으로 간주할 최소 프레임
        silence_threshold_frames: int = 25,  # 침묵으로 간주할 최소 프레임 (2초)
    ):
        self.frame_rate = frame_rate
        self.enabled = enabled
        self.overlap_threshold = overlap_threshold_frames
        self.silence_threshold = silence_threshold_frames

    def evaluate_dialogue(
        self,
        moshi_audio_codes: torch.Tensor,    # [B, 8, T] Moshi 오디오
        user_audio_codes: torch.Tensor,     # [B, 8, T] User 오디오
        moshi_text_codes: torch.Tensor,     # [B, 1, T] Moshi 텍스트
        zero_token_id: int,                 # 무음 토큰 ID
    ) -> DialogueQualityResult:
        """
        대화 품질 평가.

        과정:
        1. 각 스트림에서 발화 구간 감지
        2. 턴테이킹 패턴 분석
        3. 응답 지연시간 계산
        4. 중첩/침묵 비율 계산
        """
        pass

    def _detect_speech_segments(
        self,
        audio_codes: torch.Tensor,
        zero_token_id: int,
    ) -> List[Tuple[int, int]]:
        """발화 구간 감지 (시작/끝 프레임)."""
        # zero_token_id가 아닌 구간을 발화로 간주
        pass

    def _compute_turn_taking_score(
        self,
        moshi_segments: List[Tuple[int, int]],
        user_segments: List[Tuple[int, int]],
    ) -> float:
        """턴테이킹 자연스러움 점수 계산."""
        # 자연스러운 턴테이킹 패턴 평가
        # - 중첩 최소화
        # - 적절한 응답 타이밍
        # - 길이 균형
        pass
```

**TensorBoard 메트릭:**
```yaml
eval.dialogue/turn_taking_score: "턴테이킹 점수"
eval.dialogue/overlap_ratio: "발화 중첩 비율"
eval.dialogue/response_latency_ms: "평균 응답 지연 (ms)"
eval.dialogue/silence_ratio: "침묵 비율"
eval.dialogue/interruption_count: "끼어들기 횟수"
```

---

## 4. 통합 구현

### 4.1 Enhanced Evaluation Orchestrator

**파일:** `finetune/monitoring/enhanced_evaluation.py`

```python
class EnhancedEvaluationOrchestrator:
    """
    Enhanced evaluation orchestrator.

    모든 평가 모니터를 통합 관리하고 메트릭을 집계합니다.
    """

    def __init__(
        self,
        args: TrainArgs,
        tokenizer,
        mimi_model,
        model_config: dict,
    ):
        # 기존 모니터
        self.advanced_monitor = AdvancedTrainingMonitor(
            tokenizer=tokenizer,
            text_padding_token_id=model_config["text_padding_token_id"],
            end_of_text_padding_id=model_config["end_of_text_padding_id"],
            num_codebooks=model_config.get("dep_q", 8),
            first_codebook_weight=args.first_codebook_weight_multiplier,
            config=getattr(args, "monitoring", {}),
        )

        # 새 모니터들 (설정에 따라 활성화)
        eval_config = getattr(args, "enhanced_evaluation", {})

        self.audio_quality_monitor = None
        if eval_config.get("audio_quality", {}).get("enabled", False):
            self.audio_quality_monitor = AudioQualityMonitor(
                mimi_model=mimi_model,
                sample_rate=24000,
                **eval_config.get("audio_quality", {}),
            )

        self.semantic_monitor = None
        if eval_config.get("semantic", {}).get("enabled", False):
            self.semantic_monitor = SemanticQualityMonitor(
                tokenizer=tokenizer,
                **eval_config.get("semantic", {}),
            )

        self.alignment_monitor = None
        if eval_config.get("alignment", {}).get("enabled", False):
            self.alignment_monitor = AlignmentQualityMonitor(
                frame_rate=12.5,
                **eval_config.get("alignment", {}),
            )

        self.dialogue_monitor = None
        if eval_config.get("dialogue", {}).get("enabled", False):
            self.dialogue_monitor = DialogueQualityMonitor(
                frame_rate=12.5,
                **eval_config.get("dialogue", {}),
            )

    def evaluate_batch(
        self,
        batch,           # Batch 객체
        model_output,    # 모델 출력
    ) -> Dict[str, Any]:
        """배치 종합 평가."""
        results = {}

        # 기존 평가
        text_result = self.advanced_monitor.evaluate_text(
            model_output.text_logits,
            batch.codes[:, :1],
            model_output.text_mask,
        )
        if text_result:
            results["text"] = text_result

        # 새 평가들
        if self.audio_quality_monitor:
            audio_result = self.audio_quality_monitor.evaluate_batch(...)
            results["audio_quality"] = audio_result

        if self.semantic_monitor:
            semantic_result = self.semantic_monitor.evaluate_batch(...)
            results["semantic"] = semantic_result

        # ... 기타 모니터들

        return results

    def get_all_metrics(self) -> Dict[str, float]:
        """모든 메트릭을 TensorBoard 로깅용 딕셔너리로 반환."""
        metrics = {}

        # 기존 모니터 메트릭
        metrics.update(self.advanced_monitor.get_metrics_dict())

        # 새 모니터 메트릭 추가
        if self.audio_quality_monitor:
            aq_summary = self.audio_quality_monitor.get_summary()
            for k, v in aq_summary.items():
                metrics[f"audio_quality/{k}"] = v

        # ... 기타

        return metrics
```

### 4.2 설정 스키마 (args.py 확장)

```python
@dataclass
class EnhancedEvaluationArgs(Serializable):
    """Enhanced evaluation configuration."""

    # Audio Quality (계산 비용 높음 - 기본 비활성화)
    audio_quality: dict = field(default_factory=lambda: {
        "enabled": False,
        "compute_pesq": True,
        "compute_stoi": True,
        "compute_mcd": True,
        "max_samples": 10,
    })

    # Semantic Quality
    semantic: dict = field(default_factory=lambda: {
        "enabled": True,
        "compute_bleu": True,
        "compute_semantic": False,  # sentence-transformers 필요
        "max_samples": 50,
    })

    # Alignment Quality
    alignment: dict = field(default_factory=lambda: {
        "enabled": True,
        "tolerance_frames": 2,
    })

    # Dialogue Quality (Full-Duplex 모드 전용)
    dialogue: dict = field(default_factory=lambda: {
        "enabled": False,  # Full-Duplex 모드에서만 활성화
        "overlap_threshold_frames": 3,
        "silence_threshold_frames": 25,
    })
```

### 4.3 YAML 설정 예시

```yaml
# example/korean_v3_fsdp.yaml 에 추가

enhanced_evaluation:
  # Audio Quality Metrics
  # 주의: PESQ/STOI는 계산 비용이 높으므로 신중하게 활성화
  audio_quality:
    enabled: false           # 기본 비활성화
    compute_pesq: true
    compute_stoi: true
    compute_mcd: true
    max_samples: 10          # 평가할 최대 샘플 수

  # Semantic Quality Metrics
  semantic:
    enabled: true
    compute_bleu: true
    compute_semantic: false  # sentence-transformers 필요
    max_samples: 50

  # Alignment Quality Metrics
  alignment:
    enabled: true
    tolerance_frames: 2      # 80ms × 2 = 160ms 허용 오차

  # Dialogue Quality Metrics (Full-Duplex 전용)
  dialogue:
    enabled: true            # V3 Full-Duplex 모드이므로 활성화
    overlap_threshold_frames: 3
    silence_threshold_frames: 25
```

---

## 5. TensorBoard 통합

### 5.1 메트릭 태그 체계

```yaml
# 계층적 태그 구조
eval.loss/total: "종합 손실"
eval.loss/text: "텍스트 손실"
eval.loss/audio: "오디오 손실"
eval.loss/perplexity: "Perplexity"

eval.text_quality/wer: "Word Error Rate"
eval.text_quality/cer: "Character Error Rate"
eval.text_quality/bleu: "BLEU Score"
eval.text_quality/bleu_1: "BLEU-1"
eval.text_quality/bleu_2: "BLEU-2"
eval.text_quality/bleu_3: "BLEU-3"
eval.text_quality/bleu_4: "BLEU-4"
eval.text_quality/semantic_sim: "Semantic Similarity"

eval.audio_quality/pesq: "PESQ Score"
eval.audio_quality/stoi: "STOI Score"
eval.audio_quality/mcd: "Mel Cepstral Distortion"

eval.alignment/timing_accuracy: "Timing Accuracy"
eval.alignment/boundary_f1: "Boundary F1"
eval.alignment/sync_score: "Sync Score"

eval.dialogue/turn_taking: "Turn-Taking Score"
eval.dialogue/response_latency: "Response Latency (ms)"
eval.dialogue/overlap_ratio: "Overlap Ratio"

eval.codebook/cb0_loss: "Codebook 0 (Semantic) Loss"
eval.codebook/cb1_loss: "Codebook 1 Loss"
# ... cb2-cb7

eval.gradient/norm: "Gradient Norm"
eval.gradient/nan_count: "NaN Gradient Count"
```

### 5.2 Custom Layouts 확장

```python
def _setup_enhanced_layouts(self):
    """Enhanced TensorBoard custom layouts."""
    layout = {
        "K-Moshi Training": {
            "Loss Overview": ["Multiline", [
                "eval.loss/total",
                "eval.loss/text",
                "eval.loss/audio",
            ]],
            "Perplexity": ["Multiline", ["eval.loss/perplexity"]],
        },
        "Text Quality": {
            "Error Rates": ["Multiline", [
                "eval.text_quality/wer",
                "eval.text_quality/cer",
            ]],
            "BLEU Scores": ["Multiline", [
                "eval.text_quality/bleu",
                "eval.text_quality/bleu_1",
                "eval.text_quality/bleu_2",
                "eval.text_quality/bleu_3",
                "eval.text_quality/bleu_4",
            ]],
        },
        "Audio Quality": {
            "Objective Metrics": ["Multiline", [
                "eval.audio_quality/pesq",
                "eval.audio_quality/stoi",
                "eval.audio_quality/mcd",
            ]],
        },
        "Alignment Quality": {
            "Timing": ["Multiline", [
                "eval.alignment/timing_accuracy",
                "eval.alignment/sync_score",
            ]],
            "Boundaries": ["Multiline", [
                "eval.alignment/boundary_f1",
            ]],
        },
        "Dialogue Quality": {
            "Turn-Taking": ["Multiline", [
                "eval.dialogue/turn_taking",
                "eval.dialogue/overlap_ratio",
            ]],
            "Latency": ["Multiline", [
                "eval.dialogue/response_latency",
            ]],
        },
        "Codebook Analysis": {
            "Per-Codebook Loss": ["Multiline", [
                f"eval.codebook/cb{i}_loss" for i in range(8)
            ]],
        },
        "Training Health": {
            "Gradients": ["Multiline", [
                "eval.gradient/norm",
            ]],
            "Learning Rate": ["Multiline", [
                "train.lr_tempformer",
                "train.lr_depformer",
            ]],
        },
    }
    self.summary_writer.add_custom_scalars(layout)
```

---

## 6. 구현 우선순위

### Phase 1: 핵심 메트릭 (1주)

1. **SemanticQualityMonitor** - BLEU 구현
   - 의존성: `sacrebleu` (가벼움)
   - 난이도: 낮음
   - 가치: 높음 (텍스트 생성 품질의 다른 관점)

2. **AlignmentQualityMonitor** - 기본 타이밍 검증
   - 의존성: 없음
   - 난이도: 중간
   - 가치: 높음 (데이터/Interleaver 품질 검증)

### Phase 2: 확장 메트릭 (1-2주)

3. **DialogueQualityMonitor** - 턴테이킹 분석
   - 의존성: 없음
   - 난이도: 중간
   - 가치: 높음 (Full-Duplex 모드 핵심)

4. **AudioQualityMonitor** - PESQ/STOI
   - 의존성: `pesq`, `pystoi` (설치 필요)
   - 난이도: 중간
   - 가치: 중간 (오디오 재합성 품질)

### Phase 3: 고급 메트릭 (선택적)

5. **Semantic Similarity** (sentence-transformers)
   - 의존성: 무거움 (별도 모델 로드)
   - 난이도: 낮음
   - 가치: 중간

---

## 7. 테스트 계획

### 7.1 단위 테스트

```python
# tests/test_enhanced_evaluation.py

def test_bleu_calculation():
    """BLEU 계산 정확성 테스트."""
    monitor = SemanticQualityMonitor(...)
    refs = ["안녕하세요 반갑습니다"]
    hyps = ["안녕하세요 반가워요"]
    result = monitor.evaluate_batch(refs, hyps)
    assert result.bleu_score > 0  # 부분 일치

def test_alignment_accuracy():
    """정렬 정확도 계산 테스트."""
    monitor = AlignmentQualityMonitor(frame_rate=12.5)
    # 완벽한 정렬 케이스
    # ... 테스트 케이스

def test_turn_taking_detection():
    """턴테이킹 감지 테스트."""
    monitor = DialogueQualityMonitor(frame_rate=12.5)
    # 명확한 턴테이킹 패턴
    # ... 테스트 케이스
```

### 7.2 통합 테스트

```bash
# 단기 학습으로 전체 파이프라인 테스트
./scripts/run_training_v3.sh --test --config example/korean_v3_fsdp.yaml
```

---

## 8. 기대 효과

| 메트릭 | 용도 | 기대 인사이트 |
|--------|------|---------------|
| BLEU | 텍스트 생성 다양성 | WER와 다른 관점의 텍스트 품질 |
| Alignment | 데이터 품질 검증 | Interleaver/전사 품질 문제 조기 발견 |
| Turn-Taking | 대화 자연스러움 | Full-Duplex 모드 효과 측정 |
| PESQ/STOI | 오디오 품질 | Mimi 코덱 + 모델 결합 품질 |

---

*문서 버전: 1.0*
*작성일: 2025-12-30*
*프로젝트: K-Moshi Enhanced Evaluation System*
