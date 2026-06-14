# K-Moshi Full-Duplex Stream Processing Analysis

**Date**: 2026-01-22
**Status**: 🚨 CRITICAL BUG IDENTIFIED

---

## 1. Executive Summary

### 발견된 문제
K-Moshi의 Full-Duplex 모드에서 **User Audio Stream (8개 codebook)이 Temporal Transformer에 입력되지 않고 있음**.

| 항목 | 기대값 | 실제값 | 상태 |
|------|--------|--------|------|
| Temporal TF 입력 스트림 | 17개 (1 text + 16 audio) | 9개 (1 text + 8 audio) | ❌ 버그 |
| audio_embs 개수 | n_q=16 | dep_q=8 | ❌ 버그 |
| User Audio 처리 | embedding 후 합산 | 완전 무시 | ❌ 버그 |

---

## 2. 아키텍처 비교 다이어그램

### 2.1 올바른 Full-Duplex 아키텍처 (Original Moshi / J-Moshi)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CORRECT FULL-DUPLEX ARCHITECTURE                         │
│                    (Original Moshi / J-Moshi 방식)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT STREAMS (17 codebooks)                                               │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  ┌─────────────┐                                                            │
│  │   Text      │  idx=0   ─────────► text_emb(codes[:,0])                  │
│  │  (Inner     │                           │                                │
│  │ Monologue)  │                           │                                │
│  └─────────────┘                           │                                │
│                                            │                                │
│  ┌─────────────┐                           │                                │
│  │ Moshi Audio │  idx=1   ─────────► audio_emb[0](codes[:,1])  ─┐          │
│  │  Codebook 0 │                                                 │          │
│  ├─────────────┤                                                 │          │
│  │ Moshi Audio │  idx=2   ─────────► audio_emb[1](codes[:,2])  ─┤          │
│  │  Codebook 1 │                                                 │          │
│  ├─────────────┤                                                 │          │
│  │    ...      │   ...                     ...                   │          │
│  ├─────────────┤                                                 ├─► Σ ────┐│
│  │ Moshi Audio │  idx=8   ─────────► audio_emb[7](codes[:,8])  ─┤         ││
│  │  Codebook 7 │                                                 │         ││
│  └─────────────┘                                                 │         ││
│                                                                  │         ││
│  ┌─────────────┐                                                 │         ││
│  │ User Audio  │  idx=9   ─────────► audio_emb[8](codes[:,9])  ─┤         ││
│  │  Codebook 0 │                                                 │         ││
│  ├─────────────┤                                                 │         ││
│  │ User Audio  │  idx=10  ─────────► audio_emb[9](codes[:,10]) ─┤         ││
│  │  Codebook 1 │                                                 │         ││
│  ├─────────────┤                                                 │         ││
│  │    ...      │   ...                     ...                   │         ││
│  ├─────────────┤                                                 │         ││
│  │ User Audio  │  idx=16  ─────────► audio_emb[15](codes[:,16])─┘         ││
│  │  Codebook 7 │                                                           ││
│  └─────────────┘                                                           ││
│                                                                            ││
│                                            ┌───────────────────────────────┘│
│                                            │                                │
│                                            ▼                                │
│  TEMPORAL TRANSFORMER INPUT                                                 │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│      combined_input = text_emb + Σ(audio_emb[0:16])                        │
│                           │              │                                  │
│                           │              └── 16개 audio embedding 모두 합산 │
│                           │                  (Moshi 8개 + User 8개)         │
│                           ▼                                                 │
│                  ┌─────────────────┐                                        │
│                  │    Temporal     │                                        │
│                  │   Transformer   │  ◄─── User context 포함!              │
│                  │   (Backbone)    │                                        │
│                  └────────┬────────┘                                        │
│                           │                                                 │
│                           ▼                                                 │
│                  ┌─────────────────┐                                        │
│                  │     Depth       │                                        │
│                  │   Transformer   │  ─────► Moshi Audio 8개 예측          │
│                  │   (Depformer)   │         (dep_q=8)                      │
│                  └─────────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 현재 K-Moshi 구현 (버그 있음)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CURRENT K-MOSHI IMPLEMENTATION                           │
│                    🚨 BUG: User Audio Not Processed!                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT STREAMS (17 codebooks)                                               │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│  ┌─────────────┐                                                            │
│  │   Text      │  idx=0   ─────────► text_emb(codes[:,0])                  │
│  │  (Inner     │                           │                                │
│  │ Monologue)  │                           │                                │
│  └─────────────┘                           │                                │
│                                            │                                │
│  ┌─────────────┐                           │                                │
│  │ Moshi Audio │  idx=1   ─────────► audio_emb[0](codes[:,1])  ─┐          │
│  │  Codebook 0 │                                                 │          │
│  ├─────────────┤                                                 │          │
│  │ Moshi Audio │  idx=2   ─────────► audio_emb[1](codes[:,2])  ─┤          │
│  │  Codebook 1 │                                                 │          │
│  ├─────────────┤                                                 │          │
│  │    ...      │   ...                     ...                   ├─► Σ ────┐│
│  ├─────────────┤                                                 │         ││
│  │ Moshi Audio │  idx=8   ─────────► audio_emb[7](codes[:,8])  ─┘         ││
│  │  Codebook 7 │                                                           ││
│  └─────────────┘                                                           ││
│                                                                            ││
│  ┌─────────────┐                                                           ││
│  │ User Audio  │  idx=9   ─────────► ❌ NO EMBEDDING!                      ││
│  │  Codebook 0 │          ╔═══════════════════════════════════╗            ││
│  ├─────────────┤          ║  audio_embs has only 8 elements!  ║            ││
│  │ User Audio  │  idx=10  ║  No audio_emb[8] ~ audio_emb[15]  ║            ││
│  │  Codebook 1 │          ║                                   ║            ││
│  ├─────────────┤          ║  User audio streams are           ║            ││
│  │    ...      │   ...    ║  COMPLETELY IGNORED!              ║            ││
│  ├─────────────┤          ╚═══════════════════════════════════╝            ││
│  │ User Audio  │  idx=16  ─────────► ❌ NO EMBEDDING!                      ││
│  │  Codebook 7 │                                                           ││
│  └─────────────┘                                                           ││
│                                                                            ││
│                                            ┌───────────────────────────────┘│
│                                            │                                │
│                                            ▼                                │
│  TEMPORAL TRANSFORMER INPUT                                                 │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                             │
│      combined_input = text_emb + Σ(audio_emb[0:8])   ◄── ONLY 8!           │
│                           │              │                                  │
│                           │              └── Moshi 8개만 합산               │
│                           │                  User 8개 누락!                 │
│                           ▼                                                 │
│                  ┌─────────────────┐                                        │
│                  │    Temporal     │                                        │
│                  │   Transformer   │  ◄─── User context 없음!              │
│                  │   (Backbone)    │        🚨 대화 맥락 손실!             │
│                  └────────┬────────┘                                        │
│                           │                                                 │
│                           ▼                                                 │
│                  ┌─────────────────┐                                        │
│                  │     Depth       │                                        │
│                  │   Transformer   │  ─────► Moshi Audio 8개 예측          │
│                  │   (Depformer)   │         (dep_q=8)                      │
│                  └─────────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 코드 레벨 비교

### 3.1 Original Moshi (`moshi/models/lm.py`)

```python
# Line 135-137: n_q개의 embedding 생성
self.emb = nn.ModuleList(
    [EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]  # n_q = 16!
)

# Line 390-394: 모든 n_q audio codebook embedding
for cb_index in range(self.num_audio_codebooks):  # n_q = 16
    audio_emb = self.emb[cb_index](input_sequence[:, cb_index + self.audio_offset])
    input_ = audio_emb if input_ is None else input_ + audio_emb

# Line 294-295: num_audio_codebooks = n_q
@property
def num_audio_codebooks(self) -> int:
    return self.n_q  # 16 in full-duplex mode
```

### 3.2 J-Moshi (`finetune.py`)

```python
# Line 363-368: 모든 n_q audio codebook embedding
def tempformer_forward(moshi_lm: MoshiForFinetuning, batch: Batch):
    text_emb = moshi_lm.text_emb(batch.input_ids[:, 0])
    audio_emb = None
    for acb_index in range(moshi_lm.num_audio_codebooks):  # n_q = 16
        audio_emb_ = moshi_lm.emb[acb_index](
            batch.input_ids[:, moshi_lm.audio_offset + acb_index]
        )
        audio_emb = audio_emb_ if audio_emb is None else audio_emb + audio_emb_
    tempformer_input = text_emb + audio_emb  # 16개 모두 포함!
```

### 3.3 K-Moshi (`finetune/backbone/lm_model_wrapper.py`) - 🚨 버그

```python
# Line 564-577: num_audio_embs = dep_q (not n_q!)
@property
def num_audio_embs(self) -> int:
    """Number of audio embeddings available for processing.
    This is typically dep_q (8), not n_q (16 for full-duplex)."""
    return len(self.audio_embs)  # Returns 8, should return 16!

# Line 731-738: Only 8 audio codebooks embedded
n_audio_embs = self.num_audio_embs  # 8, not 16!
for cb_index in range(n_audio_embs):  # 0~7 only, missing 8~15!
    audio_codes = input_sequence[:, cb_index + self._audio_offset]
    audio_emb = self.audio_embs[cb_index](audio_codes)
    audio_input = audio_emb if audio_input is None else audio_input + audio_emb

# Line 745: Missing user audio!
combined_input = text_emb if audio_input is None else text_emb + audio_input
```

---

## 4. 데이터 흐름 타임라인

### 4.1 올바른 Full-Duplex 데이터 흐름

```
Time ──────────────────────────────────────────────────────────────────►
        t=0        t=1        t=2        t=3        t=4

Input Codes [B, 17, T]:
┌───────────────────────────────────────────────────────────────────────┐
│ [0]  Text        │ [안]    │ [녕]    │ [PAD]   │ [하]    │ [세]    │
├───────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ [1]  Moshi A0    │ C0,0    │ C0,1    │ C0,2    │ C0,3    │ C0,4    │
│ [2]  Moshi A1    │ C1,0    │ C1,1    │ C1,2    │ C1,3    │ C1,4    │
│      ...         │  ...    │  ...    │  ...    │  ...    │  ...    │
│ [8]  Moshi A7    │ C7,0    │ C7,1    │ C7,2    │ C7,3    │ C7,4    │
├───────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ [9]  User A0     │ U0,0    │ U0,1    │ U0,2    │ U0,3    │ U0,4    │ ◄── 처리되어야 함!
│ [10] User A1     │ U1,0    │ U1,1    │ U1,2    │ U1,3    │ U1,4    │
│      ...         │  ...    │  ...    │  ...    │  ...    │  ...    │
│ [16] User A7     │ U7,0    │ U7,1    │ U7,2    │ U7,3    │ U7,4    │
└───────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

                              │
                              ▼

Embedding (올바른 구현):
┌───────────────────────────────────────────────────────────────────────┐
│ text_emb    = text_emb_layer(codes[:,0])         → [B, T, D]        │
│ moshi_emb   = Σ(audio_emb[i](codes[:,1+i]))      → [B, T, D] (i=0~7)│
│ user_emb    = Σ(audio_emb[8+i](codes[:,9+i]))    → [B, T, D] (i=0~7)│ ◄── 필요!
│                                                                      │
│ combined    = text_emb + moshi_emb + user_emb    → [B, T, D]        │
└───────────────────────────────────────────────────────────────────────┘

                              │
                              ▼

Temporal Transformer:
┌───────────────────────────────────────────────────────────────────────┐
│ tempformer_output = temporal_transformer(combined) → [B, T, D]       │
│                                                                      │
│ ✅ User context 포함 → 대화 맥락 이해 가능                          │
└───────────────────────────────────────────────────────────────────────┘
```

### 4.2 현재 K-Moshi 데이터 흐름 (버그)

```
Time ──────────────────────────────────────────────────────────────────►
        t=0        t=1        t=2        t=3        t=4

Input Codes [B, 17, T]:
┌───────────────────────────────────────────────────────────────────────┐
│ [0]  Text        │ [안]    │ [녕]    │ [PAD]   │ [하]    │ [세]    │
├───────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ [1]  Moshi A0    │ C0,0    │ C0,1    │ C0,2    │ C0,3    │ C0,4    │ ✅
│ [2]  Moshi A1    │ C1,0    │ C1,1    │ C1,2    │ C1,3    │ C1,4    │ ✅
│      ...         │  ...    │  ...    │  ...    │  ...    │  ...    │ ✅
│ [8]  Moshi A7    │ C7,0    │ C7,1    │ C7,2    │ C7,3    │ C7,4    │ ✅
├───────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ [9]  User A0     │ U0,0    │ U0,1    │ U0,2    │ U0,3    │ U0,4    │ ❌ 무시됨!
│ [10] User A1     │ U1,0    │ U1,1    │ U1,2    │ U1,3    │ U1,4    │ ❌ 무시됨!
│      ...         │  ...    │  ...    │  ...    │  ...    │  ...    │ ❌ 무시됨!
│ [16] User A7     │ U7,0    │ U7,1    │ U7,2    │ U7,3    │ U7,4    │ ❌ 무시됨!
└───────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

                              │
                              ▼

Embedding (현재 K-Moshi):
┌───────────────────────────────────────────────────────────────────────┐
│ text_emb    = text_emb_layer(codes[:,0])         → [B, T, D]        │
│ moshi_emb   = Σ(audio_emb[i](codes[:,1+i]))      → [B, T, D] (i=0~7)│
│ user_emb    = ❌ MISSING! (audio_embs only has 8 elements)          │
│                                                                      │
│ combined    = text_emb + moshi_emb               → [B, T, D]        │
│              (user_emb 누락!)                                        │
└───────────────────────────────────────────────────────────────────────┘

                              │
                              ▼

Temporal Transformer:
┌───────────────────────────────────────────────────────────────────────┐
│ tempformer_output = temporal_transformer(combined) → [B, T, D]       │
│                                                                      │
│ ❌ User context 없음 → 대화 맥락 이해 불가!                         │
│ 🚨 모델이 User가 무슨 말을 했는지 알 수 없음!                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 5. 영향 분석

### 5.1 현재 문제의 영향

1. **대화 맥락 손실**
   - User의 발화 내용이 Temporal Transformer에 전달되지 않음
   - 모델이 User가 무슨 말을 했는지 알 수 없음
   - 턴테이킹, 응답 생성에 심각한 영향

2. **학습 효과 감소**
   - Full-Duplex 모드의 핵심인 양방향 대화 학습 불가
   - 사실상 Monologue 모드와 동일한 학습 효과

3. **추론 시 문제**
   - 실시간 대화에서 User 입력에 적절히 반응 불가
   - 대화 품질 저하

### 5.2 원인 분석

```python
# 근본 원인: audio_embs 초기화 시 dep_q 사용

# LMModelWrapper.__init__에서:
self.audio_embs = nn.ModuleList([
    nn.Embedding(...)
    for _ in range(dep_q)  # 8개만 생성!
])

# 올바른 구현:
self.audio_embs = nn.ModuleList([
    nn.Embedding(...)
    for _ in range(n_q)  # 16개 생성해야 함!
])
```

---

## 6. 수정 방안

### 6.1 방안 A: audio_embs를 n_q개로 확장 (권장)

```python
# lm_model_wrapper.py 수정

class LMModelWrapper(nn.Module):
    def __init__(self, ..., n_q: int = 8, dep_q: int = 8, ...):
        # audio_embs를 n_q개로 생성 (dep_q가 아닌!)
        self.audio_embs = nn.ModuleList([
            nn.Embedding(card + 1, dim)
            for _ in range(n_q)  # Full-duplex: 16, Monologue: 8
        ])

        # 기존 속성 유지
        self.n_q = n_q
        self.dep_q = dep_q

    @property
    def num_audio_embs(self) -> int:
        """Returns n_q (total audio streams), not dep_q."""
        return len(self.audio_embs)  # Now returns n_q

    def forward_text(self, ...):
        # 모든 n_q audio codebook embedding
        for cb_index in range(self.num_audio_embs):  # 0~15 (n_q)
            audio_codes = input_sequence[:, cb_index + self._audio_offset]
            audio_emb = self.audio_embs[cb_index](audio_codes)
            audio_input = audio_emb if audio_input is None else audio_input + audio_emb
```

### 6.2 방안 B: 별도 user_audio_embs 추가

```python
# 기존 audio_embs 유지하고 user_audio_embs 추가

class LMModelWrapper(nn.Module):
    def __init__(self, ..., n_q: int = 8, dep_q: int = 8, ...):
        # Moshi audio embeddings (예측 대상)
        self.audio_embs = nn.ModuleList([
            nn.Embedding(card + 1, dim)
            for _ in range(dep_q)  # 8
        ])

        # User audio embeddings (컨텍스트용)
        self.user_audio_embs = nn.ModuleList([
            nn.Embedding(card + 1, dim)
            for _ in range(n_q - dep_q)  # 8 (full-duplex에서)
        ])

    def forward_text(self, ...):
        # Moshi audio embedding
        for cb_index in range(len(self.audio_embs)):
            ...

        # User audio embedding (full-duplex mode only)
        if len(self.user_audio_embs) > 0:
            for cb_index in range(len(self.user_audio_embs)):
                user_audio_codes = input_sequence[:,
                    cb_index + self._audio_offset + len(self.audio_embs)]
                user_audio_emb = self.user_audio_embs[cb_index](user_audio_codes)
                audio_input = audio_input + user_audio_emb
```

---

## 7. 검증 결과 요약

| 검증 항목 | 결과 | 상세 |
|-----------|------|------|
| Temporal TF 입력 형태 | ❌ 불완전 | text_emb + 8개 audio_emb (User 누락) |
| User Audio Stream 처리 | ❌ 미처리 | audio_embs가 8개만 있어 indices 9-16 무시 |
| Original Moshi 대비 | ❌ 불일치 | Moshi는 n_q개 audio embedding 사용 |
| J-Moshi 대비 | ❌ 불일치 | J-Moshi도 n_q개 audio embedding 사용 |

---

## 8. 권장 조치

1. **즉시**: `lm_model_wrapper.py`에서 `audio_embs`를 `n_q`개로 확장
2. **검증**: 기존 체크포인트와의 호환성 확인 (추가 embedding 초기화 필요)
3. **테스트**: Full-duplex 모드 학습 후 대화 품질 검증

---

*Document created: 2026-01-22*
*Author: K-Moshi Development Team*
