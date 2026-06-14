# K-Moshi Phase 1 Training Guide

Korean Moshi 파인튜닝을 위한 완전 가이드입니다.

## 개요

### Phase 1 목표
- Moshi LLM + Moshiko vocabulary (SentencePiece 32K)
- FSDP/mpirun을 사용한 분산 학습
- AIHub broadcast 데이터셋 활용

### 데이터 경로
```
/path/to/data
├── audio/                      # 스테레오 오디오 (L=MAIN, R=USER)
├── alignment_speaker01/        # Speaker01 word-level alignments
├── alignments/                 # Moshi 형식 combined alignments
└── manifest.jsonl              # 메타데이터
```

---

## Step 1: 환경 준비

### 1.1 GPU 서버 접속 및 코드 동기화

```bash
# GPU 서버에 코드 동기화
rsync -avz --exclude '.git' --exclude '__pycache__' \
    /path/to/moshi-korean-finetune/ \
    user@gpu-server:/path/to

# GPU 서버 접속
ssh user@gpu-server
cd /path/to
```

### 1.2 의존성 설치

```bash
# Python 환경 활성화
source /path/to/venv/bin/activate

# 의존성 설치
pip install -e .

# 추가 필요 패키지
pip install fire wandb soundfile
```

### 1.3 모델 다운로드

```bash
# Moshi 모델 다운로드 (이미 완료된 경우 건너뛰기)
python scripts/download_models.py \
    --output-dir /path/to/model

# 다운로드 확인
ls -la /path/to/model
# 출력:
# - model.safetensors          (~14GB)
# - tokenizer-e351c8d8-...     (~82MB)
# - tokenizer_spm_32k_3.model  (~0.5MB)
# - default_config.json
```

---

## Step 2: 데이터 준비

### 2.1 데이터셋 검증

```bash
# 데이터셋 유효성 검사
python scripts/prepare_training_data.py \
    --input-dir /path/to/data \
    --validate-only

# 예상 출력:
# ============================================================
# Validation Results
# ============================================================
# total: 15,289
# valid: 15,250
# invalid: 39
# total_words: 2,500,000
# total_duration_hours: 839.00h
```

### 2.2 학습용 JSONL 생성

```bash
# 학습 데이터 JSONL 생성
python scripts/prepare_training_data.py \
    --input-dir /path/to/data \
    --output-jsonl ./data/korean_phase1.jsonl \
    --alignment-dir alignments \
    --min-duration 2.0 \
    --max-duration 300.0

# 생성 확인
head -5 ./data/korean_phase1.jsonl
# {"path": "/path/to", "duration": 45.32}
# {"path": "/path/to", "duration": 38.15}
# ...
```

### 2.3 데이터 형식 이해

#### JSONL 형식 (manifest)
```json
{"path": "/abs/path/to/audio.wav", "duration": 45.32}
```

#### Alignment JSON 형식 (alignments/)
```json
{
  "alignments": [
    ["안녕하세요", [0.0, 0.85], "SPEAKER_MAIN"],
    ["네", [1.2, 1.4], "SPEAKER_USER"],
    ["안녕하세요", [1.5, 2.1], "SPEAKER_USER"]
  ]
}
```

---

## Step 3: 설정 파일 수정

### 3.1 경로 확인 및 수정

`example/korean_phase1_fsdp.yaml` 파일을 열고 경로를 확인합니다:

```yaml
# 데이터 경로
data:
  train_data: './data/korean_phase1.jsonl'

# 모델 경로 (실제 경로로 수정)
moshi_paths:
  hf_repo_id: null
  moshi_path: '/path/to/model'
  mimi_path: '/path/to/model'
  tokenizer_path: '/path/to/model'
  config_path: '/path/to/model'

# 출력 디렉토리
run_dir: './runs/korean_phase1_v1'
```

### 3.2 하이퍼파라미터 조정 (선택사항)

```yaml
# 메모리 제약이 있는 경우
duration_sec: 30       # 기본 60 → 30
batch_size: 1          # 기본 2 → 1
num_microbatches: 8    # 기본 4 → 8 (effective batch size 유지)

# 학습률 조정
optim:
  lr: 5.0e-6           # 더 보수적인 학습률
```

---

## Step 4: 학습 실행

### 4.1 FSDP + mpirun (권장)

```bash
# 단일 노드, 4 GPU
mpirun -np 4 python -m train example/korean_phase1_fsdp.yaml

# 백그라운드 실행 (로그 저장)
nohup mpirun -np 4 python -m train example/korean_phase1_fsdp.yaml \
    > logs/phase1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 4.2 FSDP + torchrun

```bash
# 단일 노드, 4 GPU
torchrun --nproc-per-node 4 -m train example/korean_phase1_fsdp.yaml

# 포트 충돌 시
torchrun --nproc-per-node 4 --master_port 29501 -m train example/korean_phase1_fsdp.yaml
```

### 4.3 단일 GPU 테스트

```bash
# 단일 GPU 테스트 (설정 확인용)
CUDA_VISIBLE_DEVICES=0 python -m train example/korean_phase1_fsdp.yaml
```

### 4.4 학습 모니터링

```bash
# 로그 확인
tail -f logs/phase1_*.log

# GPU 사용량 확인
watch -n 1 nvidia-smi

# W&B 동기화 (오프라인 모드 사용 시)
cd ./runs/korean_phase1_v1
wandb sync --sync-all
```

---

## Step 5: 체크포인트 관리

### 5.1 체크포인트 구조

```
runs/korean_phase1_v1/
├── args.yaml                  # 학습 설정 백업
├── checkpoints/
│   ├── step_500/
│   │   ├── model.safetensors  # 모델 가중치
│   │   └── optimizer.pt       # 옵티마이저 상태
│   ├── step_1000/
│   └── ...
├── train.jsonl                # 학습 로그
└── wandb/                     # W&B 로그 (오프라인)
```

### 5.2 학습 재개

```bash
# 체크포인트에서 재개 (자동 감지)
# TODO: resume 기능 구현 필요
```

### 5.3 체크포인트 검증

```python
import torch
from safetensors import safe_open

# 체크포인트 로드 테스트
ckpt_path = "./runs/korean_phase1_v1/checkpoints/step_1000/model.safetensors"
with safe_open(ckpt_path, framework="pt") as f:
    keys = list(f.keys())
    print(f"Parameters: {len(keys)}")
    print(f"Sample keys: {keys[:5]}")
```

---

## 예상 리소스 사용량

### A100 80GB 기준

| 설정 | batch_size | duration_sec | VRAM | 예상 속도 |
|------|------------|--------------|------|----------|
| 보수적 | 1 | 30 | ~40GB | ~1000 steps/hr |
| 균형 | 2 | 60 | ~60GB | ~800 steps/hr |
| 적극적 | 4 | 60 | ~75GB | ~600 steps/hr |

### 전체 학습 시간 예상

- **10,000 steps, 4x A100**: ~10-15시간
- **데이터 839시간, 1 epoch**: ~15,000 steps

---

## 트러블슈팅

### CUDA Out of Memory

```yaml
# 해결책 1: 배치 크기 및 시퀀스 길이 감소
duration_sec: 30
batch_size: 1
num_microbatches: 8

# 해결책 2: Gradient checkpointing 확인
gradient_checkpointing: true
```

### MPI 초기화 실패

```bash
# OpenMPI 환경 변수 설정
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_btl_tcp_if_include=eth0

mpirun -np 4 --mca btl self,tcp python -m train ...
```

### Alignment 파일 찾기 실패

```bash
# 경로 해석 문제 디버깅
python -c "
import json
from pathlib import Path

input_dir = Path('/path/to/data')
with open(input_dir / 'manifest.jsonl') as f:
    entry = json.loads(f.readline())
    print('Entry:', entry)

    # 오디오 경로
    audio_path = input_dir / entry['audio']
    print('Audio exists:', audio_path.exists())

    # Alignment 경로
    align_path = input_dir / 'alignments' / (audio_path.stem + '.json')
    print('Alignment exists:', align_path.exists())
"
```

### NaN Loss 발생

```bash
# 진단 로그 확인
grep -i "nan\|inf" logs/phase1_*.log

# 학습률 감소
optim:
  lr: 1.0e-6  # 더 낮은 학습률

# Gradient clipping 확인
max_norm: 0.5  # 더 엄격한 클리핑
```

---

## 다음 단계

### Phase 2: KLUE Vocabulary

Phase 1 완료 후, KLUE BERT vocabulary로 전환:

```yaml
korean:
  korean_tokenizer_path: '/path/to/model'
  korean_tokenizer_type: 'klue'
```

### User Stream 학습

User stream을 포함한 17 codebook 학습:

```yaml
korean:
  enable_user_stream: true
  initialized_model_path: './models/k-moshi-init'
```

---

## 참고 자료

- [K-Moshi Implementation Plan](../README.md)
- [Data Preparation Guide](../data_preparation/README.md)
- [Moshi Paper](https://arxiv.org/abs/2410.00037)
- [Original moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)

---

*Last Updated: 2025-12-25*
