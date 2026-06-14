# Moshi Finetuning Tutorial Scripts

Moshi 파인튜닝 파이프라인을 단계별로 학습할 수 있는 튜토리얼 스크립트입니다.

## 개요

이 튜토리얼은 Moshi 음성 대화 모델을 파인튜닝하는 전체 과정을 7단계로 나누어 설명합니다:

0. **환경 설정** - 의존성 설치 (GPU 서버 필수)
1. **데이터 다운로드** - Toy 데이터셋 준비
2. **데이터 탐색** - 데이터 형식 및 구조 이해
3. **설정 생성** - 학습 설정 파일 작성
4. **모델 다운로드** - 사전학습 모델 준비
5. **학습 실행** - 단일 GPU에서 파인튜닝
6. **추론 실행** - 파인튜닝된 모델 사용

## 요구사항

### 하드웨어
- GPU: NVIDIA GPU (V100 32GB 이상 권장, A100 80GB 최적)
- RAM: 32GB 이상
- Storage: 50GB 이상 (모델 + 데이터)

### 소프트웨어
```bash
# Python 3.10+
pip install torch torchvision torchaudio
pip install huggingface_hub safetensors fire pyyaml
pip install sphn  # 오디오 처리

# 선택사항
pip install matplotlib  # 시각화
pip install wandb       # 실험 추적
```

## 빠른 시작 (mpirun + DDP 설정)

```bash
# 프로젝트 루트에서 실행
cd moshi-finetune

# 0. 환경 설정 (GPU 서버에서 최초 1회)
bash scripts/tutorials/00_setup_environment.sh

# 1. 데이터 다운로드
bash scripts/tutorials/01_download_toy_data.sh

# 2. 데이터 구조 탐색
bash scripts/tutorials/02_explore_data.sh

# 3. 설정 파일 생성 (DDP 백엔드로 설정됨)
bash scripts/tutorials/03_create_config.sh

# 4. 모델 다운로드
bash scripts/tutorials/04_download_model.sh

# 5. 학습 실행 (mpirun + DDP)
bash scripts/tutorials/05_train_single_gpu.sh --dry-run  # 설명만 보기
bash scripts/tutorials/05_train_single_gpu.sh            # 실제 학습

# 또는 train_mpi.sh 사용
bash scripts/train_mpi.sh --config configs/tutorials/toy_finetune.yaml --gpus 1

# 6. 추론 가이드
bash scripts/tutorials/06_inference.sh
```

## 스크립트 상세

### Step 1: Download Toy Data
```bash
bash scripts/tutorials/01_download_toy_data.sh
# 또는
python scripts/tutorials/01_download_toy_data.py
```

**출력:**
- `data/daily-talk-contiguous/` - DailyTalkContiguous 데이터셋
  - 스테레오 WAV 파일 (Left=AI, Right=User)
  - JSON 전사 파일 (단어별 타임스탬프)
  - JSONL 메타데이터

### Step 2: Explore Data Structure
```bash
bash scripts/tutorials/02_explore_data.sh
```

**학습 내용:**
- JSONL 메타데이터 형식
- JSON 전사 (alignment) 형식
- 스테레오 WAV 채널 구조
- Text-Audio 인터리빙 개념

### Step 3: Create Configuration
```bash
bash scripts/tutorials/03_create_config.sh
```

**출력:**
- `configs/tutorials/toy_finetune.yaml` - 튜토리얼용 설정
- `configs/tutorials/korean_finetune_template.yaml` - 한국어 파인튜닝 템플릿

**학습 내용:**
- TrainArgs 파라미터 설명
- LoRA (Low-Rank Adaptation) 개념
- GPU별 권장 설정

### Step 4: Download Pretrained Model
```bash
bash scripts/tutorials/04_download_model.sh
```

**출력:**
- `models/moshiko-pytorch-bf16/` - Moshi 7B 모델 (~14GB)
- `models/tokenizer-e351c8d8-checkpoint125.safetensors` - Mimi 코덱
- `models/tokenizer_spm_32k_3.model` - 텍스트 토크나이저

**학습 내용:**
- Moshi 모델 아키텍처
- Mimi 오디오 코덱 구조
- GPU 메모리 요구사항

### Step 5: Train with MPI + DDP
```bash
# 설명만 보기
bash scripts/tutorials/05_train_single_gpu.sh --dry-run

# 실제 학습 (mpirun + DDP, 기본값)
bash scripts/tutorials/05_train_single_gpu.sh

# 또는 train_mpi.sh 사용
bash scripts/train_mpi.sh --config configs/tutorials/toy_finetune.yaml --gpus 1

# torchrun 사용 (대안)
bash scripts/tutorials/05_train_single_gpu.sh --torchrun
```

**출력:**
- `runs/tutorial_toy/` - 학습 결과
  - `checkpoints/` - 모델 체크포인트
  - `train/metrics.jsonl` - 학습 로그

**학습 내용:**
- 분산 학습 백엔드 선택 (FSDP vs DDP)
- 런처 선택 (torchrun vs mpirun)
- 학습 모니터링 방법
- 트러블슈팅 가이드

**분산 학습 옵션:**
| 백엔드 | 런처 | 메모리 사용 | 권장 환경 |
|--------|------|-------------|-----------|
| FSDP | torchrun | 낮음 (분할) | 대규모 모델, 표준 PyTorch |
| DDP | torchrun/mpirun | 높음 (복제) | HPC 환경, MPI 기반 클러스터 |

### Step 6: Inference
```bash
bash scripts/tutorials/06_inference.sh
```

**학습 내용:**
- 체크포인트 구조
- LoRA 가중치 로딩
- Rust 서버 통합 방법
- 한국어 K-Moshi 배포 흐름

## 파일 구조

```
scripts/tutorials/
├── README.md                    # 이 파일
├── 01_download_toy_data.py      # 데이터 다운로드
├── 01_download_toy_data.sh
├── 02_explore_data.py           # 데이터 탐색
├── 02_explore_data.sh
├── 03_create_config.py          # 설정 생성
├── 03_create_config.sh
├── 04_download_model.py         # 모델 다운로드
├── 04_download_model.sh
├── 05_train_single_gpu.py       # 학습 가이드
├── 05_train_single_gpu.sh
├── 06_inference.py              # 추론 가이드
└── 06_inference.sh
```

## 한국어 파인튜닝 (K-Moshi)

튜토리얼 완료 후 한국어 Moshi를 구축하려면:

1. **데이터 준비**
   ```bash
   # 한국어 오디오 데이터 전처리
   python -m moshi_finetune.annotate \
       --input_dir /path/to/korean_audio \
       --output_dir /path/to/output \
       --lang ko \
       --device cuda
   ```

2. **설정 수정**
   ```bash
   # korean_finetune_template.yaml을 복사하여 수정
   cp configs/tutorials/korean_finetune_template.yaml configs/k-moshi.yaml
   # 경로 및 하이퍼파라미터 수정
   ```

3. **대규모 학습**
   ```bash
   # 방법 1: torchrun 사용 (FSDP 백엔드)
   torchrun --nproc_per_node=4 train.py --config configs/k-moshi.yaml

   # 방법 2: mpirun 사용 (DDP 백엔드 - HPC 환경 권장)
   bash scripts/train_mpi.sh --config configs/k-moshi.yaml --gpus 4

   # 다중 노드 MPI 학습
   bash scripts/train_mpi.sh --config configs/k-moshi.yaml \
       --hostfile scripts/hostfile
   ```

   **설정 파일에서 백엔드 선택:**
   ```yaml
   # FSDP (기본값) - 메모리 효율적
   distributed_backend: fsdp

   # DDP - MPI와 함께 사용 권장
   distributed_backend: ddp
   ```

4. **Rust 서버 배포**
   ```bash
   cd /path/to/moshi/rust
   cargo run --release --bin moshi-backend -- \
       --moshi-weights /path/to/consolidated.safetensors
   ```

자세한 내용은 프로젝트 루트의 `CLAUDE.md`를 참조하세요.

## 문제 해결

### CUDA Out of Memory
```yaml
# 설정 파일에서 조정
batch_size: 1      # 줄이기
duration: 30.0     # 줄이기
lora_rank: 8       # 줄이기
```

### 데이터 로딩 오류
- JSONL 경로 확인
- WAV 파일 유효성 확인
- `sphn` 라이브러리 설치 확인

### NCCL 통신 오류
```bash
export NCCL_DEBUG=INFO
# 단일 GPU로 먼저 테스트
torchrun --nproc_per_node=1 train.py --config ...
```

## 참고 자료

- [Moshi Paper](https://arxiv.org/abs/2410.00037)
- [Official Moshi Repository](https://github.com/kyutai-labs/moshi)
- [J-Moshi (Japanese)](https://github.com/nu-dialogue/j-moshi)
