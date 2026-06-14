# Phase 1: 데이터 전처리 가이드

Korean Moshi 파인튜닝을 위한 데이터 전처리 파이프라인입니다.

## 개요

```
Lhotse Shar 데이터 (~15,289시간)
        ↓
   Phase 1 (CPU)
   - Speaker 역할 할당 (MAIN/USER)
   - 스테레오 변환 (L=MAIN, R=USER)
   - FLAC 압축 (50-60% 용량 절감)
        ↓
   스테레오 오디오 + 메타데이터
        ↓
   Phase 2 (GPU) - 추후 진행
   - Word-level alignment
```

---

## 빠른 시작

### 단일 머신 실행

```bash
cd /path/to/moshi-finetune

# 기본 실행 (FLAC 포맷, 16 workers)
python -m data_preparation.scripts.run_phase1 --parallel

# 특정 데이터셋만 처리
python -m data_preparation.scripts.run_phase1 \
    --dataset aihub-broadcast-key463-839g-train \
    --parallel

# 실행 전 미리보기 (dry-run)
python -m data_preparation.scripts.run_phase1 --dry-run
```

### WAV 포맷 사용 (FLAC 대신)

```bash
python -m data_preparation.scripts.run_phase1 \
    --audio-format wav \
    --parallel
```

---

## 분산 처리 (여러 머신)

### Step 1: 스크립트 생성

```bash
# 10대 머신용 스크립트 생성
python -m data_preparation.scripts.generate_distributed_scripts \
    --total-machines 10 \
    --output-dir ./run_scripts

# 생성되는 파일:
# ./run_scripts/
# ├── run_machine_000.sh
# ├── run_machine_001.sh
# ├── ...
# ├── run_machine_009.sh
# ├── master.sh          (실행 가이드)
# └── merge_results.py   (결과 병합용)
```

### Step 2: 각 머신에 배포 & 실행

```bash
# 머신 0에서
scp run_scripts/run_machine_000.sh user@machine0:/workspace/
ssh user@machine0 "cd /workspace && ./run_machine_000.sh"

# 머신 1에서
scp run_scripts/run_machine_001.sh user@machine1:/workspace/
ssh user@machine1 "cd /workspace && ./run_machine_001.sh"

# ... 나머지 머신들도 동일하게
```

또는 직접 실행:

```bash
# 머신 0에서 직접 실행
python -m data_preparation.scripts.run_phase1 \
    --machine-id 0 \
    --total-machines 10 \
    --parallel

# 머신 1에서 직접 실행
python -m data_preparation.scripts.run_phase1 \
    --machine-id 1 \
    --total-machines 10 \
    --parallel
```

### Step 3: 완료 확인 & 결과 병합

```bash
# 완료 상태 확인
python -m data_preparation.scripts.merge_phase1_results \
    --total-machines 10 \
    --verify-only

# 결과 병합
python -m data_preparation.scripts.merge_phase1_results \
    --total-machines 10
```

---

## 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--parallel` | 병렬 처리 활성화 | 비활성화 |
| `--num-workers N` | 워커 수 | 16 |
| `--machine-id N` | 현재 머신 ID (0부터 시작) | 0 |
| `--total-machines N` | 총 머신 수 | 1 |
| `--audio-format` | 출력 포맷 (`flac` / `wav`) | `flac` |
| `--dataset NAME` | 특정 데이터셋만 처리 | 전체 |
| `--no-resume` | 체크포인트 무시하고 처음부터 | 자동 resume |
| `--dry-run` | 실제 처리 없이 설정만 확인 | - |
| `--config PATH` | 설정 파일 경로 | 기본 설정 |

---

## 출력 구조

```
/path/to/data
└── aihub-broadcast-key463-839g/
    └── train/
        ├── audio/
        │   ├── conv_000001.flac    # 스테레오 (L=MAIN, R=USER)
        │   └── ...
        ├── metadata/
        │   ├── conv_000001.json    # 화자 정보, 세그먼트, 타임스탬프
        │   └── ...
        ├── manifest.jsonl          # 최종 매니페스트
        ├── stats.json              # 처리 통계
        └── .checkpoints/           # 체크포인트 (resume용)
```

---

## 체크포인트 & Resume

처리 중 중단되어도 자동으로 이어서 처리합니다:

```bash
# 중단 후 재실행하면 자동 resume
python -m data_preparation.scripts.run_phase1 --parallel

# 처음부터 다시 시작하려면
python -m data_preparation.scripts.run_phase1 --parallel --no-resume
```

---

## 예상 처리 시간

| 구성 | 데이터 | 예상 시간 |
|------|--------|----------|
| 1 머신, 16 workers | 15,289시간 | ~95시간 |
| 10 머신, 16 workers | 15,289시간 | ~10시간 |
| 20 머신, 16 workers | 15,289시간 | ~5시간 |

---

## 문제 해결

### 메모리 부족
```bash
# 워커 수 줄이기
python -m data_preparation.scripts.run_phase1 --parallel --num-workers 8
```

### 특정 데이터셋 재처리
```bash
# 체크포인트 삭제 후 재실행
rm -rf /output/path/train/.checkpoints/
python -m data_preparation.scripts.run_phase1 --dataset NAME --parallel
```

### 분산 처리 중 일부 머신 실패
```bash
# 실패한 머신만 다시 실행 (자동 resume)
./run_machine_003.sh

# 완료 후 병합
python -m data_preparation.scripts.merge_phase1_results --total-machines 10
```

---

## 다음 단계 (Phase 2)

Phase 1 완료 후, Phase 2에서 word-level alignment를 수행합니다.
(whisper-timestamped 또는 WhisperX 사용 예정)

```bash
# Phase 2는 추후 진행
python -m data_preparation.scripts.run_phase2 --help
```
