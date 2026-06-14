#!/bin/bash
# =============================================================================
# Step 5: Train with MPI + DDP
# =============================================================================
# 실행: bash scripts/tutorials/05_train_single_gpu.sh
#
# 옵션:
#   --dry-run    : 실제 학습 없이 설명만 출력
#   --config     : 다른 설정 파일 사용
#   --gpus       : GPU 개수 (기본: 1)
#   --torchrun   : torchrun 사용 (기본: mpirun)
# =============================================================================

set -e  # 에러 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 기본 설정
CONFIG_FILE="configs/tutorials/toy_finetune.yaml"
DRY_RUN=false
NUM_GPUS=1
USE_TORCHRUN=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --torchrun)
            USE_TORCHRUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--config <file>] [--gpus <n>] [--torchrun]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Step 5: Train with MPI + DDP"
echo "========================================"
echo "Project Root: $PROJECT_ROOT"
echo "Config File: $CONFIG_FILE"
echo "Dry Run: $DRY_RUN"
echo "Num GPUs: $NUM_GPUS"
echo "Launcher: $([ "$USE_TORCHRUN" = true ] && echo 'torchrun' || echo 'mpirun')"
echo ""

cd "$PROJECT_ROOT"

# 설명 스크립트 실행
python scripts/tutorials/05_train_single_gpu.py

# 설정 파일 존재 확인
if [ ! -f "$CONFIG_FILE" ]; then
    echo ""
    echo "❌ 설정 파일이 존재하지 않습니다: $CONFIG_FILE"
    echo ""
    echo "먼저 설정 파일을 생성하세요:"
    echo "  python scripts/tutorials/03_create_config.py"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "========================================"
    echo "Dry run mode - 실제 학습을 실행하지 않습니다."
    echo "학습을 시작하려면 --dry-run 없이 다시 실행하세요."
    echo "========================================"
    exit 0
fi

echo ""
echo "========================================"
echo "학습 시작"
echo "========================================"

# 환경변수 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 로그 디렉토리 생성
mkdir -p logs

# 학습 실행 명령어 구성
if [ "$USE_TORCHRUN" = true ]; then
    # torchrun 사용 (FSDP 또는 DDP)
    TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS train.py --config $CONFIG_FILE"
else
    # mpirun 사용 (DDP 권장)
    MASTER_ADDR=$(hostname)
    MASTER_PORT=${MASTER_PORT:-29500}

    TRAIN_CMD="mpirun -np $NUM_GPUS --npernode $NUM_GPUS"
    TRAIN_CMD="$TRAIN_CMD -x CUDA_VISIBLE_DEVICES=0"
    TRAIN_CMD="$TRAIN_CMD -x MASTER_ADDR=$MASTER_ADDR"
    TRAIN_CMD="$TRAIN_CMD -x MASTER_PORT=$MASTER_PORT"
    TRAIN_CMD="$TRAIN_CMD -x PYTHONUNBUFFERED=1"
    TRAIN_CMD="$TRAIN_CMD python train.py --config $CONFIG_FILE"
fi

echo ""
echo "실행 명령어:"
echo "  $TRAIN_CMD"
echo ""

# 확인 프롬프트
read -p "학습을 시작하시겠습니까? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "학습 시작..."
    eval "$TRAIN_CMD"

    echo ""
    echo "========================================"
    echo "학습 완료!"
    echo "========================================"
    echo "체크포인트: runs/tutorial_toy/"
    echo ""
    echo "다음 단계: bash scripts/tutorials/06_inference.sh"
else
    echo "학습이 취소되었습니다."
fi
