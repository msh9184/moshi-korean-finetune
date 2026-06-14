#!/bin/bash
# =============================================================================
# Step 6: Run Inference with Finetuned Model
# =============================================================================
# 실행: bash scripts/tutorials/06_inference.sh
#
# 이 스크립트는 추론 가이드를 표시합니다.
# 실제 추론은 Rust 서버 또는 Python 코드를 사용합니다.
# =============================================================================

set -e  # 에러 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "Step 6: Inference with Finetuned Model"
echo "========================================"
echo "Project Root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Python 스크립트 실행
python scripts/tutorials/06_inference.py

echo ""
echo "========================================"
echo "튜토리얼 완료!"
echo "========================================"
echo ""
echo "전체 과정 요약 (mpirun + DDP 설정):"
echo "  1. bash scripts/tutorials/01_download_toy_data.sh  # 데이터 다운로드"
echo "  2. bash scripts/tutorials/02_explore_data.sh       # 데이터 구조 분석"
echo "  3. bash scripts/tutorials/03_create_config.sh      # 설정 파일 생성 (DDP 백엔드)"
echo "  4. bash scripts/tutorials/04_download_model.sh     # 모델 다운로드"
echo "  5. bash scripts/tutorials/05_train_single_gpu.sh   # 학습 실행 (mpirun)"
echo "  6. bash scripts/tutorials/06_inference.sh          # 추론 가이드"
echo ""
echo "또는 train_mpi.sh 사용:"
echo "  bash scripts/train_mpi.sh --config configs/tutorials/toy_finetune.yaml --gpus 1"
echo ""
echo "자세한 내용은 README.md를 참조하세요."
