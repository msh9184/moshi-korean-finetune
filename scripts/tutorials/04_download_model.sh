#!/bin/bash
# =============================================================================
# Step 4: Download Pretrained Model
# =============================================================================
# 실행: bash scripts/tutorials/04_download_model.sh
# =============================================================================

set -e  # 에러 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "Step 4: Download Pretrained Model"
echo "========================================"
echo "Project Root: $PROJECT_ROOT"
echo ""

# HuggingFace 캐시 디렉토리 설정 (선택사항)
# export HF_HOME=/path/to/cache

# 프록시 설정 (필요한 경우)
# export HTTP_PROXY=http://your-proxy:port
# export HTTPS_PROXY=http://your-proxy:port

cd "$PROJECT_ROOT"

# Python 스크립트 실행
python scripts/tutorials/04_download_model.py

echo ""
echo "Done! Next step: bash scripts/tutorials/05_train_single_gpu.sh"
