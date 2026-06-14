#!/bin/bash
# =============================================================================
# Step 1: Download Toy Dataset
# =============================================================================
# 실행: bash scripts/tutorials/01_download_toy_data.sh
# =============================================================================

set -e  # 에러 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "Step 1: Download Toy Dataset"
echo "========================================"
echo "Project Root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Python 스크립트 실행
python scripts/tutorials/01_download_toy_data.py

echo ""
echo "Done! Next step: bash scripts/tutorials/02_explore_data.sh"
