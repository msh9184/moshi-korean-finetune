#!/bin/bash
# =============================================================================
# Step 3: Create Training Configuration
# =============================================================================
# 실행: bash scripts/tutorials/03_create_config.sh
# =============================================================================

set -e  # 에러 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "Step 3: Create Training Configuration"
echo "========================================"
echo "Project Root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Python 스크립트 실행
python scripts/tutorials/03_create_config.py

echo ""
echo "Done! Next step: bash scripts/tutorials/04_download_model.sh"
