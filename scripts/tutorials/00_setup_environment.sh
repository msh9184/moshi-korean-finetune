#!/bin/bash
# =============================================================================
# Step 0: Setup Environment (GPU 서버용)
# =============================================================================
#
# 목적: moshi-finetune 프로젝트의 의존성을 설치합니다.
#
# 실행: bash scripts/tutorials/00_setup_environment.sh
#
# 주의: 이 스크립트는 GPU 서버에서 실행해야 합니다.
# =============================================================================

set -e  # 에러 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "Step 0: Setup Environment"
echo "========================================"
echo "Project Root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# 프록시 설정 (필요한 경우 주석 해제)
# export HTTP_PROXY=http://your-proxy:port
# export HTTPS_PROXY=http://your-proxy:port
# export NO_PROXY=localhost,127.0.0.1

echo "[1/4] Python 환경 확인..."
echo "========================================"
python --version
pip --version
echo ""

echo "[2/4] CUDA 환경 확인..."
echo "========================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "⚠️  nvidia-smi를 찾을 수 없습니다."
fi
echo ""

echo "[3/4] 프로젝트 의존성 설치..."
echo "========================================"
echo "moshi 패키지 및 기타 의존성을 설치합니다."
echo "GitHub에서 moshi를 다운로드하므로 시간이 걸릴 수 있습니다."
echo ""

# NumPy 버전 호환성 수정 (TensorFlow/TensorBoard 호환성)
echo "NumPy 버전 호환성 확인 및 수정..."
pip install "numpy<2.0" --quiet 2>/dev/null || true

# 방법 1: pip 사용
pip install -e . --no-build-isolation

# 설치 실패 시 대체 방법
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  설치 실패. 개별 패키지 설치를 시도합니다..."
    echo ""

    # moshi 먼저 설치
    pip install "moshi @ git+https://github.com/kyutai-labs/moshi.git#subdirectory=moshi"

    # 나머지 설치
    pip install -e . --no-build-isolation
fi

# TensorBoard 관련 NumPy 호환성 문제 해결
echo ""
echo "TensorBoard/TensorFlow 호환성 확인..."
pip install "numpy<2.0" --quiet 2>/dev/null || true

echo ""
echo "[4/4] 설치 확인..."
echo "========================================"

# moshi 모듈 확인
echo -n "moshi 모듈: "
if python -c "import moshi" 2>/dev/null; then
    echo "✅ OK"
else
    echo "❌ FAILED"
    echo "   moshi 모듈을 찾을 수 없습니다."
    exit 1
fi

# finetune 모듈 확인
echo -n "finetune 모듈: "
if python -c "import finetune" 2>/dev/null; then
    echo "✅ OK"
else
    echo "❌ FAILED"
    echo "   finetune 모듈을 찾을 수 없습니다."
    exit 1
fi

# torch 확인
echo -n "torch CUDA: "
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "✅ OK ($GPU_NAME)"
else
    echo "⚠️  CUDA를 사용할 수 없습니다."
fi

# sphn 확인
echo -n "sphn 모듈: "
if python -c "import sphn" 2>/dev/null; then
    echo "✅ OK"
else
    echo "❌ FAILED (pip install sphn==0.1.12)"
fi

echo ""
echo "========================================"
echo "✅ 환경 설정 완료!"
echo "========================================"
echo ""
echo "다음 단계:"
echo "  1. bash scripts/tutorials/01_download_toy_data.sh"
echo "  2. bash scripts/tutorials/02_explore_data.sh"
echo "  3. ..."
echo ""
