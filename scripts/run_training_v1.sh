#!/bin/bash
# =============================================================================
# K-Moshi Version 1 Training Script
# =============================================================================
#
# Korean Moshi Full Finetuning with mpirun + FSDP
#
# Environment:
#   - 1 Node x 8 GPU (NVIDIA A100 80GB)
#   - mpirun distributed launcher
#   - FSDP model sharding
#
# Usage:
#   ./scripts/run_training_v1.sh                    # Default: 8 GPUs
#   ./scripts/run_training_v1.sh --gpus 4           # Use 4 GPUs
#   ./scripts/run_training_v1.sh --config custom.yaml
#   ./scripts/run_training_v1.sh --test             # Quick test (10 steps)
#
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Script Location
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# -----------------------------------------------------------------------------
# Environment Check
# -----------------------------------------------------------------------------
echo "Environment:"
echo "  Python: $(which python3)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "  NumPy: $(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'not installed')"
echo "  Moshi: $(python3 -c 'import moshi; print(getattr(moshi, \"__version__\", \"installed\"))' 2>/dev/null || echo 'not installed')"
echo ""

# Detailed moshi validation
validate_moshi() {
    echo "Validating moshi installation..."

    # Check if moshi package exists
    if ! python3 -c "import moshi" 2>/dev/null; then
        echo "  [FAIL] moshi package not found"
        return 1
    fi
    echo "  [OK] moshi package imported"

    # Check moshi.models.loaders (correct import path used by train.py)
    if python3 -c "from moshi.models import loaders" 2>/dev/null; then
        echo "  [OK] moshi.models.loaders"
    else
        echo "  [FAIL] moshi.models.loaders not available"
        echo "         This is required for training. Try reinstalling:"
        echo "         pip install 'git+https://github.com/kyutai-labs/moshi.git#subdirectory=moshi' --no-deps --force-reinstall"
        return 1
    fi

    # Check moshi.models.lm.LMModel
    if python3 -c "from moshi.models.lm import LMModel" 2>/dev/null; then
        echo "  [OK] moshi.models.lm.LMModel"
    else
        echo "  [FAIL] moshi.models.lm.LMModel not available"
        return 1
    fi

    # Check moshi.models.loaders.CheckpointInfo
    if python3 -c "from moshi.models.loaders import CheckpointInfo" 2>/dev/null; then
        echo "  [OK] moshi.models.loaders.CheckpointInfo"
    else
        echo "  [WARN] moshi.models.loaders.CheckpointInfo not available"
    fi

    # Check essential modules for training
    local essential_ok=true
    for module in "torch" "einops" "safetensors" "sentencepiece"; do
        if python3 -c "import $module" 2>/dev/null; then
            echo "  [OK] $module"
        else
            echo "  [FAIL] $module not installed"
            essential_ok=false
        fi
    done

    if [ "$essential_ok" = false ]; then
        return 1
    fi

    return 0
}

if ! validate_moshi; then
    echo ""
    echo "ERROR: Environment validation failed."
    echo ""
    echo "Try running: ./scripts/setup_environment.sh"
    echo "Or check:    ./scripts/setup_environment.sh --check"
    echo ""
    echo "For detailed error, run:"
    echo "  python3 -c 'from moshi import loaders'"
    exit 1
fi
echo ""

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=3

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
NUM_GPUS=8
CONFIG_FILE="example/korean_v1_fsdp.yaml"
LOG_DIR="logs"
TEST_MODE=false
BACKGROUND=false

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus|-g)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --test|-t)
            TEST_MODE=true
            shift
            ;;
        --background|-b)
            BACKGROUND=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus, -g NUM      Number of GPUs (default: 8)"
            echo "  --config, -c FILE   Config YAML file (default: example/korean_v1_fsdp.yaml)"
            echo "  --test, -t          Quick test mode (10 steps)"
            echo "  --background, -b    Run in background with nohup"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
echo "============================================================"
echo "K-Moshi Version 1 Training"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  GPUs: ${NUM_GPUS}"
echo "  Config: ${CONFIG_FILE}"
echo "  Test mode: ${TEST_MODE}"
echo "  Background: ${BACKGROUND}"
echo ""

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# OpenMPI settings for optimal performance
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_btl_tcp_if_include=eth0
export OMP_NUM_THREADS=4

# NCCL settings
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available

echo "Environment:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo ""

# -----------------------------------------------------------------------------
# Validate Prerequisites
# -----------------------------------------------------------------------------
echo "Validating prerequisites..."

# Check config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Check NVIDIA GPUs
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "${NUM_GPUS}" -gt "${AVAILABLE_GPUS}" ]; then
    echo "ERROR: Requested ${NUM_GPUS} GPUs but only ${AVAILABLE_GPUS} available"
    exit 1
fi

echo "  Config file: OK"
echo "  GPUs available: ${AVAILABLE_GPUS} (using ${NUM_GPUS})"
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# -----------------------------------------------------------------------------
# Create Log Directory
# -----------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_v1_${TIMESTAMP}.log"

# -----------------------------------------------------------------------------
# Build Training Command
# -----------------------------------------------------------------------------
if [ "${TEST_MODE}" = true ]; then
    echo ">>> TEST MODE: Running 10 steps only <<<"
    echo ""

    # Create temporary test config
    TEST_CONFIG="${LOG_DIR}/test_config_${TIMESTAMP}.yaml"
    cp "${CONFIG_FILE}" "${TEST_CONFIG}"

    # Modify for quick test
    sed -i 's/max_steps: .*/max_steps: 10/' "${TEST_CONFIG}"
    sed -i 's/log_freq: .*/log_freq: 1/' "${TEST_CONFIG}"
    sed -i 's/ckpt_freq: .*/ckpt_freq: 10/' "${TEST_CONFIG}"
    sed -i 's/eval_freq: .*/eval_freq: 5/' "${TEST_CONFIG}"
    sed -i "s|run_dir: .*|run_dir: './runs/test_v1_${TIMESTAMP}'|" "${TEST_CONFIG}"
    sed -i 's/overwrite_run_dir: .*/overwrite_run_dir: true/' "${TEST_CONFIG}"

    CONFIG_FILE="${TEST_CONFIG}"
fi

TRAIN_CMD="mpirun -np ${NUM_GPUS} python -m train ${CONFIG_FILE}"

# -----------------------------------------------------------------------------
# Execute Training
# -----------------------------------------------------------------------------
echo "============================================================"
echo "Starting Training"
echo "============================================================"
echo ""
echo "Command: ${TRAIN_CMD}"
echo "Log file: ${LOG_FILE}"
echo ""

if [ "${BACKGROUND}" = true ]; then
    echo "Running in background..."
    nohup ${TRAIN_CMD} > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "Started with PID: ${PID}"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f ${LOG_FILE}"
    echo ""
    echo "To stop training:"
    echo "  kill ${PID}"
else
    echo "Running in foreground..."
    echo ""
    ${TRAIN_CMD} 2>&1 | tee "${LOG_FILE}"
fi

echo ""
echo "============================================================"
echo "Training Complete"
echo "============================================================"
