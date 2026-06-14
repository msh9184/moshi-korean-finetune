#!/bin/bash
# =============================================================================
# K-Moshi Version 2 Training Script
# =============================================================================
#
# Korean Moshi Full Finetuning with Optimized Settings
#
# Version 2 Improvements:
#   - GPU Memory Optimization: batch_size=6, duration_sec=90
#   - Two-rate Optimizer: Separate LRs for TempFormer/DepFormer
#   - Cosine Warmup Scheduler: Better convergence
#   - Advanced Monitoring: WER, per-codebook loss, gradient health
#   - Sample Saving: Audio/text predictions during training
#   - Research Logging: Loss curves, attention maps for papers
#
# Environment:
#   - 1 Node x 8 GPU (NVIDIA A100 80GB)
#   - mpirun distributed launcher with FSDP
#   - Expected GPU memory: ~55-60GB per GPU
#
# Usage:
#   ./scripts/run_training_v2.sh                    # Default: 8 GPUs
#   ./scripts/run_training_v2.sh --gpus 4           # Use 4 GPUs
#   ./scripts/run_training_v2.sh --config custom.yaml
#   ./scripts/run_training_v2.sh --test             # Quick test (10 steps)
#   ./scripts/run_training_v2.sh --create-manifests # Create V2 manifests first
#
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Script Location
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# -----------------------------------------------------------------------------
# Color Codes for Output
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# -----------------------------------------------------------------------------
# Environment Check
# -----------------------------------------------------------------------------
echo ""
print_header "K-Moshi Version 2 Training"
echo ""
echo "Environment:"
echo "  Python: $(which python3)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'not available')"
echo ""

# Detailed moshi validation
validate_environment() {
    echo "Validating environment..."

    local all_ok=true

    # Check moshi package
    if python3 -c "from moshi.models import loaders" 2>/dev/null; then
        print_success "moshi.models.loaders"
    else
        print_error "moshi.models.loaders not available"
        all_ok=false
    fi

    # Check essential packages
    for module in "torch" "einops" "safetensors" "sentencepiece" "sphn"; do
        if python3 -c "import $module" 2>/dev/null; then
            print_success "$module"
        else
            print_error "$module not installed"
            all_ok=false
        fi
    done

    # Check optional packages for V2 features
    echo ""
    echo "Optional packages (V2 features):"
    for module in "matplotlib" "soundfile" "torchaudio"; do
        if python3 -c "import $module" 2>/dev/null; then
            print_success "$module"
        else
            print_warning "$module not installed (some features disabled)"
        fi
    done

    if [ "$all_ok" = false ]; then
        return 1
    fi
    return 0
}

if ! validate_environment; then
    echo ""
    print_error "Environment validation failed."
    echo ""
    echo "Try running: pip install matplotlib soundfile torchaudio"
    exit 1
fi
echo ""

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=3

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
NUM_GPUS=8
CONFIG_FILE="example/korean_v2_fsdp.yaml"
LOG_DIR="logs"
TEST_MODE=false
BACKGROUND=false
CREATE_MANIFESTS=false
DRY_RUN=false

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
        --create-manifests|-m)
            CREATE_MANIFESTS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus, -g NUM          Number of GPUs (default: 8)"
            echo "  --config, -c FILE       Config YAML file (default: example/korean_v2_fsdp.yaml)"
            echo "  --test, -t              Quick test mode (10 steps)"
            echo "  --background, -b        Run in background with nohup"
            echo "  --create-manifests, -m  Create V2 train/valid manifests first"
            echo "  --dry-run               Show commands without executing"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                # Full training with 8 GPUs"
            echo "  $0 --test                         # Quick test (10 steps)"
            echo "  $0 --create-manifests --test      # Create manifests then test"
            echo "  $0 --gpus 4 --background          # 4 GPUs, run in background"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Create V2 Manifests (if requested)
# -----------------------------------------------------------------------------
if [ "${CREATE_MANIFESTS}" = true ]; then
    print_header "Creating V2 Manifests"
    echo ""

    echo "Creating training manifest (key463-train + key71314-train)..."
    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY RUN] python scripts/create_unified_manifest.py --datasets key463-train key71314-train --output ./data/korean_v2_train.jsonl"
    else
        python scripts/create_unified_manifest.py \
            --datasets key463-train key71314-train \
            --output ./data/korean_v2_train.jsonl
    fi
    echo ""

    echo "Creating validation manifest (key71314-valid)..."
    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY RUN] python scripts/create_unified_manifest.py --datasets key71314-valid --output ./data/korean_v2_valid.jsonl"
    else
        python scripts/create_unified_manifest.py \
            --datasets key71314-valid \
            --output ./data/korean_v2_valid.jsonl
    fi
    echo ""
fi

# -----------------------------------------------------------------------------
# Display Configuration
# -----------------------------------------------------------------------------
print_header "Training Configuration"
echo ""
echo "Version 2 Features:"
echo "  - GPU Memory Optimization (batch=6, duration=90s)"
echo "  - Two-rate Optimizer (TempFormer + DepFormer)"
echo "  - Cosine Warmup Scheduler"
echo "  - Advanced Monitoring (WER, codebook, gradients)"
echo "  - Sample Saving (audio + text)"
echo "  - Research Logging (plots, CSV, summary)"
echo ""
echo "Settings:"
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
export NCCL_IB_DISABLE=1

# =============================================================================
# CRITICAL: Disable torch.compile/dynamo for FSDP compatibility
# PyTorch 2.x's inductor causes stride mismatches with FSDP + gradient checkpointing
# =============================================================================
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_LOGS="-dynamo"

echo "Environment Variables:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  TORCH_COMPILE_DISABLE: ${TORCH_COMPILE_DISABLE} (FSDP compatibility)"
echo ""

# -----------------------------------------------------------------------------
# Validate Prerequisites
# -----------------------------------------------------------------------------
echo "Validating prerequisites..."

# Check config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    print_error "Config file not found: ${CONFIG_FILE}"
    echo ""
    echo "Available config files:"
    ls -la example/*.yaml 2>/dev/null || echo "  (none found)"
    exit 1
fi
print_success "Config file: ${CONFIG_FILE}"

# Check NVIDIA GPUs
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "${NUM_GPUS}" -gt "${AVAILABLE_GPUS}" ]; then
    print_error "Requested ${NUM_GPUS} GPUs but only ${AVAILABLE_GPUS} available"
    exit 1
fi
print_success "GPUs: ${NUM_GPUS}/${AVAILABLE_GPUS} available"

# Check manifest files (for V2)
if [ -f "./data/korean_v2_train.jsonl" ]; then
    TRAIN_SAMPLES=$(wc -l < ./data/korean_v2_train.jsonl)
    print_success "Training manifest: ${TRAIN_SAMPLES} samples"
else
    print_warning "Training manifest not found: ./data/korean_v2_train.jsonl"
    echo "         Run with --create-manifests to create it"
fi

if [ -f "./data/korean_v2_valid.jsonl" ]; then
    VALID_SAMPLES=$(wc -l < ./data/korean_v2_valid.jsonl)
    print_success "Validation manifest: ${VALID_SAMPLES} samples"
else
    print_warning "Validation manifest not found: ./data/korean_v2_valid.jsonl"
fi

echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# -----------------------------------------------------------------------------
# Create Log Directory
# -----------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_v2_${TIMESTAMP}.log"

# -----------------------------------------------------------------------------
# Build Training Command
# -----------------------------------------------------------------------------
if [ "${TEST_MODE}" = true ]; then
    echo -e "${YELLOW}>>> TEST MODE: Running 10 steps only <<<${NC}"
    echo ""

    # Create temporary test config
    TEST_CONFIG="${LOG_DIR}/test_config_v2_${TIMESTAMP}.yaml"
    cp "${CONFIG_FILE}" "${TEST_CONFIG}"

    # Modify for quick test
    sed -i 's/max_steps: .*/max_steps: 10/' "${TEST_CONFIG}"
    sed -i 's/log_freq: .*/log_freq: 1/' "${TEST_CONFIG}"
    sed -i 's/ckpt_freq: .*/ckpt_freq: 10/' "${TEST_CONFIG}"
    sed -i 's/eval_freq: .*/eval_freq: 5/' "${TEST_CONFIG}"
    sed -i 's/save_freq: .*/save_freq: 5/' "${TEST_CONFIG}"
    sed -i 's/plot_freq: .*/plot_freq: 5/' "${TEST_CONFIG}"
    sed -i "s|run_dir: .*|run_dir: './runs/test_v2_${TIMESTAMP}'|" "${TEST_CONFIG}"
    sed -i 's/overwrite_run_dir: .*/overwrite_run_dir: true/' "${TEST_CONFIG}"

    CONFIG_FILE="${TEST_CONFIG}"
    echo "Test config created: ${TEST_CONFIG}"
    echo ""
fi

TRAIN_CMD="mpirun -np ${NUM_GPUS} python -m train ${CONFIG_FILE}"

# -----------------------------------------------------------------------------
# Execute Training
# -----------------------------------------------------------------------------
print_header "Starting V2 Training"
echo ""
echo "Command: ${TRAIN_CMD}"
echo "Log file: ${LOG_FILE}"
echo ""

if [ "${DRY_RUN}" = true ]; then
    echo "[DRY RUN] Would execute: ${TRAIN_CMD}"
    exit 0
fi

if [ "${BACKGROUND}" = true ]; then
    echo "Running in background..."
    nohup ${TRAIN_CMD} > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo ""
    print_success "Started with PID: ${PID}"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f ${LOG_FILE}"
    echo ""
    echo "To view TensorBoard:"
    echo "  tensorboard --logdir=./runs/korean_v2/tensorboard"
    echo ""
    echo "To stop training:"
    echo "  kill ${PID}"
else
    echo "Running in foreground..."
    echo ""
    ${TRAIN_CMD} 2>&1 | tee "${LOG_FILE}"
fi

echo ""
print_header "Training Complete"
echo ""
echo "Output locations:"
echo "  Checkpoints: ./runs/korean_v2/"
echo "  TensorBoard: ./runs/korean_v2/tensorboard/"
echo "  Samples:     ./runs/korean_v2/samples/"
echo "  Research:    ./runs/korean_v2/research/"
echo "  Log file:    ${LOG_FILE}"
echo ""
