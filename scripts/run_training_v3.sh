#!/bin/bash
# =============================================================================
# K-Moshi Version 3 Training Script
# =============================================================================
#
# FULL-DUPLEX Mode Training (Original Moshi / J-Moshi Compatible)
#
# Version 3 Key Features:
#   - FULL-DUPLEX MODE: Stereo input (17 codebooks), dep_q=8 output
#   - User audio as CONTEXT ONLY (not predicted)
#   - 100% compatible with original Moshi inference pipeline
#   - J-Moshi style training configuration
#
# V3 vs V2 Comparison:
#   +-------------+-------------------+-------------------+
#   | Feature     | V2 (USER-STREAM)  | V3 (FULL-DUPLEX)  |
#   +-------------+-------------------+-------------------+
#   | Input       | 17 codebooks      | 17 codebooks      |
#   | Output      | dep_q=16          | dep_q=8           |
#   | User Audio  | Predicted         | Context only      |
#   | Inference   | Needs extension   | Original Moshi OK |
#   +-------------+-------------------+-------------------+
#
# Environment:
#   - 1 Node x 8 GPU (NVIDIA A100 80GB)
#   - mpirun distributed launcher with FSDP
#   - Expected GPU memory: ~55-60GB per GPU
#
# Usage:
#   ./scripts/run_training_v3.sh                    # Default: 8 GPUs
#   ./scripts/run_training_v3.sh --gpus 4           # Use 4 GPUs
#   ./scripts/run_training_v3.sh --config custom.yaml
#   ./scripts/run_training_v3.sh --test             # Quick test (10 steps)
#   ./scripts/run_training_v3.sh --create-manifests # Create V3 manifests first
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
CYAN='\033[0;36m'
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

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# -----------------------------------------------------------------------------
# Environment Check
# -----------------------------------------------------------------------------
echo ""
print_header "K-Moshi Version 3 Training (FULL-DUPLEX Mode)"
echo ""
echo "Environment:"
echo "  Python: $(which python3)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'not available')"
echo ""

# Display V3 mode info
print_info "V3 Mode: FULL-DUPLEX (Original Moshi / J-Moshi Compatible)"
print_info "  - Input: 17 codebooks (stereo)"
print_info "  - Output: dep_q=8 (Moshi audio only)"
print_info "  - User audio: Context only (not predicted)"
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

    # Check optional packages for V3 features
    echo ""
    echo "Optional packages (V3 features):"
    for module in "matplotlib" "soundfile" "torchaudio"; do
        if python3 -c "import $module" 2>/dev/null; then
            print_success "$module"
        else
            print_warning "$module not installed (some features disabled)"
        fi
    done

    # Check Enhanced Evaluation packages
    echo ""
    echo "Enhanced Evaluation packages (optional):"
    check_enhanced_eval_deps

    if [ "$all_ok" = false ]; then
        return 1
    fi
    return 0
}

# -----------------------------------------------------------------------------
# Enhanced Evaluation Dependencies
# -----------------------------------------------------------------------------
# These packages are optional but recommended for comprehensive evaluation metrics
#
# REQUIRED for basic evaluation:
#   sacrebleu           - BLEU score calculation (text quality) - RECOMMENDED
#
# OPTIONAL for audio quality (disabled by default due to computational cost):
#   pystoi              - STOI metric (speech intelligibility)
#   pesq                - PESQ metric (perceptual quality)
#   librosa             - MCD metric (mel cepstral distortion)
#
# OPTIONAL for semantic similarity (disabled by default):
#   sentence-transformers - Embedding-based semantic similarity (~500MB)
#                          Only needed if compute_semantic=true in config
#
# Installation:
#   pip install sacrebleu                    # Recommended (lightweight)
#   pip install pystoi pesq librosa          # Audio quality metrics
#   pip install sentence-transformers        # Only if compute_semantic=true
# -----------------------------------------------------------------------------

ENHANCED_EVAL_DEPS_MISSING=""
ENHANCED_EVAL_DEPS_OPTIONAL=""

check_enhanced_eval_deps() {
    local missing_deps=""
    local optional_deps=""

    # sacrebleu - BLEU metrics (recommended, lightweight)
    if python3 -c "import sacrebleu" 2>/dev/null; then
        print_success "sacrebleu (BLEU metrics)"
    else
        print_warning "sacrebleu not installed (BLEU metrics disabled)"
        missing_deps="${missing_deps} sacrebleu"
    fi

    # pystoi - STOI metric (optional, disabled by default)
    if python3 -c "import pystoi" 2>/dev/null; then
        print_success "pystoi (STOI metric)"
    else
        print_info "pystoi not installed (optional, STOI metric)"
        optional_deps="${optional_deps} pystoi"
    fi

    # pesq - PESQ metric (optional, disabled by default)
    if python3 -c "import pesq" 2>/dev/null; then
        print_success "pesq (PESQ metric)"
    else
        print_info "pesq not installed (optional, PESQ metric)"
        optional_deps="${optional_deps} pesq"
    fi

    # librosa - MCD metric (optional)
    if python3 -c "import librosa" 2>/dev/null; then
        print_success "librosa (MCD metric)"
    else
        print_info "librosa not installed (optional, MCD metric)"
        optional_deps="${optional_deps} librosa"
    fi

    # sentence-transformers - Semantic similarity (optional, heavy)
    # Only needed if compute_semantic=true in enhanced_evaluation config
    if python3 -c "import sentence_transformers" 2>/dev/null; then
        print_success "sentence-transformers (semantic similarity)"
    else
        print_info "sentence-transformers not installed (optional, ~500MB)"
        print_info "  → Only needed if enhanced_evaluation.semantic.compute_semantic=true"
        optional_deps="${optional_deps} sentence-transformers"
    fi

    ENHANCED_EVAL_DEPS_MISSING="${missing_deps}"
    ENHANCED_EVAL_DEPS_OPTIONAL="${optional_deps}"
}

install_enhanced_eval_deps() {
    local to_install=""

    # Always install missing required deps
    if [ -n "${ENHANCED_EVAL_DEPS_MISSING}" ]; then
        to_install="${ENHANCED_EVAL_DEPS_MISSING}"
    fi

    if [ -z "${to_install}" ]; then
        print_success "All required Enhanced Evaluation dependencies already installed"
        echo ""
        echo "Optional packages not installed:"
        echo "  Audio quality: pystoi pesq librosa"
        echo "  Semantic sim:  sentence-transformers (only if compute_semantic=true)"
        echo ""
        echo "To install optional packages manually:"
        echo "  pip install pystoi pesq librosa"
        echo "  pip install sentence-transformers  # ~500MB, only if needed"
        return 0
    fi

    echo ""
    print_info "Installing Enhanced Evaluation dependencies..."
    echo "  Required packages:${to_install}"
    echo ""

    # Install required packages
    pip install ${to_install}

    if [ $? -eq 0 ]; then
        print_success "Enhanced Evaluation dependencies installed successfully"
    else
        print_warning "Some packages failed to install. Training will continue with available metrics."
    fi
}

install_all_eval_deps() {
    # Install ALL evaluation dependencies (including optional)
    local all_deps="${ENHANCED_EVAL_DEPS_MISSING}${ENHANCED_EVAL_DEPS_OPTIONAL}"

    if [ -z "${all_deps}" ]; then
        print_success "All Enhanced Evaluation dependencies already installed"
        return 0
    fi

    echo ""
    print_info "Installing ALL Enhanced Evaluation dependencies (including optional)..."
    echo "  Packages:${all_deps}"
    echo ""

    pip install ${all_deps}

    if [ $? -eq 0 ]; then
        print_success "All Enhanced Evaluation dependencies installed successfully"
    else
        print_warning "Some packages failed to install."
    fi
}

if ! validate_environment; then
    echo ""
    print_error "Environment validation failed."
    echo ""
    echo "Try running: pip install matplotlib soundfile torchaudio"
    exit 1
fi
echo ""

# -----------------------------------------------------------------------------
# Handle Enhanced Evaluation Dependencies Installation
# -----------------------------------------------------------------------------
if [ "${INSTALL_ALL_DEPS}" = true ]; then
    # Install ALL dependencies (including optional)
    install_all_eval_deps
    echo ""
elif [ "${INSTALL_DEPS}" = true ]; then
    # Install only required dependencies
    install_enhanced_eval_deps
    echo ""
elif [ "${AUTO_INSTALL_DEPS}" = true ] && [ -n "${ENHANCED_EVAL_DEPS_MISSING}" ]; then
    # Auto-install required dependencies
    install_enhanced_eval_deps
    echo ""
elif [ -n "${ENHANCED_EVAL_DEPS_MISSING}" ]; then
    echo ""
    print_info "Missing required Enhanced Evaluation dependencies:"
    echo "  Packages:${ENHANCED_EVAL_DEPS_MISSING}"
    echo ""
    echo "  To install required: $0 --install-deps"
    echo "  To install all:      $0 --install-all-deps"
    echo ""
    echo "  Or manually:"
    echo "    pip install${ENHANCED_EVAL_DEPS_MISSING}           # Required"
    echo "    pip install pystoi pesq librosa                    # Optional (audio quality)"
    echo "    pip install sentence-transformers                  # Optional (~500MB, semantic)"
    echo ""
    echo "  Training will continue with available metrics."
    echo ""
fi

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=3

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
NUM_GPUS=8
CONFIG_FILE="example/korean_v3_fsdp.yaml"
LOG_DIR="logs"
TEST_MODE=false
BACKGROUND=false
CREATE_MANIFESTS=false
DRY_RUN=false
INSTALL_DEPS=false
AUTO_INSTALL_DEPS=false
INSTALL_ALL_DEPS=false

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
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --auto-install-deps)
            AUTO_INSTALL_DEPS=true
            shift
            ;;
        --install-all-deps)
            INSTALL_ALL_DEPS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "K-Moshi V3 Training Script (FULL-DUPLEX Mode)"
            echo ""
            echo "Options:"
            echo "  --gpus, -g NUM          Number of GPUs (default: 8)"
            echo "  --config, -c FILE       Config YAML file (default: example/korean_v3_fsdp.yaml)"
            echo "  --test, -t              Quick test mode (10 steps)"
            echo "  --background, -b        Run in background with nohup"
            echo "  --create-manifests, -m  Create V3 train/valid manifests first"
            echo "  --dry-run               Show commands without executing"
            echo "  --install-deps          Install required Enhanced Evaluation dependencies (sacrebleu)"
            echo "  --auto-install-deps     Auto-install required deps without prompting"
            echo "  --install-all-deps      Install ALL deps including optional (pystoi, pesq, librosa, sentence-transformers)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Training Modes:"
            echo "  V3 (FULL-DUPLEX): Stereo input, dep_q=8, user audio as context"
            echo "  V2 (USER-STREAM): Stereo input, dep_q=16, user audio predicted"
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
# Create V3 Manifests (if requested)
# -----------------------------------------------------------------------------
if [ "${CREATE_MANIFESTS}" = true ]; then
    print_header "Creating V3 Manifests"
    echo ""

    echo "Creating training manifest (key463-train + key71314-train)..."
    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY RUN] python scripts/create_unified_manifest.py --datasets key463-train key71314-train --output ./data/korean_v3_train.jsonl"
    else
        python scripts/create_unified_manifest.py \
            --datasets key463-train key71314-train \
            --output ./data/korean_v3_train.jsonl
    fi
    echo ""

    echo "Creating validation manifest (key71314-valid)..."
    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY RUN] python scripts/create_unified_manifest.py --datasets key71314-valid --output ./data/korean_v3_valid.jsonl"
    else
        python scripts/create_unified_manifest.py \
            --datasets key71314-valid \
            --output ./data/korean_v3_valid.jsonl
    fi
    echo ""
fi

# -----------------------------------------------------------------------------
# Display Configuration
# -----------------------------------------------------------------------------
print_header "Training Configuration"
echo ""
echo "Version 3 Features (FULL-DUPLEX Mode):"
echo "  - Stereo Input: 17 codebooks (1 text + 8 moshi + 8 user)"
echo "  - Standard Output: dep_q=8 (Moshi audio only)"
echo "  - User Audio: Context only (not predicted)"
echo "  - Inference: 100% compatible with original Moshi"
echo ""
echo "Additional Features:"
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

# Check manifest files (for V3)
if [ -f "./data/korean_v3_train.jsonl" ]; then
    TRAIN_SAMPLES=$(wc -l < ./data/korean_v3_train.jsonl)
    print_success "Training manifest: ${TRAIN_SAMPLES} samples"
else
    print_warning "Training manifest not found: ./data/korean_v3_train.jsonl"
    echo "         Run with --create-manifests to create it"
    echo "         Or copy from V2: cp ./data/korean_v2_train.jsonl ./data/korean_v3_train.jsonl"
fi

if [ -f "./data/korean_v3_valid.jsonl" ]; then
    VALID_SAMPLES=$(wc -l < ./data/korean_v3_valid.jsonl)
    print_success "Validation manifest: ${VALID_SAMPLES} samples"
else
    print_warning "Validation manifest not found: ./data/korean_v3_valid.jsonl"
    echo "         Or copy from V2: cp ./data/korean_v2_valid.jsonl ./data/korean_v3_valid.jsonl"
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
LOG_FILE="${LOG_DIR}/train_v3_${TIMESTAMP}.log"

# -----------------------------------------------------------------------------
# Build Training Command
# -----------------------------------------------------------------------------
if [ "${TEST_MODE}" = true ]; then
    echo -e "${YELLOW}>>> TEST MODE: Running 10 steps only <<<${NC}"
    echo ""

    # Create temporary test config
    TEST_CONFIG="${LOG_DIR}/test_config_v3_${TIMESTAMP}.yaml"
    cp "${CONFIG_FILE}" "${TEST_CONFIG}"

    # Modify for quick test
    sed -i 's/max_steps: .*/max_steps: 10/' "${TEST_CONFIG}"
    sed -i 's/log_freq: .*/log_freq: 1/' "${TEST_CONFIG}"
    sed -i 's/ckpt_freq: .*/ckpt_freq: 10/' "${TEST_CONFIG}"
    sed -i 's/eval_freq: .*/eval_freq: 5/' "${TEST_CONFIG}"
    sed -i 's/save_freq: .*/save_freq: 5/' "${TEST_CONFIG}"
    sed -i 's/plot_freq: .*/plot_freq: 5/' "${TEST_CONFIG}"
    sed -i "s|run_dir: .*|run_dir: './runs/test_v3_${TIMESTAMP}'|" "${TEST_CONFIG}"
    sed -i 's/overwrite_run_dir: .*/overwrite_run_dir: true/' "${TEST_CONFIG}"

    CONFIG_FILE="${TEST_CONFIG}"
    echo "Test config created: ${TEST_CONFIG}"
    echo ""
fi

TRAIN_CMD="mpirun -np ${NUM_GPUS} python -m train ${CONFIG_FILE}"

# -----------------------------------------------------------------------------
# Execute Training
# -----------------------------------------------------------------------------
print_header "Starting V3 Training (FULL-DUPLEX Mode)"
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
    echo "  tensorboard --logdir=./runs/korean_v3/tensorboard"
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
echo "  Checkpoints: ./runs/korean_v3/"
echo "  TensorBoard: ./runs/korean_v3/tensorboard/"
echo "  Samples:     ./runs/korean_v3/samples/"
echo "  Research:    ./runs/korean_v3/research/"
echo "  Log file:    ${LOG_FILE}"
echo ""
echo "V3 Model Usage:"
echo "  - Trained model is 100% compatible with original Moshi inference"
echo "  - No model extension needed for serving"
echo "  - Use with moshi.server directly"
echo ""
