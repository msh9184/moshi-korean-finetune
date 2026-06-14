#!/bin/bash
# =============================================================================
# K-Moshi Version 4 Training Script
# =============================================================================
#
# MODULAR BACKBONE Mode Training (HFLM/Custom LLM Support)
#
# Version 4 Key Features:
#   - MODULAR BACKBONE: Swappable LLM backend (Moshi, HFLM, Custom)
#   - DIMENSION ADAPTER: Automatic dimension bridging (4096 <-> 3072)
#   - HFLM SUPPORT: HuggingFace Transformers-based Korean LLM
#   - Maintains FULL-DUPLEX training mode from V3
#
# Architecture:
#   +-----------------------------------------------------------------------+
#   |                        LMModelWrapper                                 |
#   +-----------------------------------------------------------------------+
#   | Input Embedding (Moshi, 4096-dim)                                     |
#   +-----------------------------------------------------------------------+
#   | DimensionAdapter (input_proj): 4096 -> 3072                          |
#   +-----------------------------------------------------------------------+
#   | HFLMBackbone (3B): 30 layers, 24 heads, 4 KV heads (GQA)           |
#   +-----------------------------------------------------------------------+
#   | DimensionAdapter (output_proj): 3072 -> 4096                         |
#   +-----------------------------------------------------------------------+
#   | Output Heads (Moshi, 4096-dim)                                        |
#   +-----------------------------------------------------------------------+
#
# V4 vs V3 Comparison:
#   +-------------+----------------------+-------------------------+
#   | Feature     | V3 (FULL-DUPLEX)     | V4 (MODULAR BACKBONE)   |
#   +-------------+----------------------+-------------------------+
#   | Backbone    | Moshi (7B)           | HFLM (3B) / Custom    |
#   | Dimension   | 4096 (fixed)         | Adapter bridging        |
#   | Input       | 17 codebooks         | 17 codebooks            |
#   | Output      | dep_q=8              | dep_q=8                 |
#   | Memory      | ~60GB                | ~50GB (HFLM)          |
#   +-------------+----------------------+-------------------------+
#
# Environment:
#   - 1 Node x 8 GPU (NVIDIA A100 80GB)
#   - torchrun distributed launcher with FSDP
#   - Expected GPU memory: ~50-60GB per GPU (with HFLM)
#
# Usage:
#   ./scripts/run_training_v4.sh                        # Default: 8 GPUs
#   ./scripts/run_training_v4.sh --gpus 4               # Use 4 GPUs
#   ./scripts/run_training_v4.sh --config custom.yaml
#   ./scripts/run_training_v4.sh --test                 # Quick test (10 steps)
#   ./scripts/run_training_v4.sh --backbone moshi       # Use Moshi backbone
#   ./scripts/run_training_v4.sh --validate-backbone    # Validate backbone setup
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
MAGENTA='\033[0;35m'
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

print_backbone() {
    echo -e "${MAGENTA}[BACKBONE]${NC} $1"
}

# -----------------------------------------------------------------------------
# Environment Check
# -----------------------------------------------------------------------------
echo ""
print_header "K-Moshi Version 4 Training (MODULAR BACKBONE)"
echo ""
echo "Environment:"
echo "  Python: $(which python3)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'not available')"
echo ""

# Display V4 mode info
print_info "V4 Mode: MODULAR BACKBONE (HFLM/Custom LLM Support)"
print_info "  - Backbone: Swappable LLM (Moshi, HFLM, Custom)"
print_info "  - Dimension Adapter: Automatic dimension bridging"
print_info "  - Input: 17 codebooks (stereo, Full-Duplex)"
print_info "  - Output: dep_q=8 (Moshi audio only)"
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

    # Check transformers for HFLM backbone
    echo ""
    echo "Modular Backbone packages:"
    if python3 -c "import transformers" 2>/dev/null; then
        TRANSFORMERS_VERSION=$(python3 -c "import transformers; print(transformers.__version__)")
        print_success "transformers ($TRANSFORMERS_VERSION)"
    else
        print_warning "transformers not installed (HFLM backbone requires it)"
    fi

    # Check optional packages for V4 features
    echo ""
    echo "Optional packages (V4 features):"
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
# Backbone Validation
# -----------------------------------------------------------------------------
validate_backbone_setup() {
    print_header "Validating Backbone Setup"
    echo ""

    # Check backbone module
    echo "Checking backbone module..."
    if python3 -c "from finetune.backbone import BackboneFactory, LMModelWrapper" 2>/dev/null; then
        print_success "finetune.backbone module"
    else
        print_error "finetune.backbone module not available"
        return 1
    fi

    # Check HFLMBackbone
    if python3 -c "from finetune.backbone import HFLMBackbone" 2>/dev/null; then
        print_success "HFLMBackbone"
    else
        print_error "HFLMBackbone not available"
        return 1
    fi

    # Check DimensionAdapter
    if python3 -c "from finetune.backbone import DimensionAdapter" 2>/dev/null; then
        print_success "DimensionAdapter"
    else
        print_error "DimensionAdapter not available"
        return 1
    fi

    # Check available backbone types
    echo ""
    echo "Available backbone types:"
    python3 -c "
from finetune.backbone import BackboneFactory
for t in BackboneFactory.list_types():
    print(f'  - {t}')
"

    # Check HFLM model path if specified
    if [ -n "${HFLM_MODEL_PATH}" ]; then
        echo ""
        echo "Checking HFLM model path..."
        if [ -d "${HFLM_MODEL_PATH}" ]; then
            print_success "HFLM model path exists: ${HFLM_MODEL_PATH}"
            if [ -f "${HFLM_MODEL_PATH}/config.json" ]; then
                print_success "Found config.json"
            else
                print_warning "config.json not found"
            fi
        else
            print_warning "HFLM model path not found: ${HFLM_MODEL_PATH}"
        fi
    fi

    echo ""
    print_success "Backbone setup validation complete"
    return 0
}

# -----------------------------------------------------------------------------
# Enhanced Evaluation Dependencies
# -----------------------------------------------------------------------------
ENHANCED_EVAL_DEPS_MISSING=""
ENHANCED_EVAL_DEPS_OPTIONAL=""

check_enhanced_eval_deps() {
    local missing_deps=""
    local optional_deps=""

    # sacrebleu - BLEU metrics
    if python3 -c "import sacrebleu" 2>/dev/null; then
        print_success "sacrebleu (BLEU metrics)"
    else
        print_warning "sacrebleu not installed (BLEU metrics disabled)"
        missing_deps="${missing_deps} sacrebleu"
    fi

    # pystoi - STOI metric (optional)
    if python3 -c "import pystoi" 2>/dev/null; then
        print_success "pystoi (STOI metric)"
    else
        print_info "pystoi not installed (optional)"
        optional_deps="${optional_deps} pystoi"
    fi

    # pesq - PESQ metric (optional)
    if python3 -c "import pesq" 2>/dev/null; then
        print_success "pesq (PESQ metric)"
    else
        print_info "pesq not installed (optional)"
        optional_deps="${optional_deps} pesq"
    fi

    # librosa - MCD metric (optional)
    if python3 -c "import librosa" 2>/dev/null; then
        print_success "librosa (MCD metric)"
    else
        print_info "librosa not installed (optional)"
        optional_deps="${optional_deps} librosa"
    fi

    ENHANCED_EVAL_DEPS_MISSING="${missing_deps}"
    ENHANCED_EVAL_DEPS_OPTIONAL="${optional_deps}"
}

install_enhanced_eval_deps() {
    local to_install="${ENHANCED_EVAL_DEPS_MISSING}"

    if [ -z "${to_install}" ]; then
        print_success "All required Enhanced Evaluation dependencies already installed"
        return 0
    fi

    echo ""
    print_info "Installing Enhanced Evaluation dependencies..."
    pip install ${to_install}

    if [ $? -eq 0 ]; then
        print_success "Enhanced Evaluation dependencies installed"
    else
        print_warning "Some packages failed to install"
    fi
}

install_backbone_deps() {
    echo ""
    print_info "Installing backbone dependencies..."
    pip install transformers accelerate
    if [ $? -eq 0 ]; then
        print_success "Backbone dependencies installed"
    else
        print_warning "Some packages failed to install"
    fi
}

if ! validate_environment; then
    echo ""
    print_error "Environment validation failed."
    exit 1
fi
echo ""

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=3

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
NUM_GPUS=8
CONFIG_FILE="example/korean_v4_fsdp.yaml"
LOG_DIR="logs"
TEST_MODE=false
BACKGROUND=false
DRY_RUN=false
VALIDATE_BACKBONE=false
BACKBONE_TYPE=""
HFLM_MODEL_PATH=""
INSTALL_DEPS=false
INSTALL_BACKBONE_DEPS=false
FORCE_SINGLE_NODE=false
MASTER_PORT="${MASTER_PORT:-29500}"
HOSTFILE="/horovod/generated/hostfile"

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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --validate-backbone)
            VALIDATE_BACKBONE=true
            shift
            ;;
        --backbone)
            BACKBONE_TYPE="$2"
            shift 2
            ;;
        --hf_lm-model)
            HFLM_MODEL_PATH="$2"
            shift 2
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --install-backbone-deps)
            INSTALL_BACKBONE_DEPS=true
            shift
            ;;
        --single-node)
            FORCE_SINGLE_NODE=true
            shift
            ;;
        --hostfile)
            HOSTFILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "K-Moshi V4 Training Script (MODULAR BACKBONE)"
            echo ""
            echo "Options:"
            echo "  --gpus, -g NUM           Number of GPUs per node (default: 8)"
            echo "  --config, -c FILE        Config YAML file (default: example/korean_v4_fsdp.yaml)"
            echo "  --test, -t               Quick test mode (10 steps)"
            echo "  --background, -b         Run in background with nohup"
            echo "  --dry-run                Show commands without executing"
            echo ""
            echo "Backbone Options:"
            echo "  --validate-backbone      Validate backbone setup before training"
            echo "  --backbone TYPE          Override backbone type (moshi, hf_lm)"
            echo "  --hf_lm-model PATH      Path to HFLM HuggingFace model"
            echo "  --install-backbone-deps  Install backbone dependencies (transformers)"
            echo ""
            echo "Distributed Options:"
            echo "  --single-node            Force single-node mode (ignore hostfile)"
            echo "  --hostfile PATH          Path to hostfile (default: /horovod/generated/hostfile)"
            echo ""
            echo "Dependency Options:"
            echo "  --install-deps           Install required Enhanced Evaluation dependencies"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Backbone Types:"
            echo "  moshi   - Original Moshi transformer (7B, dim=4096)"
            echo "  hf_lm  - HFLM/Mistral Korean LLM (3B, dim=3072)"
            echo "  custom  - User-defined backbone (requires factory registration)"
            echo ""
            echo "Examples:"
            echo "  $0                                     # Auto-detect nodes, full training"
            echo "  $0 --test                              # Quick test (10 steps)"
            echo "  $0 --validate-backbone --test          # Validate backbone then test"
            echo "  $0 --backbone moshi --config v3.yaml   # Use Moshi backbone with V3 config"
            echo "  $0 --gpus 4 --background               # 4 GPUs/node, run in background"
            echo "  $0 --single-node                       # Force single-node mode"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Handle Dependencies Installation
# -----------------------------------------------------------------------------
if [ "${INSTALL_BACKBONE_DEPS}" = true ]; then
    install_backbone_deps
    echo ""
fi

if [ "${INSTALL_DEPS}" = true ]; then
    install_enhanced_eval_deps
    echo ""
fi

# -----------------------------------------------------------------------------
# Validate Backbone (if requested)
# -----------------------------------------------------------------------------
if [ "${VALIDATE_BACKBONE}" = true ]; then
    if ! validate_backbone_setup; then
        print_error "Backbone validation failed"
        exit 1
    fi
    echo ""
fi

# -----------------------------------------------------------------------------
# Detect Multinode Configuration
# -----------------------------------------------------------------------------
MULTINODE=false
NUM_NODES=1
TOTAL_GPUS=${NUM_GPUS}
HEAD_NODE=$(hostname)

if [ "${FORCE_SINGLE_NODE}" = false ] && [ -f "${HOSTFILE}" ]; then
    # Parse hostfile for multinode setup
    NUM_NODES=$(grep -v "^#" "${HOSTFILE}" 2>/dev/null | grep -c "slots" || echo 1)
    HEAD_NODE=$(grep -v "^#" "${HOSTFILE}" 2>/dev/null | head -1 | cut -d' ' -f1)

    if [ "${NUM_NODES}" -gt 1 ]; then
        MULTINODE=true
        TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))
        print_info "Multinode environment detected!"
        print_info "  Hostfile: ${HOSTFILE}"
        print_info "  Nodes: ${NUM_NODES}"
        print_info "  Total GPUs: ${TOTAL_GPUS}"
    else
        print_info "Single node detected from hostfile"
    fi
else
    if [ "${FORCE_SINGLE_NODE}" = true ]; then
        print_info "Single-node mode forced (--single-node)"
    else
        print_info "No hostfile found at ${HOSTFILE}, using single-node mode"
    fi
fi
echo ""

# -----------------------------------------------------------------------------
# Display Configuration
# -----------------------------------------------------------------------------
print_header "Training Configuration"
echo ""
echo "Version 4 Features (MODULAR BACKBONE):"
echo "  - Backbone: Swappable LLM (Moshi, HFLM, Custom)"
echo "  - Dimension Adapter: 4096 <-> backbone_dim bridging"
echo "  - Stereo Input: 17 codebooks (1 text + 8 moshi + 8 user)"
echo "  - Standard Output: dep_q=8 (Moshi audio only)"
echo "  - FULL-DUPLEX: User audio as context (not predicted)"
echo ""
echo "Additional Features:"
echo "  - Two-rate Optimizer (Backbone + DepFormer + Adapter)"
echo "  - Cosine Warmup Scheduler"
echo "  - Advanced Monitoring (backbone, codebook, gradients)"
echo "  - Sample Saving (audio + text)"
echo "  - Research Logging (plots, CSV, summary)"
echo ""
echo "Distributed Training:"
if [ "${MULTINODE}" = true ]; then
    echo -e "  ${GREEN}Mode: MULTINODE (mpirun)${NC}"
    echo "  Nodes: ${NUM_NODES}"
    echo "  GPUs per node: ${NUM_GPUS}"
    echo "  Total GPUs: ${TOTAL_GPUS}"
    echo "  Head node: ${HEAD_NODE}"
else
    echo -e "  ${CYAN}Mode: SINGLE-NODE (torchrun)${NC}"
    echo "  GPUs: ${NUM_GPUS}"
fi
echo ""
echo "Settings:"
echo "  Config: ${CONFIG_FILE}"
if [ -n "${BACKBONE_TYPE}" ]; then
    echo "  Backbone Override: ${BACKBONE_TYPE}"
fi
if [ -n "${HFLM_MODEL_PATH}" ]; then
    echo "  HFLM Model: ${HFLM_MODEL_PATH}"
fi
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

# HuggingFace settings for HFLM
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

echo "Environment Variables:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  TORCH_COMPILE_DISABLE: ${TORCH_COMPILE_DISABLE} (FSDP compatibility)"
echo "  HF_HOME: ${HF_HOME}"
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

# Check manifest files (for V4)
if [ -f "./data/korean_v4_train.jsonl" ]; then
    TRAIN_SAMPLES=$(wc -l < ./data/korean_v4_train.jsonl)
    print_success "Training manifest: ${TRAIN_SAMPLES} samples"
else
    print_warning "Training manifest not found: ./data/korean_v4_train.jsonl"
    echo "         You can copy from V3: cp ./data/korean_v3_train.jsonl ./data/korean_v4_train.jsonl"
fi

if [ -f "./data/korean_v4_valid.jsonl" ]; then
    VALID_SAMPLES=$(wc -l < ./data/korean_v4_valid.jsonl)
    print_success "Validation manifest: ${VALID_SAMPLES} samples"
else
    print_warning "Validation manifest not found: ./data/korean_v4_valid.jsonl"
    echo "         You can copy from V3: cp ./data/korean_v3_valid.jsonl ./data/korean_v4_valid.jsonl"
fi

# Extract and display backbone info from config
echo ""
print_backbone "Checking backbone configuration..."
BACKBONE_FROM_CONFIG=$(python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    config = yaml.safe_load(f)
backbone = config.get('backbone', {})
print(f\"Type: {backbone.get('type', 'moshi')}\")
if backbone.get('type') == 'hf_lm':
    g = backbone.get('hf_lm', {})
    print(f\"Model: {g.get('model_path', 'NOT SET')}\")
    print(f\"Hidden dim: {g.get('hidden_dim', 3072)}\")
    print(f\"Layers: {g.get('num_layers', 30)}\")
    da = backbone.get('dimension_adapter', {})
    print(f\"Adapter enabled: {da.get('enable', False)}\")
" 2>/dev/null || echo "Failed to parse config")
echo "${BACKBONE_FROM_CONFIG}"

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
LOG_FILE="${LOG_DIR}/train_v4_${TIMESTAMP}.log"

# -----------------------------------------------------------------------------
# Build Training Command
# -----------------------------------------------------------------------------
if [ "${TEST_MODE}" = true ]; then
    echo -e "${YELLOW}>>> TEST MODE: Running 10 steps only <<<${NC}"
    echo ""

    # Create temporary test config
    TEST_CONFIG="${LOG_DIR}/test_config_v4_${TIMESTAMP}.yaml"
    cp "${CONFIG_FILE}" "${TEST_CONFIG}"

    # Modify for quick test
    sed -i 's/max_steps: .*/max_steps: 10/' "${TEST_CONFIG}"
    sed -i 's/log_freq: .*/log_freq: 1/' "${TEST_CONFIG}"
    sed -i 's/ckpt_freq: .*/ckpt_freq: 10/' "${TEST_CONFIG}"
    sed -i 's/eval_freq: .*/eval_freq: 5/' "${TEST_CONFIG}"
    sed -i 's/save_freq: .*/save_freq: 5/' "${TEST_CONFIG}"
    sed -i 's/plot_freq: .*/plot_freq: 5/' "${TEST_CONFIG}"
    sed -i "s|run_dir: .*|run_dir: './runs/test_v4_${TIMESTAMP}'|" "${TEST_CONFIG}"
    sed -i 's/overwrite_run_dir: .*/overwrite_run_dir: true/' "${TEST_CONFIG}"

    # Override backbone type if specified
    if [ -n "${BACKBONE_TYPE}" ]; then
        sed -i "s/type: .*/type: \"${BACKBONE_TYPE}\"/" "${TEST_CONFIG}"
    fi

    CONFIG_FILE="${TEST_CONFIG}"
    echo "Test config created: ${TEST_CONFIG}"
    echo ""
fi

# -----------------------------------------------------------------------------
# Build Training Command
# -----------------------------------------------------------------------------
if [ "${MULTINODE}" = true ]; then
    # ==========================================================================
    # MULTINODE: Use mpirun for distributed training across nodes
    # ==========================================================================
    # MPI options
    MPI_OPTS="-np ${TOTAL_GPUS} --npernode ${NUM_GPUS}"
    MPI_OPTS="${MPI_OPTS} --hostfile ${HOSTFILE}"
    MPI_OPTS="${MPI_OPTS} --allow-run-as-root"
    MPI_OPTS="${MPI_OPTS} -bind-to none -map-by slot"

    # MCA parameters for network optimization
    MPI_OPTS="${MPI_OPTS} -mca pml ob1 -mca btl ^openib"
    MPI_OPTS="${MPI_OPTS} -mca orte_keep_fqdn_hostnames t"

    # Environment variables to pass to all nodes
    MPI_ENV="-x CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    MPI_ENV="${MPI_ENV} -x MASTER_ADDR=${HEAD_NODE}"
    MPI_ENV="${MPI_ENV} -x MASTER_PORT=${MASTER_PORT}"
    MPI_ENV="${MPI_ENV} -x PYTHONUNBUFFERED=1"
    MPI_ENV="${MPI_ENV} -x PATH -x PYTHONPATH -x LD_LIBRARY_PATH"

    # NCCL settings
    MPI_ENV="${MPI_ENV} -x NCCL_DEBUG=WARN"
    MPI_ENV="${MPI_ENV} -x NCCL_IB_DISABLE=1"
    MPI_ENV="${MPI_ENV} -x NCCL_SOCKET_IFNAME=eth0"

    # PyTorch settings
    MPI_ENV="${MPI_ENV} -x PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
    MPI_ENV="${MPI_ENV} -x TORCH_COMPILE_DISABLE=1"
    MPI_ENV="${MPI_ENV} -x TORCHDYNAMO_DISABLE=1"
    MPI_ENV="${MPI_ENV} -x OMP_NUM_THREADS=4"

    # HuggingFace settings
    MPI_ENV="${MPI_ENV} -x HF_HOME=${HF_HOME}"
    MPI_ENV="${MPI_ENV} -x TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"

    # Proxy settings (if set)
    [ -n "${http_proxy}" ] && MPI_ENV="${MPI_ENV} -x http_proxy"
    [ -n "${https_proxy}" ] && MPI_ENV="${MPI_ENV} -x https_proxy"
    [ -n "${no_proxy}" ] && MPI_ENV="${MPI_ENV} -x no_proxy"

    TRAIN_CMD="mpirun ${MPI_OPTS} ${MPI_ENV} python ${PROJECT_DIR}/train.py ${CONFIG_FILE}"

    print_info "Using mpirun for multinode training"
else
    # ==========================================================================
    # SINGLE-NODE: Use torchrun for distributed training
    # ==========================================================================
    TRAIN_CMD="torchrun --nproc-per-node ${NUM_GPUS} --master_port ${MASTER_PORT} -m train ${CONFIG_FILE}"

    print_info "Using torchrun for single-node training"
fi

# -----------------------------------------------------------------------------
# Execute Training
# -----------------------------------------------------------------------------
print_header "Starting V4 Training (MODULAR BACKBONE)"
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
    echo "  tensorboard --logdir=./runs/korean_v4_hf_lm/tensorboard"
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
echo "  Checkpoints: ./runs/korean_v4_hf_lm/"
echo "  TensorBoard: ./runs/korean_v4_hf_lm/tensorboard/"
echo "  Samples:     ./runs/korean_v4_hf_lm/samples/"
echo "  Research:    ./runs/korean_v4_hf_lm/research/"
echo "  Log file:    ${LOG_FILE}"
echo ""
echo "V4 Model Notes:"
echo "  - Backbone: Modular (HFLM/Moshi/Custom)"
echo "  - Dimension Adapter: Bridges backbone to Moshi components"
echo "  - Inference: Requires LMModelWrapper for non-Moshi backbones"
echo "  - Checkpoint: Contains backbone + adapter + Moshi components"
echo ""
