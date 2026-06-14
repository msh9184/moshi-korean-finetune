#!/bin/bash
#
# K-Moshi Training Launcher
# MPI-based distributed training for Korean Moshi finetuning
#
# Usage:
#   bash scripts/train_mpi.sh --config example/korean_ddp.yaml
#   bash scripts/train_mpi.sh --config example/korean_fsdp.yaml --dry-run
#
set -e

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOSTFILE="/horovod/generated/hostfile"
MASTER_PORT="${MASTER_PORT:-29500}"

# ─────────────────────────────────────────────────────────────────────────────
# Colors & Symbols
# ─────────────────────────────────────────────────────────────────────────────
R='\033[0;31m'    # Red
G='\033[0;32m'    # Green
Y='\033[1;33m'    # Yellow
B='\033[0;34m'    # Blue
C='\033[0;36m'    # Cyan
M='\033[0;35m'    # Magenta
W='\033[1;37m'    # White Bold
D='\033[0;90m'    # Dark Gray
N='\033[0m'       # No Color

# ─────────────────────────────────────────────────────────────────────────────
# Display Functions
# ─────────────────────────────────────────────────────────────────────────────
banner() {
    echo ""
    echo -e "${C}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
    echo -e "${W}  Moshi Finetune${N} ${D}| MPI Distributed Training${N}"
    echo -e "${C}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
}

section() {
    echo ""
    echo -e "${C}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
    echo -e "${W}  $1${N}"
    echo -e "${C}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
}

info()  { echo -e "  ${D}│${N} $1"; }
item()  { printf "  ${D}│${N} ${C}%-22s${N} %s\n" "$1" "$2"; }
ok()    { echo -e "  ${G}✓${N} $1"; }
warn()  { echo -e "  ${Y}⚠${N} $1"; }
fail()  { echo -e "  ${R}✗${N} $1"; }

# ─────────────────────────────────────────────────────────────────────────────
# YAML Parser (Simple)
# ─────────────────────────────────────────────────────────────────────────────
yaml_get() {
    local file="$1" key="$2"
    grep -E "^${key}:" "$file" 2>/dev/null | head -1 | sed "s/^${key}:[[:space:]]*//" | tr -d "'" | tr -d '"'
}

yaml_get_nested() {
    local file="$1" parent="$2" key="$3"
    awk -v parent="$parent" -v key="$key" '
        $0 ~ "^"parent":" { in_parent=1; next }
        in_parent && /^[a-zA-Z]/ { in_parent=0 }
        in_parent && $0 ~ "^[[:space:]]+"key":" {
            gsub(/^[[:space:]]+/, ""); gsub(key":[[:space:]]*", ""); gsub(/['"'"'"]/, ""); print; exit
        }
    ' "$file"
}

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
show_help() {
    echo ""
    echo -e "${W}Moshi Finetune${N} - MPI Distributed Training"
    echo ""
    echo -e "${W}USAGE${N}"
    echo "  bash scripts/train_mpi.sh --config <config.yaml> [options]"
    echo ""
    echo -e "${W}OPTIONS${N}"
    echo "  -c, --config <file>    Training configuration YAML (required)"
    echo "  -g, --gpus <n>         Number of GPUs per node (default: auto-detect)"
    echo "  -n, --dry-run          Show command without executing"
    echo "  -h, --help             Show this help"
    echo ""
    echo -e "${W}EXAMPLES${N}"
    echo -e "  ${D}# DDP training (8 GPUs)${N}"
    echo "  bash scripts/train_mpi.sh -c example/korean_ddp.yaml"
    echo ""
    echo -e "  ${D}# FSDP training (memory efficient)${N}"
    echo "  bash scripts/train_mpi.sh -c example/korean_fsdp.yaml"
    echo ""
    echo -e "  ${D}# Preview command only${N}"
    echo "  bash scripts/train_mpi.sh -c example/korean_ddp.yaml --dry-run"
    echo ""
    exit 0
}

# ─────────────────────────────────────────────────────────────────────────────
# GPU Detection
# ─────────────────────────────────────────────────────────────────────────────
detect_gpus() {
    if [ -z "$NUM_GPUS" ]; then
        if command -v nvidia-smi &>/dev/null; then
            NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        else
            NUM_GPUS=4
        fi
    fi
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Parse Arguments
# ─────────────────────────────────────────────────────────────────────────────
CONFIG=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)  CONFIG="$2"; shift 2 ;;
        -g|--gpus)    NUM_GPUS="$2"; shift 2 ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        -h|--help)    show_help ;;
        *) fail "Unknown option: $1"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "$CONFIG" ]; then
    fail "Missing required: --config"
    echo "  Run with --help for usage"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    fail "Config not found: $CONFIG"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Load Configuration from YAML
# ─────────────────────────────────────────────────────────────────────────────
detect_gpus

# Training settings
BACKEND=$(yaml_get "$CONFIG" "distributed_backend")
BATCH_SIZE=$(yaml_get "$CONFIG" "batch_size")
MAX_STEPS=$(yaml_get "$CONFIG" "max_steps")
DURATION=$(yaml_get "$CONFIG" "duration_sec")
GRAD_CKPT=$(yaml_get "$CONFIG" "gradient_checkpointing")
RUN_DIR=$(yaml_get "$CONFIG" "run_dir")

# LoRA settings
LORA_ENABLE=$(yaml_get_nested "$CONFIG" "lora" "enable")
LORA_RANK=$(yaml_get_nested "$CONFIG" "lora" "rank")
LORA_SCALING=$(yaml_get_nested "$CONFIG" "lora" "scaling")
FULL_FT=$(yaml_get "$CONFIG" "full_finetuning")

# Data settings
TRAIN_DATA=$(yaml_get_nested "$CONFIG" "data" "train_data")

# Optimizer
LR=$(yaml_get_nested "$CONFIG" "optim" "lr")

# Model paths
MOSHI_PATH=$(yaml_get_nested "$CONFIG" "moshi_paths" "moshi_path")

# Defaults
BACKEND="${BACKEND:-ddp}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"

# ─────────────────────────────────────────────────────────────────────────────
# Compute Cluster Configuration
# ─────────────────────────────────────────────────────────────────────────────
if [ -f "$HOSTFILE" ]; then
    NUM_NODES=$(grep -v "^#" "$HOSTFILE" 2>/dev/null | grep -c "slots" || echo 1)
    HEAD_NODE=$(grep -v "^#" "$HOSTFILE" 2>/dev/null | head -1 | cut -d' ' -f1)
else
    NUM_NODES=1
    HEAD_NODE=$(hostname)
fi
TOTAL_PROCS=$((NUM_NODES * NUM_GPUS))

# ─────────────────────────────────────────────────────────────────────────────
# Display Configuration
# ─────────────────────────────────────────────────────────────────────────────
banner

section "Training Configuration"
item "Config File" "$CONFIG"
item "Output Directory" "${RUN_DIR:-runs/default}"
item "Training Data" "${TRAIN_DATA:-N/A}"

section "Model Architecture"
if [ "$LORA_ENABLE" = "true" ] && [ "$FULL_FT" != "true" ]; then
    echo -e "  ${D}│${N} ${G}LoRA Finetuning${N} ${D}(Parameter Efficient)${N}"
    item "  LoRA Rank" "${LORA_RANK:-8}"
    item "  LoRA Scaling" "${LORA_SCALING:-2.0}"
    echo -e "  ${D}│${N}"
    echo -e "  ${D}│${N} ${D}┌─────────────────────────────────────────────────────────┐${N}"
    echo -e "  ${D}│${N} ${D}│${N}  Moshi 7.7B                                             ${D}│${N}"
    echo -e "  ${D}│${N} ${D}│${N}  ${C}├── Transformer Backbone${N}        ${D}(frozen)${N}              ${D}│${N}"
    echo -e "  ${D}│${N} ${D}│${N}  ${C}│   └── 32 Layers${N}                                      ${D}│${N}"
    echo -e "  ${D}│${N} ${D}│${N}  ${G}├── LoRA Adapters${N}               ${G}(trainable)${N}           ${D}│${N}"
    echo -e "  ${D}│${N} ${D}│${N}  ${G}│   ├── Query Projections${N}                             ${D}│${N}"
    echo -e "  ${D}│${N} ${D}│${N}  ${G}│   ├── Key Projections${N}                               ${D}│${N}"
    echo -e "  ${D}│${N} ${D}│${N}  ${G}│   ├── Value Projections${N}                             ${D}│${N}"
    echo -e "  ${D}│${N} ${D}│${N}  ${G}│   └── Output Projections${N}                            ${D}│${N}"
    echo -e "  ${D}│${N} ${D}│${N}  ${C}├── Audio Encoder (Mimi)${N}        ${D}(frozen)${N}              ${D}│${N}"
    echo -e "  ${D}│${N} ${D}│${N}  ${C}└── Text Tokenizer${N}              ${D}(frozen)${N}              ${D}│${N}"
    echo -e "  ${D}│${N} ${D}└─────────────────────────────────────────────────────────┘${N}"
else
    echo -e "  ${D}│${N} ${Y}Full Finetuning${N} ${D}(All Parameters)${N}"
fi
echo -e "  ${D}│${N}"
item "Base Model" "${MOSHI_PATH:-moshiko-pytorch-bf16}"
item "Codebooks" "17 (1 text + 8 audio x 2 streams)"
item "Parameters" "~7.7B total, ~24M trainable (LoRA)"

section "Distributed Backend"
if [ "$BACKEND" = "fsdp" ]; then
    echo -e "  ${D}│${N} ${G}FSDP${N} - Fully Sharded Data Parallel"
    info "Memory efficient model sharding across GPUs"
    info "Recommended for large batch sizes"
else
    echo -e "  ${D}│${N} ${C}DDP${N} - Distributed Data Parallel"
    info "Full model replica on each GPU"
    info "Simpler debugging, lower communication overhead"
fi

section "GPU Cluster"
item "Head Node" "$HEAD_NODE"
item "Total Nodes" "$NUM_NODES"
item "GPUs per Node" "$NUM_GPUS"
item "Total GPUs" "$TOTAL_PROCS"
item "CUDA Devices" "$CUDA_VISIBLE_DEVICES"

section "Training Hyperparameters"
item "Batch Size" "${BATCH_SIZE} per GPU"
item "Effective Batch" "$((BATCH_SIZE * TOTAL_PROCS)) total"
item "Learning Rate" "${LR:-1e-4}"
item "Max Steps" "$MAX_STEPS"
item "Duration (sec)" "${DURATION:-10.0}"
item "Grad Checkpoint" "${GRAD_CKPT:-true}"

# ─────────────────────────────────────────────────────────────────────────────
# Build Command
# ─────────────────────────────────────────────────────────────────────────────
MPI_OPTS="-np $TOTAL_PROCS --npernode $NUM_GPUS"
[ -f "$HOSTFILE" ] && MPI_OPTS="$MPI_OPTS -hostfile $HOSTFILE"

MPI_ENV="-x CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
MPI_ENV="$MPI_ENV -x MASTER_ADDR=$HEAD_NODE"
MPI_ENV="$MPI_ENV -x MASTER_PORT=$MASTER_PORT"
MPI_ENV="$MPI_ENV -x PYTHONUNBUFFERED=1"
MPI_ENV="$MPI_ENV -x NCCL_IB_DISABLE=0"
MPI_ENV="$MPI_ENV -x NCCL_SOCKET_IFNAME=eth0"
[ -n "$NCCL_DEBUG" ] && MPI_ENV="$MPI_ENV -x NCCL_DEBUG=$NCCL_DEBUG"

FULL_CMD="mpirun $MPI_OPTS $MPI_ENV python $PROJECT_ROOT/train.py --config $CONFIG"

section "Execution Command"
echo -e "  ${D}│${N}"
echo -e "  ${D}│${N} ${W}$FULL_CMD${N}"
echo -e "  ${D}│${N}"

# ─────────────────────────────────────────────────────────────────────────────
# Execute
# ─────────────────────────────────────────────────────────────────────────────
if [ -n "$DRY_RUN" ]; then
    echo ""
    echo -e "${Y}━━━ DRY RUN ━━━ Command not executed${N}"
    echo ""
    exit 0
fi

echo ""
echo -e "${C}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
echo -e "${W}  Starting Training...${N}"
echo -e "${C}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
echo ""

cd "$PROJECT_ROOT"
eval "$FULL_CMD"

echo ""
echo -e "${G}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
echo -e "${W}  Training Completed${N}"
echo -e "${G}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
echo ""
