#!/bin/bash
# =============================================================================
# K-Moshi Full Finetuned Model Serving Script
# =============================================================================
#
# Converts K-Moshi full finetuning checkpoint to Rust format and serves via
# moshi-backend on GPU server.
#
# Prerequisites:
#   1. K-Moshi training completed with checkpoint
#   2. moshi repository cloned and Rust backend built
#   3. GPU server with CUDA support
#
# Usage:
#   ./scripts/serve_full_finetuned.sh \
#       --checkpoint ./runs/korean_v2/checkpoints/checkpoint_010000/consolidated/consolidated.safetensors \
#       --moshi-repo /path/to/moshi \
#       --port 8998
#
# SSH Tunnel (from Windows PC):
#   ssh -L 8998:localhost:8998 user@gpu-server
#   → Access at localhost:8998
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Script Location
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# -----------------------------------------------------------------------------
# Color Codes
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
# Default Configuration
# -----------------------------------------------------------------------------
CHECKPOINT=""
MOSHI_REPO=""
PORT=8998
OUTPUT_DIR="${PROJECT_DIR}/models"
INSTANCE_NAME="korean-moshi-full"
SKIP_CONVERSION=false
DRY_RUN=false

# Model paths - Updated to use local models directory
# For Korean finetuned model, use the custom Korean tokenizer
TOKENIZER_PATH="${PROJECT_DIR}/models/tokenizer_spe_unigram_v32000_max_500_pad_bos_eos/tokenizer.model"
MIMI_PATH="${PROJECT_DIR}/models/tokenizer-e351c8d8-checkpoint125.safetensors"

# Rust server paths
LOG_DIR="${PROJECT_DIR}/logs"
CERT_DIR="."
STATIC_DIR=""
ADDR="0.0.0.0"

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint|-c)
            CHECKPOINT="$2"
            shift 2
            ;;
        --moshi-repo|-m)
            MOSHI_REPO="$2"
            shift 2
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --instance-name)
            INSTANCE_NAME="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --mimi)
            MIMI_PATH="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --static-dir)
            STATIC_DIR="$2"
            shift 2
            ;;
        --addr)
            ADDR="$2"
            shift 2
            ;;
        --skip-conversion)
            SKIP_CONVERSION=true
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
            echo "  --checkpoint, -c PATH     K-Moshi checkpoint (consolidated.safetensors)"
            echo "  --moshi-repo, -m PATH     Path to moshi repository"
            echo "  --port, -p PORT           Server port (default: 8998)"
            echo "  --output-dir, -o PATH     Output directory for converted model"
            echo "  --instance-name NAME      Server instance name"
            echo "  --tokenizer PATH          Path to tokenizer_spm_32k_3.model"
            echo "  --mimi PATH               Path to mimi tokenizer checkpoint"
            echo "  --log-dir PATH            Log directory for Rust server (default: PROJECT/logs)"
            echo "  --static-dir PATH         Static files directory for web UI"
            echo "  --addr ADDRESS            Bind address (default: 0.0.0.0)"
            echo "  --skip-conversion         Skip conversion, use existing converted model"
            echo "  --dry-run                 Show commands without executing"
            echo "  --help, -h                Show this help"
            echo ""
            echo "Example:"
            echo "  $0 --checkpoint ./runs/korean_v2/checkpoints/checkpoint_010000/consolidated/consolidated.safetensors \\"
            echo "     --moshi-repo /path/to/moshi --port 8998"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
print_header "K-Moshi Full Finetuned Model Serving"
echo ""

if [ -z "$CHECKPOINT" ]; then
    print_error "Checkpoint path required (--checkpoint)"
    exit 1
fi

if [ -z "$MOSHI_REPO" ]; then
    print_error "Moshi repository path required (--moshi-repo)"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ] && [ "$SKIP_CONVERSION" = false ]; then
    print_error "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -d "$MOSHI_REPO" ]; then
    print_error "Moshi repository not found: $MOSHI_REPO"
    exit 1
fi

RUST_DIR="${MOSHI_REPO}/rust"
if [ ! -d "$RUST_DIR" ]; then
    print_error "Rust directory not found: $RUST_DIR"
    exit 1
fi

print_success "Checkpoint: $CHECKPOINT"
print_success "Moshi repo: $MOSHI_REPO"
print_success "Port: $PORT"
echo ""

# -----------------------------------------------------------------------------
# Phase 1: Convert Checkpoint
# -----------------------------------------------------------------------------
CONVERTED_MODEL="${OUTPUT_DIR}/${INSTANCE_NAME}.safetensors"
CONFIG_FILE="${OUTPUT_DIR}/config-${INSTANCE_NAME}.json"

if [ "$SKIP_CONVERSION" = false ]; then
    print_header "Phase 1: Converting Checkpoint to Rust Format"
    echo ""

    mkdir -p "$OUTPUT_DIR"

    CONVERT_CMD="python ${SCRIPT_DIR}/convert_to_rust.py \
        --checkpoint \"$CHECKPOINT\" \
        --output \"$CONVERTED_MODEL\" \
        --dtype bf16 \
        --verbose"

    echo "Command: $CONVERT_CMD"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute conversion"
    else
        eval $CONVERT_CMD
        print_success "Conversion complete: $CONVERTED_MODEL"
    fi
else
    echo "Skipping conversion, using existing: $CONVERTED_MODEL"
    if [ ! -f "$CONVERTED_MODEL" ]; then
        print_error "Converted model not found: $CONVERTED_MODEL"
        exit 1
    fi
fi

echo ""

# -----------------------------------------------------------------------------
# Phase 2: Generate Rust Config
# -----------------------------------------------------------------------------
print_header "Phase 2: Generating Rust Backend Config"
echo ""

# Ensure log directory exists
mkdir -p "$LOG_DIR"

cat > "$CONFIG_FILE" << EOF
{
    "instance_name": "${INSTANCE_NAME}",
    "lm_model_file": "${CONVERTED_MODEL}",
    "text_tokenizer_file": "${TOKENIZER_PATH}",
    "mimi_model_file": "${MIMI_PATH}",
    "mimi_num_codebooks": 8,
    "log_dir": "${LOG_DIR}",
    "cert_dir": "${CERT_DIR}",
    "static_dir": "${STATIC_DIR}",
    "addr": "${ADDR}",
    "port": ${PORT}
}
EOF

print_success "Config generated: $CONFIG_FILE"
cat "$CONFIG_FILE"
echo ""

# -----------------------------------------------------------------------------
# Phase 3: Check/Build Rust Backend
# -----------------------------------------------------------------------------
print_header "Phase 3: Checking Rust Backend"
echo ""

RELEASE_BINARY="${MOSHI_REPO}/rust/target/release/moshi-backend"

if [ -f "$RELEASE_BINARY" ]; then
    print_success "Release binary found: $RELEASE_BINARY"
else
    print_warning "Release binary not found at: $RELEASE_BINARY"

    # Check if cargo is available
    if command -v cargo &> /dev/null; then
        print_warning "Building with cargo..."
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would run: cargo build --release -p moshi-backend"
        else
            cd "${MOSHI_REPO}/rust"
            cargo build --release -p moshi-backend
            print_success "Build complete"
            cd "$PROJECT_DIR"
        fi
    else
        print_error "cargo not found and binary doesn't exist!"
        echo ""
        echo "Options:"
        echo "  1. Build moshi-backend on a machine with Rust/cargo installed"
        echo "  2. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo "  3. Copy pre-built binary to: $RELEASE_BINARY"
        echo ""
        echo "If the binary exists elsewhere, specify with --moshi-repo pointing to the correct location."
        exit 1
    fi
fi

# Final verification
if [ ! -f "$RELEASE_BINARY" ]; then
    print_error "Binary still not found after build attempt: $RELEASE_BINARY"
    exit 1
fi

echo ""

# -----------------------------------------------------------------------------
# Phase 4: Start Server
# -----------------------------------------------------------------------------
print_header "Phase 4: Starting Rust Backend Server"
echo ""

# Server command with standalone mode
SERVER_CMD="${RELEASE_BINARY} --config \"${CONFIG_FILE}\" standalone"

echo "Server command:"
echo "  $SERVER_CMD"
echo ""
echo "============================================================"
echo "SSH Tunnel Instructions (run from your local machine):"
echo "============================================================"
echo ""
echo "  ssh -L ${PORT}:localhost:${PORT} user@$(hostname)"
echo ""
echo "Then access in browser: http://localhost:${PORT}"
echo ""
echo "============================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would start server"
else
    print_success "Starting Moshi server on port ${PORT}..."
    echo ""
    eval $SERVER_CMD
fi
