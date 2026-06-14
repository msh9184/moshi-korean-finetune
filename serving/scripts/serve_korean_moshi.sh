#!/bin/bash
#
# Korean Moshi Serving Script
# Converts LoRA weights and serves with Rust backend
#
set -e

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Model paths (modify these as needed)
BASE_MODEL="/path/to/moshi_base.safetensors"
MIMI_MODEL="/path/to/mimi.safetensors"
TOKENIZER="/path/to/tokenizer.model"
FUSED_MODEL="$PROJECT_ROOT/models/korean-moshi-fused.safetensors"

# Default LoRA path (can be overridden via argument)
LORA_WEIGHT="${1:-/path/to/lora.safetensors}"

# ─────────────────────────────────────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────────────────────────────────────
C='\033[0;36m'
G='\033[0;32m'
Y='\033[1;33m'
R='\033[0;31m'
W='\033[1;37m'
D='\033[0;90m'
N='\033[0m'

section() {
    echo ""
    echo -e "${C}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
    echo -e "${W}  $1${N}"
    echo -e "${C}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
}

item() { printf "  ${D}│${N} ${C}%-20s${N} %s\n" "$1" "$2"; }
ok()   { echo -e "  ${G}✓${N} $1"; }
fail() { echo -e "  ${R}✗${N} $1"; }

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
show_help() {
    echo ""
    echo -e "${W}Korean Moshi Serving Script${N}"
    echo ""
    echo -e "${W}USAGE${N}"
    echo "  $0 [lora_weight_path] [--skip-convert] [--port PORT]"
    echo ""
    echo -e "${W}OPTIONS${N}"
    echo "  lora_weight_path    Path to LoRA weights (default: tutorial_toy2)"
    echo "  --skip-convert      Skip LoRA fusion, use existing fused model"
    echo "  --port PORT         Server port (default: 8998)"
    echo "  --help              Show this help"
    echo ""
    echo -e "${W}EXAMPLES${N}"
    echo "  # Convert and serve"
    echo "  $0 /path/to/lora.safetensors"
    echo ""
    echo "  # Skip conversion, just serve"
    echo "  $0 --skip-convert"
    echo ""
    exit 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Parse Arguments
# ─────────────────────────────────────────────────────────────────────────────
SKIP_CONVERT=""
PORT="8998"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-convert) SKIP_CONVERT=1; shift ;;
        --port) PORT="$2"; shift 2 ;;
        --help|-h) show_help ;;
        -*) fail "Unknown option: $1"; exit 1 ;;
        *) LORA_WEIGHT="$1"; shift ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
section "Validation"

if [ -z "$SKIP_CONVERT" ]; then
    if [ ! -f "$LORA_WEIGHT" ]; then
        fail "LoRA weights not found: $LORA_WEIGHT"
        exit 1
    fi
    ok "LoRA weights found"
fi

if [ ! -f "$BASE_MODEL" ]; then
    fail "Base model not found: $BASE_MODEL"
    exit 1
fi
ok "Base model found"

if [ ! -f "$MIMI_MODEL" ]; then
    fail "Mimi model not found: $MIMI_MODEL"
    exit 1
fi
ok "Mimi model found"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Convert LoRA to Rust Format
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "$SKIP_CONVERT" ]; then
    section "Step 1: Fusing LoRA weights"
    item "LoRA weights" "$LORA_WEIGHT"
    item "Output" "$FUSED_MODEL"

    mkdir -p "$(dirname "$FUSED_MODEL")"

    cd "$PROJECT_ROOT"
    python scripts/import_rust_lora.py \
        --moshi-weight "$BASE_MODEL" \
        --mimi-weight "$MIMI_MODEL" \
        --tokenizer "$TOKENIZER" \
        --lora-weight "$LORA_WEIGHT" \
        "$FUSED_MODEL"

    if [ -f "$FUSED_MODEL" ]; then
        ok "Fused model created successfully"
        item "Size" "$(du -h "$FUSED_MODEL" | cut -f1)"
    else
        fail "Failed to create fused model"
        exit 1
    fi
else
    section "Step 1: Skipping LoRA fusion (using existing model)"
    if [ ! -f "$FUSED_MODEL" ]; then
        fail "Fused model not found: $FUSED_MODEL"
        echo "  Run without --skip-convert first"
        exit 1
    fi
    ok "Using existing fused model"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Update Config
# ─────────────────────────────────────────────────────────────────────────────
section "Step 2: Updating server config"

CONFIG_FILE="$PROJECT_ROOT/rust/moshi-backend/config-korean.json"
mkdir -p "$PROJECT_ROOT/logs"

cat > "$CONFIG_FILE" << EOF
{
  "instance_name": "korean-moshi",
  "hf_repo": "",
  "lm_model_file": "$FUSED_MODEL",
  "text_tokenizer_file": "$TOKENIZER",
  "log_dir": "$PROJECT_ROOT/logs",
  "mimi_model_file": "$MIMI_MODEL",
  "mimi_num_codebooks": 8,
  "static_dir": "../../client/dist",
  "addr": "0.0.0.0",
  "port": $PORT,
  "cert_dir": "."
}
EOF

ok "Config updated: $CONFIG_FILE"
item "Port" "$PORT"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Start Server
# ─────────────────────────────────────────────────────────────────────────────
section "Step 3: Starting Rust server"

cd "$PROJECT_ROOT/rust/moshi-backend"
echo ""
echo -e "${G}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
echo -e "${W}  Server starting on http://0.0.0.0:$PORT${N}"
echo -e "${G}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
echo ""

cargo run --release -- --config config-korean.json standalone
