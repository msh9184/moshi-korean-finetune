#!/bin/bash
# =============================================================================
# K-Moshi Stage 2: Transfer to GPU Server
# =============================================================================
#
# Purpose: Transfer downloaded resources to GPU server for synthesis
# Run this script on a local PC after completing downloads
#
# Usage:
#   chmod +x transfer_to_gpu.sh
#   ./transfer_to_gpu.sh --server user@gpu-server [--download-dir /path/to/downloads]
#
# =============================================================================

set -e

# Default settings
DOWNLOAD_DIR="${DOWNLOAD_DIR:-./k-moshi-downloads}"
GPU_SERVER=""
REMOTE_BASE="/path/to/workspace"
REMOTE_DATA="$REMOTE_BASE/data_preparation_stage2/data"
REMOTE_MODELS="/models"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server|-s)
            GPU_SERVER="$2"
            shift 2
            ;;
        --download-dir|-d)
            DOWNLOAD_DIR="$2"
            shift 2
            ;;
        --remote-base)
            REMOTE_BASE="$2"
            REMOTE_DATA="$REMOTE_BASE/data_preparation_stage2/data"
            shift 2
            ;;
        --remote-models)
            REMOTE_MODELS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 --server user@gpu-server [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --server, -s     GPU server address (required)"
            echo "  --download-dir   Local download directory"
            echo "  --remote-base    Remote base path"
            echo "  --remote-models  Remote models path"
            echo "  --dry-run        Show what would be transferred"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$GPU_SERVER" ]; then
    echo "Error: --server is required"
    echo "Usage: $0 --server user@gpu-server"
    exit 1
fi

echo "============================================================"
echo "  K-Moshi Stage 2: Transfer to GPU Server"
echo "============================================================"
echo ""
echo "Local directory:  $DOWNLOAD_DIR"
echo "GPU server:       $GPU_SERVER"
echo "Remote data:      $REMOTE_DATA"
echo "Remote models:    $REMOTE_MODELS"
echo "Dry run:          $DRY_RUN"
echo ""

# Function to run rsync
transfer() {
    local src="$1"
    local dst="$2"
    local desc="$3"

    if [ ! -e "$src" ]; then
        echo "  ⏭ Skipping $desc (source not found)"
        return
    fi

    echo "  → Transferring $desc..."

    if [ "$DRY_RUN" = true ]; then
        echo "    [DRY-RUN] rsync -avz $src → $GPU_SERVER:$dst"
    else
        rsync -avz --progress "$src" "$GPU_SERVER:$dst" || {
            echo "  ⚠ Transfer failed for $desc"
            return 1
        }
        echo "  ✓ $desc transferred"
    fi
}

# -----------------------------------------------------------------------------
# 1. Create Remote Directories
# -----------------------------------------------------------------------------
echo "[1/5] Creating remote directories..."
echo "--------------------------------------------------------------"

if [ "$DRY_RUN" = true ]; then
    echo "  [DRY-RUN] Would create directories on $GPU_SERVER"
else
    ssh "$GPU_SERVER" "mkdir -p $REMOTE_DATA/{corpus/korean,corpus/english,voices}" 2>/dev/null || {
        echo "  ⚠ Could not create remote directories. They may already exist."
    }
    ssh "$GPU_SERVER" "mkdir -p $REMOTE_MODELS" 2>/dev/null || true
    echo "  ✓ Remote directories ready"
fi
echo ""

# -----------------------------------------------------------------------------
# 2. Transfer Models
# -----------------------------------------------------------------------------
echo "[2/5] Transferring TTS models..."
echo "--------------------------------------------------------------"

# Supertonic-2
if [ -d "$DOWNLOAD_DIR/models/supertonic-2" ]; then
    transfer "$DOWNLOAD_DIR/models/supertonic-2/" \
        "$REMOTE_MODELS/supertonic-2/" \
        "Supertonic-2 TTS"
else
    echo "  ⏭ Supertonic-2 not found"
fi

# Local LLM (if downloaded)
if [ -d "$DOWNLOAD_DIR/models/gemma-2-9b-it" ]; then
    transfer "$DOWNLOAD_DIR/models/gemma-2-9b-it/" \
        "$REMOTE_MODELS/gemma-2-9b-it/" \
        "Gemma-2-9B LLM"
fi

if [ -d "$DOWNLOAD_DIR/models/solar-10.7b" ]; then
    transfer "$DOWNLOAD_DIR/models/solar-10.7b/" \
        "$REMOTE_MODELS/solar-10.7b/" \
        "SOLAR-10.7B LLM"
fi
echo ""

# -----------------------------------------------------------------------------
# 3. Transfer Korean Corpus
# -----------------------------------------------------------------------------
echo "[3/5] Transferring Korean corpus..."
echo "--------------------------------------------------------------"

# AI Hub
if [ -d "$DOWNLOAD_DIR/corpus/korean/aihub" ] && \
   [ "$(ls -A $DOWNLOAD_DIR/corpus/korean/aihub 2>/dev/null)" ]; then
    transfer "$DOWNLOAD_DIR/corpus/korean/aihub/" \
        "$REMOTE_DATA/corpus/korean/aihub/" \
        "AI Hub Korean Corpus"
else
    echo "  ⏭ AI Hub corpus not found or empty"
fi

# 모두의말뭉치
if [ -d "$DOWNLOAD_DIR/corpus/korean/nikl" ] && \
   [ "$(ls -A $DOWNLOAD_DIR/corpus/korean/nikl 2>/dev/null)" ]; then
    transfer "$DOWNLOAD_DIR/corpus/korean/nikl/" \
        "$REMOTE_DATA/corpus/korean/nikl/" \
        "모두의말뭉치 Corpus"
else
    echo "  ⏭ 모두의말뭉치 corpus not found or empty"
fi
echo ""

# -----------------------------------------------------------------------------
# 4. Transfer English Corpus
# -----------------------------------------------------------------------------
echo "[4/5] Transferring English corpus..."
echo "--------------------------------------------------------------"

# DailyDialog
if [ -d "$DOWNLOAD_DIR/corpus/english/ijcnlp_dailydialog" ]; then
    transfer "$DOWNLOAD_DIR/corpus/english/ijcnlp_dailydialog/" \
        "$REMOTE_DATA/corpus/english/dailydialog/" \
        "DailyDialog Corpus"
elif [ -d "$DOWNLOAD_DIR/corpus/english/dailydialog" ]; then
    transfer "$DOWNLOAD_DIR/corpus/english/dailydialog/" \
        "$REMOTE_DATA/corpus/english/dailydialog/" \
        "DailyDialog Corpus"
else
    echo "  ⏭ DailyDialog corpus not found"
fi

# EmpatheticDialogues
if [ -d "$DOWNLOAD_DIR/corpus/english/empathetic" ]; then
    transfer "$DOWNLOAD_DIR/corpus/english/empathetic/" \
        "$REMOTE_DATA/corpus/english/empathetic/" \
        "EmpatheticDialogues Corpus"
fi
echo ""

# -----------------------------------------------------------------------------
# 5. Transfer Voice Samples
# -----------------------------------------------------------------------------
echo "[5/5] Transferring voice samples..."
echo "--------------------------------------------------------------"

# Moshi reference voice
if [ -f "$DOWNLOAD_DIR/voices/moshi/moshi_reference.wav" ]; then
    transfer "$DOWNLOAD_DIR/voices/moshi/" \
        "$REMOTE_DATA/voices/moshi/" \
        "K-Moshi Reference Voice"
else
    echo "  ⏭ K-Moshi reference voice not found"
    echo "    Please record reference audio and save to:"
    echo "    $DOWNLOAD_DIR/voices/moshi/moshi_reference.wav"
fi

# User voices
if [ -d "$DOWNLOAD_DIR/voices/users" ] && \
   [ "$(ls -A $DOWNLOAD_DIR/voices/users 2>/dev/null)" ]; then
    transfer "$DOWNLOAD_DIR/voices/users/" \
        "$REMOTE_DATA/voices/users/" \
        "User Voice Samples"
else
    echo "  ⏭ User voice samples not found"
fi
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "============================================================"
echo "  Transfer Summary"
echo "============================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "This was a DRY RUN. No files were actually transferred."
    echo "Remove --dry-run flag to perform actual transfer."
else
    echo "Transfer complete!"
    echo ""
    echo "Data location on GPU server:"
    echo "  Corpus:   $GPU_SERVER:$REMOTE_DATA/corpus/"
    echo "  Voices:   $GPU_SERVER:$REMOTE_DATA/voices/"
    echo "  Models:   $GPU_SERVER:$REMOTE_MODELS/"
fi
echo ""

# Check what's missing
echo "Status Check:"
echo "--------------------------------------------------------------"

check_remote() {
    local path="$1"
    local desc="$2"

    if [ "$DRY_RUN" = true ]; then
        echo "  ⏸ $desc (dry-run, status unknown)"
        return
    fi

    if ssh "$GPU_SERVER" "[ -e $path ]" 2>/dev/null; then
        echo "  ✅ $desc"
    else
        echo "  ❌ $desc (missing)"
    fi
}

check_remote "$REMOTE_MODELS/supertonic-2" "Supertonic-2 TTS"
check_remote "$REMOTE_DATA/corpus/korean/aihub" "AI Hub Korean Corpus"
check_remote "$REMOTE_DATA/corpus/korean/nikl" "모두의말뭉치"
check_remote "$REMOTE_DATA/corpus/english/dailydialog" "DailyDialog English"
check_remote "$REMOTE_DATA/voices/moshi/moshi_reference.wav" "K-Moshi Reference Voice"
check_remote "$REMOTE_DATA/voices/users" "User Voice Samples"

echo ""
echo "============================================================"
echo "  Next Steps on GPU Server"
echo "============================================================"
echo ""
echo "1. SSH to GPU server:"
echo "   ssh $GPU_SERVER"
echo ""
echo "2. Verify data:"
echo "   ls -la $REMOTE_DATA/corpus/"
echo "   ls -la $REMOTE_DATA/voices/"
echo ""
echo "3. Run synthesis pipeline:"
echo "   cd $REMOTE_BASE"
echo "   python -m data_preparation_stage2.scripts.run_synthesis \\"
echo "       data_preparation_stage2/configs/default.yaml"
echo ""
echo "============================================================"
