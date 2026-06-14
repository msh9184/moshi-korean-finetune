#!/bin/bash
# =============================================================================
# K-Moshi Stage 2: Offline Data Downloader
# =============================================================================
#
# Purpose: Download all required resources for K-Moshi data synthesis
# Run this script on a local PC with internet access
#
# Usage:
#   chmod +x download_all.sh
#   ./download_all.sh [--download-dir /path/to/downloads]
#
# =============================================================================

set -e

# Default download directory
DOWNLOAD_DIR="${DOWNLOAD_DIR:-./k-moshi-downloads}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --download-dir)
            DOWNLOAD_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--download-dir /path/to/downloads]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$DOWNLOAD_DIR"/{models,corpus/english,corpus/korean}

echo "============================================================"
echo "  K-Moshi Stage 2: Offline Data Downloader"
echo "============================================================"
echo ""
echo "Download directory: $DOWNLOAD_DIR"
echo ""

# -----------------------------------------------------------------------------
# 1. TTS Models
# -----------------------------------------------------------------------------
echo "[1/5] Downloading TTS Models..."
echo "--------------------------------------------------------------"

# Supertonic-2
if [ -d "$DOWNLOAD_DIR/models/supertonic-2" ]; then
    echo "  ✓ Supertonic-2 already exists, skipping..."
else
    echo "  → Downloading Supertonic-2 from HuggingFace..."
    git lfs install
    git clone https://huggingface.co/Supertone/supertonic-2 \
        "$DOWNLOAD_DIR/models/supertonic-2" || {
        echo "  ⚠ Failed to clone Supertonic-2. Manual download required:"
        echo "    https://huggingface.co/Supertone/supertonic-2"
    }
fi

# OpenAudio S1 Mini - Check if already downloaded
echo "  → OpenAudio S1 Mini:"
echo "    This model should be pre-downloaded from NeMo-voice_agent"
echo "    If not available, check: https://github.com/NVIDIA/NeMo-voice_agent"
echo ""

# -----------------------------------------------------------------------------
# 2. English Corpora
# -----------------------------------------------------------------------------
echo "[2/5] Downloading English Corpora..."
echo "--------------------------------------------------------------"

# DailyDialog
if [ -f "$DOWNLOAD_DIR/corpus/english/dailydialog/dialogues_text.txt" ]; then
    echo "  ✓ DailyDialog already exists, skipping..."
else
    echo "  → Downloading DailyDialog..."
    DAILYDIALOG_URL="http://yanran.li/files/ijcnlp_dailydialog.zip"
    wget -q --show-progress -O "$DOWNLOAD_DIR/corpus/english/dailydialog.zip" \
        "$DAILYDIALOG_URL" || {
        echo "  ⚠ Failed to download DailyDialog from primary URL"
        echo "    Try manual download from: $DAILYDIALOG_URL"
    }

    if [ -f "$DOWNLOAD_DIR/corpus/english/dailydialog.zip" ]; then
        echo "  → Extracting DailyDialog..."
        unzip -q "$DOWNLOAD_DIR/corpus/english/dailydialog.zip" \
            -d "$DOWNLOAD_DIR/corpus/english/"
        rm "$DOWNLOAD_DIR/corpus/english/dailydialog.zip"
        echo "  ✓ DailyDialog extracted"
    fi
fi

# EmpatheticDialogues (from Facebook Research)
if [ -d "$DOWNLOAD_DIR/corpus/english/empathetic" ]; then
    echo "  ✓ EmpatheticDialogues already exists, skipping..."
else
    echo "  → EmpatheticDialogues:"
    echo "    Download from: https://github.com/facebookresearch/EmpatheticDialogues"
    echo "    Save to: $DOWNLOAD_DIR/corpus/english/empathetic/"
fi

# PersonaChat
echo "  → PersonaChat:"
echo "    Download from: https://github.com/facebookresearch/ParlAI"
echo "    (persona_chat dataset)"
echo ""

# -----------------------------------------------------------------------------
# 3. Korean Corpora (Manual Download Required)
# -----------------------------------------------------------------------------
echo "[3/5] Korean Corpora (Manual Download Required)..."
echo "--------------------------------------------------------------"

echo "  → AI Hub 한국어 일상대화:"
echo "    1. Visit: https://aihub.or.kr"
echo "    2. Sign up and login"
echo "    3. Search: '한국어 일상대화' or '감성 대화'"
echo "    4. Apply for data usage (may take 1-3 days)"
echo "    5. Download and save to: $DOWNLOAD_DIR/corpus/korean/aihub/"
echo ""

echo "  → 모두의말뭉치 일상대화:"
echo "    1. Visit: https://corpus.korean.go.kr"
echo "    2. Sign up (research purpose)"
echo "    3. Download: 일상대화 말뭉치"
echo "    4. Save to: $DOWNLOAD_DIR/corpus/korean/nikl/"
echo ""

# Create placeholder directories
mkdir -p "$DOWNLOAD_DIR/corpus/korean/aihub"
mkdir -p "$DOWNLOAD_DIR/corpus/korean/nikl"

# Create README for manual downloads
cat > "$DOWNLOAD_DIR/corpus/korean/README.md" << 'EOF'
# Korean Corpus Manual Download Instructions

## AI Hub (https://aihub.or.kr)

### Recommended Datasets:
1. **한국어 일상대화 데이터**: General daily conversations
2. **감성 대화 데이터**: Emotional conversations with sentiment labels
3. **고객 응대 데이터**: Customer service conversations

### Download Steps:
1. Create account at https://aihub.or.kr
2. Login and navigate to "AI 데이터"
3. Search for the dataset name
4. Click "이용신청" (Apply for usage)
5. Wait for approval (1-3 business days)
6. Download and extract to `aihub/` directory

## 모두의말뭉치 (https://corpus.korean.go.kr)

### Recommended Datasets:
1. **일상대화 말뭉치**: Daily conversation corpus
2. **구어 말뭉치**: Spoken language corpus

### Download Steps:
1. Create account at https://corpus.korean.go.kr
2. Select "연구" (Research) as usage purpose
3. Apply for data access
4. Download and extract to `nikl/` directory

## Directory Structure After Download:
```
korean/
├── aihub/
│   ├── daily_conversation/
│   └── emotional_dialogue/
└── nikl/
    ├── daily_dialogue/
    └── spoken_corpus/
```
EOF

echo "  ✓ Created README with download instructions"
echo ""

# -----------------------------------------------------------------------------
# 4. Voice Samples (Optional)
# -----------------------------------------------------------------------------
echo "[4/5] Voice Samples..."
echo "--------------------------------------------------------------"

mkdir -p "$DOWNLOAD_DIR/voices/moshi"
mkdir -p "$DOWNLOAD_DIR/voices/users"

echo "  → K-Moshi Reference Voice:"
echo "    Record 10-30 seconds of reference audio"
echo "    Use the following transcript:"
echo ""
echo "    ---"
echo "    안녕하세요, 저는 케이모시입니다."
echo "    한국어 음성 대화 AI예요."
echo "    무엇이든 편하게 물어봐 주세요."
echo "    오늘 하루도 좋은 하루 되세요."
echo "    ---"
echo ""
echo "    Save to: $DOWNLOAD_DIR/voices/moshi/moshi_reference.wav"
echo ""

echo "  → User Voice Samples (10-20 recommended):"
echo "    Record or collect diverse voice samples"
echo "    Save to: $DOWNLOAD_DIR/voices/users/"
echo ""

# Create reference transcript file
cat > "$DOWNLOAD_DIR/voices/moshi/reference_transcript.txt" << 'EOF'
안녕하세요, 저는 케이모시입니다.
한국어 음성 대화 AI예요.
무엇이든 편하게 물어봐 주세요.
오늘 하루도 좋은 하루 되세요.
EOF

echo "  ✓ Created reference transcript"
echo ""

# -----------------------------------------------------------------------------
# 5. Local LLM (Optional)
# -----------------------------------------------------------------------------
echo "[5/5] Local LLM for Spoken Conversion (Optional)..."
echo "--------------------------------------------------------------"

echo "  → For offline spoken-style conversion, download one of:"
echo ""
echo "    Option A: Gemma-2-9B"
echo "      huggingface-cli download google/gemma-2-9b-it \\"
echo "        --local-dir $DOWNLOAD_DIR/models/gemma-2-9b-it"
echo ""
echo "    Option B: SOLAR-10.7B-Ko"
echo "      huggingface-cli download upstage/SOLAR-10.7B-Instruct-v1.0 \\"
echo "        --local-dir $DOWNLOAD_DIR/models/solar-10.7b"
echo ""
echo "    Note: These are large models (15-30GB). Rule-based conversion"
echo "    can be used as an alternative without LLM."
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "============================================================"
echo "  Download Summary"
echo "============================================================"
echo ""
echo "Auto-downloaded:"
if [ -d "$DOWNLOAD_DIR/models/supertonic-2" ]; then
    echo "  ✅ Supertonic-2 TTS Model"
else
    echo "  ⏳ Supertonic-2 (download failed or pending)"
fi
if [ -f "$DOWNLOAD_DIR/corpus/english/ijcnlp_dailydialog/dialogues_text.txt" ] || \
   [ -f "$DOWNLOAD_DIR/corpus/english/dailydialog/dialogues_text.txt" ]; then
    echo "  ✅ DailyDialog English Corpus"
else
    echo "  ⏳ DailyDialog (download failed or pending)"
fi
echo ""

echo "Manual download required:"
echo "  ⏳ AI Hub 한국어 일상대화"
echo "  ⏳ 모두의말뭉치 일상대화"
echo "  ⏳ K-Moshi Reference Voice Recording"
echo "  ⏳ User Voice Samples"
echo ""

echo "Optional:"
echo "  ⏳ Local LLM (Gemma-2 or SOLAR)"
echo "  ⏳ EmpatheticDialogues"
echo "  ⏳ PersonaChat"
echo ""

echo "============================================================"
echo "  Next Steps"
echo "============================================================"
echo ""
echo "1. Complete manual downloads listed above"
echo "2. Record K-Moshi reference voice"
echo "3. Run transfer script to copy to GPU server:"
echo "   ./transfer_to_gpu.sh --download-dir $DOWNLOAD_DIR \\"
echo "       --server user@gpu-server"
echo ""
echo "Download directory: $DOWNLOAD_DIR"
echo "============================================================"
