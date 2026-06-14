#!/bin/bash
# =============================================================================
# K-Moshi Environment Setup Script (v2.0 - Server-Aware)
# =============================================================================
#
# Optimized for GPU servers with pre-installed PyTorch environments.
# Detects existing compatible packages and only installs what's missing.
#
# Usage:
#   ./scripts/setup_environment.sh              # Smart install (skip existing)
#   ./scripts/setup_environment.sh --force      # Force reinstall moshi/sphn only
#   ./scripts/setup_environment.sh --check      # Check installation only
#   ./scripts/setup_environment.sh --minimal    # Install only critical missing pkgs
#
# Target Environment:
#   - NVIDIA A100 80GB x 8
#   - CUDA 12.x
#   - PyTorch 2.4+ (pre-installed)
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# Configuration
# =============================================================================
FORCE_INSTALL=false
CHECK_ONLY=false
MINIMAL_MODE=false
VERBOSE=false
FIX_TORCHAUDIO_ONLY=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE_INSTALL=true
            shift
            ;;
        --check|-c)
            CHECK_ONLY=true
            shift
            ;;
        --minimal|-m)
            MINIMAL_MODE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --fix-torchaudio|--torchaudio)
            FIX_TORCHAUDIO_ONLY=true
            shift
            ;;
        --help|-h)
            echo "K-Moshi Environment Setup Script (v2.0)"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force, -f        Force reinstall moshi and sphn"
            echo "  --check, -c        Check installation status only"
            echo "  --minimal, -m      Install only critical missing packages"
            echo "  --fix-torchaudio   Fix torchaudio version to match torch (--no-deps)"
            echo "  --verbose, -v      Show detailed output"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --check              # Check current installation status"
            echo "  $0 --fix-torchaudio     # Fix torchaudio version mismatch only"
            echo "  $0                      # Full smart installation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================
print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_skip() {
    echo -e "${CYAN}[SKIP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_ok() {
    echo -e "  ${GREEN}✓${NC} $1"
}

print_fail() {
    echo -e "  ${RED}✗${NC} $1"
}

# =============================================================================
# Version Checking Functions
# =============================================================================

# Check if package is installed (any version)
is_installed() {
    python3 -c "import $1" 2>/dev/null
}

# Get installed version of a package
get_version() {
    local pkg=$1
    local import_name=${2:-$1}
    python3 -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null || echo "not_installed"
}

# Check if version satisfies constraint (handles dev versions)
# Usage: check_version torch "2.4" "2.8"
check_version() {
    local import_name=$1
    local min_ver=$2
    local max_ver=$3

    python3 << PYEOF
import sys
try:
    import $import_name as pkg
    ver_str = getattr(pkg, '__version__', '0.0.0')

    # Extract base version (handle dev builds like 2.6.0.dev20241112+cu121)
    import re
    base_match = re.match(r'^(\d+\.\d+(?:\.\d+)?)', ver_str)
    if not base_match:
        print("ERROR:parse")
        sys.exit(1)
    base_ver = base_match.group(1)

    from packaging import version
    v = version.parse(base_ver)
    min_ok = version.parse("$min_ver") <= v if "$min_ver" else True
    max_ok = v < version.parse("$max_ver") if "$max_ver" else True

    if min_ok and max_ok:
        print(f"OK:{ver_str}")
    else:
        print(f"WRONG:{ver_str}")
except ImportError:
    print("MISSING")
except Exception as e:
    print(f"ERROR:{e}")
PYEOF
}

# =============================================================================
# Check Installation Status
# =============================================================================
check_installation() {
    print_header "K-Moshi Installation Check"

    echo ""
    echo "System Information:"
    echo "  Python: $(python3 --version 2>&1)"
    echo "  Pip: $(pip3 --version 2>&1 | head -1)"
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)
        echo "  GPU: ${gpu_info}"
        local cuda_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        echo "  CUDA Runtime: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
    fi
    echo ""

    local all_ok=true
    local critical_missing=()

    # -------------------------------------------------------------------------
    # PyTorch Core (CRITICAL)
    # -------------------------------------------------------------------------
    echo "PyTorch Stack:"

    local torch_status=$(check_version torch "2.4" "2.8")
    if [[ "$torch_status" == OK:* ]]; then
        print_ok "torch ${torch_status#OK:}"
        # Check CUDA
        local cuda_ok=$(python3 -c "import torch; print('OK' if torch.cuda.is_available() else 'NO')" 2>/dev/null)
        if [[ "$cuda_ok" == "OK" ]]; then
            local gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
            print_ok "  CUDA available (${gpu_count} GPUs)"
        else
            print_warn "  CUDA not available"
        fi
    else
        print_fail "torch ${torch_status} (need >=2.4, <2.8)"
        critical_missing+=("torch")
        all_ok=false
    fi

    # torchaudio
    local ta_status=$(check_version torchaudio "2.4" "2.8")
    if [[ "$ta_status" == OK:* ]]; then
        print_ok "torchaudio ${ta_status#OK:}"
    else
        print_warn "torchaudio ${ta_status}"
    fi

    # NumPy
    local numpy_status=$(check_version numpy "1.24" "2.0")
    if [[ "$numpy_status" == OK:* ]]; then
        print_ok "numpy ${numpy_status#OK:}"
    else
        print_fail "numpy ${numpy_status} (need >=1.24, <2.0)"
        all_ok=false
    fi

    # Triton
    local triton_ver=$(get_version triton)
    if [[ "$triton_ver" != "not_installed" ]]; then
        print_ok "triton ${triton_ver}"
    else
        print_warn "triton not installed (optional for training)"
    fi

    # -------------------------------------------------------------------------
    # Moshi Core (CRITICAL)
    # -------------------------------------------------------------------------
    echo ""
    echo "Moshi Core:"

    # sphn
    if is_installed sphn; then
        local sphn_ver=$(get_version sphn)
        print_ok "sphn ${sphn_ver}"
    else
        print_fail "sphn not installed"
        critical_missing+=("sphn")
        all_ok=false
    fi

    # moshi
    if is_installed moshi; then
        local moshi_ver=$(get_version moshi)
        print_ok "moshi ${moshi_ver}"

        # Check moshi submodules (correct import path: moshi.models.loaders)
        if python3 -c "from moshi.models import loaders" 2>/dev/null; then
            print_ok "  moshi.models.loaders"
        else
            print_fail "  moshi.models.loaders"
            all_ok=false
        fi

        if python3 -c "from moshi.models.lm import LMModel" 2>/dev/null; then
            print_ok "  moshi.models.lm.LMModel"
        else
            print_fail "  moshi.models.lm.LMModel"
            all_ok=false
        fi

        # Check additional required modules
        if python3 -c "from moshi.modules.transformer import StreamingTransformerLayer" 2>/dev/null; then
            print_ok "  moshi.modules.transformer"
        else
            print_warn "  moshi.modules.transformer (optional)"
        fi
    else
        print_fail "moshi not installed"
        critical_missing+=("moshi")
        all_ok=false
    fi

    # -------------------------------------------------------------------------
    # Moshi Dependencies
    # -------------------------------------------------------------------------
    echo ""
    echo "Moshi Dependencies:"

    local moshi_deps=("einops" "bitsandbytes" "sentencepiece" "safetensors" "huggingface_hub" "aiohttp")
    for dep in "${moshi_deps[@]}"; do
        if is_installed "$dep"; then
            local ver=$(get_version "$dep")
            print_ok "$dep ${ver}"
        else
            print_fail "$dep not installed"
            all_ok=false
        fi
    done

    # -------------------------------------------------------------------------
    # Training Dependencies
    # -------------------------------------------------------------------------
    echo ""
    echo "Training Dependencies:"

    local train_deps=("tensorboard" "fire" "simple_parsing:simple-parsing" "yaml:pyyaml" "submitit" "auditok")
    for dep_spec in "${train_deps[@]}"; do
        local import_name="${dep_spec%%:*}"
        local pkg_name="${dep_spec#*:}"
        if [[ "$import_name" == "$pkg_name" ]]; then
            pkg_name="$import_name"
        fi

        if is_installed "$import_name"; then
            local ver=$(get_version "$import_name" "$import_name")
            print_ok "$pkg_name ${ver}"
        else
            print_warn "$pkg_name not installed"
        fi
    done

    # -------------------------------------------------------------------------
    # K-Moshi Project
    # -------------------------------------------------------------------------
    echo ""
    echo "K-Moshi Project:"

    if is_installed "finetune.args"; then
        print_ok "finetune package installed"
    else
        print_warn "finetune package not installed (will install)"
    fi

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    echo ""
    if [ "$all_ok" = true ]; then
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}All requirements satisfied! Ready for training.${NC}"
        echo -e "${GREEN}============================================================${NC}"
        return 0
    else
        echo -e "${YELLOW}============================================================${NC}"
        if [ ${#critical_missing[@]} -gt 0 ]; then
            echo -e "${YELLOW}Critical packages missing: ${critical_missing[*]}${NC}"
            echo -e "${YELLOW}Run: ./scripts/setup_environment.sh${NC}"
        else
            echo -e "${YELLOW}Some optional packages missing (may still work).${NC}"
        fi
        echo -e "${YELLOW}============================================================${NC}"
        return 1
    fi
}

# =============================================================================
# Installation Functions
# =============================================================================

# -----------------------------------------------------------------------------
# Fix torchaudio version to match torch (CRITICAL for audio processing)
# -----------------------------------------------------------------------------
fix_torchaudio_compatibility() {
    print_step "Checking torchaudio compatibility..."

    # Get torch version (base version without build info)
    local torch_ver=$(python3 -c "
import torch
import re
ver = torch.__version__
# Extract major.minor.patch (e.g., 2.7.1 from 2.7.1+cu121)
match = re.match(r'^(\d+\.\d+\.\d+)', ver)
print(match.group(1) if match else ver.split('+')[0])
" 2>/dev/null)

    if [ -z "$torch_ver" ]; then
        print_warn "Could not detect torch version"
        return 1
    fi

    # Get torchaudio version
    local ta_ver=$(python3 -c "
import torchaudio
import re
ver = torchaudio.__version__
match = re.match(r'^(\d+\.\d+\.\d+)', ver)
print(match.group(1) if match else ver.split('+')[0])
" 2>/dev/null || echo "not_installed")

    # Extract major.minor for comparison
    local torch_major_minor=$(echo "$torch_ver" | cut -d. -f1,2)
    local ta_major_minor=$(echo "$ta_ver" | cut -d. -f1,2)

    print_info "torch: ${torch_ver} (major.minor: ${torch_major_minor})"
    print_info "torchaudio: ${ta_ver} (major.minor: ${ta_major_minor})"

    # Check if versions match
    if [ "$torch_major_minor" = "$ta_major_minor" ]; then
        print_ok "torchaudio ${ta_ver} is compatible with torch ${torch_ver}"
        return 0
    fi

    print_warn "Version mismatch detected! Updating torchaudio..."

    # Detect CUDA version from torch
    local cuda_ver=$(python3 -c "
import torch
cuda = torch.version.cuda
if cuda:
    # Convert 12.6 -> cu126
    parts = cuda.split('.')
    print(f'cu{parts[0]}{parts[1]}')
else:
    print('cpu')
" 2>/dev/null || echo "cu126")

    print_info "Detected CUDA: ${cuda_ver}"

    # Install matching torchaudio with --no-deps to avoid dependency changes
    local pip_index="https://download.pytorch.org/whl/${cuda_ver}"

    print_step "Installing torchaudio==${torch_ver} from ${pip_index}"
    print_info "Using --no-deps to preserve other dependencies"

    if pip install "torchaudio==${torch_ver}" --index-url "${pip_index}" --no-deps 2>&1; then
        local new_ta_ver=$(python3 -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null)
        print_ok "torchaudio updated to ${new_ta_ver}"

        # Verify it works
        if python3 -c "import torchaudio; print(f'torchaudio {torchaudio.__version__} loaded successfully')" 2>/dev/null; then
            print_ok "torchaudio import verification passed"
        else
            print_warn "torchaudio installed but import may have issues"
        fi
    else
        print_warn "torchaudio ${torch_ver} not available, trying latest compatible..."

        # Try without exact version match
        if pip install torchaudio --index-url "${pip_index}" --no-deps --upgrade 2>&1; then
            local new_ta_ver=$(python3 -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null)
            print_ok "torchaudio updated to ${new_ta_ver}"
        else
            print_error "Failed to update torchaudio"
            print_info "Manual fix: pip install torchaudio --index-url ${pip_index} --no-deps"
            return 1
        fi
    fi

    return 0
}

install_sphn() {
    if is_installed sphn && [ "$FORCE_INSTALL" = false ]; then
        local ver=$(get_version sphn)
        print_skip "sphn ${ver} - already installed"
        return 0
    fi

    print_step "Installing sphn..."
    pip install 'sphn>=0.1.4,<0.2.0' --quiet 2>&1 || {
        print_warn "sphn install failed, trying alternative..."
        pip install sphn --quiet 2>&1 || {
            print_error "Failed to install sphn"
            return 1
        }
    }
    print_info "sphn installed: $(get_version sphn)"
}

install_moshi() {
    # Check if moshi is properly installed with correct modules
    if python3 -c "from moshi.models import loaders" 2>/dev/null && [ "$FORCE_INSTALL" = false ]; then
        local ver=$(get_version moshi)
        print_skip "moshi ${ver} - already installed and working"
        return 0
    fi

    print_step "Installing moshi from git (--no-deps to preserve environment)..."

    # First, ensure all moshi runtime dependencies are present
    local moshi_runtime_deps=(
        "einops>=0.7,<0.9"
        "sentencepiece==0.2.0"
        "sounddevice>=0.5"
    )

    for dep in "${moshi_runtime_deps[@]}"; do
        local pkg_name=$(echo "$dep" | sed 's/[<>=].*//')
        if ! is_installed "$pkg_name"; then
            print_info "Installing moshi dependency: $dep"
            pip install "$dep" --quiet 2>&1 || true
        fi
    done

    # Install moshi with --no-deps to avoid version conflicts
    pip install 'git+https://github.com/kyutai-labs/moshi.git#subdirectory=moshi' \
        --no-deps --quiet 2>&1 || {
        print_error "Failed to install moshi from git"
        print_info "Trying alternative: pip install moshi"
        pip install moshi --no-deps --quiet 2>&1 || {
            print_error "All moshi installation methods failed"
            return 1
        }
    }

    # Verify installation
    if python3 -c "from moshi.models import loaders" 2>/dev/null; then
        print_info "moshi installed and verified: $(get_version moshi)"
    else
        print_warn "moshi installed but modules not loading correctly"
        print_info "Installed version: $(get_version moshi)"
        print_info "Run 'python3 -c \"from moshi.models import loaders\"' to debug"
    fi
}

install_training_deps() {
    print_step "Installing training dependencies..."

    local deps=(
        "fire"
        "simple-parsing"
        "pyyaml"
        "auditok>=0.2,<0.4"
        "whisper_timestamped"
        "submitit"
    )

    for dep in "${deps[@]}"; do
        local pkg_name=$(echo "$dep" | sed 's/[<>=].*//')
        if is_installed "$pkg_name"; then
            [ "$VERBOSE" = true ] && print_skip "$pkg_name already installed"
        else
            pip install "$dep" --quiet 2>&1 || print_warn "Failed to install $dep"
        fi
    done
}

install_project() {
    print_step "Installing K-Moshi project..."
    cd "$PROJECT_DIR"
    pip install -e . --no-deps --quiet 2>&1 || {
        print_warn "Editable install failed, trying regular install..."
        pip install . --no-deps --quiet 2>&1 || {
            print_error "Project installation failed"
            return 1
        }
    }
    print_info "K-Moshi project installed"
}

# =============================================================================
# Main Installation Flow
# =============================================================================
install_all() {
    print_header "K-Moshi Environment Setup (v2.0)"

    echo ""
    echo "  Project: $PROJECT_DIR"
    echo "  Python: $(which python3)"
    echo "  Mode: $([ "$FORCE_INSTALL" = true ] && echo 'Force Install' || echo 'Smart Install')"
    echo ""

    cd "$PROJECT_DIR"

    # -------------------------------------------------------------------------
    # Step 0: Pre-flight Checks
    # -------------------------------------------------------------------------
    print_step "Pre-flight checks..."

    # Check PyTorch
    local torch_status=$(check_version torch "2.4" "2.8")
    if [[ "$torch_status" != OK:* ]]; then
        print_error "PyTorch not found or incompatible version!"
        print_info "Expected: torch >= 2.4, < 2.8"
        print_info "This environment requires pre-installed PyTorch."
        print_info ""
        print_info "If PyTorch is missing, install it first:"
        print_info "  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
        exit 1
    fi
    print_ok "PyTorch ${torch_status#OK:} - compatible"

    # Check NumPy
    local numpy_status=$(check_version numpy "1.24" "2.0")
    if [[ "$numpy_status" != OK:* ]]; then
        print_warn "NumPy version issue: ${numpy_status}"
        if [[ "$numpy_status" == *"2."* ]]; then
            print_step "Downgrading NumPy to < 2.0..."
            pip install 'numpy>=1.24,<2.0' --quiet 2>&1 || {
                pip install 'numpy==1.26.4' --force-reinstall --quiet 2>&1
            }
        fi
    else
        print_ok "NumPy ${numpy_status#OK:} - compatible"
    fi

    # -------------------------------------------------------------------------
    # Check torchaudio compatibility with torch
    # -------------------------------------------------------------------------
    fix_torchaudio_compatibility || true

    # -------------------------------------------------------------------------
    # Step 1: Install packaging (for version checks)
    # -------------------------------------------------------------------------
    pip install packaging --quiet 2>&1

    # -------------------------------------------------------------------------
    # Step 2: Install sphn (CRITICAL for moshi audio processing)
    # -------------------------------------------------------------------------
    install_sphn || exit 1

    # -------------------------------------------------------------------------
    # Step 3: Install moshi (CRITICAL)
    # -------------------------------------------------------------------------
    install_moshi || exit 1

    # -------------------------------------------------------------------------
    # Step 4: Install training dependencies
    # -------------------------------------------------------------------------
    if [ "$MINIMAL_MODE" = false ]; then
        install_training_deps
    fi

    # -------------------------------------------------------------------------
    # Step 5: Install K-Moshi project
    # -------------------------------------------------------------------------
    install_project || exit 1

    # -------------------------------------------------------------------------
    # Step 6: Final NumPy check (in case moshi reinstalled it)
    # -------------------------------------------------------------------------
    local final_numpy=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
    if [[ "$final_numpy" == 2.* ]]; then
        print_warn "NumPy was upgraded to 2.x - downgrading..."
        pip install 'numpy>=1.24,<2.0' --force-reinstall --quiet 2>&1
    fi

    # -------------------------------------------------------------------------
    # Step 7: Verification
    # -------------------------------------------------------------------------
    echo ""
    check_installation

    # -------------------------------------------------------------------------
    # Complete
    # -------------------------------------------------------------------------
    print_header "Setup Complete!"

    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Verify: ./scripts/setup_environment.sh --check"
    echo ""
    echo "  2. Test:   ./scripts/run_training_v1.sh --test"
    echo ""
    echo "  3. Train:  ./scripts/run_training_v1.sh"
    echo ""
}

# =============================================================================
# Main Entry Point
# =============================================================================
if [ "$CHECK_ONLY" = true ]; then
    check_installation
elif [ "$FIX_TORCHAUDIO_ONLY" = true ]; then
    print_header "Fixing torchaudio Compatibility"
    fix_torchaudio_compatibility
    echo ""
    print_info "Done. Run './scripts/setup_environment.sh --check' to verify."
else
    install_all
fi
