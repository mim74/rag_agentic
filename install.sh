#!/usr/bin/env bash
# =============================================================================
# RAG Agentic — Akıllı Kurulum Scripti
#
# Ne yapar:
#   1. .venv oluşturur (yoksa)
#   2. NVIDIA GPU varlığını ve CUDA sürümünü algılar
#   3. Donanıma uygun PyTorch varyantını kurar (CPU veya CUDA)
#   4. requirements.txt içindeki diğer bağımlılıkları kurar
#   5. config/settings.json içindeki cihaz ayarlarını profile göre düzenler
#
# Kullanım:
#   ./install.sh              # tam kurulum (settings güncelleme dahil)
#   ./install.sh --no-config  # settings.json'a dokunma
#   ./install.sh --dry-run    # ne yapacağını göster, kurulum yapma
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_ROOT/.venv"
PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

CONFIGURE_SETTINGS=true
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --no-config)  CONFIGURE_SETTINGS=false ;;
        --dry-run)    DRY_RUN=true ;;
    esac
done

# ─── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

info()    { echo -e "\033[0;36m[INFO]\033[0m  $*"; }
success() { echo -e "\033[0;32m[OK]\033[0m    $*"; }
warn()    { echo -e "\033[0;33m[WARN]\033[0m  $*"; }
error()   { echo -e "\033[0;31m[ERR]\033[0m   $*" >&2; }

# ─── 1. Sanal ortam ───────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       RAG Agentic — Kurulum              ║"
echo "╚══════════════════════════════════════════╝"
echo ""

if [ ! -d "$VENV" ]; then
    info ".venv oluşturuluyor..."
    $DRY_RUN || python3 -m venv "$VENV"
    success ".venv oluşturuldu"
else
    info ".venv mevcut, kullanılıyor"
fi

if $DRY_RUN; then
    info "[dry-run] .venv aktifleştirilmeyecek"
else
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
fi

# ─── 2. pip güncelle ──────────────────────────────────────────────────────────

info "pip güncelleniyor..."
$DRY_RUN || "$PIP" install --quiet --upgrade pip

# ─── 3. Donanım algılama ──────────────────────────────────────────────────────

GPU_PROFILE="cpu"
TORCH_EXTRA_INDEX=""
TORCH_SPEC="torch>=2.4.0"

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    # nvidia-smi çıktısından "CUDA Version: X.Y" al
    CUDA_VER=$(nvidia-smi 2>/dev/null \
        | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' \
        | head -1 || true)

    if [ -n "$CUDA_VER" ]; then
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

        if   [ "$CUDA_MAJOR" -ge 13 ]; then
            TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu124"
        elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
            TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu124"
        elif [ "$CUDA_MAJOR" -eq 12 ]; then
            TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu121"
        elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
            TORCH_EXTRA_INDEX="https://download.pytorch.org/whl/cu118"
        else
            warn "Eski CUDA sürümü ($CUDA_VER) — CPU PyTorch kurulacak"
        fi

        if [ -n "$TORCH_EXTRA_INDEX" ]; then
            GPU_PROFILE="cuda"
            info "NVIDIA GPU algılandı — CUDA $CUDA_VER"
        fi
    else
        warn "nvidia-smi bulundu fakat CUDA sürümü okunamadı — CPU kullanılacak"
    fi
else
    info "NVIDIA GPU bulunamadı — CPU PyTorch kurulacak"
fi

echo ""
echo "  Profil      : $GPU_PROFILE"
if [ -n "$TORCH_EXTRA_INDEX" ]; then
    echo "  Torch kaynağı: $TORCH_EXTRA_INDEX"
else
    echo "  Torch kaynağı: PyPI (CPU)"
fi
echo ""

# ─── 4. PyTorch kurulumu ──────────────────────────────────────────────────────

info "PyTorch kuruluyor ($GPU_PROFILE)..."
if $DRY_RUN; then
    if [ -n "$TORCH_EXTRA_INDEX" ]; then
        info "[dry-run] pip install '$TORCH_SPEC' --index-url '$TORCH_EXTRA_INDEX'"
    else
        info "[dry-run] pip install '$TORCH_SPEC' --index-url 'https://download.pytorch.org/whl/cpu'"
    fi
else
    if [ -n "$TORCH_EXTRA_INDEX" ]; then
        "$PIP" install "$TORCH_SPEC" --index-url "$TORCH_EXTRA_INDEX"
    else
        "$PIP" install "$TORCH_SPEC" --index-url "https://download.pytorch.org/whl/cpu"
    fi
fi
success "PyTorch kuruldu"

# ─── 5. Diğer bağımlılıklar ───────────────────────────────────────────────────

info "Diğer bağımlılıklar kuruluyor (requirements.txt)..."
$DRY_RUN || "$PIP" install -r "$PROJECT_ROOT/requirements.txt"
success "Bağımlılıklar kuruldu"

# ─── 6. Settings yapılandırması ───────────────────────────────────────────────

if $CONFIGURE_SETTINGS; then
    echo ""
    info "config/settings.json güncelleniyor (profil: $GPU_PROFILE)..."
    if $DRY_RUN; then
        "$PYTHON" "$PROJECT_ROOT/scripts/configure_settings.py" \
            --profile "$GPU_PROFILE" --dry-run
    else
        "$PYTHON" "$PROJECT_ROOT/scripts/configure_settings.py" \
            --profile "$GPU_PROFILE"
    fi
else
    info "--no-config: settings.json güncellenmedi"
fi

# ─── Tamamlandı ───────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       Kurulum tamamlandı ✓               ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Uygulamayı başlatmak için:"
echo "    ./run_chainlit.sh   (Chainlit web arayüzü)"
echo "    ./run.sh            (Terminal chat)"
echo ""
