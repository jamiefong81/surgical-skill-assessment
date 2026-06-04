#!/usr/bin/env bash
set -euo pipefail

# One-time setup for alpha-beta-CROWN in an isolated Python 3.11 venv.
# Run from the repo root. Safe to re-run — each step skips if already done.
#
# Prerequisites:
#   python3.11 and python3.11-venv installed (sudo apt install python3.11 python3.11-venv)
#
# What this creates (both gitignored):
#   alpha-beta-CROWN/   cloned verifier repo
#   envs/abcrown/       isolated Python 3.11 venv

VENV="envs/abcrown"
ABCROWN_DIR="alpha-beta-CROWN"

echo "=== Step 1: Clone alpha-beta-CROWN ==="
if [ -d "$ABCROWN_DIR/.git" ]; then
    echo "  already cloned — skipping"
else
    git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git "$ABCROWN_DIR"
fi

echo ""
echo "=== Step 2: Create Python 3.11 venv ==="
if [ -d "$VENV/bin" ]; then
    echo "  already exists — skipping"
else
    python3.11 -m venv "$VENV"
fi

echo ""
echo "=== Step 3: Upgrade pip ==="
"$VENV/bin/pip" install --upgrade pip

echo ""
echo "=== Step 4: Install PyTorch 2.8.0 with CUDA 12.8 ==="
"$VENV/bin/pip" install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "=== Step 5: Install auto_LiRPA (core CROWN library) ==="
"$VENV/bin/pip" install -e "$ABCROWN_DIR/auto_LiRPA/"

echo ""
echo "=== Step 6: Install alpha-beta-CROWN (complete_verifier via root setup.py) ==="
"$VENV/bin/pip" install -e "$ABCROWN_DIR/"

echo ""
echo "=== Done ==="
echo "Run verification with alpha-beta-CROWN:"
echo "  bash verify.sh --verifier abcrown [--T N]"
