#!/usr/bin/env bash
# Launch WeVisOne-VL Gradio Demo on Intel Arc A770 (torch.xpu)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Starting WeVisOne-VL Gradio Demo on Intel Arc A770 (XPU)..."
python "${SCRIPT_DIR}/app_xpu.py" "$@"
