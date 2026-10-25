#!/usr/bin/env python3
"""Gradio web demo for WeVisOne-VL adapted for Intel Arc A770 (torch.xpu)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_ID = "tencent/WeVisOne-VL-4B"
EXAMPLES_IMAGE_DIR = DEMO_DIR / "examples" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WeVisOne-VL Gradio Demo on Intel Arc XPU")
    parser.add_argument(
        "--model",
        "--model_path",
        dest="model",
        default=os.environ.get("WEVISONE_MODEL", DEFAULT_MODEL_ID),
    )
    parser.add_argument("--backend", default="transformers", choices=("transformers", "vllm"))
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on (e.g., 'auto', 'xpu:0', 'cpu')",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Computation precision (default: bfloat16 for Arc A770)",
    )
    parser.add_argument(
        "--sequential-loading",
        action="store_true",
        help="Enable sequential loading & aggressive VRAM cache clearing (without CPU offloading)",
    )
    parser.add_argument("--server-name", "--server_name", dest="server_name", default="0.0.0.0")
    parser.add_argument(
        "--server-port",
        "--server_port",
        dest="server_port",
        type=int,
        default=int(os.environ.get("PORT", "8080")),
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="skip model loading; return mock outputs for UI preview",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import gradio as gr
        from demo.app import MockEngine, build_demo, _style_kwargs
    except ImportError as exc:
        print("\n" + "!" * 70)
        print(f" [WeVisOne-VL-XPU] Dependency missing: {exc}")
        print(" To run the web demo UI, install Gradio with:")
        print("   conda run -n WeVisOne pip install gradio")
        print("!" * 70 + "\n")
        sys.exit(1)

    if args.mock:
        print("[WeVisOne-VL-XPU] Starting in MOCK mode...")
        engine: Any = MockEngine()
    else:
        print(f"[WeVisOne-VL-XPU] Loading {args.model} on device={args.device}, dtype={args.dtype}, sequential_loading={args.sequential_loading}...")
        from wevisone_vl_xpu import WeVisOneVL

        engine = WeVisOneVL(
            args.model,
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
            sequential_loading=args.sequential_loading,
        )

    demo = build_demo(engine)
    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=[str(EXAMPLES_IMAGE_DIR)],
        **_style_kwargs("launch"),
    )


if __name__ == "__main__":
    main()
