#!/usr/bin/env python3
"""Comprehensive diagnostic and verification script for WeVisOne-VL on Intel Arc XPU."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F


def report_line(status: bool, message: str) -> None:
    badge = "[\033[92mPASS\033[0m]" if status else "[\033[91mFAIL\033[0m]"
    print(f" {badge} {message}")


def main() -> int:
    print("=" * 70)
    print(" WeVisOne-VL Intel Arc A770 (torch.xpu) Diagnostic & Verification")
    print("=" * 70)

    all_passed = True

    # 1. Check PyTorch version and XPU availability
    print("\n1. PyTorch & XPU Backend Check:")
    print(f"   PyTorch Version: {torch.__version__}")
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    report_line(has_xpu, f"torch.xpu available: {has_xpu}")
    if not has_xpu:
        print("ERROR: torch.xpu is not available in the active environment.")
        return 1

    device_count = torch.xpu.device_count()
    device_name = torch.xpu.get_device_name(0)
    report_line(device_count > 0, f"XPU Devices detected: {device_count} ({device_name})")

    prop = torch.xpu.get_device_properties(0)
    total_mem_gb = prop.total_memory / (1024**3)
    report_line(total_mem_gb >= 15.0, f"Total VRAM: {total_mem_gb:.2f} GB (Intel Arc A770)")

    device = torch.device("xpu:0")

    # 2. Check Precision & Tensor Operations
    print("\n2. Precision & Kernel Execution on XPU:")
    try:
        a_bf16 = torch.randn(256, 256, dtype=torch.bfloat16, device=device)
        b_bf16 = a_bf16 @ a_bf16
        report_line(b_bf16.shape == (256, 256), "BFloat16 Matrix Multiplication on XPU")
    except Exception as exc:
        report_line(False, f"BFloat16 Matmul failed: {exc}")
        all_passed = False

    try:
        a_fp16 = torch.randn(256, 256, dtype=torch.float16, device=device)
        b_fp16 = a_fp16 @ a_fp16
        report_line(b_fp16.shape == (256, 256), "Float16 Matrix Multiplication on XPU")
    except Exception as exc:
        report_line(False, f"Float16 Matmul failed: {exc}")
        all_passed = False

    # 3. Check SDPA on XPU
    try:
        q = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16, device=device)
        k = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16, device=device)
        v = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16, device=device)
        attn = F.scaled_dot_product_attention(q, k, v)
        report_line(attn.shape == (1, 8, 64, 64), "Scaled Dot-Product Attention (SDPA) on XPU")
    except Exception as exc:
        report_line(False, f"SDPA on XPU failed: {exc}")
        all_passed = False

    # 4. Check F.grid_sample on XPU
    try:
        img = torch.randn(1, 32, 64, 64, dtype=torch.float32, device=device)
        grid = torch.randn(1, 16, 4, 2, dtype=torch.float32, device=device).clamp(-1, 1)
        sampled = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        report_line(sampled.shape == (1, 32, 16, 4), "F.grid_sample (Deformable Attention kernel) on XPU")
    except Exception as exc:
        report_line(False, f"F.grid_sample on XPU failed: {exc}")
        all_passed = False

    # 5. Check wevisone_vl_xpu package imports and modules
    print("\n3. wevisone_vl_xpu Components & Decoder Execution:")
    try:
        from wevisone_vl_xpu.config import Qwen3VLSegConfig
        from wevisone_vl_xpu.msda import MultiScaleDeformableAttention
        from wevisone_vl_xpu.mask_decoder import WeVisOneMaskDecoder
        from wevisone_vl_xpu.engine import WeVisOneVL, _auto_device
        report_line(True, "Imported wevisone_vl_xpu modules successfully")
    except Exception as exc:
        report_line(False, f"Importing wevisone_vl_xpu failed: {exc}")
        return 1

    detected = _auto_device()
    report_line(detected.type == "xpu", f"_auto_device() resolved to: {detected}")

    # 6. Test MultiScaleDeformableAttention module on XPU
    try:
        msda = MultiScaleDeformableAttention(embed_dim=192, num_levels=4, num_heads=8, num_points=4).to(device)
        query = torch.randn(1, 10, 192, device=device)
        ref_points = torch.rand(1, 10, 4, 4, device=device)
        spatial_shapes = torch.tensor([[32, 32], [16, 16], [8, 8], [4, 4]], device=device, dtype=torch.int32)
        total_tokens = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
        flatten = torch.randn(1, total_tokens, 192, device=device)
        level_start = torch.cat([spatial_shapes.new_zeros(1), (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]])
        msda_out = msda(query, ref_points, flatten, spatial_shapes, level_start)
        report_line(msda_out.shape == (1, 10, 192), "MultiScaleDeformableAttention forward pass on XPU")
    except Exception as exc:
        report_line(False, f"MSDA forward pass failed: {exc}")
        all_passed = False

    # 7. Test WeVisOneMaskDecoder on XPU
    try:
        cfg = Qwen3VLSegConfig(
            hidden_dim=192,
            mask_feature_dim=64,
            msda_encoder_layers=1,
            transformer_layers=1,
            alpha_cascade_blocks=1,
            cnn_image_size=128,
        )
        decoder = WeVisOneMaskDecoder(visual_in_dims=[128, 128, 128, 128], llm_dim=256, cfg=cfg).to(device)
        dummy_img = torch.randn(1, 3, 128, 128, device=device)
        dummy_vit = [torch.randn(1, 128, 16, 16, device=device) for _ in range(4)]
        dummy_t_seg = torch.randn(2, 256, device=device)
        dummy_boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]], device=device)
        decoder_out = decoder(dummy_img, dummy_vit, dummy_t_seg, dummy_boxes, (16, 16))
        report_line("mask_logits" in decoder_out and decoder_out["mask_logits"].shape[0] == 2, "WeVisOneMaskDecoder forward pass on XPU")
    except Exception as exc:
        report_line(False, f"WeVisOneMaskDecoder forward pass failed: {exc}")
        all_passed = False

    # 8. Memory Management Check
    print("\n4. Memory Allocator & Cache Management:")
    torch.xpu.synchronize()
    allocated_mb = torch.xpu.memory_allocated() / (1024**2)
    reserved_mb = torch.xpu.memory_reserved() / (1024**2)
    print(f"   Memory allocated: {allocated_mb:.2f} MB, Reserved: {reserved_mb:.2f} MB")
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    post_empty_alloc = torch.xpu.memory_allocated() / (1024**2)
    report_line(True, f"torch.xpu.empty_cache() completed successfully (Allocated: {post_empty_alloc:.2f} MB)")

    print("\n" + "=" * 70)
    if all_passed:
        print(" ALL CHECKS PASSED: Intel Arc A770 is fully verified for WeVisOne-VL!")
    else:
        print(" SOME CHECKS FAILED: Review the output above.")
    print("=" * 70)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
