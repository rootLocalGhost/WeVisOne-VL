"""Public inference runtime for WeVisOne-VL adapted for Intel Arc A770 (torch.xpu)."""

from .engine import InferenceResult, WeVisOneVL
from .prompts import TASKS, build_prompt

__all__ = ["InferenceResult", "TASKS", "WeVisOneVL", "build_prompt"]

__version__ = "1.0.0-xpu"
