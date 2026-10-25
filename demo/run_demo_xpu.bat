@echo off
REM Launch WeVisOne-VL Gradio Demo on Intel Arc A770 (torch.xpu)
echo Starting WeVisOne-VL Gradio Demo on Intel Arc A770 (XPU)...
conda run -n WeVisOne python "%~dp0app_xpu.py" %*
