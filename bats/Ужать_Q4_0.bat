@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if "%~1"=="" (
  echo Kin? .safetensors model na etot fayl (drag and drop^)
  pause
  exit /b
)
echo === XQUANT Q4_0 ===
"D:\ComfyBot\comfyui_portable\ComfyUI_windows_portable\python_embeded\python.exe" -s "D:\ComfyBot\comfyui_portable\ComfyUI_windows_portable\xquant_tool.py" "%~1" Q4_0
echo.
echo Gotovo. Fayl ryadom s ishodnym.
pause