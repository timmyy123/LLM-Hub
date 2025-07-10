@echo off
REM Manual download script for Gemma-3 models
REM Run this after accepting HuggingFace licenses

echo Creating models directory...
if not exist "models" mkdir "models"

echo.
echo Downloading Gemma-3 1B INT4 (529MB)...
curl -L -o "models/gemma3-1b-it-int4.task" "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task"

echo.
echo Downloading Gemma-3 1B INT8 (1005MB)...
curl -L -o "models/gemma3-1b-it-int8.task" "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv1280.task"

echo.
echo Downloading Gemma-3n E2B Multimodal (2.9GB)...
curl -L -o "models/gemma3n-e2b-multimodal.task" "https://huggingface.co/google/gemma-3n-E2B-it-litert-preview/resolve/main/model.task"

echo.
echo Download completed! Check the models/ directory.
pause
