@echo off
REM Convenience batch file for running Gemma-3 scripts with Anaconda Python
REM Usage: run_scripts.bat <script_name> <arguments>

set PYTHON_PATH="C:\ProgramData\anaconda3\python.exe"

if "%1"=="summary" (
    %PYTHON_PATH% public/gemma3_model_summary.py
    goto :end
)

if "%1"=="download-list" (
    %PYTHON_PATH% public/download_gemma3_mediapipe.py list
    goto :end
)

if "%1"=="download-all" (
    %PYTHON_PATH% public/download_gemma3_mediapipe.py download-all
    goto :end
)

if "%1"=="download" (
    %PYTHON_PATH% public/download_gemma3_mediapipe.py download %2
    goto :end
)

if "%1"=="convert-list" (
    %PYTHON_PATH% public/convert_gemma3_to_mediapipe.py list
    goto :end
)

if "%1"=="convert-all" (
    %PYTHON_PATH% public/convert_gemma3_to_mediapipe.py convert-all
    goto :end
)

if "%1"=="convert" (
    %PYTHON_PATH% public/convert_gemma3_to_mediapipe.py convert %2
    goto :end
)

if "%1"=="help" (
    echo Gemma-3 Model Management Scripts
    echo ================================
    echo.
    echo Usage: run_scripts.bat [command] [arguments]
    echo.
    echo Commands:
    echo   summary        - Show complete model overview
    echo   download-list  - List available models for download
    echo   download-all   - Download all available models
    echo   download MODEL - Download specific model
    echo   convert-list   - List models available for conversion  
    echo   convert-all    - Convert all available models
    echo   convert MODEL  - Convert specific model
    echo   help          - Show this help
    echo.
    echo Examples:
    echo   run_scripts.bat summary
    echo   run_scripts.bat download gemma-3-1b-int4
    echo   run_scripts.bat convert gemma-3-4b
    goto :end
)

echo Invalid command. Use 'run_scripts.bat help' for usage information.

:end 