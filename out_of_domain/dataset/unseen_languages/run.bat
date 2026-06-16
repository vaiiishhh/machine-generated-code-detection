@echo off
setlocal enabledelayedexpansion

:: ─────────────────────────────────────────────
:: dataset.py — Windows run script
:: Requires: Python 3.10+, pip, git on PATH
::            CUDA-capable GPU recommended
:: ─────────────────────────────────────────────

echo ============================================================
echo  OOD Dataset Pipeline — Setup ^& Run
echo ============================================================

:: 1. Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: 2. Check Git (needed for repo cloning in Stage 1)
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Install Git and add it to PATH.
    pause
    exit /b 1
)

:: 3. Install dependencies
echo.
echo [STEP 1] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check requirements.txt and your Python environment.
    pause
    exit /b 1
)

:: 4. Disable Google Drive mirroring (not available outside Colab)
::    The script defaults GDRIVE_OUTPUT_DIR to a /content/drive path.
::    We override it here so the script won't crash trying to write there.
echo.
echo [STEP 2] Setting environment overrides for local (non-Colab) run...
set GDRIVE_OUTPUT_DIR=

:: 5. Run the pipeline
echo.
echo [STEP 3] Running dataset.py...
echo    Output will be saved to: .\ood_dataset\
echo.
python dataset.py

if errorlevel 1 (
    echo.
    echo [ERROR] dataset.py exited with an error. Check the output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Pipeline complete. Results in .\ood_dataset\
echo ============================================================
pause
