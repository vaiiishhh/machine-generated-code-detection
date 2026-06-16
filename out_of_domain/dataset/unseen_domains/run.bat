@echo off
echo ========== STARTING PROJECT ==========

REM 1. Create virtual environment
echo [1] Setting up virtual environment...
python -m venv venv

REM Activate
call venv\Scripts\activate

REM 2. Upgrade pip
echo [2] Upgrading pip...
python -m pip install --upgrade pip

REM 3. Install requirements
echo [3] Installing dependencies...
pip install -r requirements.txt

REM 4. Run dataset processing
echo [4] Running dataset processing...
python dataset.py

REM 5. Run main pipeline
echo [5] Running main pipeline...
python extension.py

echo ========== DONE ==========
pause