@echo off
echo ========================================
echo Starting Social Debate AI (Flask Version)
echo ========================================
echo.

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM 切換到專案根目錄
cd /d "%~dp0\.."

REM 檢查虛擬環境
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment not found
    echo Consider creating one with: python -m venv venv
)

REM 安裝依賴（如果需要）
echo Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM 設置環境變數
set FLASK_APP=ui.app
set FLASK_ENV=development
set PYTHONPATH=%cd%

REM 啟動 Flask
echo.
echo Starting Flask server...
echo ========================================
python run_flask.py

pause 