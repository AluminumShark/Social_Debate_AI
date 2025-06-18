#!/bin/bash

echo "========================================"
echo "Starting Social Debate AI (Flask Version)"
echo "========================================"
echo

# 檢查 Python 是否安裝
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# 切換到專案根目錄
cd "$(dirname "$0")/.." || exit

# 檢查虛擬環境
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "[WARNING] Virtual environment not found"
    echo "Consider creating one with: python3 -m venv venv"
fi

# 安裝依賴（如果需要）
echo "Checking dependencies..."
if ! pip show flask &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# 設置環境變數
export FLASK_APP=ui.app
export FLASK_ENV=development
export PYTHONPATH=$PWD

# 啟動 Flask
echo
echo "Starting Flask server..."
echo "========================================"
python3 run_flask.py 