@echo off
echo 🚀 YOLO to ONNX Converter for FreshTrack
echo ================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python first.
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if best.pt exists
if not exist "best.pt" (
    echo ❌ best.pt not found in current directory
    echo    Please place your best.pt file in the backend directory
    pause
    exit /b 1
)

REM Install ultralytics if not already installed
echo 📦 Installing ultralytics...
pip install ultralytics

REM Run the conversion script
echo 🔄 Converting model...
python convert_model.py

if errorlevel 1 (
    echo ❌ Conversion failed
    pause
    exit /b 1
)

echo.
echo ✅ Conversion complete!
echo 📋 Next steps:
echo    1. Start your backend: npm run dev
echo    2. The system will now use your YOLO model for detection
pause
