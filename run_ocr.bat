@echo off
cd /d "D:\Project\image-ocr-mr-ogy"

echo ====================================
echo    CHECKING PACKAGES
echo ====================================

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Venv: [OK]
) else (
    echo ❌ Virtual Environment not found!
    echo.
    echo Please create a virtual environment first:
    echo python -m venv venv
    pause
    exit
)

python -c "from paddleocr import PaddleOCR, PPStructure; print('PaddleOCR: [OK]')"
if errorlevel 1 (
    echo ❌ PaddleOCR not found in venv!
    echo Installing paddleocr...
    pip install paddleocr opencv-python pandas openpyxl pillow
)

echo.
echo ====================================
echo    STARTING OCR PROCESS
echo ====================================
echo.

python main.py --input ./input --output ./output --mode ocr --lang en

echo.
echo ====================================
echo    OCR PROCESS COMPLETED
echo ====================================
echo.
pause