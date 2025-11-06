@echo off
cd /d "D:\Project\image-ocr-mr-ogy"

echo ====================================
echo    ACTIVATING VIRTUAL ENVIRONMENT
echo ====================================

REM Coba berbagai nama venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✓ Activated: venv
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo ✓ Activated: .venv
) else if exist "env\Scripts\activate.bat" (
    call env\Scripts\activate.bat
    echo ✓ Activated: env
) else (
    echo ❌ Virtual Environment not found!
    echo.
    echo Please create a virtual environment first:
    echo python -m venv venv
    pause
    exit
)

echo.
echo ====================================
echo    CHECKING PACKAGES
echo ====================================

python -c "from paddleocr import PaddleOCR, PPStructure; print('✓ PaddleOCR: OK')"
if errorlevel 1 (
    echo ❌ PaddleOCR not found in venv!
    echo Installing paddleocr...
    pip install paddleocr opencv-python pandas openpyxl pillow
)

echo.
echo ====================================
echo    STARTING OCR PROCESS
echo ====================================

@REM python main.py --input ./input --output ./output --mode ocr
python main_structured_and_raw.py --input ./input --output ./output --mode ocr

echo ====================================
echo    OCR PROCESS COMPLETED
echo ====================================
pause