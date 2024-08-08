:: install_packages_windows_venv.bat
@echo off
echo Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo Failed to create virtual environment
    exit /b %errorlevel%
)

echo Activating virtual environment...
call .venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment
    exit /b %errorlevel%
)

echo Installing bettercam...
pip install bettercam
if %errorlevel% neq 0 (
    echo Failed to install bettercam
    exit /b %errorlevel%
)

echo Installing Pillow...
pip install Pillow
if %errorlevel% neq 0 (
    echo Failed to install Pillow
    exit /b %errorlevel%
)

echo Installing opencv-python...
pip install opencv-python
if %errorlevel% neq 0 (
    echo Failed to install opencv-python
    exit /b %errorlevel%
)

echo Installing numpy...
pip install numpy
if %errorlevel% neq 0 (
    echo Failed to install numpy
    exit /b %errorlevel%
)

echo Installing torch...
pip install torch
if %errorlevel% neq 0 (
    echo Failed to install torch
    exit /b %errorlevel%
)

echo Installing ultralytics...
pip install ultralytics
if %errorlevel% neq 0 (
    echo Failed to install ultralytics
    exit /b %errorlevel%
)

echo Installing configparser...
pip install configparser
if %errorlevel% neq 0 (
    echo Failed to install configparser
    exit /b %errorlevel%
)

echo All packages installed successfully.
pause
