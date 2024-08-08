# Object Detection and Tracking with YOLO and BetterCam

This repository contains a script for object detection and tracking using the YOLO model and BetterCam for capturing frames. The script performs real-time object detection, identifies different classes, and tracks objects across frames using ByteTrack.

## Features

- **Real-time Object Detection**: Utilizes the YOLO model to detect various objects in real-time.
- **Object Tracking**: Tracks objects across frames using ByteTrack.
- **Customizable Configuration**: Allows setting box dimensions through a configuration file.
- **Logging**: Integrated logging for better tracking and debugging.

## Requirements

- Python 3.11.6 (recommended)
- CUDA-compatible GPU (for running YOLO model on GPU)
- Python packages:
  - bettercam
  - Pillow
  - opencv-python
  - numpy
  - torch
  - ultralytics
  - configparser

### Python Installation

It is recommended to use Python 3.11.6 for this project. You can download and install it from the following link:

[Python 3.11.6 Installer](https://github.com/KernFerm/Py3.11.6installer)

## Installation

1. **Download the repository**:

    - Go to the repository page in your browser.
    - Click the green "Code" button.
    - Select "Download ZIP".
    - Extract the downloaded ZIP file to your desired location.

2. **Navigate to the project directory**:
    ```bash
    cd path/to/extracted/repo
    ```

3. **Create a virtual environment and activate it**:
    ```
    python -m venv venv
    ```

    ## On Linux / Mac Os use
    ```
    source venv/bin/activate  
    ```
    
    ## On Windows use 
    ```
    `venv\Scripts\activate`
    ```

4. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

    or
    - you can use the `install_packages_windows.bat` in **Scripts** Folder.

    or
    - you can use the `install_packages_windows_venv.bat` in **Scripts** Folder.

5. **Ensure you have CUDA installed and configured for running the YOLO model on GPU**.

## Configuration

### `box_config.ini`

Create a `box_config.ini` file to set the box dimensions for the region of interest:

```ini
[BoxDimensions]
size = 320
```

## `game.yaml`

- The `game.yaml` file should be in the same directory and look like this:
```yaml
# paths
path: game  # path to the dataset
train: images  # path to the training images
val: val  # path to the validation images

# class names
names:
  0: player
  1: bot
  2: weapon
  3: outline
  4: dead_body
  5: hideout_target_human
  6: hideout_target_balls
  7: head
  8: smoke
  9: fire
  10: third_person
```
## Usage

- **Run the script with:**

  ```
  python main.py
  ```

## Logging

- The script uses Python's built-in logging module to log important events. Logs will be printed to the console with timestamps, log levels, and messages.

## Batch Script for Installation

- You can use the following batch script to install each package separately, but all at once.


`install_requirements.bat`

```
@echo off
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
```

## Acknowledgements

- [YOLO by Ultralytics](https://github.com/ultralytics/yolov5)

- [ByteTrack](https://github.com/ifzhang/ByteTrack)

- [OpenCV](https://opencv.org/)

- [PyTorch](https://pytorch.org/)

For always being there to educate and help
- [Rootkit-Org]https://github.com/RootKit-Org/
## Contributing
- Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

- If you want to contribute: **Recommended** `Python 3.11.6`
- [Python 3.11.6 Installer](https://github.com/KernFerm/Py3.11.6installer)
