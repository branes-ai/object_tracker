#!/bin/bash
# setup_opencv.sh - Install appropriate OpenCV based on environment

# First, clean up any existing OpenCV installations
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python

# Detect if we are in a headless or windowed environment
if [[ -n "$SSH_CLIENT" ]] || [[ -n "$SSH_TTY" ]] || [[ -z "$DISPLAY" && "$OSTYPE" == "linux-gnu"* ]]; then
	echo "Headless environment detected - installing opencv-python-headless"
	pip install opencv-python-headless==4.10.0.84
else
	echo "GUI environment detected - installing opencv-python"
	pip install opencv-python==4.11.0.86
fi

# Verify installation
python -c "import cv2; print(f'OpenCV {cv2.__version__} installed successfully')"
