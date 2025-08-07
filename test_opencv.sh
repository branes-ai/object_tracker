#!/bin/bash
# test_opencv.sh - report on the appropriate OpenCV based on environment

# Detect if we are in a headless or windowed environment
if [[ -n "$SSH_CLIENT" ]] || [[ -n "$SSH_TTY" ]] || [[ -z "$DISPLAY" && "$OSTYPE" == "linux-gnu"* ]]; then
	echo "Headless environment detected - installing opencv-python-headless"
else
	echo "GUI environment detected - installing opencv-python"
fi

# Verify installation
python -c "import cv2; print(f'OpenCV {cv2.__version__} installed successfully')"
