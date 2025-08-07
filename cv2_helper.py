# cv2_helper.py - Smart OpenCV import module
import os
import sys
import subprocess
import logging

logger = logging.getLogger(__name__)


def ensure_correct_opencv():
    """Ensure the correct OpenCV version is installed for the environment."""

    # Check if we're in a headless environment
    is_headless = (
        os.environ.get('SSH_CLIENT') or
        os.environ.get('SSH_TTY') or
        (sys.platform == 'linux' and not os.environ.get('DISPLAY'))
    )

    # Try to import cv2 and check if it works
    try:
        import cv2
        # Test if GUI functions are available
        if not is_headless:
            # Try to create a dummy window to test GUI
            try:
                cv2.namedWindow('test')
                cv2.destroyWindow('test')
                logger.info("OpenCV with GUI support is working")
                return cv2
            except Exception as e:
                logger.warning(f"OpenCV GUI test failed: {e}")
                if not is_headless:
                    # We want GUI but it's not working
                    raise RuntimeError(
                        "GUI environment detected but OpenCV GUI not working. "
                        "Please run: pip uninstall opencv-python-headless && pip install opencv-python"
                    )
        logger.info("OpenCV (headless) is working")
        return cv2
    except ImportError as e:
        logger.error(f"OpenCV not installed: {e}")

        # Auto-install the correct version
        if is_headless:
            logger.info("Installing opencv-python-headless for headless environment...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
        else:
            logger.info("Installing opencv-python for GUI environment...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])

        # Try importing again
        import cv2
        return cv2


# Import and verify OpenCV
cv2 = ensure_correct_opencv()

# Wrapper functions that gracefully handle GUI operations
class CV2Wrapper:
    """Wrapper that gracefully handles GUI operations in headless mode."""

    def __init__(self, cv2_module):
        self.cv2 = cv2_module
        self.gui_available = self._check_gui()

    def _check_gui(self):
        """Check if GUI operations are available."""
        try:
            self.cv2.namedWindow('test')
            self.cv2.destroyWindow('test')
            return True
        except:
            return False

    def imshow(self, window_name, image):
        """Show image if GUI available, otherwise skip."""
        if self.gui_available:
            self.cv2.imshow(window_name, image)
        else:
            logger.debug(f"Skipping imshow('{window_name}') - GUI not available")

    def waitKey(self, delay=0):
        """Wait for key if GUI available, otherwise return -1."""
        if self.gui_available:
            return self.cv2.waitKey(delay)
        return -1

    def destroyAllWindows(self):
        """Destroy windows if GUI available."""
        if self.gui_available:
            self.cv2.destroyAllWindows()

    def __getattr__(self, name):
        """Pass through all other attributes to cv2."""
        return getattr(self.cv2, name)


# Export the wrapper
cv2_wrapped = CV2Wrapper(cv2)
