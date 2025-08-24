import cv2
import numpy as np
from PIL import ImageGrab

def get_screen_frame() -> np.ndarray:
    """
    Capture the entire screen.
    TODO: restrict capture to Bluestacks window.
    """
    img = ImageGrab.grab()
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
