import cv2
import numpy as np

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using luminosity method."""
    weights = np.array([0.299, 0.587, 0.114], dtype=img.dtype)
    return np.tensordot(img, weights, axes=([2], [0]))

def background_correction(img: np.ndarray, blur_size: int = 127) -> np.ndarray:
    """Subtract blurred background."""
    bg = cv2.blur(img.astype(float), (blur_size, blur_size))
    return (img.astype(float) - (bg - bg.mean())).astype(img.dtype)
