import cv2
import numpy as np
from skimage.measure import label, regionprops

# Code to preprocess the original image

def correct_illumination(gray: np.ndarray, blur_radius: int = 51) -> np.ndarray:
    """
    Subtract a blurred version of the image to remove slow-varying
    background illumination gradients (rolling-ball equivalent).

    Args:
        gray:        grayscale image
        blur_radius: size of the Gaussian blur kernel (must be odd).
                     Larger = removes broader gradients.
                     Rule of thumb: ~1/4 of the root width in pixels.

    Returns:
        corrected: float32 image, normalized to [0, 255]
    """
    # Ensure odd kernel size
    if blur_radius % 2 == 0:
        blur_radius += 1

    background = cv2.GaussianBlur(gray.astype(np.float32), (blur_radius, blur_radius), 0)
    corrected  = gray.astype(np.float32) - background

    # Shift so minimum is 0, scale to [0, 255]
    corrected -= corrected.min()
    corrected  = (corrected / corrected.max() * 255).astype(np.uint8)
    return corrected

def denoise(gray: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """
    Reduce image noise while preserving edges.

    Args:
        gray:   grayscale uint8 image
        method: "gaussian" (fast, general purpose)
                "median"   (good for salt-and-pepper noise)
                "bilateral"(edge-preserving, slower)

    Returns:
        denoised uint8 image
    """

    if method == "gaussian":
        return cv2.GaussianBlur(gray, (3, 3), sigmaX=1.0)
    elif method == "median":
        return cv2.medianBlur(gray, ksize=3)
    elif method == "bilateral":
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"Unknown method: {method}")
    
def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Improves local contrast without over-amplifying noise.
    Especially useful when root hairs are faint relative to the root body.
    """

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def preprocess_pipeline(
    img: np.ndarray,
    correct_bg:      bool = True,
    bg_blur_radius:  int  = 51,
    denoise_method:  str  = "bilateral",
    enhance:         bool = False,
) -> np.ndarray:
    """
    Run the full preprocessing chain. Adjust flags based on image quality.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if correct_bg:
        img = correct_illumination(img, blur_radius=bg_blur_radius)

    img = denoise(img, method=denoise_method)

    if enhance:
        img = enhance_contrast(img)

    return img