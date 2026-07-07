from src.helper_functions.image_functions import show_image
from src.helper_functions.image_preprocessing import preprocess_pipeline
from src.helper_functions.image_postprocessing import cut_leftmost_hairs

from skimage.feature import blob_log
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_inside_roots(image_path, root, verbose =False, min_sigma=5, max_sigma=8, threshold=0.10, overlap=0.3):

    image = cv2.imread(image_path)
    carved = cv2.bitwise_and(image, image, mask=root)
    gray = preprocess_pipeline(carved, correct_bg=True, denoise_method="bilateral", bg_blur_radius=80, enhance=True)

    # Invert — dark blobs become bright blobs
    inverted = cv2.bitwise_not(gray)

    # Detect blobs
    blobs = blob_log(
        inverted,
        min_sigma=5,    # minimum blob size
        max_sigma=8,    # maximum blob size  
        threshold=0.10, # sensitivity — lower = more blobs detected
        overlap=0.3
    )

    # Build binary mask
    bump_mask = np.zeros(gray.shape, dtype=np.uint8)
    for blob in blobs:
        y, x, sigma = blob
        radius = int(math.sqrt(2) * sigma)
        cv2.circle(bump_mask, (int(x), int(y)), max(radius, 2), 255, thickness=cv2.FILLED)

    # Erode the root mask to create an interior-only region
    # This removes the edge band where the cylinder outline blobs are
    interior_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    interior_mask = cv2.erode(root, interior_kernel, iterations=1)

    # Filter blobs — only keep ones whose centre falls inside the interior mask
    filtered_blobs = []
    for blob in blobs:
        y, x, sigma = blob
        if interior_mask[int(y), int(x)] > 0:
            filtered_blobs.append(blob)

    # Rebuild mask from filtered blobs
    bump_mask = np.zeros(gray.shape, dtype=np.uint8)
    for blob in filtered_blobs:
        y, x, sigma = blob
        radius = int(math.sqrt(2) * sigma)
        cv2.circle(bump_mask, (int(x), int(y)), max(radius, 2), 255, thickness=cv2.FILLED)

    # _, bump_mask = cut_leftmost_hairs(bump_mask, fraction=1/3)


    # Visualise
    if verbose:
        show_image(carved)
        show_image(gray)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title("Preprocessed")
        axes[0].axis("off")

        axes[1].imshow(bump_mask, cmap="gray")
        axes[1].set_title(f"Bump Mask ({len(blobs)} bumps)")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    return bump_mask
