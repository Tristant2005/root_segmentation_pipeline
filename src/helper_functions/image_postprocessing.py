from src.helper_functions.image_functions import *

import cv2, os
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
import pandas as pd

def post_process_root(root_mask, verbose=False):
    root_mask = fill_mask(root_mask)

    # Step 1 — Close small gaps
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    root_closed = cv2.morphologyEx(root_mask, cv2.MORPH_CLOSE, close_kernel)

    # root_closed = return_largest_area(root_closed)

    # Step 2 — Find the largest contour and fill it completely
    contours, _ = cv2.findContours(root_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw filled contour on blank image
    root_filled = np.zeros_like(root_mask)
    cv2.drawContours(root_filled, [largest_contour], -1, 255, thickness=cv2.FILLED)

    root_filled = ndimage.binary_fill_holes(root_filled)
    root_filled = (root_filled * 255).astype(np.uint8)

    if verbose:
        show_image(root_filled, title="Root filled", size=(10, 6))

    return root_filled


def get_better_root(hairs_mask, original_binary_mask, erosion_radius, verbose=False):
    removed = cv2.subtract(original_binary_mask, hairs_mask)
    removed = return_largest_area(removed)
    removed = fill_mask(removed)
    removed = post_process_root(removed, verbose=verbose)

    # Erosion radius from outside. It needs to be greater than the maximum root hair to not get any accidentally
    # Create circular structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius * 2 + 1, erosion_radius * 2 + 1))

    # Apply erosion
    root_cylinder = cv2.erode(removed, kernel, iterations=1)

    # Overlay show what was removed in red
    removed = cv2.subtract(removed, root_cylinder)
    overlay = np.zeros((*removed.shape, 3), dtype=np.uint8)
    overlay[root_cylinder > 0] = (255, 255, 255)   # white = cylinder
    overlay[removed > 0] = (255, 0, 0)             # red = removed hairs

    if verbose:
        show_image(overlay, title="Erosion Visualization")

    return root_cylinder

def cut_leftmost_hairs(hairs, fraction=1/15, root=None):
    """
    Remove all connected components that have any pixel
    in the leftmost fraction of the image.
    If root mask is provided, removed components are added to root.

    Args:
        hairs:    binary uint8 hair mask
        fraction: portion of image width to use as cutoff (default 1/15)
        root:     optional binary uint8 root mask to absorb removed components

    Returns:
        (cleaned_hairs, updated_root) if root provided
        (cleaned_hairs, None) if no root provided
    """
    img_width = hairs.shape[1]
    cutoff_x = int(img_width * fraction)

    labeled = label(hairs > 0)
    regions = regionprops(labeled)

    cleaned_hairs = hairs.copy()
    updated_root = root.copy() if root is not None else None
    removed = 0

    for region in regions:
        coords = region.coords
        if np.any(coords[:, 1] < cutoff_x):
            cleaned_hairs[labeled == region.label] = 0
            if updated_root is not None:
                updated_root[labeled == region.label] = 255
            removed += 1

    # print(f"Removed {removed} components touching leftmost {fraction:.2%} of image")
    return updated_root, cleaned_hairs


def haircut(hair_mask, percentile_threshold=25):
    # Label each connected hair as a separate region
    labeled = label(hair_mask > 0)
    regions = regionprops(labeled)

    # Extract areas
    areas = [r.area for r in regions]
    df = pd.DataFrame({"area": areas})

    # Calculate cutoff
    area_cutoff = np.percentile(areas, percentile_threshold)
    # print(f"Removing regions below {percentile_threshold}th percentile: < {area_cutoff:.1f} px")

    # Filter regions
    filtered_mask = np.zeros_like(hair_mask)
    for r in regions:
        if r.area >= area_cutoff:
            filtered_mask[labeled == r.label] = 255

    return filtered_mask
    
def save_the_segments(plate, experiment, root_hairs, root_mask):
    # define output path
    output_path = f"outputs/Plate_{plate}/{experiment:06d}"
    os.makedirs(output_path, exist_ok=True)

    # define filenames for hairs
    hairs_filename = "root_hairs.png"
    root_filename = "root_body.png"
    whole_root_filename = "whole_root.png"

    # make whole root mask
    whole_root_mask = cv2.add(root_hairs, root_mask)

    # save images
    cv2.imwrite(os.path.join(output_path, hairs_filename), root_hairs)
    cv2.imwrite(os.path.join(output_path, root_filename), root_mask)
    cv2.imwrite(os.path.join(output_path, whole_root_filename), whole_root_mask)
