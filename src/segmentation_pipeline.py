from src.helper_functions.image_functions import *
from src.helper_functions.image_preprocessing import preprocess_pipeline
from src.helper_functions.image_functions import return_largest_area, show_image, show_multiple_images


import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops



def find_endpoints(skeleton_mask):
    """Find endpoint pixels (exactly one neighbor) of a thin skeleton."""
    from scipy.ndimage import convolve
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    neighbor_count = convolve(skeleton_mask.astype(np.uint8), kernel, mode="constant")
    # pixel value 11 = itself (10) + exactly 1 neighbor
    return np.argwhere(neighbor_count == 11)


def determine_optimal_erosion_radius(mask):

    radii = range(1, 40)
    areas_remaining = []
    areas_removed = []

    for erosion_radius in radii:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion_radius * 2 + 1, erosion_radius * 2 + 1)
        )
        eroded = cv2.erode(mask, kernel, iterations=1)
        areas_remaining.append(np.sum(eroded > 0))
        areas_removed.append(np.sum(mask > 0) - np.sum(eroded > 0))

    # Compute area loss between each step
    areas_remaining = np.array(areas_remaining)
    delta = np.diff(areas_remaining)  

    # Find where area loss jumps significantly
    delta2 = np.diff(delta)
    elbow_idx = np.argmin(delta2) + 1
    optimal_radius = list(radii)[elbow_idx]

    return optimal_radius


def dilate_mask(mask_to_dilate, dilation_radius):
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
    dilated_mask = cv2.dilate(mask_to_dilate, dilation_kernel, iterations=1)

    return dilated_mask


# def has_hole(mask: np.ndarray) -> bool:
#     mask = mask.astype(bool)

#     # connectivity=1 means 4-connectivity for foreground (standard choice
#     # for hole-counting on binary images; using 8-connectivity for the
#     # foreground would treat some background as "hole" differently)
#     e = euler_number(mask, connectivity=1)

#     # number of connected components in the foreground
#     n_objects = label(mask, connectivity=1).max()

#     # Euler number = objects - holes  =>  holes = objects - euler_number
#     n_holes = n_objects - e

#     return n_holes > 0




def fill_small_holes(mask, max_hole_size=205):
    mask = mask.astype(bool)

    # Invert to get background/holes
    inverted = ~mask

    # Label connected components of the background
    labeled, num_features = ndimage.label(inverted)

    # Identify which background component touches the image border
    # (this is the "true" outside — everything else is a hole)
    border_labels = set(labeled[0, :]) | set(labeled[-1, :]) | \
                    set(labeled[:, 0]) | set(labeled[:, -1])
    border_labels.discard(0)  # 0 isn't a label

    # Compute size of each component
    component_sizes = ndimage.sum(inverted, labeled, range(1, num_features + 1))

    # Build output: start from original mask, fill small enclosed holes
    output = mask.copy()
    for i in range(1, num_features + 1):
        if i in border_labels:
            continue  # skip the actual background
        if component_sizes[i - 1] <= max_hole_size:
            output[labeled == i] = True

    return output



def segment_root(full_mask, max_hole_size=205, verbose=False):

    filled_mask = fill_small_holes(full_mask, max_hole_size)
    filled_mask = (filled_mask > 0).astype(np.uint8)

    # Get best parameters
    erosion_radius = determine_optimal_erosion_radius(filled_mask)
    dilation_radius = erosion_radius * 2

    # # perform erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius * 2 + 1, erosion_radius * 2 + 1))
    root_cylinder = cv2.erode(filled_mask, kernel, iterations=1)

    # Perform Dilation
    root_cylinder_dilated = dilate_mask(root_cylinder, dilation_radius)
    root = return_largest_area(root_cylinder_dilated)


    if verbose:
        show_image(filled_mask, title="Attempted root filling")
        show_image(root, title="Segmented Root")


    return root



def remove_small_blobs(binary_mask, min_area=100):
    
    mask_uint8 = (binary_mask > 0).astype(np.uint8)

    # label connected components -- returns count, label map, stats, centroids
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    cleaned = np.zeros_like(mask_uint8)

    # label 0 is always background, so skip it
    for label_id in range(1, n_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 255


    return cleaned


def filter_hair_map(hair_map, root, max_distance=20, circularity_threshold=0.8):
    """
    Filter connected components in a binary mask based on distance from
    a reference mask and shape circularity.
    """

    hair_map = hair_map.astype(bool)
    reference_mask = root.astype(bool)

    # Distance transform: distance from every pixel to the nearest
    # reference_mask pixel (0 inside the reference shape itself)
    dist_to_ref = ndimage.distance_transform_edt(~reference_mask)

    # Label connected components in the callous map
    labeled, num_features = ndimage.label(hair_map)

    output = np.zeros_like(hair_map, dtype=bool)

    for i in range(1, num_features + 1):
        component_mask = (labeled == i)

        # --- Distance check ---
        min_dist = dist_to_ref[component_mask].min()
        if min_dist > max_distance:
            continue  # too far, delete

        # --- Circularity check ---
        component_u8 = (component_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue  # degenerate shape, skip

        # Circularity: 1.0 = perfect circle, lower = more elongated/irregular
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity >= circularity_threshold:
            continue  # too circular, delete

        # Passed both checks — keep it
        output[component_mask] = True

    return output


def segment_hairs(binary_mask, root, min_area=100, max_distance=20, 
                  circularity_threshold=0.8, verbose = False):

    hairs_messy = cv2.subtract(binary_mask, root)

    # 1. Removing before dilating:
    # cleaned_hairs = remove_small_blobs(hairs_messy, min_area=min_area)
    # dilated_hairs = dilate_mask(cleaned_hairs, dilation_radius=1)
    # processed_hairs = dilated_hairs

    #2. Dilate then remove:
    dilated_hairs = dilate_mask(hairs_messy, dilation_radius=1)
    cleaned_hairs = remove_small_blobs(dilated_hairs, min_area=min_area)
    processed_hairs = cleaned_hairs

    hairs = filter_hair_map(
        processed_hairs,
        root,
        max_distance=max_distance,         
        circularity_threshold=circularity_threshold
    )

    # connections = link_nearby_endpoints(hairs_messy)
    # hairs = apply_connections(hairs_messy, connections, line_thickness=2)
    # cleaned_hairs = remove_small_blobs(hairs, min_area=min_area)

    if verbose:
        # show_image(hairs_messy, title="Hair Messy")
        show_multiple_images([hairs_messy, dilated_hairs], ["Hairs from Erosion", "Dilated"], ["grey", "grey"])
        show_image(cleaned_hairs, title=f"Cleaned hairs (removed area of {min_area})")
        show_image(hairs, title="Removing Artifacts")


    return hairs

def segmentation_pipeline(img_path, max_hole_size=205, min_area=100, max_distance=20, 
                          circularity_threshold=0.8, verbose=False, show_overlay=True):

    # Load the image and apply preprocessing
    img = cv2.imread(img_path)
    img_preproc = preprocess_pipeline(img, gaussian_sigma=0.5, percentiles=(2, 98))
    img_preproc = (img_preproc * 255).clip(0, 255).astype(np.uint8)

    # binary otsu mask
    _, binary_mask = cv2.threshold(img_preproc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = cv2.bitwise_not(binary_mask)


    if verbose:
        show_image(img_preproc, title="preprocessed Image")
        show_image(binary_mask, title="Otsu Binary Mask")

    root = segment_root(binary_mask, max_hole_size=max_hole_size, verbose=verbose)


    hairs = segment_hairs(binary_mask, root, min_area=min_area, max_distance=max_distance, 
                          circularity_threshold=circularity_threshold, verbose=verbose) 

    if verbose or show_overlay:
        # Show an overlay between the original image and predicted roots and hairs
        show_overlay_with_root_and_hairs(img, root, hairs)

    return root, hairs




