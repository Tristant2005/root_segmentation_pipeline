# # Imports
# import numpy as np
# import os, json, cv2, random
# from pathlib import Path
# import matplotlib.pyplot as plt
# from skimage.morphology import skeletonize, remove_small_objects
# import pandas as pd
# from skimage.measure import label, regionprops
# from skan import Skeleton, summarize
# from sklearn.decomposition import PCA
# from scipy import ndimage
# import re, heapq 

# # Custom Imports
# from src.helper_functions.image_preprocessing import preprocess_pipeline
# from src.helper_functions.image_functions import *
# from src.helper_functions.convex_hull_of_skeletons import *
# from src.helper_functions.draw_centerline import * 
# from src.helper_functions.image_postprocessing import *
# from src.helper_functions.inside_root import get_inside_roots

# def find_centerline_path(skan_skeleton, branch_data):
#     def endpoint_key(coord):
#         return (int(coord[0]), int(coord[1]))

#     graph = {}
#     for idx, branch in branch_data.iterrows():
#         coords = skan_skeleton.path_coordinates(idx)
#         src = endpoint_key(coords[0])
#         dst = endpoint_key(coords[-1])
#         cost = 1.0 / (branch["branch-distance"] + 1e-6)

#         if src not in graph: graph[src] = []
#         if dst not in graph: graph[dst] = []
#         graph[src].append((dst, idx, cost))
#         graph[dst].append((src, idx, cost))

#     node_degree = {node: len(graph[node]) for node in graph}
#     all_nodes = list(graph.keys())

#     def is_connected_to_main(node, graph, min_col=500):
#         for neighbour, _, _ in graph.get(node, []):
#             if neighbour[1] > min_col:
#                 return True
#         return False

#     endpoint_nodes = [n for n in all_nodes if node_degree[n] == 1]
#     candidate_nodes = [n for n in endpoint_nodes if is_connected_to_main(n, graph)]
#     if not candidate_nodes:
#         candidate_nodes = all_nodes

#     start_node = min(candidate_nodes, key=lambda n: n[1])
#     end_node   = max(candidate_nodes, key=lambda n: n[1])

#     heap = [(0, start_node, [])]
#     visited = set()

#     while heap:
#         cost, node, path = heapq.heappop(heap)
#         if node in visited:
#             continue
#         visited.add(node)
#         if node == end_node:
#             full_coords = []
#             for branch_idx in path:
#                 coords = skan_skeleton.path_coordinates(branch_idx)
#                 full_coords.append(coords)
#             return np.vstack(full_coords), set(path)  # return indices too

#         for neighbour, branch_idx, edge_cost in graph.get(node, []):
#             if neighbour not in visited:
#                 heapq.heappush(heap, (cost + edge_cost, neighbour, path + [branch_idx]))

#     return None, set()

# def extract_path_info(image_path):
#     plate_match = re.search(r"Plate_(\d+)", image_path)
#     experiment_match = re.search(r"/(\d+).tif", image_path)

#     if plate_match:
#         plate_num = int(plate_match.group(1))

#     if experiment_match:
#         exp_num = int(experiment_match.group(1))

#     return plate_num, exp_num




# def convex_segmentation(img_path, verbose=False, show_overlay=True, save=True, denoise_method='bilateral', bg_blur_radius=80):
#     # Load image
#     image = cv2.imread(img_path)
#     grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # First preprocessing pipleing
#     preprocessed = preprocess_pipeline(image, correct_bg=True, denoise_method=denoise_method, bg_blur_radius=bg_blur_radius, enhance=True)
    
#     # New preprocessing pipelien
#     # preprocessed = preprocess_pipeline(grey, gaussian_sigma=0.8)
#     # preprocessed = (preprocessed * 255).clip(0, 255).astype(np.uint8)
    
    
#     otsu_binary_mask = threshold_otsu(preprocessed, True)


#     if verbose:
#         show_image(image, title="Original image", grey=False)
#         show_image(preprocessed, title="preprocessed")
#         show_image(otsu_binary_mask, title="Otsu Binary Mask")



#     _, hairs = skeletonization_pipeline(otsu_binary_mask, verbose=verbose)
#     # root, hairs = cut_leftmost_hairs(hairs, fraction=1/3, root=root)

#     original_binary_mask = threshold_otsu(grey, True)
#     root = get_better_root(hairs, original_binary_mask, erosion_radius=6, verbose=verbose)

#     # hairs = haircut(hairs)
#     # hairs = haircut(hairs)

#     # inside_hairs = get_inside_roots(img_path, root_filled)
#     # hair_total = cv2.add(inside_hairs, hairs)
#     root, hairs = cut_leftmost_hairs(hairs, fraction=1/3, root=root)

#     if verbose or show_overlay:
#         # Show an overlay between the original image and predicted roots and hairs
#         show_overlay_with_root_and_hairs(image, root, hairs)
    
#     if save:
#         # save image as pngs to "outputs/Plate_{plate}/{experiment:06d}"
#         plate, experiment = extract_path_info(img_path)
#         save_the_segments(plate, experiment, hairs, root)

#     # return masks
#     return root, hairs









# def erode_and_dilate(root_mask, hair_skeleton, erosion_radius=10, dilation_radius=2):
#     # Create circular structuring element
#     erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius * 2 + 1, erosion_radius * 2 + 1))
#     dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))

#     # Apply erosion
#     root_cylinder = cv2.erode(root_mask, erosion_kernel, iterations=1)
#     hairs = cv2.subtract(hair_skeleton, root_cylinder)

#     removed = cv2.bitwise_and(hair_skeleton, root_cylinder)


#     hairs = cv2.dilate(hairs, dilation_kernel, iterations=1)

#     return root_cylinder, hairs





# def binary_fill_convex_segmentation(img_path, verbose=False, show_overlay=True, save=True, denoise_method='bilateral', bg_blur_radius=80):

#     # Load image
#     image = cv2.imread(img_path)
#     grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     original_otsu_binary_mask = threshold_otsu(grey, True)
#     preprocessed = preprocess_pipeline(image, correct_bg=True, denoise_method=denoise_method, bg_blur_radius=bg_blur_radius, enhance=True)
#     procesed_otsu_binary_mask = threshold_otsu(preprocessed, True)

#     otsu_binary_mask = cv2.add(original_otsu_binary_mask, procesed_otsu_binary_mask)
#     root_mask_filled = return_largest_area(fill_mask(otsu_binary_mask))



#     skeleton, skeleton_skan, branch_data = build_skeleton(root_mask_filled)
#     centerline_branch_indices = set()
#     # Run
#     centerline_path, centerline_indices = find_centerline_path(skeleton_skan, branch_data)

#     # Split into two dictionaries
#     root_branches = {}
#     hair_branches = {}

#     for idx, branch in branch_data.iterrows():
#         if idx in centerline_indices:
#             root_branches[idx] = branch
#         else:
#             hair_branches[idx] = branch

#     hair_skeleton = reconstruct_skeleton(skeleton_skan, hair_branches.keys(), skeleton.shape)
#     root, hairs = erode_and_dilate(root_mask_filled, hair_skeleton)

#     # inside_hairs = get_inside_roots(img_path, root, min_sigma=5, max_sigma=8, threshold=0.09, overlap=0.3)
#     # hair_total = cv2.add(inside_hairs, hairs)
#     root, hairs = cut_leftmost_hairs(hairs, fraction=1/3, root=root)

#     if verbose or show_overlay:
#         # Show an overlay between the original image and predicted roots and hairs
#         show_overlay_with_root_and_hairs(image, root, hairs)



#     if verbose:
#         fig, axes = plt.subplots(1, 2, figsize=(10, 6))

#         axes[0].imshow(original_otsu_binary_mask, cmap="grey")
#         axes[0].set_title("Original Binary Mask")
#         axes[0].axis("off")

#         axes[1].imshow(procesed_otsu_binary_mask, cmap="grey")
#         axes[1].set_title("Preprocessed Binary Mask")
#         axes[1].axis("off")
#         show_image(root_mask_filled, grey=True)
#         show_image(skeleton)

#     return root, hairs










'''
OTHER BS
//////////////////////////////////////////////////////////////////////////////////////
'''






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


def link_nearby_endpoints(mask, max_gap=25, max_angle_diff=25):
    """
    Connect dash endpoints that are close and roughly collinear with the
    dash's own direction -- this is what actually distinguishes 'this is
    a continuation of the same line' from 'this is an unrelated nearby line'.
    """
    labeled = label(mask)
    props = regionprops(labeled)

    # get each component's endpoint(s) and its local direction
    fragments = []
    for prop in props:
        coords = prop.coords  # (row, col) pixels of this fragment
        # direction via PCA on the fragment's own pixels
        centered = coords - coords.mean(axis=0)
        _, _, vh = np.linalg.svd(centered)
        direction = vh[0]  # dominant axis of this fragment

        # the two extreme points along that direction = the fragment's ends
        proj = centered @ direction
        end1 = coords[np.argmin(proj)]
        end2 = coords[np.argmax(proj)]
        fragments.append({"end1": end1, "end2": end2, "direction": direction})

    connections = []
    all_ends = [(f["end1"], f["direction"], i) for i, f in enumerate(fragments)] + \
               [(f["end2"], f["direction"], i) for i, f in enumerate(fragments)]

    points = np.array([e[0] for e in all_ends])
    tree = cKDTree(points)

    for idx, (pt, direction, frag_id) in enumerate(all_ends):
        nearby_idx = tree.query_ball_point(pt, r=max_gap)
        for j in nearby_idx:
            other_pt, other_dir, other_frag = all_ends[j]
            if other_frag == frag_id:
                continue
            angle = np.degrees(np.arccos(np.clip(abs(np.dot(direction, other_dir)), -1, 1)))
            if angle <= max_angle_diff:
                connections.append((tuple(pt), tuple(other_pt)))

    return connections


def apply_connections(mask, connections, line_thickness=2):
    """
    Draw the detected endpoint-to-endpoint connections onto the original
    mask, producing a single merged binary mask with gaps bridged.
    """
    result = (mask > 0).astype(np.uint8) * 255
    result = result.copy()  # avoid mutating the original mask in place

    for pt_a, pt_b in connections:
        # coords are (row, col) -- cv2.line wants (x, y) = (col, row)
        start = (int(pt_a[1]), int(pt_a[0]))
        end = (int(pt_b[1]), int(pt_b[0]))
        cv2.line(result, start, end, color=255, thickness=line_thickness)

    return result

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




def fill_small_holes(mask, max_hole_size):
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



def segment_root(full_mask, verbose=False):

    filled_mask = fill_small_holes(full_mask, max_hole_size=205)
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



def remove_small_blobs(binary_mask, min_area=25):
    
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


def filter_hair_map(hair_map, root, max_distance=20, circularity_threshold=0.8, elongation_threshold=3.0):
    """
    Filter connected components in a binary mask based on distance from
    a reference mask and shape circularity.
    """


    hair_map = hair_map.astype(bool)
    root = root.astype(bool)

    # Distance transform: distance from every pixel to the nearest
    # reference_mask pixel (0 inside the reference shape itself)
    dist_to_ref = ndimage.distance_transform_edt(~root)

    # Label connected components in the callous map
    labeled, num_features = ndimage.label(hair_map)

    output = np.zeros_like(hair_map, dtype=bool)
    for i in range(1, num_features + 1):
        component_mask = (labeled == i)
        component_u8 = (component_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0 or len(cnt) < 5:
            continue

        # --- Elongation check (via fitted ellipse or min area rect) ---
        (_, _), (w, h), _ = cv2.minAreaRect(cnt)
        minor, major = sorted([w, h])
        elongation = major / minor if minor > 0 else np.inf
        is_stick_like = elongation >= elongation_threshold

        # --- Distance check (skip entirely if stick-like) ---
        if not is_stick_like:
            min_dist = dist_to_ref[component_mask].min()
            if min_dist > max_distance:
                continue  # too far, delete

        # --- Circularity check (applies to everything) ---
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity >= circularity_threshold:
            continue  # too circular, delete

        output[component_mask] = True

    return output


def segment_hairs(binary_mask, root, min_area=100, verbose = False):
    hairs_messy = cv2.subtract(binary_mask, root)


    area = np.sum(hairs_messy > 0)

    print(area)



    # cleaned_hairs = remove_small_blobs(hairs_messy, min_area=min_area)

    dilated_hairs = dilate_mask(hairs_messy, dilation_radius=1)
    cleaned_hairs = remove_small_blobs(dilated_hairs, min_area=min_area)

    hairs = filter_hair_map(
        cleaned_hairs,
        root,
        max_distance=20,          # pixels
        circularity_threshold=0.8  # closer to 1 = stricter circle filtering
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

def segmentation_pipeline(img_path, percentiles=(2, 98), verbose=False, show_overlay=True, save=True):

    # Load the image and apply preprocessing
    img = cv2.imread(img_path)
    img_preproc = preprocess_pipeline(img, gaussian_sigma=0.5, percentiles=percentiles)
    img_preproc = (img_preproc * 255).clip(0, 255).astype(np.uint8)

    # binary otsu mask
    _, binary_mask = cv2.threshold(img_preproc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = cv2.bitwise_not(binary_mask)


    if verbose:
        show_image(img_preproc, title="preprocessed Image")
        show_image(binary_mask, title="Otsu Binary Mask")

    root = segment_root(binary_mask, verbose=verbose)
    hairs = segment_hairs(binary_mask, root, min_area=100, verbose=verbose) 

    if verbose or show_overlay:
        # Show an overlay between the original image and predicted roots and hairs
        show_overlay_with_root_and_hairs(img, root, hairs)

    return root, hairs




