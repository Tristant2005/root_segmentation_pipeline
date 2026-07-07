# Imports
import numpy as np
import os, json, cv2, random
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
import pandas as pd
from skimage.measure import label, regionprops
from skan import Skeleton, summarize
from sklearn.decomposition import PCA
from scipy import ndimage
import re, heapq 

# Custom Imports
from src.helper_functions.image_preprocessing import preprocess_pipeline
from src.helper_functions.image_functions import *
from src.helper_functions.convex_hull_of_skeletons import *
from src.helper_functions.draw_centerline import * 
from src.helper_functions.image_postprocessing import *
from src.helper_functions.inside_root import get_inside_roots

def find_centerline_path(skan_skeleton, branch_data):
    def endpoint_key(coord):
        return (int(coord[0]), int(coord[1]))

    graph = {}
    for idx, branch in branch_data.iterrows():
        coords = skan_skeleton.path_coordinates(idx)
        src = endpoint_key(coords[0])
        dst = endpoint_key(coords[-1])
        cost = 1.0 / (branch["branch-distance"] + 1e-6)

        if src not in graph: graph[src] = []
        if dst not in graph: graph[dst] = []
        graph[src].append((dst, idx, cost))
        graph[dst].append((src, idx, cost))

    node_degree = {node: len(graph[node]) for node in graph}
    all_nodes = list(graph.keys())

    def is_connected_to_main(node, graph, min_col=500):
        for neighbour, _, _ in graph.get(node, []):
            if neighbour[1] > min_col:
                return True
        return False

    endpoint_nodes = [n for n in all_nodes if node_degree[n] == 1]
    candidate_nodes = [n for n in endpoint_nodes if is_connected_to_main(n, graph)]
    if not candidate_nodes:
        candidate_nodes = all_nodes

    start_node = min(candidate_nodes, key=lambda n: n[1])
    end_node   = max(candidate_nodes, key=lambda n: n[1])

    heap = [(0, start_node, [])]
    visited = set()

    while heap:
        cost, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == end_node:
            full_coords = []
            for branch_idx in path:
                coords = skan_skeleton.path_coordinates(branch_idx)
                full_coords.append(coords)
            return np.vstack(full_coords), set(path)  # return indices too

        for neighbour, branch_idx, edge_cost in graph.get(node, []):
            if neighbour not in visited:
                heapq.heappush(heap, (cost + edge_cost, neighbour, path + [branch_idx]))

    return None, set()

def extract_path_info(image_path):
    plate_match = re.search(r"Plate_(\d+)", image_path)
    experiment_match = re.search(r"/(\d+).tif", image_path)

    if plate_match:
        plate_num = int(plate_match.group(1))

    if experiment_match:
        exp_num = int(experiment_match.group(1))

    return plate_num, exp_num




def convex_segmentation(img_path, verbose=False, show_overlay=True, save=True, denoise_method='bilateral', bg_blur_radius=80):
    # Load image
    image = cv2.imread(img_path)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    preprocessed = preprocess_pipeline(image, correct_bg=True, denoise_method=denoise_method, bg_blur_radius=bg_blur_radius, enhance=True)
    otsu_binary_mask = threshold_otsu(preprocessed, True)


    if verbose:
        show_image(image, title="Original image", grey=False)
        show_image(preprocessed, title="preprocessed")
        show_image(otsu_binary_mask, title="Otsu Binary Mask")



    _, hairs = skeletonization_pipeline(otsu_binary_mask, verbose=verbose)
    # root, hairs = cut_leftmost_hairs(hairs, fraction=1/3, root=root)

    original_binary_mask = threshold_otsu(grey, True)
    root = get_better_root(hairs, original_binary_mask, erosion_radius=6, verbose=verbose)

    # hairs = haircut(hairs)
    # hairs = haircut(hairs)

    # inside_hairs = get_inside_roots(img_path, root_filled)
    # hair_total = cv2.add(inside_hairs, hairs)
    root, hairs = cut_leftmost_hairs(hairs, fraction=1/3, root=root)

    if verbose or show_overlay:
        # Show an overlay between the original image and predicted roots and hairs
        show_overlay_with_root_and_hairs(image, root, hairs)
    
    if save:
        # save image as pngs to "outputs/Plate_{plate}/{experiment:06d}"
        plate, experiment = extract_path_info(img_path)
        save_the_segments(plate, experiment, hairs, root)

    # return masks
    return root, hairs
