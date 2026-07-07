# from model_run import *

import numpy as np
from skimage.measure import label, regionprops
import cv2
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
from skan import Skeleton, summarize

import imageio
import numpy as np
import matplotlib.pyplot as plt
import io


def build_skeleton(full_mask):
    binary = full_mask > 0
    skeleton = skeletonize(binary)
    skeleton_skan = Skeleton(skeleton, keep_images=True)
    branch_data = summarize(skeleton_skan, separator='-')
    return skeleton, skeleton_skan, branch_data

def find_root_side(instance_mask, root_mask, min_col, max_col):
    """
    Compare how much of the instance is above vs below the root centroid.
    The dominant side is where the hairs grow.
    Returns 'above' or 'below' and the root centroid y.
    """

    try:
        # trim the root to be just above the hair object and find centroid
        trimmed_root = root_mask[:, min_col:max_col]
        root_centroid_y = int(np.mean(np.where(trimmed_root > 0)[0]))
    except:
        return 'above'


    instance_pixels = np.where(instance_mask > 0)
    
    # count the number of pixels above and below the center point
    above = np.sum(instance_pixels[0] < root_centroid_y)
    below = np.sum(instance_pixels[0] > root_centroid_y)
    
    # if there are more points above the centroid, return 'above' else 'below'
    side = 'above' if above > below else 'below'
    return side


def sweep_line_find_bases(skeleton, direction='above', min_gap=15):
    """
    Sweep from the root side outward, scanning row by row.
    At each row, find clusters of skeleton pixels.
    
    Returns the first row where 2+ distinct clusters are found,
    and the x positions of each cluster at that row.
    
    min_gap: minimum black pixel gap to consider two clusters separate
    """
    h, _ = skeleton.shape

    ratio = 0.60
    h_climb = int(h * ratio)

    if direction == 'above':
        height = range(h-1, (h-h_climb), -1)
    else:
        height = range(0, h_climb)
    

    for y_val in height:
        row_pixels = np.where(skeleton[y_val] > 0)[0]

        if len(row_pixels) < 2:
            continue

        # Find clusters separated by min_gap
        branch_bases = []
        current_base = row_pixels[0]

        for pixel in row_pixels[1:]:
            if pixel - current_base > min_gap:
                branch_bases.append(current_base)
                current_base = pixel

            else:
                current_base = pixel

        branch_bases.append(current_base)

        
        if len(branch_bases) >= 2:
            return y_val, branch_bases
        
    return None, []

    

def trace_hair_from_base(skeleton, start_y, start_x, direction='above', 
                          search_radius=3):
    """
    From a base point, trace upward/downward following the skeleton.
    Returns list of (y, x) coordinates tracing the hair path.
    """
    h, w = skeleton.shape
    path  = []
    cur_y = start_y
    cur_x = start_x
    
    dy = -1 if direction == 'above' else 1 
    
    visited = set()
    
    while 0 <= cur_y < h:
        visited.add((cur_y, cur_x))
        path.append((cur_y, cur_x))
        
        # Look in the next row within search radius for next skeleton pixel
        next_y = cur_y + dy
        if not (0 <= next_y < h):
            break
        
        # Find skeleton pixels in next row within search radius
        x_min = max(0, cur_x - search_radius)
        x_max = min(w, cur_x + search_radius)
        next_row_pixels = np.where(skeleton[next_y, x_min:x_max] > 0)[0]
        
        if len(next_row_pixels) == 0:
            break
        
        # Pick closest pixel to current x
        next_row_pixels += x_min  # adjust back to full image coords
        cur_x = next_row_pixels[np.argmin(np.abs(next_row_pixels - cur_x))]
        cur_y = next_y
    
    return path


def algorithm_4(trimmed, location, min_gap=15, search_radius=3, verbose=False):
    """
    ALGORITHM 4 
    """

    DILATION_RADIUS = 2
    COLOURS = [(255, 100, 100), (100, 100, 255), (100, 255, 100),
               (255, 255, 100), (255, 100, 255), (100, 255, 255)]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATION_RADIUS * 2 + 1, DILATION_RADIUS * 2 + 1))

    # Step 1. Skeletonize
    try:
        skeleton, skeleton_skan, branch_data = build_skeleton(trimmed.astype(bool))
    except:
        return []

    # Step 2. When we trace a pixel, know if it's type 1 or type 2
    pixel_branch_type = {}
    for idx in branch_data.index:
        btype = branch_data.loc[idx, "branch-type"]
        coords = skeleton_skan.path_coordinates(idx)
        for coord in coords:
            key = (int(coord[0]), int(coord[1]))

            # Type 2 can overwrite type 1 — shared pixels default to type 2
            if key not in pixel_branch_type or btype == 2:
                pixel_branch_type[key] = btype

    # Step 3. Sweep to find bases
    base_y, hair_x_bases = sweep_line_find_bases(skeleton, direction=location, min_gap=min_gap)

    if base_y is None:
        return []

    # Step 4. Trace each hair from its base
    hair_paths = []
    for base_x in hair_x_bases:
        path = trace_hair_from_base(skeleton, base_y, base_x, direction=location, search_radius=search_radius)
        hair_paths.append(path)

    # Step 5. Build shared type 2 mask (belongs to all instances)
    type2_mask = np.zeros_like(trimmed, dtype=np.uint8)
    for idx in branch_data[branch_data["branch-type"] == 2].index:
        coords = skeleton_skan.path_coordinates(idx)
        for coord in coords:
            type2_mask[int(coord[0]), int(coord[1])] = 255
    type2_mask_dilated = cv2.dilate(type2_mask, kernel, iterations=1)

    # Step 6. Build exclusive type 1 masks per hair
    # Track which pixels are already claimed by a type 1 trace
    claimed_pixels = np.zeros_like(trimmed, dtype=np.uint8)

    hair_masks = []
    viz = cv2.cvtColor((trimmed * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    for i, path in enumerate(hair_paths):
        hair_mask = np.zeros_like(trimmed, dtype=np.uint8)
        colour = COLOURS[i % len(COLOURS)]

        for y, x in path:
            btype = pixel_branch_type.get((y, x), 1) 

            if btype == 2:
                continue
            
            elif claimed_pixels[y, x] == 0:  
                hair_mask[y, x] = 255
                claimed_pixels[y, x] = 1
                viz[y, x] = colour

        # Dilate the type 1 mask
        hair_mask_dilated = cv2.dilate(hair_mask, kernel, iterations=1)

        # Add shared type 2 pixels to every instance
        hair_mask_final = cv2.add(hair_mask_dilated, type2_mask_dilated)
        hair_masks.append(hair_mask_final)

        # Colour type 2 pixels white in viz
        viz[type2_mask > 0] = (50, 100, 50)

    if verbose:
        cv2.line(viz, (0, base_y), (trimmed.shape[1], base_y), (200, 200, 200), thickness=1)
        plt.figure(figsize=(14, 6))
        plt.imshow(viz)
        plt.axis("off")
        plt.title(f"Algorithm 4 — {len(hair_masks)} hairs (white=shared type 2)")
        plt.show()

    return hair_masks





def instance_regions(hair_mask, root_mask, min_length=20, verbose=False) -> list:
    """Label connected components, filter small ones, detect multiple instances."""
    labeled = label(hair_mask > 0)
    regions = regionprops(labeled)
    regions.sort(key=lambda r: r.area, reverse=True)

    masks = []

    for i, region in enumerate(regions):
        instance_mask = (labeled == region.label).astype(np.uint8)

        min_row, min_col, max_row, max_col = region.bbox
        trimmed = instance_mask[min_row:max_row, min_col:max_col]

        # Skip tiny instances
        if trimmed.shape[0] < 100 and trimmed.shape[1] < 100:
            masks.append(instance_mask.astype(bool))
            continue

        area = np.sum(trimmed > 0)
        if area >= 5000:
            # these are probrabally tangled root hairs. we should preserve these
            masks.append(instance_mask.astype(bool))
            continue

        if area <= 1000:
            continue



        location = find_root_side(instance_mask, root_mask, min_col, max_col)
        segmented_hair_masks = algorithm_4(trimmed, location, min_gap=10, search_radius=5, verbose=verbose)

        if len(segmented_hair_masks) == 0:
            masks.append(instance_mask.astype(bool))

        else:
            for segmented_mask in segmented_hair_masks:
                # min_row, min_col, max_row, max_col 
                rebuilt_mask = np.zeros_like(instance_mask, dtype=np.uint8)

                rebuilt_mask[min_row:max_row, min_col:max_col] = segmented_mask
                masks.append(rebuilt_mask.astype(bool))


    return masks

def make_instance_gif(masks, img_path, output_path="instances.gif", fps=1):
    frames = []

    image = cv2.imread(img_path)
    
    index = 0
    for mask_instance in masks:    
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(image)
        
        # Green overlay
        overlay = np.zeros((*mask_instance.shape, 4), dtype=np.uint8)
        overlay[mask_instance] = (0, 255, 0, 128)
        ax.imshow(overlay)
        
        ax.axis("off")
        ax.set_title(f"ID: {index}", fontsize=14)
        
        # Render figure to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=80)
        buf.seek(0)
        frame = imageio.imread(buf)
        frames.append(frame)
        plt.close()

        index += 1
    
    # Save as gif
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"Saved {len(frames)} frames to {output_path}")

