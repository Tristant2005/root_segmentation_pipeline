from src.helper_functions.draw_centerline import *
from src.helper_functions.image_functions import *

import cv2
import numpy as np
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

def build_skeleton(full_mask):
    # Apply gaussian blur to denoise image
    blurred = cv2.GaussianBlur(full_mask, (21, 21), 0)
    _, root_cylinder = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Skeletonize 
    binary = root_cylinder > 0
    skeleton = skeletonize(binary)
    skeleton_skan = Skeleton(skeleton)

    # Summarize all branches as a dataframe
    branch_data = summarize(skeleton_skan, separator='-')

    return skeleton, skeleton_skan, branch_data

def branch_angle(skeleton_skan, branch_idx):
    coords = skeleton_skan.path_coordinates(branch_idx)
    start = coords[0]
    end = coords[-1]
    delta = end - start
    return np.degrees(np.arctan2(delta[0], delta[1]))

def angle_difference(a1, a2):
    diff = abs(a1 - a2) % 180
    return min(diff, 180 - diff) 

def classify_branches(skeleton_skan, branch_data, root_angle):
    # Classify all branches
    root_branches = {}
    hair_branches = {}
    perpendicular_threshold = 45

    branch_data = branch_data[branch_data["branch-type"] != 3]



    for idx, branch in branch_data.iterrows():
        angle = branch_angle(skeleton_skan, idx)
        diff = angle_difference(angle, root_angle)
        if diff > perpendicular_threshold:
            hair_branches[idx] = branch
        else:
            root_branches[idx] = branch

    # Filter short branches — move to root rather than discard
    # short_branches = {idx: branch for idx, branch in hair_branches.items()
    #                 if branch["branch-distance"] <= 10}
    # hair_branches = {idx: branch for idx, branch in hair_branches.items()
    #                 if branch["branch-distance"] > 10}
    # # root_branches.update(short_branches)

    # Filter left third — move to root rather than discard
    # img_width = skeleton_skan.skeleton_image.shape[1]
    # leftmost_third = img_width / 3

    # left_branches = {idx: branch for idx, branch in hair_branches.items()
    #                 if branch["image-coord-src-1"] <= leftmost_third or
    #                     branch["image-coord-dst-1"] <= leftmost_third}
    # hair_branches = {idx: branch for idx, branch in hair_branches.items()
    #                 if branch["image-coord-src-1"] > leftmost_third and
    #                     branch["image-coord-dst-1"] > leftmost_third}
    # root_branches.update(left_branches)

    # print(f"Root branches: {len(root_branches)}")
    # print(f"Hair branches: {len(hair_branches)}")

    return hair_branches, root_branches

# Reconstruct skeleton images for each
def reconstruct_skeleton(skeleton_skan, branch_indices, shape):
    img = np.zeros(shape, dtype=np.uint8)
    for idx in branch_indices:
        coords = skeleton_skan.path_coordinates(idx)
        for coord in coords:
            img[int(coord[0]), int(coord[1])] = 255
    return img

def return_masks(hair_skeleton, full_mask, dilation_radius=2):
    # Convert to uint8 for opencv
    root_hair_skeletons = (hair_skeleton * 255).astype(np.uint8)

    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_radius * 2 + 1, dilation_radius * 2 + 1)
    )

    hair_mask = cv2.dilate(root_hair_skeletons, dilation_kernel, iterations=1)
    hair_mask = cv2.bitwise_and(hair_mask, full_mask)


    # Build root
    hair_mask = (hair_mask * 255).astype(np.uint8)
    root_mask = cv2.subtract(full_mask, hair_mask)

    # morph fill 
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    root_mask = cv2.morphologyEx(root_mask, cv2.MORPH_CLOSE, close_kernel)

    return root_mask, hair_mask


def skeletonization_pipeline(mask, dilation_radius=6, verbose=False):
    skeleton, skeleton_skan, branch_data = build_skeleton(mask)

    root_angle, coords = find_centerline_angle(branch_data, skeleton_skan)

    hair_branches, root_branches = classify_branches(skeleton_skan, branch_data, root_angle)

    root_skeleton = reconstruct_skeleton(skeleton_skan, root_branches.keys(), skeleton.shape)
    hair_skeleton = reconstruct_skeleton(skeleton_skan, hair_branches.keys(), skeleton.shape)

    root, hairs = return_masks(hair_skeleton, mask, dilation_radius=dilation_radius)

    root = ndimage.binary_fill_holes(root > 0).astype(np.uint8) * 255
    root = return_largest_area(root)

    if verbose:
        show_image(skeleton, title="Skeletal Structure")
        draw_axis_line((skeleton * 255).astype(np.uint8), coords, root_angle, is_gray=True)

        greys = ["grey", "grey"]

        images = [root_skeleton, hair_skeleton]
        titles = [f"Root Branches ({len(root_branches)})", f"Hair Branches ({len(hair_branches)})"]
        show_multiple_images(images, titles, greys)

        images = [root, hairs]
        titles = ["Root Branches", "Hair Branches"]
        show_multiple_images(images, titles, greys)
    
    return root, hairs
