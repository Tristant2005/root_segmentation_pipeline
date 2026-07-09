# Imports
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
from matplotlib import pyplot as plt
from src.model_run import CLASSES


# Take the instances output by the model and place them in a custom json-like format
def load_model_instances(instances: list[dict], confidence_threshold: float = 0.5) -> dict:

    # define variables
    masks = []
    scores = []
    categories = []
    boxes = []

    # iterate through instances from the Detectron2 model 
    for mask, score, category, box in zip(instances.pred_masks, instances.scores, instances.pred_classes, instances.pred_boxes.tensor):
        if float(score) < confidence_threshold:
            continue
        masks.append(mask.numpy().astype(bool))
        scores.append(float(score))

        # shift all indexes by one to account for the root class being slotted in
        # Basically because the model was trained without the root class, the indexing is like this:
        # 0: "root_hair",            
        # 1: "bump",   
        # 2: "background_root_hair", 
        # 3: "tangled_root_hair",                 
        # 4: "edge_root_hair",                 
        # 5: "embedded_root_hair",               
        # 6: "bubble",   
        # 7: "dirt"

        categories.append(int(category) + 1)
            
        boxes.append(box)

    return {"masks": masks, "scores": scores, "classes": categories, "box": boxes}


# Compute the IOU of the model instances and the classic instances
def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two binary masks. Returns 0.0 if union is empty."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union        = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0

# Compute IoU matrix between N classical and M DL instance masks.
def build_iou_matrix(classical_masks: list[np.ndarray], dl_masks: list[np.ndarray]) -> np.ndarray:
    """Entry [i, j] = IoU(classical_masks[i], dl_masks[j])."""
    n, m = len(classical_masks), len(dl_masks)
    iou_matrix = np.zeros((n, m), dtype=np.float32)
    for i, cm in enumerate(classical_masks):
        for j, dm in enumerate(dl_masks):
            iou_matrix[i, j] = compute_mask_iou(cm, dm)
    return iou_matrix


# Pair classical and DL instances using the Hungarian algorithm.
def pair_instances(classical_masks: list[np.ndarray], dl_masks: list[np.ndarray], iou_threshold: float = 0.1) -> dict:
    """

    The IoU threshold here is intentionally LOW (default 0.1) compared
    to the F1 evaluation threshold (0.5). The reason: a DL hair whose
    base originates inside the root polygon will have very low overlap
    with the classical mask (which was cropped at the root boundary),
    but they still represent the same physical hair and should be merged.
    Use 0.1-0.2 to catch these partial overlaps.

    If two masks genuinely represent different hairs, their IoU will be
    close to 0.0 even with this lenient threshold.

    Args:
        classical_masks: list of bool masks from Part 1 splitting
        dl_masks:        list of bool masks from DL model
        iou_threshold:   minimum IoU to consider two masks the same hair

    Returns:
        dict with keys:
            matched_pairs — list of (classical_idx, dl_idx, iou) for Case A
            classical_only — list of classical_idx for Case B
            dl_only        — list of dl_idx for Case C
            iou_matrix     — full float array (useful for debugging)
    """

    if len(classical_masks) == 0 and len(dl_masks) == 0:
        return {"matched_pairs": [],
                "classical_only": [],
                "dl_only": [],
                "iou_matrix": np.zeros((0, 0))}

    if len(classical_masks) == 0:
        return {"matched_pairs": [],
                "classical_only": [],
                "dl_only": list(range(len(dl_masks))),
                "iou_matrix": np.zeros((0, len(dl_masks)))}

    if len(dl_masks) == 0:
        return {"matched_pairs": [],
                "classical_only": list(range(len(classical_masks))),
                "dl_only": [],
                "iou_matrix": np.zeros((len(classical_masks), 0))}

    iou_matrix = build_iou_matrix(classical_masks, dl_masks)

    # Hungarian algorithm: find globally optimal one-to-one assignment
    classical_idxs, dl_idxs = linear_sum_assignment(-iou_matrix)

    matched_pairs = []
    matched_classical = set()
    matched_dl = set()

    for c_idx, d_idx in zip(classical_idxs, dl_idxs):
        iou = iou_matrix[c_idx, d_idx]
        if iou >= iou_threshold:
            matched_pairs.append((c_idx, d_idx, float(iou)))
            matched_classical.add(c_idx)
            matched_dl.add(d_idx)

    classical_only = [i for i in range(len(classical_masks)) if i not in matched_classical]
    dl_only = [j for j in range(len(dl_masks)) if j not in matched_dl]

    print(f"Pairing results:")
    print(f"  Case A (matched):  {len(matched_pairs)}")
    print(f"  Case B (classical only): {len(classical_only)}")
    print(f"  Case C (DL only): {len(dl_only)}")

    return {
        "matched_pairs":  matched_pairs,
        "classical_only": classical_only,
        "dl_only": dl_only,
        "iou_matrix": iou_matrix,
    }

# Bitwise or between two paired instances to create the combined instance
def fuse_instance_pair(classical_mask: np.ndarray, dl_mask: np.ndarray, ) -> np.ndarray:
    return (classical_mask | dl_mask).astype(bool)



# Apply the four-case fusion logic to produce the final instance list.
def fuse_all_instances(classical_masks: list[np.ndarray], model_dict: list[np.ndarray], pairing: dict) -> list[dict]:
    """
    Args:
        classical_masks: list of bool masks
        dl_masks:        list of bool masks
        pairing:         output of pair_instances()

    Returns:
        List of dicts, each representing one final root hair instance:
        {
            "mask":   bool array,
            "case":   "A", "B", or "C",
            "source": human-readable description,
            "iou":    float IoU of the pair (Case A only, else None),
            "area_px": int pixel area of the fused mask,
        }
    """
    final_instances = []
    dl_masks = model_dict["masks"]
    index = 0

    # --- Case A: matched pairs ---
    for (c_idx, d_idx, iou) in pairing["matched_pairs"]:
        fused = fuse_instance_pair(classical_masks[c_idx], dl_masks[d_idx])
        final_instances.append({
            "mask":             fused,
            "case":             "A",
            "bbox":             model_dict["box"][d_idx],
            "category_id":      model_dict["classes"][d_idx],
            "iou":              iou,
            "area_px":          int(fused.sum()),
            "id":               index,
            "model_instance":   d_idx,
            "class_instance":   c_idx,
        })

        index += 1

    # --- Case B: classical only  ---
    for c_idx in pairing["classical_only"]:
        mask = classical_masks[c_idx].astype(bool)
        final_instances.append({
            "mask":             mask,
            "case":             "B",
            "bbox":             [],
            "category_id":      1,
            "iou":              None,
            "area_px":          int(mask.sum()),
            "id":               index,
            "model_instance":   None,
            "class_instance":   c_idx,
        })

        index += 1

    # --- Case C: DL only  ---
    for d_idx in pairing["dl_only"]:
        mask = dl_masks[d_idx].astype(bool)
        final_instances.append({
            "mask":             mask,
            "case":             "C",
            "bbox":             model_dict["box"][d_idx],
            "category_id":      model_dict["classes"][d_idx],
            "iou":              None,
            "area_px":          int(mask.sum()),
            "id":               index,
            "model_instance":   d_idx,
            "class_instance":   None,
        })

        index += 1

    return final_instances


# Function to run through all fusion logic
def fusion_of_masks(classic_instances: list[np.ndarray], model_instances: list[dict], verbose= False) -> list[dict]:
    # First turn the model instances into the right data format
    model_dict = load_model_instances(model_instances)

    # Get results of pairing algorithm
    results = pair_instances(classic_instances, model_dict["masks"])
    
    # fuse instances together
    final_instances = fuse_all_instances(classic_instances, model_dict, results)

    # visualize the pairing
    if verbose:
        for index, tuple_pair in enumerate(results["matched_pairs"]):

            classic_instance_idx = int(tuple_pair[0])
            model_instance_idx = int(tuple_pair[1])

            # print(classic_instance_idx, model_instance_idx)


            classic_mask = classic_instances[classic_instance_idx]
            model_mask = model_dict["masks"][model_instance_idx]

            # Convert bool array to uint8 image
            classic_mask_image = (classic_mask * 255).astype(np.uint8)
            model_mask_image = (model_mask * 255).astype(np.uint8)

            union = cv2.add(model_mask_image, classic_mask_image)

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(13, 10))


            axes[0].imshow(classic_mask_image, cmap="gray")
            axes[0].axis("off")
            axes[0].set_title(f"Classic Instance {classic_instance_idx}")

            axes[1].imshow(model_mask_image, cmap="gray")
            axes[1].axis("off")
            axes[1].set_title(f"Model Instance {model_instance_idx}")

            axes[2].imshow(union, cmap="gray")
            axes[2].axis("off")
            axes[2].set_title("Union of Instances")


            plt.show()


            if index == 3:
                break


    # This is a list of dicts containing all segmented instances
    return final_instances
