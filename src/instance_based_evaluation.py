from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_utils

import numpy as np
import json


def polygon_to_mask(segmentation, height, width):
    """Convert one annotation's polygon segmentation into a binary mask."""
    rles = mask_utils.frPyObjects(segmentation, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle)

def load_coco(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union > 0 else 0.0


def per_instance_evaluation(pred_anns, gt_anns, height, width,
                                     iou_threshold=0.5, limit=None):
    """
    Match predicted instances to ground truth instances by area overlap
    (IoU) only.

    MANDATORY PARAMETERS:
        pred_anns -- list of predicted COCO annotation dicts (single image)
        gt_anns   -- list of ground truth COCO annotation dicts (single image)
        height, width -- image dimensions for polygon decoding

    OPTIONAL PARAMETERS:
        iou_threshold -- minimum IoU to count as a valid match (default 0.5)
        limit -- if set, only process the first N predicted instances
                 (excluding root, category_id == 0)

    OUTPUT:
        list of match dicts, one per predicted instance considered:
        {
            "pred_id": ...,
            "pred_category": ...,
            "matched_gt_id": ... or None,
            "matched_gt_category": ... or None,
            "iou": ...,
            "category_match": True/False/None,
        }
    """
    # Tristan note: filter out root (category_id == 0) from BOTH sides --
    # root is handled separately and shouldn't compete for matches or
    # inflate/deflate the instance-matching stats
    pred_filtered = [a for a in pred_anns if a["category_id"] != 0]
    gt_filtered = [a for a in gt_anns if a["category_id"] != 0]

    if limit is not None:
        pred_filtered = pred_filtered[:limit]

    if len(pred_filtered) == 0 or len(gt_filtered) == 0:
        return [
            {
                "pred_id": a["id"],
                "pred_category": a["category_id"],
                "matched_gt_id": None,
                "matched_gt_category": None,
                "iou": 0.0,
            }
            for a in pred_filtered
        ]

    # decode masks once, up front, so we're not re-decoding polygons
    # inside the IoU double loop
    pred_masks = [polygon_to_mask(a["segmentation"], height, width) for a in pred_filtered]
    gt_masks = [polygon_to_mask(a["segmentation"], height, width) for a in gt_filtered]

    # build IoU cost matrix (area-only, category-agnostic)
    n_pred, n_gt = len(pred_masks), len(gt_masks)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)

    for i in range(n_pred):
        for j in range(n_gt):
            iou_matrix[i, j] = compute_iou(pred_masks[i], gt_masks[j])

    # Hungarian algorithm -- guarantees a 1-to-1 assignment, so no
    # predicted instance can be paired with more than one ground truth
    # instance and vice versa. This is what enforces "no duplicates."
    cost_matrix = -iou_matrix
    pred_idx, gt_idx = linear_sum_assignment(cost_matrix)

    # map assignment results back onto predicted instances
    assignment = {p_i: g_i for p_i, g_i in zip(pred_idx, gt_idx)}

    results = []
    for i, pred_ann in enumerate(pred_filtered):
        if i in assignment and iou_matrix[i, assignment[i]] >= iou_threshold:
            gt_i = assignment[i]
            gt_ann = gt_filtered[gt_i]
            results.append({
                "pred_id": pred_ann["id"],
                "pred_category": pred_ann["category_id"],
                "matched_gt_id": gt_ann["id"],
                "matched_gt_category": gt_ann["category_id"],
                "iou": float(iou_matrix[i, gt_i]),
            })
        else:
            # no ground truth instance overlapped this prediction enough
            results.append({
                "pred_id": pred_ann["id"],
                "pred_category": pred_ann["category_id"],
                "matched_gt_id": None,
                "matched_gt_category": None,
                "iou": 0.0,
            })

    return results


def compare_coco(truth_path, predicted_path, iou_threshold=0.35):
    truth = load_coco(truth_path)
    predicted = load_coco(predicted_path)

    # get every image_id present in the ground truth file
    image_ids = [img["id"] for img in truth["images"]]

    for image_id in image_ids:
        image_info = next(img for img in truth["images"] if img["id"] == image_id)
        height, width = image_info["height"], image_info["width"]

        predicted_segments = [a for a in predicted["annotations"] if a["image_id"] == image_id and a["category_id"] != 0]
        true_segments = [a for a in truth["annotations"] if a["image_id"] == image_id and a["category_id"] != 0]

        matches = per_instance_evaluation(predicted_segments, true_segments, height, width, iou_threshold=iou_threshold)

        matched_gt_ids = {m["matched_gt_id"] for m in matches if m["matched_gt_id"] is not None}
        gt_by_id = {a["id"]: a for a in true_segments}
        pred_by_id = {a["id"]: a for a in predicted_segments}

        false_negatives = [gt for gt in true_segments if gt["id"] not in matched_gt_ids]
        FN = len(false_negatives)

        true_positives = [gt_by_id[m["matched_gt_id"]] for m in matches if m["matched_gt_id"] is not None]
        TP = len(true_positives)

        false_positives = [pred_by_id[m["pred_id"]] for m in matches if m["matched_gt_id"] is None]
        FP = len(false_positives)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print("Image ID:", image_id)
        print(f"TP={TP}  FP={FP}  FN={FN}")
        print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
        print()