import numpy as np
import json
from pycocotools import mask as mask_utils
from src.model_run import CLASSES
from pathlib import Path
import cv2


def mask_to_coco_polygons(binary_mask, epsilon=0.0):
    """
    binary_mask: 2D bool/uint8 array.
    Returns (segmentation, bbox, area) in COCO polygon format.
    epsilon: cv2.approxPolyDP tolerance. 0.0 = no simplification (keeps full contour fidelity).
    """
    mask_uint8 = (binary_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        if epsilon > 0:
            contour = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(contour) < 3:
            continue  # not a valid polygon
        flattened = contour.flatten().astype(float).tolist()
        segmentation.append(flattened)

    if not segmentation:
        return None, None, None

    ys, xs = np.where(mask_uint8 > 0)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

    area = float(mask_uint8.sum())  # pixel-count area, matches your area_px convention

    return segmentation, bbox, area



def export_to_coco(instances, file_path, categories, image_id = 0, width = 2048, 
                height = 1536, epsilon=2.0, output_path = "outputs/segmentation_outputs_coco.json"):
    """
    instances: list of dicts, each with at least a boolean/uint8 "mask" and "category_id"
    image_id: int id for this image in the COCO file
    file_name, width, height: image metadata
    categories: list of {"id": int, "name": str} — defaults to your current class map
    """

    file_name = Path(file_path).name

    coco = {
        "images": [{
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        }],
        "annotations": [],
        "categories": categories,
    }

    ann_id = 1
    for inst in instances:
        binary_mask = inst["mask"]
        segmentation, bbox, area = mask_to_coco_polygons(binary_mask, epsilon=epsilon)
        if segmentation is None:
            # empty mask, skip
            continue 

        coco["annotations"].append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": int(inst["category_id"]),
            "segmentation": segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })
        ann_id += 1

    

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    return coco




def write_coco_dataset(images, annotations, categories, output_path):
    """Write the accumulated dataset-wide COCO structure to disk."""
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    return coco



def build_annotations_for_image(instances, image_id, ann_id_start=0, epsilon=2.0):
    """
    Build COCO annotation entries for ONE image's instances.


    Returns (annotations, next_ann_id) so the caller can keep annotation
    IDs globally unique across the whole accumulated dataset.
    """
    annotations = []
    ann_id = ann_id_start

    for inst in instances:
        binary_mask = inst["mask"]
        segmentation, bbox, area = mask_to_coco_polygons(binary_mask, epsilon=epsilon)
        if segmentation is None:
            continue

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": int(inst["category_id"]),
            "segmentation": segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })
        ann_id += 1

    return annotations, ann_id


def labelme_to_coco(annotation_json, file_path, CLASSES, image_id=0, output_path="outputs/ground_truth_coco.json"):
    with open(annotation_json) as f:
        data = json.load(f)
    
    file_name = Path(file_path).name
    annotations = []
    ann_id = 0
    
    for shape in data["shapes"]:
        label = shape["label"]
        
        # Normalise tangled variants
        if label in {"tangled", "tangled_root_hairs"}:
            label = "tangled_root_hair"
        
        if label not in CLASSES:
            print(f"Skipping unknown label: {label}")
            continue
        
        pts = np.array(shape["points"], dtype=np.float32)
        if len(pts) < 3:
            continue
        
        segmentation = pts.flatten().tolist()
        area = float(cv2.contourArea(pts.astype(np.int32)))
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        bbox = [float(x_min), float(y_min),
                float(x_max - x_min), float(y_max - y_min)]
        
        category_id = CLASSES.index(label)
        
        annotations.append({
            "id":           ann_id,
            "image_id":     image_id,
            "category_id":  category_id,
            "segmentation": [segmentation],
            "bbox":         bbox,
            "area":         area,
            "iscrowd":      0
        })
        ann_id += 1
    
    coco = {
        "images": [{
            "id":           image_id,
            "file_name":    file_name,
            "width":        data["imageWidth"],
            "height":       data["imageHeight"]
        }],
        "annotations": annotations,
        "categories":  CLASSES
    }
    
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    return coco


def build_annotations_for_labelme_file(annotation_json, CLASSES, image_id, ann_id_start):
    """
    Parse ONE LabelMe json file into COCO-style annotations + its image entry.
    Does not write anything to disk -- caller accumulates across files.

    Returns (image_entry, annotations, next_ann_id)
    """
    with open(annotation_json) as f:
        data = json.load(f)

    file_name = Path(annotation_json).stem + ".tif"  # adjust extension as needed, see note below

    image_entry = {
        "id": image_id,
        "file_name": file_name,
        "width": data["imageWidth"],
        "height": data["imageHeight"],
    }

    annotations = []
    ann_id = ann_id_start

    for shape in data["shapes"]:
        label = shape["label"]

        if label in {"tangled", "tangled_root_hairs"}:
            label = "tangled_root_hair"

        if label not in CLASSES:
            print(f"Skipping unknown label: {label}")
            continue

        pts = np.array(shape["points"], dtype=np.float32)
        if len(pts) < 3:
            continue

        segmentation = pts.flatten().tolist()
        area = float(cv2.contourArea(pts.astype(np.int32)))
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        bbox = [float(x_min), float(y_min),
                float(x_max - x_min), float(y_max - y_min)]
        category_id = CLASSES.index(label)

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [segmentation],
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })
        ann_id += 1

    return image_entry, annotations, ann_id


def labelme_files_to_coco(annotation_jsons, CLASSES, output_path="outputs/ground_truth_coco.json"):
    """
    Convert a LIST of LabelMe json files into one dataset-wide COCO file.

    annotation_jsons: list of paths to individual LabelMe .json files
    """
    all_images = []
    all_annotations = []
    ann_id = 0

    for image_id, annotation_json in enumerate(annotation_jsons):
        img_entry, anns, ann_id = build_annotations_for_labelme_file(
            annotation_json, CLASSES, image_id, ann_id
        )
        all_images.append(img_entry)
        all_annotations.extend(anns)

    coco = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": CLASSES,
    }

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    return coco