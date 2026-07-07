# Detectron imports 
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Imports
import os, json, cv2, random, re
from matplotlib import pyplot as plt
import numpy as np
from src.helper_functions.image_postprocessing import cut_leftmost_hairs

# These are the final classes in order
CLASSES = [
    "root",                
    "root_hair",            
    "bump",   
    "background_root_hair", 
    "tangled_root_hair",                 
    "edge_root_hair",                 
    "embedded_root_hair",               
    "bubble",   
    "dirt"    
]

FAKE_CLASSES = [         
    "root_hair",            
    "bump",   
    "background_root_hair", 
    "tangled_root_hair",                 
    "edge_root_hair",                 
    "embedded_root_hair",               
    "bubble",   
    "dirt"    
]

# Given a path to a model pth file, load it in and return 
def start_prediction_machine(path_to_model="models/binary_segmentation_model.pth", instance_based=False):

    # Setup configuration files 
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    # Mark destinction
    if instance_based:
        custom_metadata = MetadataCatalog.get("custom_root_v2")
        MetadataCatalog.get("custom_root_v2").set(thing_classes=FAKE_CLASSES)

    # Custom configuration
    cfg.MODEL.WEIGHTS = path_to_model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.MASK_ON = True

    # Put on cpu
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    return predictor


# This is for predicting binary masks
def detectron2_mask(predictor, img_path):
    image = cv2.imread(img_path)

    # Run prediction model (Detectron2)
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    # Visualise predictions directly — no manual annotation overlay needed
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0]), scale=1.0)
    out = v.draw_instance_predictions(instances)

    # Convert back to BGR for saving
    result_image = out.get_image()[:, :, ::-1]

    # Merge all instance masks into one binary mask
    if instances.has("pred_masks") and len(instances.pred_masks) > 0:
        masks = instances.pred_masks.numpy()
        merged_mask = np.any(masks, axis=0).astype(np.uint8) * 255  # Binary mask
    else:
        merged_mask = np.zeros(im.shape[:2], dtype=np.uint8)

    _, merged_mask = cut_leftmost_hairs(merged_mask, fraction=1/5)

    # merged_mask = cv2.cvtColor(merged_mask, cv2.COLOR_BGR2GRAY)
    return merged_mask

# Predict instance based masks
def detectron2_instances(predictor, img_path):
    image = cv2.imread(img_path)

    # Run prediction model (Detectron2)
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    # Use custom metadata in Visualizer
    v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get("custom_root_v2"), scale=1.0)
    out = v.draw_instance_predictions(instances)
    result_image = out.get_image()[:, :, ::-1]

    plt.figure(figsize=(10, 8))
    plt.imshow(result_image)
    plt.axis("off")
    plt.title("Detectron2 Result")
    plt.show()

    return instances