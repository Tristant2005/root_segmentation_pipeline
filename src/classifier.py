from skimage.measure import label, regionprops, find_contours
import numpy as np
import cv2, os
import pandas as pd
import joblib
from sklearn.base import TransformerMixin, BaseEstimator
from shapely import Polygon, Point, MultiPolygon

def mask_to_polygons(mask):
    contours = find_contours(mask.astype(np.uint8), 0.5)
    polygons = []
    for contour in contours:
        # Reverse coordinate order to (x, y)
        if len(contour) >= 3:
            polygons.append(Polygon([(x, y) for y, x in contour]))
    return polygons


def get_dist_from_root(root_mask, hair_instance):

    root_polygons = mask_to_polygons(root_mask)
    if not root_polygons: return None

    root_geom = root_polygons[0] if len(root_polygons) == 1 else MultiPolygon(root_polygons)

    # Get all coordinates of the hair mask
    ys, xs = np.where(hair_instance)
    if len(ys) == 0: return None

    # Compute distance from each hair pixel to the root to get base of hair
    min_dist = float('inf')
    for x, y in zip(xs, ys):
        pt = Point(x, y)
        dist = root_geom.distance(pt)
        if dist < min_dist:
            min_dist = dist

    return min_dist


class Category_Classifier():

    def __init__(self, hair_masks, hair_cases, root_mask, img_path,
                 bump_embedded_classifier_path, bg_regular_classifier_path):

        self.hair_masks = hair_masks
        self.hair_cases = hair_cases
        self.root_mask = root_mask
        self.img_path = img_path

        self.bump_embedded_classifier = bump_embedded_classifier_path
        self.bg_regular_classifier = bg_regular_classifier_path


    
    def find_edge_hair(self, hair_instance):
        """
        Check if a binary mask touches any edge of the image.
        Returns 1 if it touches an edge, 0 otherwise.
        """
        height, width = hair_instance.shape
        
        coords = np.argwhere(hair_instance > 0)  # find all non-zero pixels
        
        if len(coords) == 0:
            return 1
        
        touches_edge = np.any(
            (coords[:, 0] == 0) | (coords[:, 0] == height - 1) |  # top or bottom
            (coords[:, 1] == 0) | (coords[:, 1] == width - 1)     # left or right
        )
        
        return "edge" if touches_edge else False


    def overlap(self, hair_instance):
        overlap = np.logical_and(hair_instance, self.root_mask)
        overlap_area = np.count_nonzero(overlap)
        hair_area = np.count_nonzero(hair_instance)

        if hair_area > 0:
            overlap_percent = (overlap_area/hair_area) * 100
        else:
            return "outside of root"
        
        if overlap_percent >= 50:
            img = cv2.imread(self.img_path)
            gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            labeled_hair = label(hair_instance.astype(np.uint8))
            props = regionprops(labeled_hair, intensity_image = gray_image)

            try:
                area = props[0].area
                e = props[0].eccentricity
                intensity = props[0].intensity_mean
                aml = props[0].axis_major_length
                ar = props[0].axis_major_length/props[0].axis_minor_length
                
                test_data = pd.DataFrame({
                    "area": [area], 
                    "eccentricity": [e], 
                    "intensity": [intensity], 
                    "axis_major_length": [aml],
                    "aspect_ratio": [ar]
                })

                predictions = self.bump_embedded_classifier.predict(test_data)

            except:
                return "error"

            if predictions == ["bump"]: 
                return "bump"
            else: 
                return "embedded"


    def background_foreground(self, hair_instance):
        
        img = cv2.imread(self.img_path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        labeled_hair = label(hair_instance.astype(np.uint8))
        props = regionprops(labeled_hair, intensity_image = gray_image)

        try:
            area = props[0].area
            solidity = props[0].solidity
            aml = props[0].axis_major_length
            dist = get_dist_from_root(self.root_mask, hair_instance)
            
            test_data = pd.DataFrame({
                "area": [area], 
                "solidity": [solidity], 
                "axis_major_length": [aml],
                "dist": [dist]
            })

            predictions = self.bg_regular_classifier.predict(test_data)

        except:
            return "error"


        if predictions == ["root_hair"]: 
            return "foreground"
        else:
            return "background"


    def detect_classes(self):
        """
        Classify each hair mask as edge-touching (incomplete) or not.
        Returns a list of labels, one per hair.
        """
        labels = []
        # idx = 0
        
        for hair_mask, hair_case in zip(self.hair_masks, self.hair_cases):
            # Setup hair mask
            hair_instance = hair_mask.astype(np.uint8) * 255

            area = np.sum(hair_instance > 0)
            if area <= 500 and hair_case == "B":
                # noise so remove
                labels.append(-1)
                continue

            if area >= 5000:
                # Tangled root hairs
                labels.append(4)
                continue
            
            # determine edge hairs
            is_edge = self.find_edge_hair(hair_instance)

            if is_edge == "edge":
                labels.append(5)
                continue

            # find overlap (embedded root hairs + bumps)
            bump_embedded = self.overlap(hair_instance)

            if bump_embedded == "bump":
                labels.append(2)
                continue

            elif bump_embedded == "embedded":
                labels.append(6)
                continue

            # find overlap (embedded root hairs + bumps)
            is_background = self.background_foreground(hair_instance)
            

            if is_background == "background":
                labels.append(3)
                continue

            if bump_embedded == "error" or is_background == "error":
                labels.append(-1)
                continue

            labels.append(1)
        
        return labels
    



def load_classifiers(bmp_cls="../classification/classifiers/bumps_embedded_classifier.pkl", 
                     bg_cls="../classification/classifiers/bg_regular_classifier.pkl"):
    
    bump_embedded_classifier = joblib.load(bmp_cls)
    bg_regular_classifier = joblib.load(bg_cls)  

    return bump_embedded_classifier, bg_regular_classifier


def classify_instances(instances, root, img_path, classifier):
    hair_masks = [item["mask"] for item in instances]
    hair_cases = [item["case"] for item in instances]

    classifier.img_path = img_path
    classifier.hair_masks = hair_masks
    classifier.hair_cases = hair_cases
    classifier.root_mask = root

    # Run the classifier class
    catagory_list = classifier.detect_classes()

    kept_instances = []
    for item, category in zip(instances, catagory_list):
        if category == -1:
            continue

        item["category_id"] = category
        kept_instances.append(item)

    kept_instances.append({
        "mask":           root > 0,
        "case":           "B",
        "bbox":           [],
        "category_id":    0,
        "iou":            None,
        "area_px":        int(root.sum()),
        "id":             len(kept_instances),
        "model_instance": None,
        "class_instance": None,
    })
    return kept_instances
