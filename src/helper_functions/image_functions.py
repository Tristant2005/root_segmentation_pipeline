import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt

# Plot an image to show
def show_image(image, size=(10,6), title="", grey=True):
    plt.figure(figsize=size)

    if grey:
        plt.imshow(image, cmap="grey")
    else:
        plt.imshow(image)

    plt.axis("off")
    plt.title(title)
    plt.show()

def show_multiple_images(images, titles, greys, row=1, column=2, size=(10, 6)):
    fig, axes = plt.subplots(row, column, figsize=size)

    axes[0].imshow(images[0], cmap=greys[0])
    axes[0].set_title(f"{titles[0]}")
    axes[0].axis("off")

    axes[1].imshow(images[1], cmap=greys[1])
    axes[1].set_title(f"{titles[1]}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def threshold_otsu(img, invert: bool = False) -> np.ndarray:
    """
    Apply Otsu's threshold to produce a binary mask.

    Args:
        img:    uint8 grayscale image
        invert: set True if root tissue is DARKER than background
                (bright-field images where roots appear dark).
                Leave False if root tissue is BRIGHTER (fluorescence).

    Returns:
        binary mask: bool array, True = foreground (root tissue)
    """

    # Convert to uint8 if 16-bit (common in microscopy)
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    otsu_thresh, binary_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # print(f"Otsu threshold value: {otsu_thresh:.0f}")

    # mask = binary_mask.astype(bool)
    return cv2.bitwise_not(binary_mask) if invert else binary_mask

def return_largest_area(img):
    # Label connected components
    labeled = label(img)

    # Get region properties
    regions = regionprops(labeled)

    # Find the largest region by area
    largest_region = max(regions, key=lambda r: r.area)
    # print(f"Largest region area: {largest_region.area} pixels")
    # print(f"Total regions found: {len(regions)}")

    # Keep only the largest region
    img_clean = np.zeros_like(img)
    img_clean[labeled == largest_region.label] = 255

    return img_clean

def fill_mask(mask):
    root_mask_filled = ndimage.binary_fill_holes(mask)
    root_mask_filled = (root_mask_filled * 255).astype(np.uint8)

    # closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # root_mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)

    return root_mask_filled


def show_overlay_with_root_and_hairs(original_image, root_filled, hairs):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(original_image)

    root_overlay = np.zeros((*root_filled.shape, 4), dtype=np.uint8)
    root_overlay[root_filled > 0] = (0, 255, 0, 50)
    ax.imshow(root_overlay)

    hair_overlay = np.zeros((*hairs.shape, 4), dtype=np.uint8)
    hair_overlay[hairs > 0] = (255, 0, 0, 50)
    ax.imshow(hair_overlay)

    ax.axis("off")
    plt.tight_layout()
    plt.title("Overlay")
    plt.show()
