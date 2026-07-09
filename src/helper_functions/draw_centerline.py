import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def find_centerline_angle(branch_data, skan_skeleton):
    # Get coordinates of the main centreline branch
    root_body_branches = branch_data[branch_data["branch-type"] == 2]

    try:
        all_coords = np.vstack([skan_skeleton.path_coordinates(idx) for idx in root_body_branches.index])
        coords = all_coords

    except:
        return None, None

    # PCA to find primary axis direction
    pca = PCA(n_components=2)
    pca.fit(coords)
    root_direction = pca.components_[0]  # main axis vector
    root_angle = np.degrees(np.arctan2(root_direction[0], root_direction[1]))
    # print(f"Root axis angle: {root_angle:.1f}°")
    # print(coords)

    return root_angle, coords

# Draw red line on images
def draw_axis_line(img, coords, root_angle, is_gray=False):
    # Find centre of root body to anchor the line
    centre_y, centre_x = np.mean(coords, axis=0).astype(int)

    # Line length — make it long enough to cross the whole image
    line_length = 2000

    # Convert angle to direction vector
    dx = int(np.cos(np.radians(root_angle)) * line_length)
    dy = int(np.sin(np.radians(root_angle)) * line_length)

    if is_gray:
        viz = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        viz = img.copy()

    cv2.line(viz, (centre_x - dx, centre_y - dy), (centre_x + dx, centre_y + dy), (255, 0, 0), thickness=3)

    plt.figure(figsize=(14, 6))
    plt.imshow(viz)
    plt.title(f"Root Axis (angle={root_angle:.1f}°)")
    plt.axis("off")
    plt.show()
