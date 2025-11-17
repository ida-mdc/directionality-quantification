import math
import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.draw import disk, line
from skimage.measure import regionprops
from skimage.morphology import skeletonize

from directionality_quantification.process import region_extension_analysis, \
    angle_between


def _make_single_cell(angles_rad):
    """Generates a single cell and returns its direction AND magnitude."""
    img_size = 200
    labels_image = np.zeros((img_size, img_size), dtype=np.int32)
    center_y, center_x = img_size // 2, img_size // 2
    radius = 20
    ext_len = 60
    rr, cc = disk((center_y, center_x), radius)
    labels_image[rr, cc] = 1

    vectors = [np.array([math.sin(a), math.cos(a)]) for a in angles_rad]

    for vec in vectors:
        start_y = int(center_y + radius * vec[0])
        start_x = int(center_x + radius * vec[1])
        end_y = int(start_y + ext_len * vec[0])
        end_x = int(start_x + ext_len * vec[1])
        rr_line, cc_line = line(start_y, start_x, end_y, end_x)
        labels_image[rr_line, cc_line] = 1

    sum_vec = np.sum(vectors, axis=0)
    magnitude = np.linalg.norm(sum_vec)

    if magnitude > 1e-6:
        direction = sum_vec / magnitude
    else:
        direction = np.array([0.0, 0.0])

    return labels_image, direction, magnitude


class TestCellExtensionOrientation(unittest.TestCase):

    def test_visual(self):
        test_cases = [
            {"name": "Spread Apart", "angles_deg": [-90, 45, 135]},
            {"name": "Same Direction", "angles_deg": [30, 45, 60]},
            {"name": "Symmetrical Spread", "angles_deg": [90, 210, 330]},
        ]

        fig, axes = plt.subplots(len(test_cases), 5, figsize=(24, 15))
        fig.suptitle(f"Visual Debugging with Multiple Test Cases", fontsize=20)

        for i, case in enumerate(test_cases):
            print(f"\n--- RUNNING CASE: {case['name']} ---")

            # 1. Generate Data
            angles_rad = [np.deg2rad(a) for a in case['angles_deg']]
            labels_image, expected_direction, expected_magnitude = _make_single_cell(angles_rad)

            regions = regionprops(labels_image, intensity_image=(labels_image > 0))
            region = regions[0]

            # 2. Run Analysis
            (skeleton, center_translated, maxradius,
             length_cell_vector, absolute_angle, relative_angle,
             rolling_ball_angle, orientation_vector, condition_outside) = region_extension_analysis(region, image_target=None)

            # 4. Visualization
            binary_image = region.intensity_image
            dist_map = ndimage.distance_transform_edt(binary_image)
            center = np.unravel_index(np.argmax(dist_map), dist_map.shape)
            radius = np.max(dist_map)
            skeleton = skeletonize(binary_image)

            ax_row = axes[i]
            ax_row[0].set_ylabel(case['name'], fontsize=16, fontweight='bold')

            # Plotting logic for panels 1-3
            ax_row[0].imshow(binary_image, cmap='gray', origin='lower')
            ax_row[1].imshow(dist_map, cmap='viridis', origin='lower')
            ax_row[1].plot(center[1], center[0], 'r+', ms=10)
            ax_row[1].add_patch(plt.Circle((center[1], center[0]), radius, color='r', fill=False, ls='--'))
            ax_row[2].imshow(binary_image, cmap='gray_r', origin='lower')
            sk_rows, sk_cols = np.where(skeleton)
            ax_row[2].plot(sk_cols, sk_rows, '.', color='cyan', markersize=2)
            ax_row[3].imshow(binary_image, cmap='gray_r', origin='lower')
            ax_row[3].imshow(np.ma.masked_where(condition_outside == 0, condition_outside), cmap='cool', origin='lower')

            ax_row[4].imshow(binary_image, cmap='gray', origin='lower')

            ax_row[4].arrow(center[1], center[0], -length_cell_vector * np.cos(absolute_angle),
                            length_cell_vector * np.sin(absolute_angle),
                            head_width=7, fc='lime', ec='black')

        column_titles = ["1. Original Cell", "2. Distance Map", "3. Skeleton", "4. Extensions Used", "5. Vector Comparison"]
        for j, title in enumerate(column_titles):
            axes[0, j].set_title(title, fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        print("\nSaving summary plot to 'visual_test_summary.png'...")
        plt.savefig('visual_test_summary.png', dpi=200)


if __name__ == "__main__":
    unittest.main()