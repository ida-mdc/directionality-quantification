import math
import sys
import unittest
from pathlib import Path

import numpy as np
import tifffile
from skimage.draw import disk, line

from directionality_quantification.main import run, angle_between


class TestCellExtensionOrientation(unittest.TestCase):
    def setUp(self):
        # Ensure output directory exists
        self.output_dir = Path("sample/result")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_example_run(self):
        # Run the command from the README example
        sys.argv = ["directionality-quantification",
            "--input_raw", "../sample/input_raw.tif",
            "--input_labeling", "../sample/input_labels.tif",
            "--input_target", "../sample/input_target.tif",
            "--output", str(self.output_dir),
            "--pixel_in_micron", "0.65",
            "--output_res", "7:10"]

        run()

        # Verify the output folder has content (example: expected result files)
        output_files = list(self.output_dir.glob("*.png"))  # Modify if files are not PNGs
        self.assertGreater(len(output_files), 0, "Output directory should contain result images.")

    def test_example_run2(self):
        # Define a larger image size.
        width, height = 3000, 2000
        # Create a raw image with a uniform gray background.
        raw_image = np.full((height, width), 100, dtype=np.uint8)

        # Create the labeling image (segmentation) as an empty image.
        labels_image = np.zeros((height, width), dtype=np.int32)

        # Grid parameters: 20 rows and 30 columns.
        n_rows, n_cols = 20, 30
        circle_radius = 15

        # Define extension length range.
        max_extension = 100

        # Safe margin: ensure cells and extensions remain inside the image.
        margin = circle_radius + max_extension  # here, 15 + 30 = 45

        # Calculate grid spacing restricted to the safe region.
        x_space = (width - 2 * margin) / (n_cols - 1) if n_cols > 1 else 0
        y_space = (height - 2 * margin) / (n_rows - 1) if n_rows > 1 else 0

        # The center of the image will be used as target/reference.
        center_x, center_y = width // 2, height // 2

        label_count = 1
        total_cells = n_rows * n_cols
        for i in range(n_rows):
            for j in range(n_cols):
                # Compute the cell center inside the safe region.
                cx = int(margin + j * x_space)
                cy = int(margin + i * y_space)

                # Draw a filled circle representing the cell body.
                rr, cc = disk((cy, cx), circle_radius, shape=labels_image.shape)
                labels_image[rr, cc] = label_count

                # Determine extension length deterministically.
                cell_index = i * n_cols + j

                angle = (cell_index / total_cells) * 4 * math.pi
                extension_dir = np.array([math.cos(angle), math.sin(angle)])

                extension_length = 15+40*i*j*1./n_rows*1./n_cols

                # Compute the starting point at the circle boundary along the extension direction.
                start_x = cx + int(circle_radius * extension_dir[0])
                start_y = cy + int(circle_radius * extension_dir[1])
                # Compute the end point using the chosen extension length.
                end_x = start_x + int(extension_length * extension_dir[0])
                end_y = start_y + int(extension_length * extension_dir[1])

                # Draw the extension line.
                rr_line, cc_line = line(start_y, start_x, end_y, end_x)
                labels_image[rr_line, cc_line] = label_count

                label_count += 1

        # Create a target mask with a circle at the image center (radius 100 pixels).
        target_mask = np.zeros((height, width), dtype=bool)
        rr_target, cc_target = disk((center_y, center_x), 100, shape=target_mask.shape)
        target_mask[rr_target, cc_target] = True

        # Write the input images to temporary TIFF files.
        temp_dir = Path("sample2")
        temp_dir.mkdir(exist_ok=True)
        raw_path = temp_dir / "input_raw.tif"
        labels_path = temp_dir / "input_labels.tif"
        target_path = temp_dir / "input_target.tif"

        tifffile.imwrite(raw_path, raw_image.astype(np.uint8))
        tifffile.imwrite(labels_path, labels_image.astype(np.uint16))
        # Save the target mask as uint8.
        tifffile.imwrite(target_path, target_mask.astype(np.uint8))

        # Create an output directory.
        output_dir = temp_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Set up command line arguments to point to these temporary files.
        sys.argv = [
            "directionality-quantification",
            "--input_raw", str(raw_path),
            "--input_labeling", str(labels_path),
            # "--input_target", str(target_path),
            "--output", str(output_dir),
            "--pixel_in_micron", "0.65",
            "--output_res", "7:10"
        ]

        # Run your analysis main function.
        run()

        # Verify that the output folder has produced result images (e.g., PNG files).
        output_files = list(output_dir.glob("*.png"))
        assert len(output_files) > 0, "Output directory should contain result images."
        print("Generated output files:", output_files)


    def test_angle_between(self):
        print(angle_between((1, 0), (0, 1)))
        print(angle_between((1, 0), (0, -1)))
        print(angle_between((1, 0), (1, 0)))
        print(angle_between((1, 0), (-1, 0)))

    def tearDown(self):
        pass
        # uncomment the following code to clean up the output
        # # Clean up output directory after test
        # for file in self.output_dir.glob("*"):
        #     file.unlink()
        # self.output_dir.rmdir()

if __name__ == "__main__":
    unittest.main()
