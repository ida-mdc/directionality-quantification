import math
import sys
import unittest
from pathlib import Path

import numpy as np
import tifffile
from skimage.draw import disk, line

from directionality_quantification.main import run
from directionality_quantification.process import angle_between

TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent


def _write_inputs(base_dir: Path, raw_image, labels_image, target_mask):
    raw_path = base_dir / "input_raw.tif"
    labels_path = base_dir / "input_labels.tif"
    target_path = base_dir / "input_target.tif"

    tifffile.imwrite(raw_path, raw_image.astype(np.uint8))
    tifffile.imwrite(labels_path, labels_image.astype(np.uint16))
    tifffile.imwrite(target_path, target_mask.astype(np.uint8))

    return raw_path, labels_path, target_path


class TestCellExtensionOrientation(unittest.TestCase):
    def setUp(self):
        self.test_results_dir = TEST_DIR / "test_results"
        self.test_results_dir.mkdir(parents=True, exist_ok=True)

    def test_example_run(self):
        output_dir = self.test_results_dir / "sample"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sys.argv = ["directionality-quantification",
            "--input_raw", str(PROJECT_ROOT / "report" / "test_data" / "sample" / "input_raw.tif"),
            "--input_labeling", str(PROJECT_ROOT / "report" / "test_data" / "sample" / "input_labels.tif"),
            "--input_target", str(PROJECT_ROOT / "report" / "test_data" / "sample" / "input_target.tif"),
            "--output", str(output_dir),
            "--pixel_in_micron", "0.65",
            "--output_res", "10:7"]

        run()

        output_files = list(output_dir.glob("*.png"))
        self.assertGreater(len(output_files), 0, "Output directory should contain result images.")

    def _make_inputs_uniform(self):
        """Original uniform grid."""
        width, height = 3000, 2000
        raw_image = np.full((height, width), 100, dtype=np.uint8)
        labels_image = np.zeros((height, width), dtype=np.int32)

        n_rows, n_cols = 20, 30
        circle_radius = 15
        max_extension = 100
        margin = circle_radius + max_extension

        x_space = (width - 2 * margin) / (n_cols - 1) if n_cols > 1 else 0
        y_space = (height - 2 * margin) / (n_rows - 1) if n_rows > 1 else 0

        center_x, center_y = width // 2, height // 2

        label_count = 1
        total_cells = n_rows * n_cols
        for i in range(n_rows):
            for j in range(n_cols):
                cx = int(margin + j * x_space)
                cy = int(margin + i * y_space)

                rr, cc = disk((cy, cx), circle_radius, shape=labels_image.shape)
                labels_image[rr, cc] = label_count

                cell_index = i * n_cols + j
                angle = (cell_index / total_cells) * 4 * math.pi
                extension_dir = np.array([math.cos(angle), math.sin(angle)])

                extension_length = 15 + 40 * i * j / n_rows / n_cols

                start_x = cx + int(circle_radius * extension_dir[0])
                start_y = cy + int(circle_radius * extension_dir[1])
                end_x = start_x + int(extension_length * extension_dir[0])
                end_y = start_y + int(extension_length * extension_dir[1])

                rr_line, cc_line = line(start_y, start_x, end_y, end_x)
                labels_image[rr_line, cc_line] = label_count

                label_count += 1

        target_mask = np.zeros((height, width), dtype=bool)
        rr_t, cc_t = disk((center_y, center_x), 100, shape=target_mask.shape)
        target_mask[rr_t, cc_t] = True

        return raw_image, labels_image, target_mask

    def _make_inputs_uneven(self):
        """
        Non-uniform grid: cells become sparser along x (to the right),
        and extensions become longer with y (downward).
        """
        width, height = 3000, 2000
        raw_image = np.full((height, width), 100, dtype=np.uint8)
        labels_image = np.zeros((height, width), dtype=np.int32)

        n_rows, n_cols = 20, 30
        circle_radius = 15
        # We’ll allow longer tails in principle but still keep a safety margin.
        max_extension = 160
        margin = circle_radius + max_extension

        def quad_map(u: float) -> float:
            return u * u

        x_positions = []
        for j in range(n_cols):
            if n_cols == 1:
                u = 0.0
            else:
                u = j / (n_cols - 1)
            t = quad_map(u)
            x = int(margin + t * (width - 2 * margin))
            x_positions.append(x)

        y_positions = []
        for i in range(n_rows):
            if n_rows == 1:
                v = 0.0
            else:
                v = i / (n_rows - 1)
            y = int(margin + v * (height - 2 * margin))
            y_positions.append(y)

        min_len = 20
        max_len = 140
        center_x, center_y = width // 2, height // 2

        label_count = 1
        total_cells = n_rows * n_cols
        for i in range(n_rows):
            for j in range(n_cols):
                cx = x_positions[j]
                cy = y_positions[i]

                rr, cc = disk((cy, cx), circle_radius, shape=labels_image.shape)
                labels_image[rr, cc] = label_count

                cell_index = i * n_cols + j
                angle = (cell_index / total_cells) * 4 * math.pi
                extension_dir = np.array([math.cos(angle), math.sin(angle)])

                v = 0.0 if n_rows == 1 else i / (n_rows - 1)
                extension_length = min_len + v * (max_len - min_len)

                start_x = cx + int(circle_radius * extension_dir[0])
                start_y = cy + int(circle_radius * extension_dir[1])
                end_x = start_x + int(extension_length * extension_dir[0])
                end_y = start_y + int(extension_length * extension_dir[1])

                rr_line, cc_line = line(start_y, start_x, end_y, end_x)
                labels_image[rr_line, cc_line] = label_count

                label_count += 1

        target_mask = np.zeros((height, width), dtype=bool)
        rr_t, cc_t = disk((center_y, center_x), 100, shape=target_mask.shape)
        target_mask[rr_t, cc_t] = True

        return raw_image, labels_image, target_mask

    def _make_inputs_multi_extension(self):
        """
        Grid where the number of extensions grows with y (1-4), and
        the angular spread of extensions grows with x (from a single
        direction to multiple directions).
        """
        width, height = 3000, 2000
        labels_image = np.zeros((height, width), dtype=np.int32)

        n_rows, n_cols = 5, 10
        circle_radius = 15
        extension_length = 80
        margin = circle_radius + extension_length + 10  # a bit of safety margin

        x_space = (width - 2 * margin) / (n_cols - 1) if n_cols > 1 else 0
        y_space = (height - 2 * margin) / (n_rows - 1) if n_rows > 1 else 0

        center_x, center_y = width // 2, height // 2

        label_count = 1
        for i in range(n_rows):
            for j in range(n_cols):
                cx = int(margin + j * x_space)
                cy = int(margin + i * y_space)

                rr, cc = disk((cy, cx), circle_radius, shape=labels_image.shape)
                labels_image[rr, cc] = label_count

                v_norm = i / (n_rows - 1) if n_rows > 1 else 0
                num_extensions = 1 + int(round(3 * v_norm))

                u_norm = j / (n_cols - 1) if n_cols > 1 else 0
                max_spread_angle = math.pi
                spread_angle = u_norm * max_spread_angle

                vec_to_center = np.array([center_x - cx, center_y - cy])
                if np.linalg.norm(vec_to_center) < 1e-6:
                    base_angle = 0
                else:
                    base_angle = np.arctan2(vec_to_center[1], vec_to_center[0])

                for m in range(num_extensions):
                    if num_extensions == 1:
                        offset = 0
                    else:
                        offset = (m / (num_extensions - 1) - 0.5) * spread_angle

                    angle = base_angle + offset
                    extension_dir = np.array([math.cos(angle), math.sin(angle)])

                    start_x = cx + int(circle_radius * extension_dir[0])
                    start_y = cy + int(circle_radius * extension_dir[1])
                    end_x = start_x + int(extension_length * extension_dir[0])
                    end_y = start_y + int(extension_length * extension_dir[1])

                    rr_line, cc_line = line(start_y, start_x, end_y, end_x)
                    valid_indices = (rr_line >= 0) & (rr_line < height) & (cc_line >= 0) & (cc_line < width)
                    labels_image[rr_line[valid_indices], cc_line[valid_indices]] = label_count

                label_count += 1
        target_mask = np.zeros((height, width), dtype=bool)
        rr_t, cc_t = disk((center_y, center_x), 100, shape=target_mask.shape)
        target_mask[rr_t, cc_t] = True

        return labels_image, labels_image, target_mask

    def _run_case(self, raw_image, labels_image, target_mask, output_name):
        for include_target in (False, True):
            with self.subTest(include_target=include_target):
                test_case_name = output_name + ("_with_target" if include_target else "_without_target")
                temp_dir = self.test_results_dir / test_case_name
                output_dir = temp_dir / "output"
                output_dir.mkdir(exist_ok=True, parents=True)

                raw_path, labels_path, target_path = _write_inputs(
                    temp_dir, raw_image, labels_image, target_mask
                )

                old_argv = sys.argv[:]
                try:
                    sys.argv = [
                        "directionality-quantification",
                        "--input_raw", str(raw_path),
                        "--input_labeling", str(labels_path),
                        "--output", str(output_dir),
                        "--pixel_in_micron", "0.65",
                        "--output_res", "10:7",
                    ]
                    if include_target:
                        sys.argv.extend(["--input_target", str(target_path)])

                    run()
                finally:
                    sys.argv = old_argv

                output_files = list(output_dir.glob("*.png"))
                self.assertGreater(
                    len(output_files), 0,
                    f"Output directory should contain result images (include_target={include_target})."
                )

    def test_uniform_grid_with_and_without_target(self):
        """Original behavior (uniform spacing) for regression."""
        raw_image, labels_image, target_mask = self._make_inputs_uniform()
        self._run_case(raw_image, labels_image, target_mask, "generated_uniform")

    def test_uneven_grid_sparser_x_longer_y_with_and_without_target(self):
        """New dataset: sparser along x, longer extensions along y."""
        raw_image, labels_image, target_mask = self._make_inputs_uneven()
        self._run_case(raw_image, labels_image, target_mask, "generated_uneven")

    def test_multi_extension_grid_with_and_without_target(self):
        """
        New dataset: number of extensions grows along y,
        angular spread grows along x.
        """
        raw_image, labels_image, target_mask = self._make_inputs_multi_extension()
        self._run_case(raw_image, labels_image, target_mask, "generated_multi_extension")

    def test_angle_between(self):
        """Test angle_between function with various vector pairs."""
        import math
        
        angle1 = angle_between((1, 0), (0, 1))
        self.assertAlmostEqual(angle1, -math.pi / 2, delta=0.1, msg="Perpendicular vectors (1,0)->(0,1) should give -π/2 radians")
        
        angle2 = angle_between((1, 0), (0, -1))
        self.assertAlmostEqual(angle2, math.pi / 2, delta=0.1, msg="Perpendicular vectors (1,0)->(0,-1) should give π/2 radians")
        
        angle3 = angle_between((1, 0), (1, 0))
        self.assertAlmostEqual(angle3, 0.0, delta=0.1, msg="Parallel vectors should give 0 radians")
        
        angle4 = angle_between((1, 0), (-1, 0))
        self.assertAlmostEqual(angle4, -math.pi, delta=0.1, msg="Opposite vectors should give -π radians")

    def test_fullres_rectangles_with_alpha(self):
        """Test that full-resolution rectangles are drawn with alpha blending."""
        from directionality_quantification.plot import save_fullres_with_rectangles
        from directionality_quantification.process import compute_and_write_avg_dir_tables
        from pandas import DataFrame
        import tempfile
        
        raw_image = np.full((100, 100), 128, dtype=np.uint8)
        roi_fullres = [0, 100, 0, 100]
        
        cell_table = DataFrame({
            'YC': [50],
            'XC': [50],
            'DY': [10],
            'DX': [10],
            'Relative angle': [45],
            'Absolute angle': [45]
        })
        
        avg_tables = compute_and_write_avg_dir_tables(
            cell_table, raw_image, roi_fullres, None, "100", None
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            
            try:
                save_fullres_with_rectangles(
                    output, raw_image, avg_tables, roi_fullres, None, "100"
                )
                png_files = list(output.glob("fullres_with_rectangles_tile*.png"))
                self.assertGreater(len(png_files), 0, "Should create PNG files")
            except TypeError as e:
                if "can't multiply sequence" in str(e):
                    self.fail(f"Alpha blending failed: {e}")
                raise

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()