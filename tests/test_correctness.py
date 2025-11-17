import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.draw import disk, line

from directionality_quantification.main import run  # uses current CLI pipeline

# helper that writes TIFFs; keep your existing import if you already have it
from tests.test_sample import _write_inputs


class TestCellExtensionOrientation(unittest.TestCase):
    def test_three_cells_and_tiles(self):
        """
        Build three cells (three arrows) in different tiles and verify:
        - per-cell metrics for Cell A (like before),
        - tile aggregation correctness at tile_size=100.
        """
        # --- image + ROI ---
        H, W = 200, 400
        roi = [0, H, 0, W]      # your module treats ROI as [x_min, x_max, y_min, y_max]
        tile_size = 100         # -> 2 (x) by 4 (y) tiles in your current build code

        # --- cells ---
        # A: rightward extension (original case)
        cellA_center = (90, 190)     # (row, col)
        cellA_end    = (90, 240)

        # B: leftward extension, different tile
        cellB_center = (40, 40)
        cellB_end    = (40, 0)

        # C: upward extension, another tile
        cellC_center = (140, 340)
        cellC_end    = (110, 340)

        target_center = (40, 190)     # north of A
        radius = 10

        # expected (keep your original for A)
        expected_abs_angle_A = 90.0
        expected_rel_angle_A = 90.0
        expected_len_A = 50.0
        expected_radius = radius

        # --- build synthetic images ---
        labels_image = np.zeros((H, W), dtype=np.int32)
        raw_image = np.zeros((H, W), dtype=np.uint8)
        target_mask = np.zeros((H, W), dtype=bool)

        for center, end in [(cellA_center, cellA_end),
                            (cellB_center, cellB_end),
                            (cellC_center, cellC_end)]:
            rr, cc = disk(center, radius, shape=labels_image.shape)
            labels_image[rr, cc] = 1
            rr_line, cc_line = line(center[0], center[1], end[0], end[1])
            labels_image[rr_line, cc_line] = 1

        rr_t, cc_t = disk(target_center, radius, shape=target_mask.shape)
        target_mask[rr_t, cc_t] = True

        # run twice: without and with target
        for include_target in [False, True]:
            with self.subTest(include_target=include_target):
                temp_dir = Path("three_cells_tiles_test")
                output_dir = temp_dir / ("out_rel" if include_target else "out_abs")
                output_dir.mkdir(parents=True, exist_ok=True)

                raw_path, labels_path, target_path = _write_inputs(
                    temp_dir, raw_image, labels_image, target_mask
                )

                # --- run CLI pipeline once; request only tile_size=100 to simplify outputs ---
                old_argv = sys.argv[:]
                try:
                    sys.argv = [
                        "directionality-quantification",
                        "--input_raw", str(raw_path),
                        "--input_labeling", str(labels_path),
                        "--output", str(output_dir),
                        "--tiles", "100",          # ensure we get average_directions_tile100.csv
                    ]
                    if include_target:
                        sys.argv.extend(["--input_target", str(target_path)])
                    run()
                finally:
                    sys.argv = old_argv

                # --- per-cell checks from cells.csv ---
                cells_csv = output_dir / "cells.csv"
                self.assertTrue(cells_csv.exists(), "cells.csv was not created.")
                df = pd.read_csv(cells_csv)
                self.assertEqual(len(df), 3, "Expected three cells in the results table.")

                # Pick the row closest to A's center
                idxA = ((df["YC"] - cellA_center[0]).abs()
                        + (df["XC"] - cellA_center[1]).abs()).argmin()
                rowA = df.iloc[int(idxA)]

                # geometry at A
                self.assertAlmostEqual(rowA["XC"], cellA_center[1], delta=3,
                                       msg="Center X (cell A) is incorrect.")
                self.assertAlmostEqual(rowA["YC"], cellA_center[0], delta=3,
                                       msg="Center Y (cell A) is incorrect.")
                self.assertAlmostEqual(rowA["Radius biggest circle"], expected_radius, delta=2,
                                       msg="Radius (cell A) is incorrect.")
                self.assertAlmostEqual(rowA["Length cell vector"], expected_len_A, delta=8,
                                       msg="Length (cell A) is incorrect.")
                self.assertAlmostEqual(rowA["Absolute angle"], expected_abs_angle_A, delta=10,
                                       msg="Absolute angle (cell A) is incorrect.")
                if include_target:
                    self.assertAlmostEqual(rowA["Relative angle"], expected_rel_angle_A, delta=10,
                                           msg="Relative angle (cell A) is incorrect.")

                # --- tile checks from average_directions_tile100.csv ---
                avg_csv = output_dir / "average_directions_tile100.csv"
                self.assertTrue(avg_csv.exists(), "average_directions_tile100.csv was not created.")
                avg_df = pd.read_csv(avg_csv)

                def to_tile_indices(center_rc, tile_size):
                    r, c = center_rc  # (row, col)
                    ix = int(c // tile_size)
                    iy = int(r // tile_size)
                    return ix, iy

                # pull per-cell rows for easy access
                # (match each by nearest centers, like for A)
                def find_row_by_center(target_rc):
                    j = ((df["YC"] - target_rc[0]).abs()
                         + (df["XC"] - target_rc[1]).abs()).argmin()
                    return df.iloc[int(j)]

                rowB = find_row_by_center(cellB_center)
                rowC = find_row_by_center(cellC_center)

                # which tiles should each cell occupy?
                tA = to_tile_indices(cellA_center, tile_size)
                tB = to_tile_indices(cellB_center, tile_size)
                tC = to_tile_indices(cellC_center, tile_size)

                # fetch the corresponding tile rows from avg_df
                def tile_row(ix, iy):
                    hit = avg_df[(avg_df["tile_x"] == ix) & (avg_df["tile_y"] == iy)]
                    self.assertEqual(len(hit), 1, f"Tile ({ix},{iy}) not found or duplicated.")
                    return hit.iloc[0]

                trA = tile_row(*tA)
                trB = tile_row(*tB)
                trC = tile_row(*tC)

                # exactly one arrow per occupied tile
                self.assertEqual(int(trA["count"]), 1, "Tile A count should be 1.")
                self.assertEqual(int(trB["count"]), 1, "Tile B count should be 1.")
                self.assertEqual(int(trC["count"]), 1, "Tile C count should be 1.")

                if include_target:
                    # relative mode: avg_df stores angle (u) and avg_length (v) differently.
                    # Check avg_length ~= per-cell length for each single-arrow tile.
                    self.assertAlmostEqual(trA["avg_length"], rowA["Length cell vector"], delta=5,
                                           msg="Tile A avg_length mismatch (relative mode).")
                    self.assertAlmostEqual(trB["avg_length"], rowB["Length cell vector"], delta=5,
                                           msg="Tile B avg_length mismatch (relative mode).")
                    self.assertAlmostEqual(trC["avg_length"], rowC["Length cell vector"], delta=5,
                                           msg="Tile C avg_length mismatch (relative mode).")
                    self.assertEqual(trA["color_mode"], "relative")
                else:
                    # absolute mode: u = mean(DX), v = mean(DY) for the tile
                    self.assertAlmostEqual(trA["u"], rowA["DX"], delta=3, msg="Tile A u!=DX")
                    self.assertAlmostEqual(trA["v"], rowA["DY"], delta=3, msg="Tile A v!=DY")
                    self.assertAlmostEqual(trB["u"], rowB["DX"], delta=3, msg="Tile B u!=DX")
                    self.assertAlmostEqual(trB["v"], rowB["DY"], delta=3, msg="Tile B v!=DY")
                    self.assertAlmostEqual(trC["u"], rowC["DX"], delta=3, msg="Tile C u!=DX")
                    self.assertAlmostEqual(trC["v"], rowC["DY"], delta=3, msg="Tile C v!=DY")
                    self.assertEqual(trA["color_mode"], "absolute")


if __name__ == "__main__":
    unittest.main()

