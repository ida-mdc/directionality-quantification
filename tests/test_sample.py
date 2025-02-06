import unittest
import subprocess
from pathlib import Path


class TestCellExtensionOrientation(unittest.TestCase):
    def setUp(self):
        # Ensure output directory exists
        self.output_dir = Path("sample/result")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_example_run(self):
        # Run the command from the README example
        result = subprocess.run([
            "directionality-quantification",
            "--input_raw", "sample/input_raw.tif",
            "--input_labeling", "sample/input_labels.tif",
            "--input_target", "sample/input_target.tif",
            "--output", str(self.output_dir),
            "--pixel_in_micron", "0.65",
            "--output_res", "7:10"
        ], capture_output=True, text=True)

        # Check that the command ran successfully
        self.assertEqual(result.returncode, 0, msg=f"Error: {result.stderr}")

        # Verify the output folder has content (example: expected result files)
        output_files = list(self.output_dir.glob("*.png"))  # Modify if files are not PNGs
        self.assertGreater(len(output_files), 0, "Output directory should contain result images.")

    def tearDown(self):
        pass
        # uncomment the following code to clean up the output
        # # Clean up output directory after test
        # for file in self.output_dir.glob("*"):
        #     file.unlink()
        # self.output_dir.rmdir()

if __name__ == "__main__":
    unittest.main()
