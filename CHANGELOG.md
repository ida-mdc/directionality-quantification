# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-02-24

### Added

- **Tile Filtering by Extension Length**:
  - New `--min_extension_length` parameter to include only cells with sufficiently long extension vectors in tile statistics.
  - Tile tables now store `min_extension_length` and `excluded_cells_below_min_extension_length` to document the threshold and how many cells were excluded per tile.
  - Added `angle_std_deg` per tile (standard deviation of cell angles in degrees) as a measure of directional coherence.

- **Simplified Color Strategies**:
  - Replaced the previous three-strategy system with two clearer strategies:
    - `alpha_from_count` (new default): color from average angle, opacity from cell count.
    - `alpha_from_angle_std`: color from average angle, opacity from angular coherence (low std dev = high opacity).
  - Strategy name and alpha legend descriptions are stored in the tile CSVs (`color_strategy`, `alpha_description_low`, `alpha_description_high`).

### Changed

- **Relative Angle Aggregation in Tiles**:
  - Relative tile angles are now computed as a **simple (unweighted) average** of per‑cell relative angles (no weighting by extension length).

- **Relative Angle Colormap**:
  - Updated to a **red–white–blue** colormap for relative angles:
    - Red (0°) = towards target
    - White (90°) = parallel
    - Blue (180°) = away from target
  - This replaces the red–cyan–blue colormap introduced in 0.4.0.

- **CLI Inputs**:
  - `--input_raw` is now optional. If omitted, the labeling image is also used as the background for thumbnails and plots.

## [0.4.0] - 2026-02-03

### Added

- **Interactive HTML Report**: New client-side interactive report (`docs/report.html`) for visualizing analysis results
  - GitHub Pages hosting support with example dataset
  - Interactive visualizations: summary statistics, distribution charts, filtering, cell gallery

- **Color Strategy System**: New modular color strategy system (`directionality_quantification/color_strategy.py`)
  - `count_alpha_saturation` strategy (new default): saturation encodes vector strength
  - `alpha_from_count_and_length` strategy: alpha from count and length
  - `0.2.0` strategy: legacy hardcoded normalization
  - Color strategy information stored in output CSV files for proper legend generation

- **Enhanced Colormap**: Custom red-cyan-blue colormap for relative angles (replaces `coolwarm_r`)
  - Red (0°) = towards target (matches target visualization color)
  - Cyan (90°) = parallel movement
  - Blue (180°) = away from target

- **Full-Resolution Export**: New `--fullres` flag for generating full-resolution output images
  - Requires more RAM but produces high-quality visualizations
  - Includes full-resolution images with rectangles and arrows

### Changed

- **Test and Sample Data Location**:
  - All test data consolidated in `docs/test_data/`
