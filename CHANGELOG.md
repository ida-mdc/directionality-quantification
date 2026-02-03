# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
