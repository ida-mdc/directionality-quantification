# Cell Extension Orientation

## How to install
Tested with Python 3.11.
```
pip install directionality-quantification
```

## How to use

List arguments:

```
directionality-quantification --help
```

Example use case:

```
directionality-quantification --input_raw report/test_data/sample/input_raw.tif --input_labeling report/test_data/sample/input_labels.tif --input_target 
report/test_data/sample/input_target.tif --output report/test_data/sample --pixel_in_micron 0.65 --output_res 7:10
```

Generate exemplary output on sample data via unit test:

```
python -m unittest tests/test_sample.py
```


### **Step-by-Step Workflow**

#### **1. Parse Input Arguments**
- The script accepts various user-defined arguments, including:
  - Raw image and segmentation label map (required).
  - Optional parameters like ROIs, mask images, cell tables, and output settings.
- Arguments control preprocessing, analysis filters (e.g., size thresholds), and visualization options.

#### **2. Load Input Data**
- Reads the raw microscopy image and segmentation label map using `tifffile`.
- Optionally, loads:
  - A binary mask for specific regions.
  - A cell table with predefined properties for analysis.

#### **3. Crop Images to ROI**
- If ROIs are specified, the script extracts subregions of the raw image, label map, and mask for focused analysis.

#### **4. Segment and Label Regions**
- Segmentation regions are labeled using connected component analysis (`label`).
- The script identifies individual objects (cells) for further analysis.

#### **5. Filter Regions**
- Filters segmented regions based on:
  - **Minimum Size:** Excludes regions below a specified pixel count.
  - **Maximum Size:** Excludes regions exceeding a specified pixel count.

#### **6. Analyze Each Segment**
For each segmented region:
1. **Skeletonization**:
   - Generates the skeleton (centerline) of the region to identify structure and direction.
2. **Distance Map**:
   - Calculates the distance of each pixel to the nearest boundary (used for determining the region's center).
3. **Identify Key Points**:
   - Locates the central pixel and categorizes pixels as "inside" or "outside" the region relative to the center.
4. **Direction Calculation**:
   - Computes the mean directional vector for extensions outside the region.
   - Optionally, calculates directional vectors relative to a target distance map if provided.

#### **7. Store Results**

- For each region, records:
  - Direction vector length.
  - Absolute angle of the direction vector.
  - Relative angle compared to the target vector (if applicable).
  - Stores results in the cell table if provided.

#### **8. Generate Visualizations**
- Plots include:
  - **All Directions**: Visualizes individual cell extensions with directional arrows.
  - **Average Directions**: Computes average directions in user-defined tiles and visualizes them with heatmaps.
  - **ROIs**: Highlights selected regions of interest.

#### **9. Save Outputs**
- Results are saved to the specified output folder:
  - **CSV File**: Contains cell-level analysis metrics.
  - **Images**: Includes plots of directions, average directions, and ROIs.


### **Outputs**

### **1. Individual Cell Data (`cells.csv`)**

This file contains detailed metrics for each individual cell (or region) identified in the image.

| **Column Name**         | **Description** |
|:------------------------| :--- |
| `Label`                 | A unique integer identifier for each detected cell region. |
| `Area in px²`           | The total area of the cell region in square pixels. |
| `Area in um²`           | The total area of the cell region in square micrometers (µm²). |
| `Mean`                  | The mean pixel intensity within the cell region. |
| `XM`, `YM`              | The (X, Y) coordinates of the region's geometric center (centroid). |
| `Circ.`                 | **Circularity** of the region, a value from 0 (a line) to 1 (a perfect circle). |
| `%Area`                 | The percentage of the region's bounding box that is filled by the cell's pixels. |
| `MScore`                | A custom "morphology score" calculated from the cell's circularity and area. |
| `XC`, `YC`              | The (X, Y) coordinates for the center of the largest circle that can be inscribed within the cell. |
| `Radius biggest circle` | The radius (in pixels) of the largest inscribed circle. |
| `Length cell vector`    | The length (in pixels) of the primary orientation vector calculated for the cell's extensions. |
| `Anisotropy`            | A measure of the cell's elongation, ranging from 0 (not elongated) to 1 (highly elongated). |
| `Rolling ball angle`    | The angle of the local target vector (derived from the target image gradient), measured in **degrees** (0-360) from the upward vertical axis. |
| `Absolute angle`        | The absolute orientation of the cell's main vector, measured in **degrees** (0-360) from the upward vertical axis. |
| `Relative angle`        | The absolute difference between the `Absolute angle` and the `Rolling ball angle` in **degrees**, indicating how aligned the cell is to the target vector. |
| `DX`, `DY`              | The X and Y components of the cell's orientation vector (`Length cell vector`). |

### **2. Tiled Average Data (`average_directions_tile{tile_size}.csv`)**

For each tile size specified, a separate CSV file is generated. These files summarize the cell orientation data within a grid of tiles covering the image, which is useful for visualizing general trends.

| **Column Name** | **Description** |
| :--- | :--- |
| `tile_x`, `tile_y` | The column and row index of the tile in the grid. |
| `x`, `y` | The pixel coordinates of the top-left corner of the tile in the image. |
| `u`, `v` | Components of the average vector. Their meaning depends on the `color_mode`: <br> • **Absolute mode**: Mean X (`u`) and Y (`v`) components of the cell vectors. <br> • **Relative mode**: Length-weighted mean relative angle in radians (`u`) and the mean cell vector length (`v`). |
| `count` | The number of cells whose center falls within the tile. |
| `avg_length` | The average `Length cell vector` for all cells within the tile. |
| `tile_size` | The side length of the square tile in pixels. |
| `color_mode` | The mode used for calculation, either `absolute` or `relative` (if a target mask was provided). |
| `color_scalar_deg` | The final angle in degrees used to determine the tile's color in visualizations. |
| `color_hex` | The hexadecimal color code representing the `color_scalar_deg`. |
| `alpha` | The calculated transparency (0.0 to 1.0) for the tile, based on `count` and `avg_length`. |
| `max_count` | The 90th percentile of cell counts across all tiles, used for normalizing the alpha value. |
| `max_length` | The 90th percentile of average lengths across all tiles, used for normalizing the alpha value. |


#### **3. Visualization Outputs**
- Saved as image files in the specified output folder. The exact outputs depend on the options provided:

1. **All Directions (`directions_<region>.png`)**:
   - Visualizes individual cell extension vectors for each region.
   - Directional arrows are color-coded by angle or relative directionality (if a target map is used).

2. **Average Directions (`directions_<region>_tile<size>.png`)**:
   - Computes and visualizes average directional vectors within tiles of a specified size (e.g., 100px, 250px).
   - Heatmap-like visualization where tile colors represent average directions and lengths.

3. **ROIs (`ROIs.png`)**:
   - Highlights the selected regions of interest with color-coded bounding boxes.



## Interactive HTML Report

An interactive HTML report is available to visualize and explore your analysis results. The report is hosted via GitHub Pages.

### Viewing Reports

**Example Report:**
- [View example report with sample dataset](https://ida-mdc.github.io/directionality-quantification/report/?data=https://ida-mdc.github.io/directionality-quantification/report/test_data/sample/cells.csv)

**Using Your Own Data:**
To view your own analysis results:
1. Generate a `cells.csv` file using the tool
2. Open the [interactive report](https://ida-mdc.github.io/directionality-quantification/report/)
3. Click the "Load CSV File" button and select your `cells.csv` file

### Features

The interactive report includes:
- **Summary Statistics**: Total cells, mean area, circularity, vector length, and distance to target
- **Distribution Charts**: Histograms for area, circularity, angle, and vector length
- **Filtering**: Filter cells by extension length with real-time visualization updates
- **Cell Gallery**: Visual grid of cell thumbnails (20×20) with sorting options
- **Relative Angle Analysis**: Heatmap visualization of relative angle vs distance to target (when target data is available)
