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
directionality-quantification --input_raw sample/input_raw.tif --input_labeling sample/input_labels.tif --input_target 
sample/input_target.tif --output sample/result --pixel_in_micron 0.65 --output_res 7:10
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

#### **1. CSV File**
- A CSV file named `cells.csv` is generated in the output directory (if a cell table is provided or regions are analyzed). 
- The file contains the following columns:

| **Column Name**           | **Description**                                                                                                   |
|---------------------------|-------------------------------------------------------------------------------------------------------------------|
| `length_cell_vector`      | Length of the calculated directional vector for the cell extension.                                               |
| `absolute_angle`          | Absolute angle of the directional vector in radians relative to the vertical axis.                               |
| `rolling_ball_angle`      | Angle of the directional vector relative to a target map (if provided).                                           |
| `relative_angle`          | Angle between the directional vector and the target vector (if applicable).                                       |

- If an input cell table is provided, these columns are appended to it, preserving the existing structure.

#### **2. Visualization Outputs**
- Saved as image files in the specified output folder. The exact outputs depend on the options provided:

1. **All Directions (`directions_<region>.png`)**:
   - Visualizes individual cell extension vectors for each region.
   - Directional arrows are color-coded by angle or relative directionality (if a target map is used).

2. **Average Directions (`directions_<region>_tile<size>.png`)**:
   - Computes and visualizes average directional vectors within tiles of a specified size (e.g., 100px, 250px).
   - Heatmap-like visualization where tile colors represent average directions and lengths.

3. **ROIs (`ROIs.png`)**:
   - Highlights the selected regions of interest with color-coded bounding boxes.

