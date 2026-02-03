import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from pandas import DataFrame
from scipy import ndimage
from skimage.morphology import skeletonize
from tqdm import tqdm

from directionality_quantification.plot import ABS_CMAP, ABS_NORM, REL_CMAP, REL_NORM
from directionality_quantification.color_strategy import ColorStrategy, get_color_strategy
from directionality_quantification.report import extract_cell_thumbnail


def analyze_segments(regions, image_target, pixel_in_micron, raw_image: Optional[np.ndarray] = None, extract_thumbnails: bool = False) -> DataFrame:
    """
    Analyze segmented regions and extract cell properties.
    
    Args:
        regions: List of RegionProps objects
        image_target: Target distance map image
        pixel_in_micron: Pixel size in microns
        raw_image: Raw image array for thumbnail extraction
        extract_thumbnails: Whether to extract and encode thumbnails
    
    Returns:
        DataFrame with cell analysis results
    """
    rows = []

    for index, region in enumerate(tqdm(regions, desc="Processing Regions")):
        label = region.label

        row = {
            "Label": label,
            "Area in px²": region.area,
            "Area in um²": region.area * (pixel_in_micron ** 2) if pixel_in_micron else None,
            "Mean": region.intensity_mean,
            "XM": region.centroid[1],
            "YM": region.centroid[0],
        }

        # Derived properties
        circularity = max(0, min(4 * math.pi * region.area / math.pow(region.perimeter, 2), 1.0))
        row["Circ."] = circularity
        row["%Area"] = region.area / region.area_filled * 100

        if pixel_in_micron:
            row["MScore"] = circularity * ((row["Area in um²"] - 27) / 27)

        skeleton, center, radius, L, abs_rad, rel_raw, rolling_ball_angle, orientation_vector, condition_outside, distance_to_target = region_extension_analysis(region, image_target)

        rel_rad = np.abs(rel_raw) % (2 * np.pi)

        abs_deg = (np.degrees(abs_rad) + 360) % 360
        rel_deg = (np.degrees(rel_rad) + 360) % 360
        rolling_ball_deg = (np.degrees(rolling_ball_angle) + 360) % 360
        dx = L * np.sin(abs_rad)
        dy = L * np.cos(abs_rad)

        row["XC"] = center[1]
        row["YC"] = center[0]
        row["Radius biggest circle"] = radius
        row["Length cell vector"] = L
        # Convert length to micrometers if available
        if pixel_in_micron and L is not None:
            row["Length cell vector (um)"] = L * pixel_in_micron
        else:
            row["Length cell vector (um)"] = None
        row["Rolling ball angle"] = rolling_ball_deg

        row["Absolute angle"] = abs_deg
        row["Relative angle"] = rel_deg

        row["DX"] = dx
        row["DY"] = dy
        
        row["Distance to target"] = distance_to_target
        if pixel_in_micron and distance_to_target is not None:
            row["Distance to target (um)"] = distance_to_target * pixel_in_micron
        else:
            row["Distance to target (um)"] = None

        if extract_thumbnails and raw_image is not None:
            thumbnail = extract_cell_thumbnail(raw_image, region)
            row["thumbnail"] = thumbnail
        elif extract_thumbnails:
            row["thumbnail"] = None

        rows.append(row)
    cell_table = pd.DataFrame(rows)
    return cell_table


def calculate_average_extension_vector(skeleton_points, root):
    """
    Calculates a vector representing the average coherent extension.
    Its direction is the average direction of all extension pixels.
    Its length is the maximum extension length modulated by a coherence score.
    """
    points = np.argwhere(skeleton_points)
    num_points = points.shape[0]
    if num_points == 0:
        return np.array([0.0, 0.0])

    points_centered = points - root
    norms = np.linalg.norm(points_centered, axis=1)
    non_zero_norms = norms > 1e-6
    unit_vectors = points_centered[non_zero_norms] / norms[non_zero_norms, np.newaxis]
    sum_of_units = np.sum(unit_vectors, axis=0)
    coherence_vector = sum_of_units / num_points

    coherence_score = np.linalg.norm(coherence_vector)
    average_direction = coherence_vector / coherence_score if coherence_score > 1e-6 else np.array([0., 0.])

    max_extension_length = np.max(norms)

    final_vector_length = max_extension_length * coherence_score
    final_vector = average_direction * final_vector_length

    return final_vector

def region_extension_analysis(region, image_target):
    # skeletonize
    skeleton = skeletonize(region.intensity_image)
    # calculate distance map
    distance_region = ndimage.distance_transform_edt(region.intensity_image)
    miny, minx, maxy, maxx = region.bbox
    # calculate center
    maxradius = np.max(distance_region, axis=None)
    center = np.unravel_index(np.argmax(distance_region, axis=None), distance_region.shape)
    condition_outside = (skeleton > 0)

    orientation_vector = calculate_average_extension_vector(condition_outside, center)
    length = np.linalg.norm(orientation_vector)

    # pixel_locations_relevant_to_direction = np.column_stack(np.where(condition_outside))
    # pixel_locations_relevant_to_direction = pixel_locations_relevant_to_direction - center
    center_translated = [center[0] + miny, center[1] + minx]
    target_vector = [0, 0]
    distance_to_target = None
    if image_target is not None:
        neighbor_y = [center_translated[0] + 1, center_translated[1]]
        neighbor_x = [center_translated[0], center_translated[1] + 1]
        if neighbor_x[1] < image_target.shape[1] and neighbor_y[0] < image_target.shape[0]:
            value_at_center = image_target[center_translated[0], center_translated[1]]
            value_at_neighbor_x = image_target[neighbor_x[0], neighbor_x[1]]
            value_at_neighbor_y = image_target[neighbor_y[0], neighbor_y[1]]
            target_vector = [value_at_center - value_at_neighbor_y, value_at_center - value_at_neighbor_x]
            distance_to_target = float(value_at_center)
    length_cell_vector = 0
    absolute_angle = 0
    rolling_ball_angle = 0
    relative_angle = 0
    if length > maxradius:
        length_cell_vector = length
        absolute_angle = angle_between((-1, 0), orientation_vector)
        rolling_ball_angle = angle_between((-1, 0), target_vector)
        relative_angle = angle_between(orientation_vector, target_vector)
    return skeleton, center_translated, maxradius, length_cell_vector, absolute_angle, relative_angle, rolling_ball_angle, orientation_vector, condition_outside, distance_to_target


def build_average_directions_table(cell_table, shape, crop_extend, tile_size, image_target_mask, color_strategy: Optional[ColorStrategy] = None):
    """
    Build average directions table with configurable color strategy.
    
    Args:
        cell_table: DataFrame with cell data
        shape: Shape of the image
        crop_extend: Crop/extend coordinates [y_min, y_max, x_min, x_max]
        tile_size: Size of tiles in pixels
        image_target_mask: Target mask image (None for absolute mode)
        color_strategy: ColorStrategy instance. If None, uses "count_alpha_saturation" strategy.
    
    Returns:
        DataFrame with tile data including colors and alpha values
    """
    if color_strategy is None:
        color_strategy = get_color_strategy("count_alpha_saturation")
    
    tiles_num_y = int(shape[0] / tile_size) + 1
    tiles_num_x = int(shape[1] / tile_size) + 1

    ix = ((cell_table["XC"] - crop_extend[2]) // tile_size).astype(int)
    iy = ((cell_table["YC"] - crop_extend[0]) // tile_size).astype(int)

    rows = []
    is_relative = image_target_mask is not None

    for tile_x, tile_y in np.ndindex(tiles_num_x, tiles_num_y):
        x = int(tile_x * tile_size + crop_extend[2])
        y = int(tile_y * tile_size + crop_extend[0])

        mask = (ix == tile_x) & (iy == tile_y)
        idx = np.where(mask.to_numpy())[0]
        count = int(idx.size)

        if count == 0:
            row = {
                "tile_x": tile_x, "tile_y": tile_y, "x": x, "y": y,
                "u": 0.0, "v": 0.0, "count": 0, "avg_length": 0.0,
                "tile_size": tile_size, "color_mode": "relative" if is_relative else "absolute",
                "color_scalar_deg": 0.0, "color_hex": to_hex((0, 0, 0)),
                # alpha filled later by color strategy
            }
            rows.append(row)
            continue

        if is_relative:
            rel_rad = np.radians(cell_table.loc[idx, "Relative angle"])
            L = cell_table.loc[idx, "Length cell vector"]
            wsum = np.nansum(L)
            rel_tile = (np.nansum(rel_rad * L) / wsum) if wsum > 0 else 0.0
            u = rel_tile
            v = float(np.nanmean(L))
            avg_length = v
            color_scalar_deg = float(np.degrees(rel_tile))
            color_hex = to_hex(REL_CMAP(REL_NORM(color_scalar_deg)))
        else:
            dx_bar = float(np.nanmean(cell_table.loc[idx, "DX"]))
            dy_bar = float(np.nanmean(cell_table.loc[idx, "DY"]))
            u, v = dx_bar, dy_bar
            avg_length = float(np.hypot(u, v))
            angle_deg = (np.degrees(np.arctan2(u, v))) % 360.0
            color_scalar_deg = angle_deg
            color_hex = to_hex(ABS_CMAP(ABS_NORM(angle_deg)))

        row = {
            "tile_x": tile_x, "tile_y": tile_y, "x": x, "y": y,
            "u": u, "v": v, "count": count, "avg_length": avg_length,
            "tile_size": tile_size, "color_mode": "relative" if is_relative else "absolute",
            "color_scalar_deg": color_scalar_deg, "color_hex": color_hex,
        }
        rows.append(row)

    color_strategy.compute_color_and_alpha(rows, is_relative, tile_size)
    alpha_desc_low, alpha_desc_high = color_strategy.get_alpha_description()
    for r in rows:
        r["alpha_description_low"] = alpha_desc_low
        r["alpha_description_high"] = alpha_desc_high

    return pd.DataFrame(rows)


def angle_between(v1, v2):
    """Signed clockwise angle from v1 -> v2 (vectors in (y, x)), in (-π, π]."""
    v1 = np.asarray(v1, float); v2 = np.asarray(v2, float)
    if (v1[0] == 0 and v1[1] == 0) or (v2[0] == 0 and v2[1] == 0):
        return 0.0
    v1 /= np.linalg.norm(v1); v2 /= np.linalg.norm(v2)
    # convert (y,x)->(x,y) but keep image 'y down' convention
    x1, y1 = v1[1], v1[0]
    x2, y2 = v2[1], v2[0]
    dot = np.clip(x1*x2 + y1*y2, -1.0, 1.0)
    det = x1*y2 - y1*x2
    return float(np.arctan2(det, dot))


def write_table(cell_table_content: DataFrame, output):
    if cell_table_content is not None:
        if output:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            cell_table_content.to_csv(output.joinpath("cells.csv"))


def compute_and_write_avg_dir_tables(cell_table: DataFrame, raw_image, roi, image_target_mask, tiles, output, color_strategy: Optional[ColorStrategy] = None):

    dfs = []

    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

    for tile in tiles.split(','):
        tile_size = int(tile)

        avg_df = build_average_directions_table(
            cell_table=cell_table,
            shape=raw_image.shape,
            crop_extend=roi,
            tile_size=tile_size,
            image_target_mask=image_target_mask,
            color_strategy=color_strategy
        )

        dfs.append(avg_df)

        if output:
            avg_csv = output.joinpath(f'average_directions_tile{tile_size}.csv')
            avg_df_with_cells = avg_df[avg_df["count"] > 0]
            avg_df_with_cells.to_csv(avg_csv, index=False)
            print(f"Saved average directions table: {Path(avg_csv).absolute()} ({len(avg_df_with_cells)} tiles with cells)")


    if output:
        print(f"Results written to {output}")

    return dfs
