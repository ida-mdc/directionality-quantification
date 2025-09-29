import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from pandas import DataFrame
from scipy import ndimage
from skimage.morphology import skeletonize
from tqdm import tqdm

from directionality_quantification.plot import ABS_CMAP, ABS_NORM, REL_CMAP, REL_NORM


def analyze_segments(regions, image_target, pixel_in_micron) -> DataFrame:
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

        # angles from region_extension_analysis(...)
        skeleton, center, radius, L, anisotropy, abs_rad, rel_raw, rolling_ball_angle = region_extension_analysis(region, image_target)

        # --- normalize for math (canonical) ---
        rel_rad = np.abs(rel_raw) % (2 * np.pi)

        # --- normalize for display/color (standard 0 at +X) ---
        abs_deg = (np.degrees(abs_rad) + 360) % 360
        rel_deg = (np.degrees(rel_rad) + 360) % 360
        rolling_ball_deg = (np.degrees(rolling_ball_angle) + 360) % 360

        # vectors for downstream math, from canonical abs_rad
        dx = L * np.sin(abs_rad)
        dy = L * np.cos(abs_rad)

        row["X center biggest circle"] = center[1]
        row["Y center biggest circle"] = center[0]
        row["Radius biggest circle"] = radius
        row["Length cell vector"] = L
        row["Anisotropy"] = anisotropy
        row["Rolling ball angle"] = rolling_ball_deg

        row["Absolute angle"] = abs_deg
        row["Relative angle"] = rel_deg

        row["DX"] = dx
        row["DY"] = dy

        # row["Relative angle color"] = REL_CMAP(REL_NORM(rel_deg))
        # row["Absolute angle color"] = ABS_CMAP(ABS_NORM(abs_deg))

        rows.append(row)

    # Convert list of dicts to DataFrame
    cell_table = pd.DataFrame(rows)
    return cell_table


def orientation_with_anisotropy_from_root(skeleton_image, root_point):
    """
    Args:
        skeleton_image: 2D binary array
        root_point: (y, x) in image coords

    Returns:
        anisotropy: float in [0,1]
        net_vector_yx: np.ndarray([dy, dx])  # (y, x) order
        net_length: float
    """
    sk = skeleton_image.astype(bool)
    ys, xs = np.where(sk)
    n = len(xs)
    if n < 2:
        return 0.0, np.array([0.0, 0.0]), 0.0

    pts = np.column_stack((ys, xs)).astype(float)  # (y, x)
    root = np.array(root_point, dtype=float)       # (y, x)

    # PCA in (y, x)
    mean = pts.mean(axis=0)
    X = pts - mean
    C = (X.T @ X) / n
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)
    lam_min, lam_max = vals[order[0]], vals[order[1]]

    # major axis (unit) in (y, x)
    u_yx = vecs[:, order[1]]
    u_yx /= (np.linalg.norm(u_yx) + 1e-12)

    # anisotropy
    denom = lam_max + lam_min
    anisotropy = 0.0 if denom <= 0 else float((lam_max - lam_min) / denom)

    # project onto u (all in (y, x))
    proj = pts @ u_yx
    proj_min, proj_max = float(proj.min()), float(proj.max())
    proj_root = float(root @ u_yx)

    # If root lies between min/max, use (right - left) for sign.
    # If root lies outside, use the side the skeleton is on (mean vs root) for sign.
    if proj_min <= proj_root <= proj_max:
        left  = proj_root - proj_min   # >=0
        right = proj_max - proj_root   # >=0
        net_scalar = right - left      # signed
    else:
        # Entire skeleton is on one side of the root along the axis.
        side_sign = 1.0 if proj.mean() > proj_root else -1.0
        # distance to the far end on that side (positive magnitude)
        one_side_len = max(proj_max - proj_root, proj_root - proj_min)
        net_scalar = side_sign * one_side_len

    net_vector_yx = net_scalar * u_yx
    net_length = float(abs(net_scalar))
    return float(anisotropy), net_vector_yx, net_length


def region_extension_analysis(region, image_target):
    # skeletonize
    skeleton = skeletonize(region.intensity_image)
    # calculate distance map
    distance_region = ndimage.distance_transform_edt(region.intensity_image)
    miny, minx, maxy, maxx = region.bbox
    # calculate center
    maxradius = np.max(distance_region, axis=None)
    center = np.unravel_index(np.argmax(distance_region, axis=None), distance_region.shape)
    distance_center = np.linalg.norm(distance_region[center])
    distances_center = np.indices(region.image.shape) - np.array(center)[:, None, None]
    distances_center = np.apply_along_axis(np.linalg.norm, 0, distances_center)
    # label inside/outside cell
    condition_outside = (skeleton > 0) & (distances_center - distance_center >= 0)

    anisotropy, orientation_vector, length = orientation_with_anisotropy_from_root(condition_outside, center)


    # pixel_locations_relevant_to_direction = np.column_stack(np.where(condition_outside))
    # pixel_locations_relevant_to_direction = pixel_locations_relevant_to_direction - center
    center_translated = [center[0] + miny, center[1] + minx]
    target_vector = [0, 0]
    if image_target is not None:
        neighbor_y = [center_translated[0] + 1, center_translated[1]]
        neighbor_x = [center_translated[0], center_translated[1] + 1]
        if neighbor_x[1] < image_target.shape[1] and neighbor_y[0] < image_target.shape[0]:
            value_at_center = image_target[center_translated[0], center_translated[1]]
            value_at_neighbor_x = image_target[neighbor_x[0], neighbor_x[1]]
            value_at_neighbor_y = image_target[neighbor_y[0], neighbor_y[1]]
            target_vector = [value_at_center - value_at_neighbor_y, value_at_center - value_at_neighbor_x]
    length_cell_vector = 0
    absolute_angle = 0
    rolling_ball_angle = 0
    relative_angle = 0
    if length > maxradius:
        # mean_outside = np.mean(pixel_locations_relevant_to_direction, axis=0)
        length_cell_vector = length
        absolute_angle = angle_between((-1, 0), orientation_vector)
        rolling_ball_angle = angle_between((-1, 0), target_vector)
        relative_angle = angle_between(orientation_vector, target_vector)
    return skeleton, center_translated, maxradius, length_cell_vector, anisotropy, absolute_angle, relative_angle, rolling_ball_angle


def build_average_directions_table(cell_table, shape, crop_extend, tile_size, image_target_mask):
    tiles_num_y = int(shape[0] / tile_size) + 1
    tiles_num_x = int(shape[1] / tile_size) + 1

    ix = ((cell_table["X center biggest circle"] - crop_extend[2]) // tile_size).astype(int)
    iy = ((cell_table["Y center biggest circle"] - crop_extend[0]) // tile_size).astype(int)

    rows, counts_all, avg_lengths_all = [], [], []
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
                # alpha filled later
            }
            rows.append(row)
            counts_all.append(0.0)
            avg_lengths_all.append(0.0)
            continue

        if is_relative:
            # length-weighted mean relative angle (in radians)
            rel_rad = np.radians(cell_table.loc[idx, "Relative angle"])
            L = cell_table.loc[idx, "Length cell vector"]
            wsum = np.nansum(L)
            rel_tile = (np.nansum(rel_rad * L) / wsum) if wsum > 0 else 0.0  # [0, π]
            u = rel_tile
            v = float(np.nanmean(L))
            avg_length = v
            color_scalar_deg = float(np.degrees(rel_tile))  # 0..180
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
            # alpha filled later
        }
        rows.append(row)
        counts_all.append(float(count))
        avg_lengths_all.append(float(avg_length))

    counts_all = np.asarray(counts_all, dtype=float)
    avg_lengths_all = np.asarray(avg_lengths_all, dtype=float)

    max_count = float(np.nanpercentile(counts_all, 90))
    max_length = float(np.nanpercentile(avg_lengths_all, 90))

    for r in rows:
        c = r["count"]
        L = r["avg_length"]
        alpha = min(1.0, c / max_count) * min(1.0, L / max_length) * 0.9 if (max_count > 0 and max_length > 0) else 0.0
        r["alpha"] = alpha
        r["max_count"] = float(max_count)
        r["max_length"] = float(max_length)

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


def compute_and_write_avg_dir_tables(cell_table: DataFrame, raw_image, roi, image_target_mask, tiles, output):

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
            image_target_mask=image_target_mask
        )

        dfs.append(avg_df)

        if output:
            avg_csv = output.joinpath(f'average_directions_tile{tile_size}.csv')
            avg_df.to_csv(avg_csv, index=False)
            print(f"Saved average directions table: {avg_csv}")


    if output:
        print(f"Results written to {output}")

    return dfs
