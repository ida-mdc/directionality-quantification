import math
from pathlib import Path

import math
from pathlib import Path

import numpy as np
from matplotlib import cm
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.pyplot import get_cmap
from matplotlib_scalebar.scalebar import ScaleBar
from pandas import DataFrame
from tqdm import tqdm

REL_NORM  = Normalize(0, 180)
ABS_NORM  = Normalize(0, 360)
REL_CMAP = get_cmap("coolwarm_r")
ABS_CMAP = get_cmap("hsv")

import gc
from skimage.draw import line_aa, polygon  # Add this import at the top of your script

def generate_target_contour(image_target_mask):
    if image_target_mask is None:
        return
    ax = plt.gca()
    ax.contour(image_target_mask, levels=[0.5], origin='upper',
               colors='red', linewidths=1.0)
    plt.contour(image_target_mask, 1, origin='upper', colors='red')
    cs = plt.contourf(image_target_mask, 1, hatches=['', 'O'], origin='upper', colors='none')
    cs.set_edgecolor((1, 0, 0.2, 1))


def plot_all_directions(output, output_res, cell_table, bg_image_display, roi, additional_rois, additional_roi_colors,
                        image_target_mask, pixel_in_micron):
    """
    Draws all vectors with dynamically scaled thickness and size directly onto an RGBA image overlay.
    """
    print("Plotting all directions...")

    rois = [roi]
    rois.extend(additional_rois)
    region_colors = ['black']
    region_colors.extend(additional_roi_colors)

    fig, ax = plt.subplots(figsize=output_res, num="All directions")

    def _blend_rgba(overlay, rr, cc, rgb, alpha):
        # Alpha-max blend: accumulate visibility
        valid = (rr >= 0) & (rr < overlay.shape[0]) & (cc >= 0) & (cc < overlay.shape[1])
        rr, cc, alpha = rr[valid], cc[valid], alpha[valid]

        overlay[rr, cc, 0] = rgb[0]
        overlay[rr, cc, 1] = rgb[1]
        overlay[rr, cc, 2] = rgb[2]
        overlay[rr, cc, 3] = np.maximum(overlay[rr, cc, 3], alpha)

    def _draw_scaled_arrow_aa(overlay, r0, c0, r1, c1, color, scale_factor=1.0):
        """Draws an anti-aliased arrow with scalable line width and head size."""
        # --- Define base sizes (what looks good on a ~1000px image) ---
        base_head_len = 10.0
        base_head_width = 8.0
        base_line_width = 2.0

        # --- Scale dimensions ---
        head_len = base_head_len * scale_factor
        head_width = base_head_width * scale_factor
        line_width = base_line_width * scale_factor

        # --- Vector calculations ---
        dy, dx = (r1 - r0), (c1 - c0)
        norm = np.hypot(dy, dx) + 1e-9
        uy, ux = dy / norm, dx / norm  # Unit direction vector
        py, px = -ux, uy  # Perpendicular vector

        # --- Draw Shaft (as a thin rectangle) ---
        p0 = np.array([r0, c0])
        p1 = np.array([r1, c1])

        # Define the 4 corners of the rectangle for the shaft
        shaft_half_width = line_width / 2.0
        r_coords = np.array([
            p0[0] - shaft_half_width * py,  # Start-left
            p0[0] + shaft_half_width * py,  # Start-right
            p1[0] + shaft_half_width * py,  # End-right
            p1[0] - shaft_half_width * py,  # End-left
        ])
        c_coords = np.array([
            p0[1] - shaft_half_width * px,
            p0[1] + shaft_half_width * px,
            p1[1] + shaft_half_width * px,
            p1[1] - shaft_half_width * px,
        ])

        rr_shaft, cc_shaft = polygon(r_coords, c_coords, shape=overlay.shape[:2])
        _blend_rgba(overlay, rr_shaft, cc_shaft, color[:3], np.full_like(rr_shaft, 1.0, dtype=float))

        # --- Draw Arrowhead (as a triangle) ---
        tip = np.array([r1, c1], float)
        base = tip - head_len * np.array([uy, ux])
        left = base + (head_width / 2.0) * np.array([py, px])
        right = base - (head_width / 2.0) * np.array([py, px])

        pr_head = np.array([tip[0], left[0], right[0]])
        pc_head = np.array([tip[1], left[1], right[1]])
        rr_head, cc_head = polygon(pr_head, pc_head, shape=overlay.shape[:2])
        _blend_rgba(overlay, rr_head, cc_head, color[:3], np.full_like(rr_head, 1.0, dtype=float))

    ax.imshow(bg_image_display, extent=[roi[2], roi[3], roi[0], roi[1]], origin='upper', cmap='gray')

    overlay = np.zeros((bg_image_display.shape[0], bg_image_display.shape[1], 4), dtype=np.float32)

    # We base the scale on a reference dimension, e.g., 1000 pixels.
    # An arrow on a 2000px image will be twice as big as on a 1000px image.
    reference_dimension = 1000.0
    image_dimension = max(overlay.shape)
    scale_factor = image_dimension / reference_dimension

    is_relative = image_target_mask is not None
    angles = cell_table["Relative angle"] if is_relative else cell_table["Absolute angle"]
    colors = (REL_CMAP(REL_NORM(angles.to_numpy())) if is_relative
              else ABS_CMAP(ABS_NORM(angles.to_numpy())))

    y_min, y_max, x_min, x_max = roi

    for row, color in tqdm(zip(cell_table.itertuples(index=False), colors),
                           total=len(cell_table), desc="Drawing vectors"):
        r0 = float(row.YM) - y_min
        c0 = float(row.XM) - x_min
        r1 = r0 - float(row.DY)
        c1 = c0 + float(row.DX)

        if not (0 <= r0 < overlay.shape[0] and 0 <= c0 < overlay.shape[1]):
            continue

        _draw_scaled_arrow_aa(overlay, r0, c0, r1, c1, color, scale_factor=scale_factor)

    ax.imshow(overlay, extent=[x_min, x_max, y_min, y_max], origin='upper')

    if pixel_in_micron:
        scalebar = ScaleBar(pixel_in_micron, 'um', location='upper right', color='white', box_color='black')
        ax.add_artist(scalebar)

    if image_target_mask is not None:
        generate_target_contour(image_target_mask)  # This still needs transpose for contouring
        sm = plt.cm.ScalarMappable(cmap=REL_CMAP, norm=REL_NORM)
        cbar = plt.colorbar(sm, ax=ax, location='bottom', pad=0.05, aspect=50)
        cbar.set_ticks([0, 180])
        cbar.set_ticklabels(['Towards target (0°)', 'Away from target (180°)'])
        cbar.set_label("Angle (deg)")
        cbar.ax.xaxis.get_majorticklabels()[0].set_horizontalalignment('left')
        cbar.ax.xaxis.get_majorticklabels()[-1].set_horizontalalignment('right')
    else:
        plot_compass_legend()

    plt.margins(0, 0)
    plt.tight_layout(pad=1)


    if output:
        for i, region in enumerate(rois):
            adjust_to_region(roi[0] + roi[1], [region[2], region[3], region[0], region[1]], region_colors[i], scalebar if pixel_in_micron else None)
            plt.savefig(output / f"directions_{region[0]}-{region[1]}-{region[2]}-{region[3]}.png")
        plt.close()

    plt.close()
    gc.collect()  # Important: Clean up memory
    print("Done plotting all directions.")

def plot(cell_table: DataFrame, raw_image, label_image, roi, additional_rois,
         image_target_mask, pixel_in_micron, tiles, output, output_res, avg_tables):

    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

    roi_colors = []
    if len(additional_rois) > 0:
        roi_colors = plot_rois(output, output_res, label_image, roi, additional_rois)

    plot_all_directions(output, output_res, cell_table, label_image, roi,
                        additional_rois, roi_colors, image_target_mask, pixel_in_micron)

    for i, tile in enumerate(tiles.split(',')):
        tile_size = int(tile)

        avg_df = avg_tables[i]

        scalebar = plot_average_directions(
            output_res=output_res, avg_df=avg_df,
            bg_image=raw_image, roi=roi, image_target_mask=image_target_mask,
            pixel_in_micron=pixel_in_micron
        )

        if output:
            rois = [roi] + list(additional_rois)
            colors = ["black"] + list(roi_colors)
            for i, region in enumerate(rois):
                adjust_to_region(roi[1] + roi[0], [region[2], region[3], region[0], region[1]], colors[i], scalebar if pixel_in_micron else None)
                plt.savefig(output.joinpath(
                    f'directions_{region[0]}-{region[1]}-{region[2]}-{region[3]}_tile{tile_size}.png'))
            plt.close()
        else:
            plt.show()

    if output:
        print(f"Results written to {output}")

def plot_average_directions(output_res, avg_df, bg_image, roi, image_target_mask, pixel_in_micron):
    tile_size = int(avg_df["tile_size"].iloc[0])
    print(f"Plotting average directions from table (tile size {tile_size})...")

    fig, ax = plt.subplots(figsize=output_res, num=f"Average directions tile size {tile_size}")

    # Unpack ROI consistently for display:
    # roi = [y_min, y_max, x_min, x_max] in your codebase
    y_min, y_max, x_min, x_max = roi

    # Correct extent order: [x_min, x_max, y_min, y_max]
    ax.imshow(bg_image, extent=[x_min, x_max, y_min, y_max], origin='upper', cmap='gray')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')  # make tiles square in data units
    ax.margins(0)

    for s in ax.spines.values():
        s.set_visible(False)

    scalebar = None
    if pixel_in_micron:
        scalebar = ScaleBar(pixel_in_micron, 'um', location='upper right', color='white', box_color='black')
        ax.add_artist(scalebar)

    plot_grid_from_table(avg_df, image_target_mask, roi)

    if image_target_mask is not None:
        generate_target_contour(image_target_mask)

    plt.margins(0, 0)
    plt.tight_layout(pad=1)
    return scalebar

def plot_grid_from_table(avg_df, image_target_mask, roi):
    ax = plt.gca()

    tx = avg_df["tile_x"].astype(int).to_numpy()
    ty = avg_df["tile_y"].astype(int).to_numpy()
    nx = int(tx.max())
    ny = int(ty.max())

    rgba = np.zeros((ny, nx, 4), dtype=np.float32)
    cols = np.array([to_rgba(c, a) for c, a in zip(avg_df["color_hex"], avg_df["alpha"])], dtype=np.float32)

    keep = (tx >= 0) & (tx < nx) & (ty >= 0) & (ty < ny)
    rgba[ty[keep], tx[keep], :] = cols[keep]

    # no flip needed: our tile_y is 0 at top (image rows), origin='upper' draws row 0 at top
    tile_size = int(avg_df["tile_size"].iloc[0])

    # ROI order in this code path: [y_min, y_max, x_min, x_max]
    y_min, y_max, x_min, x_max = roi

    # grid-aligned extent (each cell is tile_size wide/high in data units)
    grid_xmin = x_min
    grid_xmax = x_min + nx * tile_size
    grid_ymin = y_min
    grid_ymax = y_min + ny * tile_size

    ax.imshow(
        rgba,
        extent=[grid_xmin, grid_xmax, grid_ymin, grid_ymax],
        origin='upper',
        interpolation='nearest',
        resample=False,
    )

    # make tiles square and clip to ROI limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    _add_opacity_legend(ax)

    if image_target_mask is not None:
        sm = plt.cm.ScalarMappable(cmap=REL_CMAP); sm.set_clim(0, 180)
        cbar = plt.colorbar(sm, ax=ax, location='bottom', pad=0.04, aspect=50, fraction=0.03, use_gridspec=True)
        cbar.set_ticks([0, 180])
        cbar.set_ticklabels(['Towards target (0°)', 'Away from target (180°)'])
        cbar.set_label("Angle (deg)", labelpad=0)
        cbar.ax.tick_params(pad=0)
        cbar.ax.xaxis.get_majorticklabels()[0].set_horizontalalignment('left')
        cbar.ax.xaxis.get_majorticklabels()[-1].set_horizontalalignment('right')
    else:
        plot_compass_legend()

def plot_compass_legend():
    ph = np.linspace(0,2*math.pi, 13)
    scale_start, offset = 30.0, 40.0
    x_legend = scale_start * np.cos(ph) + offset
    y_legend = scale_start * np.sin(ph) + offset
    u_legend = np.cos(ph) * scale_start * 0.5 + offset
    v_legend = np.sin(ph) * scale_start * 0.5 + offset
    colors_legend = (np.degrees(np.arctan2(np.cos(ph), np.sin(ph))) + 360.0) % 360.0
    for i in range(len(ph)):
        pos1 = [x_legend[i], y_legend[i]]
        pos2 = [u_legend[i], v_legend[i]]
        plt.annotate('', pos1, xytext=pos2, xycoords='axes pixels', arrowprops={
            'width': 3., 'headlength': 4.4, 'headwidth': 7., 'edgecolor': 'black',
            'facecolor': ABS_CMAP(ABS_NORM(colors_legend[i]))
        })

def _add_opacity_legend(ax):
    sm = plt.cm.ScalarMappable(cmap=get_cmap("binary"))
    sm.set_clim(0, 1)
    cbar = plt.colorbar(
        sm, ax=ax, location='bottom',
        pad=0.04,
        aspect=50,
        fraction=0.03,
        use_gridspec=True
    )

    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Less cells, shorter extensions (transparent)', 'More cells, longer extensions (opaque)'])
    cbar.set_label("Opacity", labelpad=0)
    cbar.ax.tick_params(pad=0)
    cbar.ax.xaxis.get_majorticklabels()[0].set_horizontalalignment('left')
    cbar.ax.xaxis.get_majorticklabels()[-1].set_horizontalalignment('right')

def adjust_to_region(data_height, region, region_color, scalebar):
    plt.setp(plt.gca().spines.values(), color=region_color)
    plt.setp([plt.gca().get_xticklines(), plt.gca().get_yticklines()], color=region_color)
    [x.set_linewidth(2) for x in plt.gca().spines.values()]
    plt.xlim(region[0], region[1])
    plt.ylim(data_height - region[3], data_height - region[2])
    if scalebar:
        scalebar.remove()
        plt.gca().add_artist(scalebar)

def plot_rois(output, output_res, bg_image, roi, additional_rois):
    print("Plotting ROIs...")
    plt.figure("ROIs", output_res)
    plt.imshow(bg_image, extent=roi, origin='upper', cmap='gray', vmin=0, vmax=1)
    indices = [i for i, _ in enumerate(additional_rois)]
    norm = Normalize()
    norm.autoscale(indices)
    colormap = cm.rainbow
    colors = colormap(norm(indices))
    for i, region in enumerate(additional_rois):
        rect = patches.Rectangle((region[0], bg_image.shape[0] - region[3]), region[1] - region[0],
                                 region[3] - region[2],
                                 linewidth=1, edgecolor=colors[i], facecolor='none')
        plt.gca().add_patch(rect)
    plt.margins(0, 0)
    plt.tight_layout(pad=1)
    plt.savefig(output.joinpath('ROIs.png'))
    plt.close()
    return colors
