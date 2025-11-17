import copy
import gc
import math
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import cm
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.pyplot import get_cmap
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import DataFrame
from skimage.transform import rescale
from tqdm import tqdm

from directionality_quantification.plot_utils import _draw_scaled_arrow_aa, apply_hatch_to_background

REL_NORM  = Normalize(0, 180)
ABS_NORM  = Normalize(0, 360)
REL_CMAP = get_cmap("coolwarm_r")
ABS_CMAP = get_cmap("hsv")


def generate_target_contour(ax, image_target_mask, roi): # Added 'ax'
    if image_target_mask is None:
        return
    # y_min, y_max, x_min, x_max = roi
    # plot_extent = [x_min, x_max, y_min, y_max]
    # ax.contour(image_target_mask, levels=[0.5], origin='upper',
    #            colors='red', linewidths=1.0, extent=plot_extent)
    # cs = ax.contourf(image_target_mask, 1, hatches=['', 'O'], origin='upper', colors='none', extent=plot_extent) # Changed to ax.
    # cs.set_edgecolor((1, 0, 0.2, 1))

def plot_all_directions(output, output_res, cell_table, bg_image_display,
                        roi, additional_rois, additional_roi_colors,
                        image_target_mask, pixel_in_micron,
                        roi_display, additional_rois_display,
                        pixel_in_micron_display):
    """
    Draws all vectors with dynamically scaled thickness and size directly onto an RGBA image overlay.
    """
    print("Plotting all directions...")

    rois = [roi]
    rois.extend(additional_rois)
    region_colors = ['black']
    region_colors.extend(additional_roi_colors)

    fig, ax = plt.subplots(figsize=output_res, num="All directions")

    divider = make_axes_locatable(ax)

    y_min_disp, y_max_disp, x_min_disp, x_max_disp = roi_display  # BIG roi, for display
    bg_image_to_display = bg_image_display
    if image_target_mask is not None:
        ax.imshow(bg_image_to_display, extent=[x_min_disp, x_max_disp, y_min_disp, y_max_disp], origin='upper',
                  zorder=1)
    else:
        ax.imshow(bg_image_to_display, extent=[x_min_disp, x_max_disp, y_min_disp, y_max_disp], origin='upper',
                  cmap="grey", zorder=1)

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
        r0 = float(row.YC) - y_min
        c0 = float(row.XC) - x_min
        r1 = r0 - float(row.DY)
        c1 = c0 + float(row.DX)

        if not (0 <= r0 < overlay.shape[0] and 0 <= c0 < overlay.shape[1]):
            continue

        _draw_scaled_arrow_aa(overlay, r0, c0, r1, c1, color, scale_factor=scale_factor)

    ax.imshow(overlay, extent=[x_min_disp, x_max_disp, y_min_disp, y_max_disp], origin='upper', zorder=2)

    if pixel_in_micron_display:
        scalebar = ScaleBar(pixel_in_micron_display, 'um', location='upper right', color='white', box_color='black')
        ax.add_artist(scalebar)

    if image_target_mask is not None:
        plot_target_legend(ax)
    else:
        plot_compass_legend(ax)

    plt.margins(0, 0)

    if output:
        rois = [roi_display] + list(additional_rois_display)  # Use display ROIs
        region_colors = ['black'] + additional_roi_colors
        for i, region in enumerate(rois):
            adjust_to_region(ax, roi_display[1] + roi_display[0],  # Use display height
                             [region[2], region[3], region[0], region[1]], region_colors[i],
                             scalebar if pixel_in_micron_display else None)
            plt.tight_layout(pad=1)
            plt.savefig(
                output / f"directions_{region[2]}-{region[3]}-{region[0]}-{region[1]}.png")  # Filename is correct
        plt.close()
    plt.close()
    gc.collect()  # Important: Clean up memory
    print("Done plotting all directions.")

def plot(cell_table: DataFrame, raw_image, roi, additional_rois,
         image_target_mask, pixel_in_micron, tiles, output, output_res, avg_tables):

    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

    roi_display = copy.copy(roi)
    additional_rois_display = copy.deepcopy(additional_rois)
    pixel_in_micron_display = pixel_in_micron

    # 1. Determine target pixel resolution from (inches, dpi)
    dpi = plt.rcParams['figure.dpi']
    target_pixels_x = int(output_res[0] * dpi)
    target_pixels_y = int(output_res[1] * dpi)
    target_max_dim = max(target_pixels_x, target_pixels_y)

    # 2. Determine data resolution
    data_max_dim = max(raw_image.shape)

    # 3. Calculate scale factor
    scale_factor = target_max_dim / data_max_dim

    if scale_factor < 1.0:
        print(f"Data res {raw_image.shape} vs Target res ~({target_pixels_x}, {target_pixels_y}).")
        print(f"Applying downsampling factor: {scale_factor:.4f}")

        # 4. Rescale IMAGES
        # We rescale the raw_image. The normalization block below
        # will then normalize this new, smaller image.
        raw_image = rescale(raw_image, scale_factor, anti_aliasing=True, preserve_range=True)

        if image_target_mask is not None:
            image_target_mask = rescale(image_target_mask, scale_factor, anti_aliasing=False, preserve_range=True)
            image_target_mask = (image_target_mask > 0.5).astype(float)  # Re-threshold blurred mask

        # 5. Rescale COORDINATES
        roi = [int(v * scale_factor) for v in roi]
        additional_rois = [[int(v * scale_factor) for v in r] for r in additional_rois]
        if pixel_in_micron:
            pixel_in_micron = pixel_in_micron / scale_factor  # The size of one (new) pixel in microns

        # 6. Rescale DATA TABLES
        cell_table = cell_table.copy()
        cols_to_scale = ['YC', 'XC', 'DY', 'DX']
        for col in cols_to_scale:
            if col in cell_table.columns:
                cell_table[col] = cell_table[col] * scale_factor

        avg_tables = copy.deepcopy(avg_tables)  # Use deep copy to be safe
        cols_to_scale_avg = ['x', 'y', 'u', 'v']
        for df in avg_tables:
            for col in cols_to_scale_avg:
                if col in df.columns:
                    df[col] = df[col] * scale_factor
    else:
        print("Target resolution is >= data. No downsampling performed.")

    print(f"Normalizing background image from range [{raw_image.min():.2f}..{raw_image.max():.2f}] to [0..1]")

    v_min = raw_image.min()
    v_max = raw_image.max()

    if (v_max - v_min) == 0:
        raw_image = np.zeros_like(raw_image, dtype=float)
    else:
        raw_image = (raw_image - v_min) / (v_max - v_min)

    if image_target_mask is not None:
        bg_image_to_display = apply_hatch_to_background(raw_image, image_target_mask)
    else:
        bg_image_to_display = raw_image


    roi_colors = []
    if len(additional_rois) > 0:
        roi_colors = plot_rois(output, output_res, bg_image_to_display, roi, additional_rois)

    plot_all_directions(output, output_res, cell_table, bg_image_to_display, roi,
                        additional_rois, roi_colors, image_target_mask, pixel_in_micron, roi_display, additional_rois_display, pixel_in_micron_display)

    for i, tile in enumerate(tiles.split(',')):
        tile_size = int(tile)

        avg_df = avg_tables[i]

        fig_avg, ax_avg, scalebar_avg = plot_average_directions(
            output_res=output_res, avg_df=avg_df,
            bg_image=bg_image_to_display,
            roi=roi,  # This is the small, data-scaled ROI
            image_target_mask=image_target_mask,
            pixel_in_micron=pixel_in_micron,
            roi_display=roi_display,
            pixel_in_micron_display=pixel_in_micron_display
        )

        if output:
            rois = [roi_display] + list(additional_rois_display)  # Use display ROIs
            colors = ["black"] + list(roi_colors)
            for i, region in enumerate(rois):
                adjust_to_region(ax_avg, roi_display[1] + roi_display[0],
                                 [region[2], region[3], region[0], region[1]], colors[i],
                                 scalebar_avg if pixel_in_micron_display else None)
                plt.tight_layout(pad=1)
                plt.savefig(output.joinpath(
                    f'directions_{region[2]}-{region[3]}-{region[0]}-{region[1]}_tile{tile_size}.png'))
            plt.close(fig_avg)
            plt.close(fig_avg)
        else:
            plt.show()

    if output:
        print(f"Results written to {output}")


def plot_average_directions(output_res, avg_df, bg_image,
                            roi, image_target_mask, pixel_in_micron,
                            roi_display, pixel_in_micron_display):

    tile_size = int(avg_df["tile_size"].iloc[0])
    print(f"Plotting average directions from table (tile size {tile_size})...")

    fig, ax = plt.subplots(figsize=output_res, num=f"Average directions tile size {tile_size}")

    # Create a divider for the axes
    divider = make_axes_locatable(ax)

    # Unpack ROI consistently for display:
    # roi = [y_min, y_max, x_min, x_max] in your codebase
    # y_min, y_max, x_min, x_max = roi
    y_min_disp, y_max_disp, x_min_disp, x_max_disp = roi_display

    if image_target_mask is not None:
        ax.imshow(bg_image, extent=[x_min_disp, x_max_disp, y_min_disp, y_max_disp], origin='upper', zorder=1)
    else:
        ax.imshow(bg_image, extent=[x_min_disp, x_max_disp, y_min_disp, y_max_disp], origin='upper', cmap="grey",
                  zorder=1)

        # --- UPDATE AXIS LIMITS ---
    ax.set_xlim(x_min_disp, x_max_disp)
    ax.set_ylim(y_min_disp, y_max_disp)

    ax.set_aspect('equal', adjustable='box')  # make tiles square in data units
    ax.margins(0)

    for s in ax.spines.values():
        s.set_visible(False)

    scalebar = None
    if pixel_in_micron_display:
        scalebar = ScaleBar(pixel_in_micron_display, 'um', location='upper right', color='white', box_color='black')
        ax.add_artist(scalebar)

    plot_grid_from_table(avg_df, image_target_mask,
                         roi,  # Pass the SMALL roi for data logic
                         roi_display,  # Pass the BIG roi for extent/limits
                         divider)

    if image_target_mask is not None:
        plot_target_legend(ax)

    plt.margins(0, 0)
    return fig, ax, scalebar


def plot_target_legend(ax):
    hatch_legend_patch = mpatches.Patch(
        facecolor='#550000',  # Match your tint color
        hatch='.....',  # Match your hatch_style ('o' for circles, '/' for stripes)
        edgecolor='red',  # This colors the hatch pattern itself
        linewidth=0,  # <-- This removes the patch border
        label='Target Area'
    )
    ax.legend(
        handles=[hatch_legend_patch],
        loc='upper left',  # Or 'upper right', etc.
        facecolor='black',  # <-- Sets legend background to black
        labelcolor='white',  # <-- Sets legend text to white
        framealpha=1.0  # Makes the black background opaque
    )


def plot_grid_from_table(avg_df, image_target_mask,
                         roi, roi_display,
                         divider):
    ax = plt.gca()

    # --- START OF FIX ---
    # Handle sparse grids (where min tile index > 0)
    tx = avg_df["tile_x"].astype(int).to_numpy()
    ty = avg_df["tile_y"].astype(int).to_numpy()
    min_tx, max_tx = tx.min(), tx.max()
    min_ty, max_ty = ty.min(), ty.max()

    nx = max_tx - min_tx  # True width of grid in tiles
    ny = max_ty - min_ty  # True height of grid in tiles

    tx_norm = tx - min_tx  # Normalized 0-based x-indices
    ty_norm = ty - min_ty  # Normalized 0-based y-indices

    rgba = np.zeros((ny, nx, 4), dtype=np.float32)
    cols = np.array([to_rgba(c, a) for c, a in zip(avg_df["color_hex"], avg_df["alpha"])], dtype=np.float32)

    keep = (tx_norm >= 0) & (tx_norm < nx) & (ty_norm >= 0) & (ty_norm < ny)
    rgba[ty_norm[keep], tx_norm[keep], :] = cols[keep]

    # Get the *unscaled* tile_size from the dataframe.
    tile_size = int(avg_df["tile_size"].iloc[0])

    # Get the *unscaled* (display) ROI limits
    # roi_display = [y_min(top), y_max(bottom), x_min(left), x_max(right)]
    y_top_disp, y_bottom_disp, x_left_disp, x_right_disp = roi_display

    # Calculate the grid's *true* extent based on its *unscaled* properties
    # This might be larger than the background image!
    # We assume the grid *origin* (min_tx, min_ty) aligns with the ROI origin (x_left_disp, y_top_disp).
    grid_x_left = x_left_disp + min_tx * tile_size
    grid_x_right = x_left_disp + (max_tx) * tile_size
    grid_y_top = y_top_disp + min_ty * tile_size
    grid_y_bottom = y_top_disp + (max_ty) * tile_size

    # Draw the overlay image using its *true* calculated extent
    # extent = [left, right, bottom, top]
    ax.imshow(
        rgba,
        extent=[grid_x_left, grid_x_right, grid_y_top, grid_y_bottom],
        origin='upper',
        interpolation='nearest',
        resample=False,
        zorder=2
    )

    ax.set_aspect('equal', adjustable='box')

    _add_opacity_legend(ax, divider)

    if image_target_mask is not None:
        sm = plt.cm.ScalarMappable(cmap=REL_CMAP)
        sm.set_clim(0, 180)

        cax_angle = divider.append_axes("bottom", size="5%", pad=0.6)
        cbar = plt.colorbar(sm, cax=cax_angle, orientation='horizontal')

        cbar.set_ticks([0, 180])
        cbar.set_ticklabels(['Towards target (0°)', 'Away from target (180°)'])
        cbar.set_label("Angle (deg)")
        cbar.ax.xaxis.get_majorticklabels()[0].set_horizontalalignment('left')
        cbar.ax.xaxis.get_majorticklabels()[-1].set_horizontalalignment('right')
        plt.sca(ax)
    else:
        plot_compass_legend(ax)

def _add_opacity_legend(ax, divider):
    sm = plt.cm.ScalarMappable(cmap=get_cmap("binary"))
    sm.set_clim(0, 1)

    cax_opacity = divider.append_axes("bottom", size="5%", pad=0.6)
    cbar = plt.colorbar(sm, cax=cax_opacity, orientation='horizontal')

    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Less cells, shorter extensions (transparent)', 'More cells, longer extensions (opaque)'])
    cbar.set_label("Opacity")
    # cbar.ax.tick_params(pad=0)
    cbar.ax.xaxis.get_majorticklabels()[0].set_horizontalalignment('left')
    cbar.ax.xaxis.get_majorticklabels()[-1].set_horizontalalignment('right')

    plt.sca(ax)

def plot_compass_legend(ax): # Added 'ax'
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
        ax.annotate('', pos1, xytext=pos2, xycoords='axes pixels', arrowprops={ # Changed to ax.
            'width': 3., 'headlength': 4.4, 'headwidth': 7., 'edgecolor': 'black',
            'facecolor': ABS_CMAP(ABS_NORM(colors_legend[i]))
        })

def adjust_to_region(ax, data_height, region, region_color, scalebar): # Added 'ax'
    plt.setp(ax.spines.values(), color=region_color) # Changed to ax.
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=region_color) # Changed to ax.
    [x.set_linewidth(2) for x in ax.spines.values()] # Changed to ax.
    ax.set_xlim(region[0], region[1]) # Changed to ax.
    ax.set_ylim(data_height - region[3], data_height - region[2]) # Changed to ax.
    if scalebar:
        scalebar.remove()
        ax.add_artist(scalebar)


def plot_rois(output, output_res, bg_image, roi, additional_rois):  # roi is roi_display
    print("Plotting ROIs...")
    plt.figure("ROIs", output_res)
    # roi is [y_min, y_max, x_min, x_max]
    # extent is [x_min, x_max, y_min, y_max]
    plt.imshow(bg_image, extent=[roi[2], roi[3], roi[0], roi[1]], origin='upper', cmap='gray', vmin=0, vmax=1)
    indices = [i for i, _ in enumerate(additional_rois)]
    norm = Normalize()
    norm.autoscale(indices)
    colormap = cm.rainbow
    colors = colormap(norm(indices))

    for i, region in enumerate(additional_rois):  # region is [y_min, y_max, x_min, x_max]
        # patches.Rectangle wants (x_min, y_min), width, height
        rect = patches.Rectangle(
            (region[2], region[0]),  # (x_min, y_min)
            region[3] - region[2],  # width (x_max - x_min)
            region[1] - region[0],  # height (y_max - y_min)
            linewidth=1, edgecolor=colors[i], facecolor='none'
        )
        plt.gca().add_patch(rect)
    plt.margins(0, 0)
    plt.tight_layout(pad=1)
    plt.savefig(output.joinpath('ROIs.png'))
    plt.close()
    return colors