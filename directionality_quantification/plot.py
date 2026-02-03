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
from skimage.draw import rectangle
from skimage.transform import rescale
from tqdm import tqdm

from directionality_quantification.plot_utils import _draw_scaled_arrow_aa, apply_hatch_to_background

REL_NORM  = Normalize(0, 180)
ABS_NORM  = Normalize(0, 360)
REL_CMAP = get_cmap("coolwarm_r")
ABS_CMAP = get_cmap("hsv")


def save_fullres_with_rectangles(output, raw_image_fullres, avg_tables_fullres, roi_fullres, image_target_mask_fullres, tiles):
    """Save full-resolution image with tile rectangles drawn on top."""
    if raw_image_fullres.size == 0 or raw_image_fullres.shape[0] <= 0 or raw_image_fullres.shape[1] <= 0:
        print("Warning: Invalid input image dimensions, skipping full-resolution rectangle save")
        return
    
    v_min = raw_image_fullres.min()
    v_max = raw_image_fullres.max()
    if (v_max - v_min) == 0:
        normalized_image = np.zeros_like(raw_image_fullres, dtype=float)
    else:
        normalized_image = (raw_image_fullres.astype(np.float32) - v_min) / (v_max - v_min)
    
    if image_target_mask_fullres is not None:
        bg_image_to_display = apply_hatch_to_background(normalized_image, image_target_mask_fullres)
        del normalized_image
    else:
        bg_image_to_display = normalized_image
    
    if len(bg_image_to_display.shape) == 2:
        rgb_image = np.stack([bg_image_to_display] * 3, axis=-1)
    elif len(bg_image_to_display.shape) == 3 and bg_image_to_display.shape[2] == 1:
        rgb_image = np.repeat(bg_image_to_display, 3, axis=2)
    elif len(bg_image_to_display.shape) == 3 and bg_image_to_display.shape[2] == 3:
        rgb_image = bg_image_to_display
    else:
        rgb_image = bg_image_to_display[:, :, :3]
    
    if bg_image_to_display is not rgb_image:
        del bg_image_to_display
    tile_list = tiles.split(',')
    for idx, tile_str in enumerate(tile_list):
        tile_size = int(tile_str)
        avg_df = avg_tables_fullres[idx]
        
        tiles_with_cells = avg_df[avg_df["count"] > 0]
        
        if len(tiles_with_cells) > 0:
            image_with_rects = rgb_image.copy()
        else:
            image_with_rects = rgb_image
        for _, row in tqdm(tiles_with_cells.iterrows(), total=len(tiles_with_cells), desc=f"Drawing rectangles (tile {tile_size})"):
            x = int(row["x"])
            y = int(row["y"])
            tile_size_actual = int(row["tile_size"])
            
            rgba_color = to_rgba(row["color_hex"], row["alpha"])
            rgb_color = np.array(rgba_color[:3])
            alpha = rgba_color[3]
            
            y_min = max(0, y)
            y_max = min(image_with_rects.shape[0], y + tile_size_actual)
            x_min = max(0, x)
            x_max = min(image_with_rects.shape[1], x + tile_size_actual)
            
            if y_max > y_min and x_max > x_min:
                image_with_rects[y_min:y_max, x_min:x_max] = (
                    image_with_rects[y_min:y_max, x_min:x_max] * (1 - alpha) + 
                    rgb_color * alpha
                )
        
        output_path = output / f"fullres_with_rectangles_tile{tile_size}.png"
        image_with_rects = np.clip(image_with_rects, 0.0, 1.0)
        image_uint8 = (image_with_rects * 255).astype(np.uint8)
        if image_uint8.shape[0] <= 0 or image_uint8.shape[1] <= 0:
            print(f"Warning: Invalid image dimensions {image_uint8.shape}, skipping save")
            continue
        print(f"Saving PNG with shape {image_uint8.shape}, dtype {image_uint8.dtype}, size {image_uint8.nbytes / 1024 / 1024:.2f} MB")
        try:
            plt.imsave(output_path, image_uint8)
            print(f"Saved full-resolution image with rectangles to {output_path}")
        except Exception as e:
            print(f"Error saving PNG: {e}")
            print(f"Image shape: {image_uint8.shape}, dtype: {image_uint8.dtype}")
            raise


def save_fullres_with_arrows(output, raw_image_fullres, cell_table_fullres, roi_fullres, image_target_mask_fullres):
    """Save full-resolution image with arrows drawn on top."""
    # Validate input image
    if raw_image_fullres.size == 0 or raw_image_fullres.shape[0] <= 0 or raw_image_fullres.shape[1] <= 0:
        print("Warning: Invalid input image dimensions, skipping full-resolution arrow save")
        return
    
    v_min = raw_image_fullres.min()
    v_max = raw_image_fullres.max()
    if (v_max - v_min) == 0:
        normalized_image = np.zeros_like(raw_image_fullres, dtype=np.float32)
    else:
        normalized_image = (raw_image_fullres.astype(np.float32) - v_min) / (v_max - v_min)
    
    if image_target_mask_fullres is not None:
        bg_image_to_display = apply_hatch_to_background(normalized_image, image_target_mask_fullres)
        del normalized_image
    else:
        bg_image_to_display = normalized_image
    if len(bg_image_to_display.shape) == 2:
        rgb_image = np.stack([bg_image_to_display] * 3, axis=-1)
    elif bg_image_to_display.shape[2] == 1:
        rgb_image = np.repeat(bg_image_to_display, 3, axis=2)
    elif bg_image_to_display.shape[2] == 3:
        rgb_image = bg_image_to_display
    else:
        rgb_image = bg_image_to_display[:, :, :3]
    
    if bg_image_to_display is not rgb_image:
        del bg_image_to_display
    
    overlay = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 4), dtype=np.float32)
    
    reference_dimension = 1000.0
    image_dimension = max(overlay.shape)
    scale_factor = np.sqrt(image_dimension / reference_dimension)
    
    is_relative = image_target_mask_fullres is not None
    angles = cell_table_fullres["Relative angle"] if is_relative else cell_table_fullres["Absolute angle"]
    colors = (REL_CMAP(REL_NORM(angles.to_numpy())) if is_relative
              else ABS_CMAP(ABS_NORM(angles.to_numpy())))
    
    y_min, y_max, x_min, x_max = roi_fullres
    
    for row, color in tqdm(zip(cell_table_fullres.itertuples(index=False), colors),
                           total=len(cell_table_fullres), desc="Drawing arrows"):
        r0 = float(row.YC) - y_min
        c0 = float(row.XC) - x_min
        r1 = r0 - float(row.DY)
        c1 = c0 + float(row.DX)
        
        if not (0 <= r0 < overlay.shape[0] and 0 <= c0 < overlay.shape[1]):
            continue
        
        _draw_scaled_arrow_aa(overlay, r0, c0, r1, c1, color, scale_factor=scale_factor, opacity=0.8)
    
    alpha = overlay[:, :, 3:4]
    rgb_overlay = overlay[:, :, :3]
    rgb_image = rgb_image * (1 - alpha) + rgb_overlay * alpha
    
    output_path = output / "fullres_with_arrows.png"
    rgb_image = np.clip(rgb_image, 0.0, 1.0)
    image_uint8 = (rgb_image * 255).astype(np.uint8)
    if image_uint8.shape[0] <= 0 or image_uint8.shape[1] <= 0:
        print(f"Warning: Invalid image dimensions {image_uint8.shape}, skipping save")
        return
    print(f"Saving PNG with shape {image_uint8.shape}, dtype {image_uint8.dtype}, size {image_uint8.nbytes / 1024 / 1024:.2f} MB")
    try:
        plt.imsave(output_path, image_uint8)
        print(f"Saved full-resolution image with arrows to {output_path}")
    except Exception as e:
        print(f"Error saving PNG: {e}")
        print(f"Image shape: {image_uint8.shape}, dtype: {image_uint8.dtype}")
        raise


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

    reference_dimension = 1000.0
    image_dimension = max(overlay.shape)
    scale_factor = np.sqrt(image_dimension / reference_dimension)

    is_relative = image_target_mask is not None
    angles = cell_table["Relative angle"] if is_relative else cell_table["Absolute angle"]
    colors = (REL_CMAP(REL_NORM(angles.to_numpy())) if is_relative
              else ABS_CMAP(ABS_NORM(angles.to_numpy())))

    y_min, y_max, x_min, x_max = roi
    y_min_disp, y_max_disp, x_min_disp, x_max_disp = roi_display

    for row, color in tqdm(zip(cell_table.itertuples(index=False), colors),
                           total=len(cell_table), desc="Drawing vectors"):
        r0 = float(row.YC) - y_min
        c0 = float(row.XC) - x_min
        r1 = r0 - float(row.DY)
        c1 = c0 + float(row.DX)

        if not (0 <= r0 < overlay.shape[0] and 0 <= c0 < overlay.shape[1]):
            continue

        _draw_scaled_arrow_aa(overlay, r0, c0, r1, c1, color, scale_factor=scale_factor, opacity=0.8)

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
            adjust_to_region(ax, roi_display[1] + roi_display[0],
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
         image_target_mask, pixel_in_micron, tiles, output, output_res, avg_tables, generate_fullres: bool = False):

    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

    roi_display = copy.copy(roi)
    additional_rois_display = copy.deepcopy(additional_rois)
    pixel_in_micron_display = pixel_in_micron
    raw_image_fullres = None
    cell_table_fullres = None
    avg_tables_fullres = None
    roi_fullres = None
    image_target_mask_fullres = None
    
    if generate_fullres:
        raw_image_fullres = raw_image.copy()
        cell_table_fullres = cell_table.copy()
        avg_tables_fullres = copy.deepcopy(avg_tables)
        roi_fullres = copy.copy(roi)
        image_target_mask_fullres = image_target_mask.copy() if image_target_mask is not None else None

    dpi = plt.rcParams['figure.dpi']
    target_pixels_x = int(output_res[0] * dpi)
    target_pixels_y = int(output_res[1] * dpi)
    target_max_dim = max(target_pixels_x, target_pixels_y)

    data_max_dim = max(raw_image.shape)

    scale_factor = target_max_dim / data_max_dim

    if scale_factor < 1.0:
        print(f"Data res {raw_image.shape} vs Target res ~({target_pixels_x}, {target_pixels_y}).")
        print(f"Applying downsampling factor: {scale_factor:.4f}")

        raw_image = rescale(raw_image, scale_factor, anti_aliasing=True, preserve_range=True)

        if image_target_mask is not None:
            image_target_mask = rescale(image_target_mask, scale_factor, anti_aliasing=False, preserve_range=True)
            image_target_mask = (image_target_mask > 0.5).astype(float)

        roi = [int(v * scale_factor) for v in roi]
        additional_rois = [[int(v * scale_factor) for v in r] for r in additional_rois]
        if pixel_in_micron:
            pixel_in_micron = pixel_in_micron / scale_factor

        cell_table = cell_table.copy()
        cols_to_scale = ['YC', 'XC', 'DY', 'DX']
        for col in cols_to_scale:
            if col in cell_table.columns:
                cell_table[col] = cell_table[col] * scale_factor
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
            roi=roi,
            image_target_mask=image_target_mask,
            pixel_in_micron=pixel_in_micron,
            roi_display=roi_display,
            pixel_in_micron_display=pixel_in_micron_display
        )

        if output:
            rois = [roi_display] + list(additional_rois_display)
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

    # Save full-resolution versions with overlays (only if requested)
    if output and generate_fullres:
        print("Saving full-resolution versions with overlays...")
        save_fullres_with_rectangles(output, raw_image_fullres, avg_tables_fullres, roi_fullres, image_target_mask_fullres, tiles)
        save_fullres_with_arrows(output, raw_image_fullres, cell_table_fullres, roi_fullres, image_target_mask_fullres)
        print(f"Results written to {output}")
    elif output:
        print(f"Results written to {output} (full-resolution output skipped, use --fullres to enable)")


def plot_average_directions(output_res, avg_df, bg_image,
                            roi, image_target_mask, pixel_in_micron,
                            roi_display, pixel_in_micron_display):

    tile_size = int(avg_df["tile_size"].iloc[0])
    print(f"Plotting average directions from table (tile size {tile_size})...")

    fig, ax = plt.subplots(figsize=output_res, num=f"Average directions tile size {tile_size}")

    divider = make_axes_locatable(ax)

    y_min_disp, y_max_disp, x_min_disp, x_max_disp = roi_display

    if image_target_mask is not None:
        ax.imshow(bg_image, extent=[x_min_disp, x_max_disp, y_min_disp, y_max_disp], origin='upper', zorder=1)
    else:
        ax.imshow(bg_image, extent=[x_min_disp, x_max_disp, y_min_disp, y_max_disp], origin='upper', cmap="grey",
                  zorder=1)

    ax.set_aspect('equal', adjustable='box')  # make tiles square in data units
    ax.margins(0)
    
    ax.set_xlim(x_min_disp, x_max_disp)
    ax.set_ylim(y_min_disp, y_max_disp)

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
        facecolor='black',
        labelcolor='white',
        framealpha=1.0  # Makes the black background opaque
    )


def plot_grid_from_table(avg_df, image_target_mask,
                         roi, roi_display,
                         divider):
    ax = plt.gca()

    tile_size = int(avg_df["tile_size"].iloc[0])
    
    y_min_disp, y_max_disp, x_min_disp, x_max_disp = roi_display
    
    tiles_with_cells = avg_df[avg_df["count"] > 0]
    
    for _, row in tiles_with_cells.iterrows():
        x = float(row["x"])
        y = float(row["y"])
        
        if x + tile_size < x_min_disp or x > x_max_disp:
            continue
        if y + tile_size < y_min_disp or y > y_max_disp:
            continue
        
        rgba_color = to_rgba(row["color_hex"], row["alpha"])
        
        y_inverted = y_max_disp + y_min_disp - y - tile_size
        rect = patches.Rectangle(
            (x, y_inverted),
            tile_size,
            tile_size,
            facecolor=rgba_color,
            edgecolor='none',
            zorder=2
        )
        ax.add_patch(rect)
    
    alpha_desc_low = avg_df["alpha_description_low"].iloc[0] if "alpha_description_low" in avg_df.columns else "Low alpha (transparent)"
    alpha_desc_low = avg_df["alpha_description_low"].iloc[0] if "alpha_description_low" in avg_df.columns else "Low alpha (transparent)"
    alpha_desc_high = avg_df["alpha_description_high"].iloc[0] if "alpha_description_high" in avg_df.columns else "High alpha (opaque)"
    _add_opacity_legend(ax, divider, alpha_desc_low, alpha_desc_high)

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

def _add_opacity_legend(ax, divider, low_label="Low alpha (transparent)", high_label="High alpha (opaque)"):
    sm = plt.cm.ScalarMappable(cmap=get_cmap("binary"))
    sm.set_clim(0, 1)

    cax_opacity = divider.append_axes("bottom", size="5%", pad=0.6)
    cbar = plt.colorbar(sm, cax=cax_opacity, orientation='horizontal')

    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([low_label, high_label])
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