import argparse
import math
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tifffile
from matplotlib import cm
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


def run():
    parser = argparse.ArgumentParser(description="Analyze cell extension orientation")

    # Define arguments
    parser.add_argument('--input_raw', type=str, required=True,
                        help="The input raw data as TIFF (2D, 1 channel).")
    parser.add_argument('--input_target', type=str, required=False,
                        help="Masked areas used for orientation calculation (optional).")
    parser.add_argument('--output', type=str, required=False,
                        help="Output folder for saving plots; if omitted, plots are displayed.")
    parser.add_argument('--output_res', type=str, default="12:9",
                        help="Resolution of output plots as WIDTH:HEIGHT, e.g., 800:600.")
    parser.add_argument('--roi', type=str, required=False,
                        help="Region of interest as MIN_X:MAX_X:MIN_Y:MAX_Y. Multiple ROIs are comma-separated.")
    parser.add_argument('--tiles', type=str, default="100,250,500",
                        help="Tile sizes for average plots, e.g., SIZE1,SIZE2,SIZE3.")
    parser.add_argument('--min_length_orientation', type=str, required=False,
                        help="Minimum length of orientation vectors (pixels).")
    parser.add_argument('--max_size', type=str, required=False,
                        help="Exclude segments with area above this size (pixels).")
    parser.add_argument('--min_size', type=str, required=False,
                        help="Exclude segments with area below this size (pixels).")
    parser.add_argument('--pixel_in_micron', type=float, required=False,
                        help="Pixel width in microns, for adding a scalebar.")
    parser.add_argument('--input_table', type=str, required=False,
                        help="Table of cells to analyze, with first column as label IDs.")
    parser.add_argument('--input_labeling', type=str, required=True,
                        help="Label map for segmentation analysis (2D, 1 channel).")

    # Parse arguments
    args = parser.parse_args()
    
    print('Reading raw image %s and segmentation %s..' % (args.input_raw, args.input_labeling))
    image_raw = tifffile.imread(args.input_raw)
    pixel_in_micron = args.pixel_in_micron
    image = tifffile.imread(args.input_labeling).astype(int)
    image_target_mask = None
    image_target = None
    if args.input_target is not None:
        image_target_mask = tifffile.imread(args.input_target).astype(bool)
        image_target = ndimage.distance_transform_edt(np.invert(image_target_mask))

    # crop input images to ROI
    roi, additional_rois = get_roi(args.roi, image)  # returns array with [min_x, max_x, min_y, max_y]
    image = image[roi[2]:roi[3], roi[0]:roi[1]]
    image_raw = image_raw[roi[2]:roi[3], roi[0]:roi[1]]
    if image_target_mask is not None:
        image_target = image_target[roi[2]:roi[3], roi[0]:roi[1]]
        image_target_mask = image_target_mask[roi[2]:roi[3], roi[0]:roi[1]]

    # analyze the segments, get resulting directions and an image labeled based on the analysis
    directions, labeled_result, cell_table_content = analyze_segments(image, roi, image_target, args.min_size, args.max_size, args.min_length_orientation, args.output)

    plot(directions, image_raw, labeled_result, roi, additional_rois, image_target_mask, pixel_in_micron, args.tiles, args.output, args.output_res)


def write_table(cell_table_content, output):
    if cell_table_content is not None:
        if output:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(data=cell_table_content).to_csv(output.joinpath("cells.csv"))


def analyze_segments(labeled, roi, image_target, min_size, max_size, min_length_orientation, output):
    # obtain labels
    print("Labeling segmentation..")
    n_components = np.max(labeled)
    if n_components == 1:
        labeled, n_components = label(labeled, return_num=True)
    print(f'%s objects detected.' % n_components)

    # calculate region properties
    segmentation = labeled > 0
    regions = regionprops(label_image=labeled, intensity_image=segmentation)

    regions = filter_regions_by_size(min_size, max_size, n_components, regions)

    n_components = len(regions)

    cell_table_content = {
        "Label": {},
        "Area": {},
        "Mean": {},
        "XM": {},
        "YM": {},
        "XM.": {},
        "%Area": {},
        "AR": {},
        "Circ.": {},
        "Round": {},
        "Solidity": {},
        "MScore": {},
        "length_cell_vector": {},
        "absolute_angle": {},
        "rolling_ball_angle": {},
        "relative_angle": {},
    }

    # iterate over remaining regions
    arrows = []
    count_not_moving = 0
    count_no_extensions = 0
    count_considered = 0
    min_length_orientation = min_length_orientation
    labeled_result = np.array(labeled, dtype=int)
    for index, region in enumerate(regions):
        if index % 100 == 0:
            print('%s/%s...' % (index, n_components))

        # write regionprops into table
        # cell_table_content["Label"] = ""
        cell_table_content["Area"][region.label] = region.area
        cell_table_content["Mean"][region.label] = region.intensity_mean
        cell_table_content["XM"][region.label] = region.centroid[1]
        cell_table_content["YM"][region.label] = region.centroid[0]
        circularity = 4 * math.pi * region.area / math.pow(region.perimeter, 2)
        cell_table_content["Circ."][region.label] = circularity
        cell_table_content["%Area"][region.label] = region.area / region.area_filled * 100
        # cell_table_content["AR"][region.label] = ""
        # cell_table_content["Round"][region.label] = ""
        # cell_table_content["Solidity"][region.label] = ""
        cell_table_content["MScore"][region.label] = circularity * ((cell_table_content["Area"][region.label] - 27) / 27)

        # skeletonize
        skeleton = skeletonize(region.intensity_image)

        # calculate distance map
        distance_region = ndimage.distance_transform_edt(region.intensity_image)

        miny, minx, maxy, maxx = region.bbox

        # calculate center
        center = np.unravel_index(np.argmax(distance_region, axis=None), distance_region.shape)
        distance_center = np.linalg.norm(distance_region[center])
        distances_center = np.indices(region.image.shape) - np.array(center)[:, None, None]
        distances_center = np.apply_along_axis(np.linalg.norm, 0, distances_center)

        # label inside/outside cell
        condition_inside = (skeleton > 0) & (distances_center - distance_center < 0)
        condition_outside = (skeleton > 0) & (distances_center - distance_center >= 0)

        # label pixels in displayed image
        clip = labeled_result[miny:maxy, minx:maxx]
        clip[condition_inside] = 4
        clip[condition_outside] = 3
        labeled_result[miny:maxy, minx:maxx] = clip

        pixel_locations_relevant_to_direction = np.column_stack(np.where(condition_outside))
        pixel_locations_relevant_to_direction = pixel_locations_relevant_to_direction - center

        center_translated = [center[1] + minx + roi[0], roi[3] - center[0] - miny]
        target_vector = [0, 0]
        if image_target is not None:
            neighbor_x = [center_translated[0] + 1, center_translated[1]]
            neighbor_y = [center_translated[0], center_translated[1] + 1]
            if neighbor_x[0] < image_target.shape[1] and neighbor_y[1] < image_target.shape[0]:
                value_at_center = image_target[-center_translated[1], center_translated[0]]
                value_at_neighbor_x = image_target[-neighbor_x[1], neighbor_x[0]]
                value_at_neighbor_y = image_target[-neighbor_y[1], neighbor_y[0]]
                target_vector = [value_at_center - value_at_neighbor_x, value_at_center - value_at_neighbor_y]

        matched_row = region.label
        print(matched_row)
        factor = 1.5385

        length_cell_vector = 0
        absolute_angle = 0
        rolling_ball_angle = 0
        relative_angle = 0

        if len(pixel_locations_relevant_to_direction) > 1:
            mean_outside = np.mean(pixel_locations_relevant_to_direction, axis=0)
            # the mean vector needs to be mirrored in Y direction to match the displayed image
            mean_outside = [mean_outside[1], -mean_outside[0]]
            length = np.linalg.norm(mean_outside)
            relative_angle = angle_between(target_vector, mean_outside)
            relative_angle = np.pi / 2 - relative_angle
            arrow_length = np.linalg.norm(mean_outside)
            if not min_length_orientation or length > float(min_length_orientation):
                arrows.append([center_translated, mean_outside, [relative_angle, arrow_length]])
                count_considered += 1
            else:
                count_not_moving += 1
            length_cell_vector = length/factor
            absolute_angle = angle_between((0, 1), mean_outside)
            rolling_ball_angle = angle_between((0, 1), target_vector)

        else:
            if min_length_orientation and float(min_length_orientation) > 0:
                count_no_extensions += 1
            else:
                arrows.append([center_translated, np.array([0., 0.]), [0, 0]])

                count_considered += 1

        cell_table_content["length_cell_vector"][region.label] = length_cell_vector
        cell_table_content["absolute_angle"][region.label] = absolute_angle
        cell_table_content["rolling_ball_angle"][region.label] = rolling_ball_angle
        cell_table_content["relative_angle"][region.label] = relative_angle

    write_table(cell_table_content, output)

    if min_length_orientation:
        print("Ignored %s labels because their directional vector is shorter than %s pixels" % (
            count_not_moving, min_length_orientation))
    print("Ignored %s labels because they don't have extensions" % count_no_extensions)
    return np.array(arrows), labeled_result, cell_table_content


def filter_regions_by_size(min_size, max_size, n_components, regions):
    # sort out regions which are too big
    max_area = max_size
    if max_area:
        regions = [region for region in regions if region.area < int(max_area)]
        region_count = len(regions)
        print(
            "Ignored %s labels because their region is bigger than %s pixels" % (n_components - region_count, max_area))
    # sort out regions which are too small
    min_area = min_size
    if min_area:
        regions = [region for region in regions if region.area >= int(min_area)]
        region_count = len(regions)
        print("Ignored %s labels because their region is smaller than %s pixels" % (
            n_components - region_count, min_area))
    return regions


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if (v1[0] == 0 and v1[1] == 0) or (v2[0] == 0 and v2[1] == 0):
        return 0
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_roi(crop, image):
    crop_min_x = 0
    crop_max_x = image.shape[1]
    crop_min_y = 0
    crop_max_y = image.shape[0]
    print('Input image dimensions: %sx%s' % (crop_max_x, crop_max_y))
    additional_rois = []
    roi = [crop_min_x, crop_max_x, crop_min_y, crop_max_y]
    if crop:
        crops = crop.split(",")
        for single_crop in crops:
            if len(str(single_crop).strip()) != 0:
                crop_parts = single_crop.split(":")
                if len(crop_parts) != 4:
                    exit(
                        "Please provide crop in the following form: MIN_X:MAX_X:MIN_Y:MAX_Y - for example 100:200:100:200")
                additional_rois.append([int(crop_parts[0]), int(crop_parts[1]), int(crop_parts[2]), int(crop_parts[3])])
        if len(additional_rois) == 1:
            roi = additional_rois[0]
            additional_rois = []
    return roi, additional_rois


def plot(directions, raw_image, bg_image, roi, additional_rois, image_target_mask, pixel_in_micron, tiles, output, output_res):
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    output_res = output_res.split(':')
    output_res = [int(output_res[0]), int(output_res[1])]
    roi_colors = []
    if len(additional_rois) > 0:
        roi_colors = plot_rois(output, output_res, bg_image, roi, additional_rois)
    plot_all_directions(output, output_res, directions, bg_image, roi, additional_rois, roi_colors, image_target_mask, pixel_in_micron)
    for tile in tiles.split(','):
        plot_average_directions(output, output_res, directions, raw_image, roi, additional_rois, roi_colors,
                                tile_size=int(tile), image_target_mask=image_target_mask, pixel_in_micron=pixel_in_micron)
    if output:
        print("Results writen to %s" % output)


def plot_average_directions(output, output_res, arrows, bg_image, roi, additional_rois, roi_colors, tile_size,
                            image_target_mask, pixel_in_micron):
    shape = bg_image.shape
    print("Calculating average directions, tile size %s..." % tile_size)
    u, v, x, y, counts = calculate_average_directions(arrows, shape, roi, tile_size, image_target_mask)
    rois = [roi]
    rois.extend(additional_rois)
    colors = ['black']
    colors.extend(roi_colors)
    print("Plotting average directions...")
    plt.figure("Average directions tile size %s" % tile_size, figsize=output_res)
    plt.imshow(bg_image, extent=roi, origin='upper', cmap='gray')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if pixel_in_micron:
        scalebar = ScaleBar(pixel_in_micron, 'um', location='upper right', color='white', box_color='black')
        plt.gca().add_artist(scalebar)
    plot_grid(roi, x, y, u, v, counts, tile_size, image_target_mask)
    if image_target_mask is not None:
        generate_target_contour(image_target_mask)
    plt.margins(0, 0)
    plt.tight_layout(pad=1)
    if output:
        for i, region in enumerate(rois):
            adjust_to_region(roi[3] + roi[2], region, colors[i], scalebar if pixel_in_micron else None)
            # plt.tight_layout(pad=1)
            plt.savefig(output.joinpath(
                'directions_%s-%s-%s-%s_tile%s.png' % (region[0], region[1], region[2], region[3], tile_size)))
        plt.close()
    else:
        plt.show()


def calculate_average_directions(directions, shape, crop_extend, tile_size, image_target_mask):
    tiles_num_x = int(shape[1] / tile_size)
    tiles_num_y = int(shape[0] / tile_size)

    # tile centers
    x = np.array([tile_x * tile_size + crop_extend[0] for tile_x, _ in np.ndindex(tiles_num_x, tiles_num_y)], dtype=int)
    y = np.array([tile_y * tile_size + crop_extend[2] for _, tile_y in np.ndindex(tiles_num_x, tiles_num_y)], dtype=int)

    arrow_indices_x = np.array([int((arrow[0][0] - crop_extend[0]) / tile_size) for arrow in directions])
    arrow_indices_y = np.array([int((arrow[0][1] - crop_extend[2]) / tile_size) for arrow in directions])
    counts = [np.count_nonzero((arrow_indices_x == index_x) & (arrow_indices_y == index_y)) for index_x, index_y in
              np.ndindex(tiles_num_x, tiles_num_y)]
    where = [np.asarray((arrow_indices_x == index_x) & (arrow_indices_y == index_y)).nonzero() for index_x, index_y in
             np.ndindex(tiles_num_x, tiles_num_y)]
    if image_target_mask is not None:
        # weighted sum of the relative angle of an arrow in relation to a target (weights: length of the arrow)
        sum_relative_angle = [np.sum(directions[arrow_indices[0]][:, 2, 0] * directions[arrow_indices[0]][:, 2, 1]) for
                              arrow_indices in where]
        sum_weights = [np.sum(directions[arrow_indices[0]][:, 2, 1]) for arrow_indices in where]
        relative_angle = np.divide(sum_relative_angle, sum_weights, out=np.zeros_like(sum_relative_angle),
                                   where=np.array(sum_weights, dtype=int) != 0)
        avg_length = np.divide(sum_weights, counts, out=np.zeros_like(sum_relative_angle),
                               where=np.array(counts, dtype=int) != 0)
        u = relative_angle
        v = avg_length
    else:
        sum_u = [np.sum(directions[arrow_indices[0]][:, 1, 0]) for arrow_indices in where]
        sum_v = [np.sum(directions[arrow_indices[0]][:, 1, 1]) for arrow_indices in where]
        u = np.divide(sum_u, counts, out=np.zeros_like(sum_u), where=np.array(counts, dtype=int) != 0)
        v = np.divide(sum_v, counts, out=np.zeros_like(sum_v), where=np.array(counts, dtype=int) != 0)
    return u, v, x, y, counts


def plot_arrows(x, y, u, v, scale=1.):
    norm = Normalize()
    colors = np.arctan2(u, v)
    colormap = cm.hsv
    norm.autoscale(colors)
    return plt.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy', scale=scale, width=0.003)


def plot_arrows_relative(x, y, u, v, relative_angle, scale=1.):
    norm = Normalize(-np.pi / 2, np.pi / 2)
    colors = relative_angle
    colormap = cm.coolwarm
    return plt.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy', scale=0.4, width=2, units='dots')


def plot_grid(roi, x, y, u, v, counts, tile_size, image_target_mask):

    if image_target_mask is not None:
        norm = Normalize(-np.pi / 2, np.pi / 2)
        colors_legend = u
        colormap = cm.coolwarm
        colors = u
    else:
        norm = Normalize()
        ph = np.linspace(0, 2 * np.pi, 13)
        scale_start = 30.
        offset = 40.
        x_legend = scale_start * np.cos(ph) + offset
        y_legend = scale_start * np.sin(ph) + offset
        u_legend = np.cos(ph) * scale_start * 0.5 + offset
        v_legend = np.sin(ph) * scale_start * 0.5 + offset
        colors_legend = np.arctan2(np.cos(ph), np.sin(ph))
        norm.autoscale(colors_legend)
        colormap = cm.hsv
        colors = np.arctan2(u, v)

    max_length = 10.
    max_count = tile_size * tile_size / 10000.
    for index, _x in enumerate(x):
        _y = y[index]
        if image_target_mask is not None:
            average_length = v[index]
        else:
            average_length = np.linalg.norm([u[index], v[index]])
        cell_count = float(counts[index])
        alpha = min(1., cell_count / max_count) * min(1., average_length / max_length) * 0.9
        facecolor = to_rgba(colormap(norm(colors[index])), alpha)
        plt.gca().add_patch(Rectangle((_x, _y), tile_size, tile_size, facecolor=facecolor))

    if image_target_mask is None:
        for index, _x in enumerate(x_legend):
            pos1 = [_x, y_legend[index]]
            pos2 = [u_legend[index], v_legend[index]]
            plt.annotate('', pos1, xytext=pos2, xycoords='axes pixels', arrowprops={
                'width': 3., 'headlength': 4.4, 'headwidth': 7., 'edgecolor': 'black',
                'facecolor': colormap(norm(colors_legend[index]))
            })
    else:
        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_array(norm(colors))
        # sm.set_clim(vmin=-2, vmax=2)
        cbar = plt.colorbar(sm, ax=plt.gca(), location='bottom', pad=0.01, aspect=50)
        vmin, vmax = cbar.vmin, cbar.vmax
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(['Moving away from target', 'Moving towards target'])
        cbar.ax.xaxis.get_majorticklabels()[0].set_horizontalalignment('left')
        cbar.ax.xaxis.get_majorticklabels()[-1].set_horizontalalignment('right')
        circ1 = mpatches.Rectangle((0,0), 1, 1, edgecolor='#ff0000', facecolor='#000000', hatch=r'O', label='target')
        plt.legend(handles=[circ1], loc=2, handlelength=4, handleheight=4)
        # legend = plt.gca().legend(handles=[cbar, patch], loc='lower center', bbox_to_anchor=(0.5, -0.3))
    # plt.quiver(x+tile_size/2., y+tile_size/2., u, v, color=colormap(norm(colors)), angles='xy', scale_units='xy', scale=0.5)


def plot_all_directions(output, output_res, directions, bg_image, roi, additional_rois, additional_roi_colors,
                        image_target_mask, pixel_in_micron):
    print("Plotting all directions...")
    rois = [roi]
    rois.extend(additional_rois)
    colors = ['black']
    colors.extend(additional_roi_colors)
    fig = plt.figure("All directions", output_res)
    plt.imshow(bg_image, extent=roi, origin='upper', cmap='gray')
    scalebar = None
    if pixel_in_micron:
        scalebar = ScaleBar(pixel_in_micron, 'um', location='upper right', color='white', box_color='black')
        plt.gca().add_artist(scalebar)
    # bg_image_with_arrows = np.array(bg_image)

    if image_target_mask is not None:
        generate_target_contour(image_target_mask)

    x = directions[:, 0, 0]
    y = directions[:, 0, 1]
    u = directions[:, 1, 0]
    v = directions[:, 1, 1]
    rel_angle = directions[:, 2, 0]

    # plt.scatter(x, y, color='white', s=15)
    if image_target_mask is not None:
        quiver = plot_arrows_relative(x, y, u, v, rel_angle, scale=0.4 * (roi[1] - roi[0]))
        # draw_arrows(bg_image_with_arrows, x, y, u, v, rel_angle)
    else:
        quiver = plot_arrows(x, y, u, v, scale=0.4 * (roi[1] - roi[0]))

    # Image.fromarray(bg_image_with_arrows).save('directions.tif')

    plt.margins(0, 0)
    plt.tight_layout(pad=1)
    if output:
        for i, region in enumerate(rois):
            adjust_to_region(roi[3] + roi[2], region, colors[i], scalebar if pixel_in_micron else None)
            plt.savefig(output.joinpath('directions_%s-%s-%s-%s.png' % (region[0], region[1], region[2], region[3])))
        plt.close()
    print("Done printing all directions")


def generate_target_contour(image_target_mask):
    plt.contour(image_target_mask, 1, origin='upper', colors='red')
    cs = plt.contourf(image_target_mask, 1, hatches=['', 'O'], origin='upper', colors='none')
    cs.set_edgecolor((1, 0, 0.2, 1))


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

