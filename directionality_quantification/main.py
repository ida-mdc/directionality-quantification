import argparse

import numpy as np
import tifffile
from scipy import ndimage
from skimage.measure import label, regionprops

from directionality_quantification.plot import plot
from directionality_quantification.process import analyze_segments, write_table, compute_and_write_avg_dir_tables


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
    image = tifffile.imread(args.input_labeling).astype(int)
    image_target_mask = None
    image_target_distances = None
    if args.input_target is not None:
        image_target_mask = tifffile.imread(args.input_target).astype(bool)
        image_target_distances = ndimage.distance_transform_edt(np.invert(image_target_mask))

    # crop input images to ROI
    roi, additional_rois = get_roi(args.roi, image)  # returns array with [min_x, max_x, min_y, max_y]
    image = image[roi[0]:roi[1], roi[2]:roi[3]]
    image_raw = image_raw[roi[0]:roi[1], roi[2]:roi[3]]
    if image_target_mask is not None:
        image_target_distances = image_target_distances[roi[0]:roi[1], roi[2]:roi[3]]
        image_target_mask = image_target_mask[roi[0]:roi[1], roi[2]:roi[3]]

    pixel_in_micron = args.pixel_in_micron

    W, H = int(args.output_res.split(':')[0]), int(args.output_res.split(':')[1])
    output_res = [W, H]

    regions = get_regions(image, args.min_size, args.max_size)
    cell_table_content  = analyze_segments(regions, image_target_distances, pixel_in_micron)
    write_table(cell_table_content, args.output)

    avg_tables = compute_and_write_avg_dir_tables(cell_table_content, image_raw, roi, image_target_mask, args.tiles, args.output)

    plot(cell_table_content, image_raw, roi, additional_rois, image_target_mask, pixel_in_micron, args.tiles,
         args.output, output_res, avg_tables)



def get_regions(labeled, min_size, max_size):
    # obtain labels
    print("Labeling segmentation..")
    # Heuristic: if the image has only two unique values and one is 0, assume it's a binary mask
    unique_vals = np.unique(labeled)
    if len(unique_vals) == 2 and 0 in unique_vals:
        # Binary mask case (e.g., 0 and 255)
        binary_mask = labeled != 0  # Covers 255 or 1 as foreground
        labeled, n_components = label(binary_mask, return_num=True)

    else:
        n_components = len(unique_vals)
    print(f'{n_components} objects detected.')
    # calculate region properties
    segmentation = labeled > 0
    regions = regionprops(label_image=labeled, intensity_image=segmentation)
    regions = filter_regions_by_size(min_size, max_size, n_components, regions)
    return regions


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


def get_roi(crop, image):
    crop_min_x = 0
    crop_max_x = image.shape[0]
    crop_min_y = 0
    crop_max_y = image.shape[1]
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


if __name__ == '__main__':
    run()