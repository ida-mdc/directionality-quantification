import numpy as np
from skimage.color import gray2rgb
from skimage.draw import disk
from skimage.draw import polygon

def _blend_rgba(overlay, rr, cc, rgb, alpha):
    valid = (rr >= 0) & (rr < overlay.shape[0]) & (cc >= 0) & (cc < overlay.shape[1])
    rr = rr[valid]
    cc = cc[valid]

    overlay[rr, cc, 0] = rgb[0]
    overlay[rr, cc, 1] = rgb[1]
    overlay[rr, cc, 2] = rgb[2]

    overlay[rr, cc, 3] = np.maximum(overlay[rr, cc, 3], alpha)

def _draw_scaled_arrow_aa(overlay, r0, c0, r1, c1, color, scale_factor=1.0, opacity=0.8):
    """Draws an anti-aliased arrow with scalable line width and head size."""
    base_head_len = 7.0
    base_head_width = 6.0
    base_line_width = 2.0

    head_len = base_head_len * scale_factor
    head_width = base_head_width * scale_factor
    line_width = base_line_width * scale_factor

    dy, dx = (r1 - r0), (c1 - c0)
    norm = np.hypot(dy, dx) + 1e-9
    uy, ux = dy / norm, dx / norm
    py, px = -ux, uy

    p0 = np.array([r0, c0])
    p1 = np.array([r1, c1])
    
    if norm < head_len:
        arrowhead_base = p0
        arrowhead_tip = p0 + head_len * np.array([uy, ux])
        draw_shaft = False
    else:
        arrowhead_base = p1 - head_len * np.array([uy, ux])
        arrowhead_tip = p1
        draw_shaft = True
        shaft_end = p1 - head_len * 0.8 * np.array([uy, ux])

    if draw_shaft:
        shaft_half_width = line_width / 2.0
        r_coords = np.array([
            p0[0] - shaft_half_width * py,
            p0[0] + shaft_half_width * py,
            shaft_end[0] + shaft_half_width * py,
            shaft_end[0] - shaft_half_width * py,
        ])
        c_coords = np.array([
            p0[1] - shaft_half_width * px,
            p0[1] + shaft_half_width * px,
            shaft_end[1] + shaft_half_width * px,
            shaft_end[1] - shaft_half_width * px,
        ])

        rr_shaft, cc_shaft = polygon(r_coords, c_coords, shape=overlay.shape[:2])
        _blend_rgba(overlay, rr_shaft, cc_shaft, color[:3], opacity)
    tip = arrowhead_tip
    base = arrowhead_base
    left = base + (head_width / 2.0) * np.array([py, px])
    right = base - (head_width / 2.0) * np.array([py, px])

    pr_head = np.array([tip[0], left[0], right[0]])
    pc_head = np.array([tip[1], left[1], right[1]])
    rr_head, cc_head = polygon(pr_head, pc_head, shape=overlay.shape[:2])
    _blend_rgba(overlay, rr_head, cc_head, color[:3], opacity)


def create_hatch_pattern(shape, style='stripes', tile_size=25, line_width=4, circle_radius_ratio=0.3):
    """
    Creates a boolean mask for a fast, tileable raster hatch pattern.

    :param shape: The (rows, cols) of the full output array.
    :param style: 'circles', 'stripes' (diagonal), or 'checkerboard'.
    :param tile_size: The size of the repeating pattern (e.g., 25 pixels).
    :param line_width: The thickness of the stripes.
    :param circle_radius_ratio: The size of the circle relative to the tile_size.
    """

    tile = np.zeros((tile_size, tile_size), dtype=bool)

    if style == 'stripes':
        indices = np.indices((tile_size, tile_size))
        stripe_spacing = tile_size // 2
        if stripe_spacing == 0: stripe_spacing = 1
        tile = (indices[0] + indices[1]) % stripe_spacing < line_width

    elif style == 'circles':
        center = (tile_size // 2, tile_size // 2)
        radius = int(tile_size * circle_radius_ratio)
        rr, cc = disk(center, radius, shape=(tile_size, tile_size))
        tile[rr, cc] = True

    elif style == 'checkerboard':
        half = tile_size // 2
        if half == 0: half = 1
        tile[0:half, 0:half] = True
        tile[half:tile_size, half:tile_size] = True

    num_repeats_y = int(np.ceil(shape[0] / tile_size))
    num_repeats_x = int(np.ceil(shape[1] / tile_size))

    tiled = np.tile(tile, (num_repeats_y, num_repeats_x))

    return tiled[:shape[0], :shape[1]]


def apply_hatch_to_background(bg_image, image_target_mask,
                              hatch_style='circles',
                              hatch_tile_size=20,
                              hatch_color_rgb=[1.0, 0.0, 0.0],
                              tint_alpha=0.3):
    """
    Bakes a hatch pattern directly into the background image
    where the mask is active, AND tints the background of that area.
    """
    print("Baking hatch texture and tint into background image...")

    bg_image_rgb = gray2rgb(bg_image)

    color_layer = np.zeros_like(bg_image_rgb)
    color_layer[:] = hatch_color_rgb

    target_area_mask = (image_target_mask > 0.5)
    target_area_mask_3d = np.stack([target_area_mask] * 3, axis=-1)

    hatch_mask = create_hatch_pattern(
        bg_image.shape,
        style=hatch_style,
        tile_size=hatch_tile_size
    )
    hatch_mask_3d = np.stack([hatch_mask] * 3, axis=-1)

    tinted_bg = (color_layer * tint_alpha) + (bg_image_rgb * (1.0 - tint_alpha))

    bg_with_tint = np.where(target_area_mask_3d, tinted_bg, bg_image_rgb)

    solid_hatch_area = target_area_mask_3d & hatch_mask_3d

    final_image = np.where(solid_hatch_area, color_layer, bg_with_tint)

    return final_image