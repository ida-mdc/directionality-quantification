"""
Thumbnail extraction for cell analysis results.
"""
import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def extract_cell_thumbnail(raw_image: np.ndarray, region, thumbnail_size: int = 64, padding: int = 5) -> Optional[str]:
    """
    Extract a thumbnail image for a cell region and encode as PNG base64.
    
    Args:
        raw_image: The raw image array
        region: RegionProps object with bbox attribute
        thumbnail_size: Target size for thumbnail (will be resized to this)
        padding: Padding around the cell in pixels
    
    Returns:
        Base64-encoded PNG string (data URI format) or None if extraction fails
    """
    try:
        miny, minx, maxy, maxx = region.bbox
        
        # Add padding
        h, w = raw_image.shape[:2] if len(raw_image.shape) == 2 else raw_image.shape[:2]
        miny = max(0, miny - padding)
        minx = max(0, minx - padding)
        maxy = min(h, maxy + padding)
        maxx = min(w, maxx + padding)
        
        # Extract region from raw image
        if len(raw_image.shape) == 2:
            cell_patch = raw_image[miny:maxy, minx:maxx]
        else:
            cell_patch = raw_image[miny:maxy, minx:maxx, :]
        
        if cell_patch.size == 0:
            return None
        
        # Normalize to 0-255 range
        if cell_patch.dtype != np.uint8:
            patch_min = cell_patch.min()
            patch_max = cell_patch.max()
            if patch_max > patch_min:
                cell_patch = ((cell_patch - patch_min) / (patch_max - patch_min) * 255).astype(np.uint8)
            else:
                cell_patch = np.zeros_like(cell_patch, dtype=np.uint8)
        
        # Convert to PIL Image and resize
        if len(cell_patch.shape) == 2:
            pil_image = Image.fromarray(cell_patch, mode='L')
        else:
            pil_image = Image.fromarray(cell_patch, mode='RGB')
        
        # Resize maintaining aspect ratio
        pil_image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
        
        # Convert to base64 PNG
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Warning: Failed to extract thumbnail for region {region.label}: {e}")
        return None


