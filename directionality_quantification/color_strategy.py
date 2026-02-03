"""
Color strategy system for determining rectangle colors and alpha values.

This module provides configurable strategies for how tiles are colored based on
their cell counts, vector lengths, and angles.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.colors import to_hex, hsv_to_rgb, rgb_to_hsv
from matplotlib.pyplot import get_cmap
from matplotlib.colors import Normalize

from directionality_quantification.plot import REL_CMAP, REL_NORM, ABS_CMAP, ABS_NORM


class ColorStrategy(ABC):
    """Abstract base class for color strategies."""
    
    @abstractmethod
    def compute_color_and_alpha(
        self,
        rows: List[Dict],
        is_relative: bool,
        tile_size: int
    ) -> None:
        """
        Compute color_hex and alpha for each row in place.
        
        Args:
            rows: List of dictionaries containing tile data. Each dict should have:
                - count: number of cells
                - avg_length: average vector length
                - u, v: vector components
                - color_scalar_deg: angle in degrees
            is_relative: Whether using relative angles (target-based)
            tile_size: Size of tiles in pixels
        """
        pass
    
    def get_alpha_description(self) -> Tuple[str, str]:
        """
        Get description labels for low and high alpha values.
        
        Returns:
            Tuple of (low_label, high_label) for the opacity legend
        """
        return ("Low alpha (transparent)", "High alpha (opaque)")


class AlphaFromCountAndLengthStrategy(ColorStrategy):
    """
    Alpha from count and length strategy:
    - Alpha: based on count and avg_length, normalized using 90th percentile
    - Color: based on angle (hue only, full saturation)
    """
    
    def get_alpha_description(self) -> Tuple[str, str]:
        return ("Less cells, shorter extensions (transparent)", "More cells, longer extensions (opaque)")
    
    def compute_color_and_alpha(
        self,
        rows: List[Dict],
        is_relative: bool,
        tile_size: int
    ) -> None:
        # Collect all counts and lengths for normalization
        counts_all = []
        avg_lengths_all = []
        
        for r in rows:
            counts_all.append(float(r["count"]))
            avg_lengths_all.append(float(r["avg_length"]))
        
        counts_all = np.asarray(counts_all, dtype=float)
        avg_lengths_all = np.asarray(avg_lengths_all, dtype=float)
        counts_all = counts_all[counts_all > 0]
        avg_lengths_all = avg_lengths_all[avg_lengths_all > 0]
        
        max_count = float(np.nanpercentile(counts_all, 90)) if len(counts_all) > 0 else 1.0
        max_length = float(np.nanpercentile(avg_lengths_all, 90)) if len(avg_lengths_all) > 0 else 1.0
        
        # Compute alpha for each row (color_hex already computed in build_average_directions_table)
        for r in rows:
            c = r["count"]
            L = r["avg_length"]
            
            # Alpha based on count and length
            alpha = min(1.0, c / max_count) * min(1.0, L / max_length) * 0.9 if (max_count > 0 and max_length > 0) else 0.0
            
            r["alpha"] = alpha
            # color_hex already set in build_average_directions_table, keep it
            r["max_count"] = float(max_count)
            r["max_length"] = float(max_length)


class Version020ColorStrategy(ColorStrategy):
    """
    Version 0.2.0 strategy:
    - Alpha: based on count and avg_length, normalized using hardcoded values
    - Color: based on angle (hue only, full saturation)
    """
    
    def get_alpha_description(self) -> Tuple[str, str]:
        return ("Less cells, shorter extensions (transparent)", "More cells, longer extensions (opaque)")
    
    def compute_color_and_alpha(
        self,
        rows: List[Dict],
        is_relative: bool,
        tile_size: int
    ) -> None:
        # Hardcoded normalization values from version 0.2.0
        max_length = 10.0
        max_count = tile_size * tile_size / 10000.0
        
        # Compute alpha for each row (color_hex already computed in build_average_directions_table)
        for r in rows:
            c = r["count"]
            L = r["avg_length"]
            
            # Alpha based on count and length (old formula with hardcoded max values)
            alpha = min(1.0, c / max_count) * min(1.0, L / max_length) * 0.9 if (max_count > 0 and max_length > 0) else 0.0
            
            r["alpha"] = alpha
            # color_hex already set in build_average_directions_table, keep it
            r["max_count"] = float(max_count)
            r["max_length"] = float(max_length)


class CountAlphaSaturationStrategy(ColorStrategy):
    """
    New strategy:
    - Alpha: determined by number of cells (normalized)
    - Color saturation: determined by strength of average vector (normalized)
    - Color hue: determined by average angle
    """
    
    def get_alpha_description(self) -> Tuple[str, str]:
        return ("Fewer cells (transparent)", "More cells (opaque)")
    
    def compute_color_and_alpha(
        self,
        rows: List[Dict],
        is_relative: bool,
        tile_size: int
    ) -> None:
        # Collect all counts and vector strengths for normalization
        counts_all = []
        vector_strengths_all = []
        
        for r in rows:
            counts_all.append(float(r["count"]))
            # Vector strength is the magnitude of (u, v)
            strength = float(np.hypot(r["u"], r["v"]))
            vector_strengths_all.append(strength)
        
        counts_all = np.asarray(counts_all, dtype=float)
        vector_strengths_all = np.asarray(vector_strengths_all, dtype=float)
        counts_all = counts_all[counts_all > 0]
        vector_strengths_all = vector_strengths_all[vector_strengths_all > 0]
        
        max_count = float(np.nanpercentile(counts_all, 90)) if len(counts_all) > 0 else 1.0
        max_strength = float(np.nanpercentile(vector_strengths_all, 90)) if len(vector_strengths_all) > 0 else 1.0
        
        # Compute colors and alpha for each row
        for r in rows:
            c = r["count"]
            strength = float(np.hypot(r["u"], r["v"]))
            
            # Alpha from cell count
            alpha = min(1.0, c / max_count) * 0.9 if (max_count > 0) else 0.0
            
            # Color: use same hue as other strategies (from colormap), adjust saturation from vector strength
            if c == 0:
                color_hex = to_hex((0, 0, 0))
            else:
                color_scalar_deg = r["color_scalar_deg"]
                
                # Get base color from the same colormap as other strategies (preserves hue)
                if is_relative:
                    base_rgba = REL_CMAP(REL_NORM(color_scalar_deg))
                else:
                    base_rgba = ABS_CMAP(ABS_NORM(color_scalar_deg))
                
                # Convert RGB to HSV to modify saturation
                base_rgb = np.array(base_rgba[:3])  # Extract RGB (ignore alpha if present)
                hsv = rgb_to_hsv(base_rgb.reshape(1, 1, 3))[0, 0]  # Convert to HSV
                original_saturation = hsv[1]  # Preserve original saturation level
                
                # Modify saturation based on vector strength (normalized)
                # Scale from 0 to original_saturation (not 1.0) to preserve exact color match at max strength
                strength_ratio = min(1.0, strength / max_strength) if max_strength > 0 else 0.0
                new_saturation = original_saturation * strength_ratio
                
                # Keep original hue and value, update saturation
                modified_hsv = np.array([hsv[0], new_saturation, hsv[2]])
                
                # Convert back to RGB
                modified_rgb = hsv_to_rgb(modified_hsv.reshape(1, 1, 3))[0, 0]
                color_hex = to_hex(modified_rgb)
            
            r["alpha"] = alpha
            r["color_hex"] = color_hex
            r["max_count"] = float(max_count)
            r["max_length"] = float(max_strength)  # Store max_strength as max_length for compatibility


# Strategy registry
STRATEGIES = {
    "alpha_from_count_and_length": AlphaFromCountAndLengthStrategy,
    "0.2.0": Version020ColorStrategy,
    "count_alpha_saturation": CountAlphaSaturationStrategy,
}


def get_color_strategy(name: str) -> ColorStrategy:
    """
    Get a color strategy by name.
    
    Args:
        name: Strategy name. Options:
            - "alpha_from_count_and_length": Alpha from count and length (percentile-based normalization)
            - "0.2.0": Version 0.2.0 strategy (hardcoded normalization)
            - "count_alpha_saturation": Alpha from count, saturation from strength, hue from angle
    
    Returns:
        ColorStrategy instance
    """
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown color strategy '{name}'. Available: {available}")
    return STRATEGIES[name]()
