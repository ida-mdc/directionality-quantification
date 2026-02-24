"""
Color strategy system for determining rectangle colors and alpha values.

This module provides configurable strategies for how tiles are colored based on
their cell counts, vector lengths, and angles.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


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


class AlphaFromCountStrategy(ColorStrategy):
    """
    Alpha from count strategy:
    - Color: determined by average angle (precomputed in build_average_directions_table)
    - Alpha: based only on cell count, normalized using the 90th percentile
    """

    def get_alpha_description(self) -> Tuple[str, str]:
        return ("Fewer cells (transparent)", "More cells (opaque)")

    def compute_color_and_alpha(
        self,
        rows: List[Dict],
        is_relative: bool,
        tile_size: int
    ) -> None:
        # Collect all counts for normalization
        counts_all = []
        for r in rows:
            counts_all.append(float(r["count"]))

        counts_all = np.asarray(counts_all, dtype=float)
        counts_all = counts_all[counts_all > 0]

        max_count = float(np.nanpercentile(counts_all, 90)) if len(counts_all) > 0 else 1.0

        # Compute alpha for each row (color_hex already computed in build_average_directions_table)
        for r in rows:
            c = float(r["count"])

            if max_count > 0 and c > 0:
                alpha = min(1.0, c / max_count) * 0.9
            else:
                alpha = 0.0

            r["alpha"] = alpha
            r["max_count"] = float(max_count)


class AlphaFromAngleStdStrategy(ColorStrategy):
    """
    Alpha from angular standard deviation strategy:
    - Color: determined by average angle (precomputed in build_average_directions_table)
    - Alpha: based on angular coherence (1 - normalized std dev of angles)
      -> Low std dev (coherent direction) = high opacity
      -> High std dev (diverse directions) = low opacity
    """

    def get_alpha_description(self) -> Tuple[str, str]:
        return ("High angle variability (transparent)", "Low angle variability (opaque)")

    def compute_color_and_alpha(
        self,
        rows: List[Dict],
        is_relative: bool,
        tile_size: int
    ) -> None:
        # Collect all angle standard deviations for normalization
        std_all = []
        for r in rows:
            c = float(r["count"])
            if c > 0:
                std_all.append(float(r.get("angle_std_deg", 0.0)))

        std_all = np.asarray(std_all, dtype=float)
        std_all = std_all[std_all > 0]

        max_std = float(np.nanpercentile(std_all, 90)) if len(std_all) > 0 else 0.0

        for r in rows:
            c = float(r["count"])
            if c == 0 or max_std <= 0.0:
                alpha = 0.0
            else:
                std_deg = float(r.get("angle_std_deg", 0.0))
                ratio = min(1.0, std_deg / max_std) if max_std > 0 else 0.0
                coherence = 1.0 - ratio
                alpha = coherence * 0.9

            r["alpha"] = alpha
            r["max_angle_std_deg"] = float(max_std)


# Strategy registry
STRATEGIES = {
    "alpha_from_count": AlphaFromCountStrategy,
    "alpha_from_angle_std": AlphaFromAngleStdStrategy,
}


def get_color_strategy(name: str) -> ColorStrategy:
    """
    Get a color strategy by name.
    
    Args:
        name: Strategy name. Options:
            - "alpha_from_count": Alpha from cell count (90th percentile normalization)
            - "alpha_from_angle_std": Alpha from angular coherence (1 - normalized std dev)
    
    Returns:
        ColorStrategy instance
    """
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown color strategy '{name}'. Available: {available}")
    return STRATEGIES[name]()
