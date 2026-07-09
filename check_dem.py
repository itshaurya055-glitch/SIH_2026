#!/usr/bin/env python3
"""
Quick sanity check for a DEM file before you drop it into ./data/dem.tif.

Usage:
    python3 check_dem.py /path/to/your/downloaded_dem.tif

This confirms rasterio can open it, prints its size/resolution, and runs it
through the same DEMProcessor the backend uses - so you catch problems
(corrupt file, unsupported format, all-nodata) before spinning up Docker.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from dem_processor import DEMProcessor  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 check_dem.py /path/to/dem.tif")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading {path} ...")
    dem = DEMProcessor(path, grid_size=128)

    print(f"Grid size: {dem.grid_size}x{dem.grid_size}")
    print(f"Elevation range: {dem.elevation.min():.1f}m to {dem.elevation.max():.1f}m")
    print(f"Slope range: {dem.slope_deg.min():.1f}° to {dem.slope_deg.max():.1f}°")

    import numpy as np
    blocked = int(np.isinf(dem.cost_grid).sum())
    total = dem.cost_grid.size
    print(f"Blocked cells (too steep to traverse): {blocked}/{total} ({100*blocked/total:.1f}%)")

    if blocked == total:
        print("\nWARNING: entire grid is blocked - your DEM may be too extreme "
              "or the slope threshold too strict. Check dem_processor.py's "
              "max_traversable_slope.")
    else:
        print("\nLooks good. Copy this file to ./data/dem.tif and run docker compose up.")


if __name__ == "__main__":
    main()
