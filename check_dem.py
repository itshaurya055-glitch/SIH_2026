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

    print(f"CRS: {dem._source_crs}")
    print(f"Bounds: {dem._source_bounds}")
    print(f"Real-world size per grid cell: {dem.pixel_size_x_m:.1f}m x {dem.pixel_size_y_m:.1f}m")
    approx_width_km = (dem.pixel_size_x_m * dem.grid_size) / 1000
    approx_height_km = (dem.pixel_size_y_m * dem.grid_size) / 1000
    print(f"Approx real-world coverage: {approx_width_km:.1f}km x {approx_height_km:.1f}km")

    print(f"Grid size: {dem.grid_size}x{dem.grid_size}")
    print(f"Elevation range: {dem.elevation.min():.1f}m to {dem.elevation.max():.1f}m")
    print(f"Slope range: {dem.slope_deg.min():.1f}° to {dem.slope_deg.max():.1f}°")

    import numpy as np
    blocked = int(np.isinf(dem.cost_grid).sum())
    total = dem.cost_grid.size
    print(f"Blocked cells (too steep to traverse): {blocked}/{total} ({100*blocked/total:.1f}%)")

    if blocked == total:
        print("\nWARNING: entire grid is blocked.")
        if approx_width_km > 200:
            print(f"Likely cause: this file covers ~{approx_width_km:.0f}km x {approx_height_km:.0f}km, "
                  "which is far too large an area for a rover-scale demo (you want more like "
                  "10-50km across). Go back to Map a Planet and set a tighter lat/lon bounding "
                  "box before re-downloading.")
        else:
            print("The clipped area itself may genuinely be extreme terrain (e.g. a crater wall "
                  "or mountain range). Try a different region, or loosen max_traversable_slope "
                  "in dem_processor.py's _compute_cost_grid().")
    else:
        print("\nLooks good. Copy this file to ./data/dem.tif and run docker compose up.")


if __name__ == "__main__":
    main()