"""
DEM (Digital Elevation Model) processor.

Takes a real GeoTIFF (e.g. USGS/NASA lunar or Mars elevation data) and
converts it into:
  1. A downsampled heightmap grid (for Three.js terrain rendering)
  2. A slope-derived cost grid (for A* pathfinding - steep slopes cost more)

Real DEM files can be tens of thousands of pixels per side, which is
overkill (and slow) for a browser to render directly, so we resample
down to a grid size that's smooth to animate (default 128x128).
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling


class DEMProcessor:
    def __init__(self, tif_path: str, grid_size: int = 128):
        self.tif_path = tif_path
        self.grid_size = grid_size
        self.elevation = None      # 2D numpy array, meters
        self.slope_deg = None      # 2D numpy array, degrees
        self.cost_grid = None      # 2D numpy array, A* traversal cost
        self._load_and_process()

    def _load_and_process(self):
        with rasterio.open(self.tif_path) as src:
            # Resample directly to target grid size while reading -
            # much faster than loading full-res then downsampling.
            data = src.read(
                1,
                out_shape=(self.grid_size, self.grid_size),
                resampling=Resampling.average,
            ).astype("float64")

            # Some DEMs use a large negative sentinel for "no data"
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
                if np.isnan(data).any():
                    # fill gaps with the mean of valid neighbors (simple approach,
                    # good enough for a hackathon - real pipelines would interpolate)
                    data = np.where(np.isnan(data), np.nanmean(data), data)

        self.elevation = data
        self._compute_slope()
        self._compute_cost_grid()

    def _compute_slope(self):
        """Approximate slope in degrees using elevation gradient between
        neighboring grid cells."""
        dz_dy, dz_dx = np.gradient(self.elevation)
        # gradient is in elevation-units per grid-cell; convert to an angle
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2) / 10.0)  # /10 = horizontal scale factor
        self.slope_deg = np.degrees(slope_rad)

    def _compute_cost_grid(self, max_traversable_slope: float = 25.0):
        """Cost grid for A*: flat ground is cheap, steep slopes are expensive,
        anything steeper than max_traversable_slope is blocked (infinite cost),
        modeling a real rover's actual climbing limits."""
        cost = 1.0 + (self.slope_deg / max_traversable_slope) ** 2 * 5.0
        cost = np.where(self.slope_deg > max_traversable_slope, np.inf, cost)
        self.cost_grid = cost

    def get_heightmap_payload(self) -> dict:
        """JSON-serializable payload for the frontend to build a Three.js terrain mesh."""
        elev = self.elevation
        normalized = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-6)
        return {
            "grid_size": self.grid_size,
            "elevation_min_m": float(elev.min()),
            "elevation_max_m": float(elev.max()),
            "heightmap": normalized.round(4).tolist(),   # 0-1 normalized for easy rendering
            "raw_elevation_m": elev.round(2).tolist(),    # actual meters, for telemetry/tilt calc
        }

    def get_cost_grid(self) -> np.ndarray:
        return self.cost_grid

    def slope_at(self, row: int, col: int) -> float:
        """Used by the telemetry simulator to set realistic tilt values
        based on the rover's actual grid position."""
        row = int(np.clip(row, 0, self.grid_size - 1))
        col = int(np.clip(col, 0, self.grid_size - 1))
        return float(self.slope_deg[row, col])
