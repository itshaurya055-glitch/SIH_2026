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
    # Body radius in meters, used to convert degree-based pixel spacing to
    # real-world meters for geographic (lat/lon) CRS DEMs. Change this if
    # you're working with Mars data instead of lunar.
    BODY_RADIUS_M = {
        "moon": 1_737_400.0,
        "mars": 3_389_500.0,
        "earth": 6_371_000.0,
    }

    def __init__(self, tif_path: str, grid_size: int = 128, body: str = "moon"):
        self.tif_path = tif_path
        self.grid_size = grid_size
        self.body_radius_m = self.BODY_RADIUS_M.get(body, self.BODY_RADIUS_M["moon"])
        self.elevation = None      # 2D numpy array, meters
        self.slope_deg = None      # 2D numpy array, degrees
        self.cost_grid = None      # 2D numpy array, A* traversal cost
        self.pixel_size_x_m = None # real-world meters per grid cell, computed from the file
        self.pixel_size_y_m = None
        self._load_and_process()

    def _load_and_process(self):
        with rasterio.open(self.tif_path) as src:
            native_width, native_height = src.width, src.height
            native_res_x, native_res_y = src.res  # pixel size in the file's CRS units

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

            # Downsampling factor: each grid cell in our 128x128 output now
            # represents this many native pixels.
            factor_x = native_width / self.grid_size
            factor_y = native_height / self.grid_size

            if src.crs and src.crs.is_geographic:
                # native_res is in DEGREES - convert to real meters using the
                # body's radius. Longitude spacing shrinks with latitude, so
                # use the center latitude of the raster's bounds.
                bounds = src.bounds
                center_lat_rad = np.radians((bounds.bottom + bounds.top) / 2.0)
                meters_per_deg_lat = (np.pi / 180.0) * self.body_radius_m
                meters_per_deg_lon = meters_per_deg_lat * np.cos(center_lat_rad)

                self.pixel_size_y_m = native_res_y * factor_y * meters_per_deg_lat
                self.pixel_size_x_m = native_res_x * factor_x * meters_per_deg_lon

                self._source_bounds = bounds
                self._source_crs = str(src.crs)
            else:
                # Already a projected CRS (units are meters) - just scale by
                # the downsampling factor.
                self.pixel_size_x_m = native_res_x * factor_x
                self.pixel_size_y_m = native_res_y * factor_y
                self._source_bounds = src.bounds
                self._source_crs = str(src.crs) if src.crs else "unknown"

        self.elevation = data
        self._compute_slope()
        self._compute_cost_grid()

    def _compute_slope(self):
        """Slope in degrees using elevation gradient between neighboring grid
        cells, converted using the DEM's REAL geographic pixel spacing (not
        a made-up constant) - this is what makes slope values physically
        meaningful instead of arbitrary."""
        dz_dy, dz_dx = np.gradient(self.elevation)
        # dz_dx/dz_dy are in meters of elevation change per grid cell;
        # divide by the real horizontal distance per grid cell to get a
        # true rise/run ratio.
        slope_x = dz_dx / max(self.pixel_size_x_m, 1e-6)
        slope_y = dz_dy / max(self.pixel_size_y_m, 1e-6)
        slope_rad = np.arctan(np.sqrt(slope_x**2 + slope_y**2))
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