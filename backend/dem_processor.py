"""
DEM (Digital Elevation Model) processor.

Takes a real GeoTIFF (e.g. USGS/NASA lunar or Mars elevation data) and
converts it into:
  1. A downsampled heightmap grid (for Three.js terrain rendering)
  2. A physics-based cost grid via Bekker-Wong terramechanics
     (replaces old slope² heuristic — cost now reflects real sinkage,
     wheel-slip, and motion resistance under lunar gravity 1.62 m/s²)
  3. Per-cell soil type map (synthetically generated — flagged as such)

Real DEM files can be tens of thousands of pixels per side, which is
overkill (and slow) for a browser to render directly, so we resample
down to a grid size that's smooth to animate (default 128x128).
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling

from lunar_physics import LunarPhysicsModel, _generate_synthetic_soil_map


class DEMProcessor:
    # Body radius in meters, used to convert degree-based pixel spacing to
    # real-world meters for geographic (lat/lon) CRS DEMs.
    BODY_RADIUS_M = {
        "moon":  1_737_400.0,
        "mars":  3_389_500.0,
        "earth": 6_371_000.0,
    }

    def __init__(self, tif_path: str, grid_size: int = 128, body: str = "moon"):
        self.tif_path = tif_path
        self.grid_size = grid_size
        self.body_radius_m = self.BODY_RADIUS_M.get(body, self.BODY_RADIUS_M["moon"])

        self.elevation = None       # 2D numpy array, meters
        self.slope_deg = None       # 2D numpy array, degrees
        self.cost_grid = None       # 2D numpy array, A* traversal cost (physics-based)
        self.slip_grid = None       # 2D numpy array, slip_ratio per cell
        self.soil_map  = None       # 2D int array: 0=SOFT, 1=MEDIUM, 2=FIRM
        self.pixel_size_x_m = None  # real-world meters per grid cell
        self.pixel_size_y_m = None

        # Physics model (stateless, reusable)
        self._physics = LunarPhysicsModel()

        self._load_and_process()

    def _load_and_process(self):
        with rasterio.open(self.tif_path) as src:
            native_width, native_height = src.width, src.height
            native_res_x, native_res_y = src.res

            data = src.read(
                1,
                out_shape=(self.grid_size, self.grid_size),
                resampling=Resampling.average,
            ).astype("float64")

            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
                if np.isnan(data).any():
                    data = np.where(np.isnan(data), np.nanmean(data), data)

            factor_x = native_width  / self.grid_size
            factor_y = native_height / self.grid_size

            if src.crs and src.crs.is_geographic:
                bounds = src.bounds
                center_lat_rad = np.radians((bounds.bottom + bounds.top) / 2.0)
                meters_per_deg_lat = (np.pi / 180.0) * self.body_radius_m
                meters_per_deg_lon = meters_per_deg_lat * np.cos(center_lat_rad)

                self.pixel_size_y_m = native_res_y * factor_y * meters_per_deg_lat
                self.pixel_size_x_m = native_res_x * factor_x * meters_per_deg_lon
                self._source_bounds = bounds
                self._source_crs = str(src.crs)
            else:
                self.pixel_size_x_m = native_res_x * factor_x
                self.pixel_size_y_m = native_res_y * factor_y
                self._source_bounds = src.bounds
                self._source_crs = str(src.crs) if src.crs else "unknown"

        self.elevation = data
        self._compute_slope()

        # Generate soil map once (spatially correlated, seeded — deterministic)
        self.soil_map = _generate_synthetic_soil_map(self.grid_size, self.grid_size)
        print(f"[dem_processor] Soil map generated: "
              f"{(self.soil_map==0).sum()} SOFT, "
              f"{(self.soil_map==1).sum()} MEDIUM, "
              f"{(self.soil_map==2).sum()} FIRM cells")

        self._compute_cost_grid()

    def _compute_slope(self):
        """Slope in degrees using real geographic pixel spacing — physically meaningful."""
        dz_dy, dz_dx = np.gradient(self.elevation)
        slope_x = dz_dx / max(self.pixel_size_x_m, 1e-6)
        slope_y = dz_dy / max(self.pixel_size_y_m, 1e-6)
        slope_rad = np.arctan(np.sqrt(slope_x**2 + slope_y**2))
        self.slope_deg = np.degrees(slope_rad)

    def _compute_cost_grid(self, max_traversable_slope: float = None):
        """
        Build cost grid via Bekker-Wong terramechanics (not slope² heuristic).

        If max_traversable_slope is provided it overrides the physics model's
        built-in limit (used by the agent's emergency-replan logic).
        """
        print(f"[dem_processor] Computing physics-based cost grid ({self.grid_size}×{self.grid_size})…")
        cost_grid, slip_grid = self._physics.compute_cost_grid(self.slope_deg, self.soil_map)

        # Agent-override: tighten traversability limit if requested
        if max_traversable_slope is not None:
            cost_grid = np.where(
                self.slope_deg > max_traversable_slope, np.inf, cost_grid
            )

        self.cost_grid = cost_grid
        self.slip_grid = slip_grid
        print(f"[dem_processor] Cost grid done. "
              f"Blocked cells: {np.isinf(cost_grid).sum()} / {cost_grid.size}")

    # ── Public accessors ──────────────────────────────────────────────────────

    def get_heightmap_payload(self) -> dict:
        """JSON-serialisable payload for Three.js terrain mesh + physics overlay."""
        elev = self.elevation
        normalized = (elev - elev.min()) / max(elev.max() - elev.min(), 1e-6)
        slip_norm = self.slip_grid / max(self.slip_grid.max(), 1e-6)
        return {
            "grid_size":        self.grid_size,
            "elevation_min_m":  float(elev.min()),
            "elevation_max_m":  float(elev.max()),
            "heightmap":        normalized.round(4).tolist(),
            "raw_elevation_m":  elev.round(2).tolist(),
            "slip_map":         slip_norm.round(4).tolist(),   # 0-1 normalised slip per cell
            "soil_map":         self.soil_map.tolist(),        # 0/1/2 soil type index
        }

    def get_cost_grid(self) -> np.ndarray:
        return self.cost_grid

    def get_slip_grid(self) -> np.ndarray:
        return self.slip_grid

    def slope_at(self, row: int, col: int) -> float:
        row = int(np.clip(row, 0, self.grid_size - 1))
        col = int(np.clip(col, 0, self.grid_size - 1))
        return float(self.slope_deg[row, col])

    def slip_at(self, row: int, col: int) -> float:
        """Wheel slip ratio at a grid cell (0 = no slip, 1 = spinout)."""
        row = int(np.clip(row, 0, self.grid_size - 1))
        col = int(np.clip(col, 0, self.grid_size - 1))
        return float(self.slip_grid[row, col])

    def soil_type_at(self, row: int, col: int) -> str:
        row = int(np.clip(row, 0, self.grid_size - 1))
        col = int(np.clip(col, 0, self.grid_size - 1))
        return ["SOFT", "MEDIUM", "FIRM"][int(self.soil_map[row, col])]

    def get_cell_physics_detail(self, row: int, col: int) -> dict:
        """Full Bekker-Wong breakdown for a single cell — for API and agent diagnostics."""
        row = int(np.clip(row, 0, self.grid_size - 1))
        col = int(np.clip(col, 0, self.grid_size - 1))
        slope = float(self.slope_deg[row, col])
        soil  = self.soil_type_at(row, col)
        detail = self._physics.get_cell_detail(slope, soil)
        detail["row"] = row
        detail["col"] = col
        return detail