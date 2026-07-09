"""
Generate a synthetic crater DEM GeoTIFF for testing.
Run once to create backend/test_data/synthetic_crater_dem.tif
"""
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
except ImportError:
    print("rasterio not installed locally — this script runs inside Docker.")
    print("The backend Dockerfile will handle it. Creating a minimal placeholder.")
    import os
    os.makedirs(os.path.dirname(__file__) or ".", exist_ok=True)
    # We'll generate a proper one via Docker build
    exit(0)


def generate_synthetic_crater(size=512, output_path="test_data/synthetic_crater_dem.tif"):
    """Create a synthetic crater-like DEM for testing."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)

    # Base elevation (flat plain at ~1000m)
    elevation = np.full((size, size), 1000.0)

    # Main crater: bowl depression
    crater_mask = r < 0.7
    crater_depth = 200 * (1 - (r / 0.7)**2)
    elevation[crater_mask] -= crater_depth[crater_mask]

    # Crater rim: raised edge
    rim_mask = (r >= 0.55) & (r < 0.75)
    rim_height = 50 * np.exp(-((r - 0.65) / 0.05)**2)
    elevation[rim_mask] += rim_height[rim_mask]

    # Small noise for realism
    elevation += np.random.normal(0, 2, (size, size))

    # Write as GeoTIFF
    transform = from_bounds(0, 0, 10000, 10000, size, size)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=size, width=size,
        count=1, dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(elevation.astype("float32"), 1)

    print(f"Wrote synthetic crater DEM to {output_path} ({size}x{size})")


if __name__ == "__main__":
    generate_synthetic_crater()
