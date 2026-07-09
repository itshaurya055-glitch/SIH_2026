FROM python:3.12-slim

# rasterio needs GDAL - the pip wheel bundles it, but a couple of system
# libs (for certain GeoTIFF compressions/CRS handling) are safer to have
# present explicitly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate synthetic fallback DEM at build time so the service always boots
RUN python generate_test_dem.py

# DEM_PATH points at the mounted data volume - see docker-compose.yml.
# GRID_SIZE controls heightmap resolution sent to the frontend.
ENV DEM_PATH=/app/data/dem.tif
ENV GRID_SIZE=128

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
