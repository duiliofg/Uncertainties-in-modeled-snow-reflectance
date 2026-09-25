"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@anori.cl
- GitHub: https://github.com/duiliofg

Script Title: Build a Landsat-equivalent vector grid from a reference raster

Description:
Generates a polygon grid matching the pixel footprint of a reference Landsat raster and
exports it as a GeoPackage. The grid defines the cells over which hyperspectral and
Landsat reflectance are later compared.

Requirements:
- Python 3.x
- Libraries: geopandas, rasterio, shapely
"""

import rasterio
import geopandas as gpd
from shapely.geometry import box

print('Loading create_grid_from_raster')

def create_grid_from_raster(raster_path, output_gpkg, resolution=None):
    """
    Generates a grid from a raster and saves it as a GeoPackage.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    This function extracts spatial metadata from a raster dataset and generates a vector grid with 
    polygons representing equally spaced cells. The grid is saved as a GeoPackage, facilitating its 
    use in GIS and remote sensing applications.

    Parameters:
    - raster_path: str, path to the raster file (e.g., Landsat image).
    - output_gpkg: str, path to save the GeoPackage containing the grid.
    - resolution: float, grid resolution in the same units as the raster's CRS.
                  If not specified, the function defaults to the raster's native resolution.

    Returns:
    - None. The function saves the GeoPackage to the specified path.

    Requirements:
    - Python 3.x
    - rasterio for raster data handling
    - geopandas for vector data manipulation
    - shapely for geometric operations
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        if resolution is None:
            resolution = src.res[0]  # Use raster pixel resolution

        # Calculate the number of rows and columns
        xmin, ymin, xmax, ymax = bounds
        rows = int((ymax - ymin) / resolution)
        cols = int((xmax - xmin) / resolution)

        # Generate grid polygons
        polygons = []
        for row in range(rows):
            for col in range(cols):
                x_min = xmin + col * resolution
                y_min = ymin + row * resolution
                x_max = x_min + resolution
                y_max = y_min + resolution
                polygons.append(box(x_min, y_min, x_max, y_max))

        # Create a GeoDataFrame with the grid
        grid = gpd.GeoDataFrame({"geometry": polygons}, crs=crs)

        # Save the grid as a GeoPackage
        grid.to_file(output_gpkg, driver="GPKG")
        print(f"Grid generated and saved to {output_gpkg}")
