"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@anori.cl
- GitHub: https://github.com/duiliofg

Script Title: Apply the trained CNN to the Greenland hypercubes

Description:
Greenland counterpart of 7_run_CNN_all_hsi.py, kept separate because of differences in
the input directory layout and scene naming.

Requirements:
- Python 3.x
- Libraries: geopandas, joblib, numpy, rasterio, shapely, tensorflow
"""

import os
import glob
import numpy as np
import rasterio
from tensorflow.keras.models import load_model
import joblib
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

print('Loading classify_and_vectorize_cubes')

CLASS_MAP = {1: 'Ice', 2: 'Rock_1', 3: 'Rock_2', 4: 'Debri Patch', 5: 'Water', 6:'Shadow/Crevasse',7: 'No_data'}

def load_hyperspectral_image(path):
    """
    Loads a hyperspectral GeoTIFF and its metadata.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Reads all bands of a hyperspectral reflectance GeoTIFF with rasterio and returns the
    pixel array together with the raster profile.

    Parameters:
    - path: str, path to the hyperspectral GeoTIFF.

    Returns:
    - image_data: numpy.ndarray, array of shape (bands, rows, cols).
    - metadata: dict, rasterio metadata of the raster (driver, dtype, CRS, transform, size).
    """
    with rasterio.open(path) as src:
        image_data = src.read()
        metadata = src.meta.copy()
    return image_data, metadata

def prepare_full_dataset(hyperspectral_data, pca_model):
    """
    Prepares every pixel of a cube for CNN prediction.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Reorders the image to (rows, cols, bands), flattens the spatial dimensions, projects the
    spectra onto the fitted PCA and reshapes the scores to the 1 x 1 x n_components input
    expected by the CNN.

    Parameters:
    - hyperspectral_data: numpy.ndarray, image array of shape (bands, rows, cols).
    - pca_model: sklearn.decomposition.PCA, PCA fitted during training.

    Returns:
    - numpy.ndarray, array of shape (rows * cols, 1, 1, n_components).
    """
    data = np.transpose(hyperspectral_data, (1, 2, 0))  # (rows, cols, bands)
    reshaped_data = data.reshape(-1, data.shape[2])     # Flatten spatial dimensions
    pca_transformed_data = pca_model.transform(reshaped_data)
    return pca_transformed_data.reshape(-1, 1, 1, pca_transformed_data.shape[1])

def save_classification_map(classification_map, metadata, output_path):
    """
    Saves a classification map as a single-band GeoTIFF.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Writes the class map as an unsigned 8-bit raster using the georeferencing of the source
    cube. The metadata dictionary is updated in place.

    Parameters:
    - classification_map: numpy.ndarray, class codes of shape (rows, cols).
    - metadata: dict, rasterio metadata of the source cube.
    - output_path: str, path of the output GeoTIFF.

    Returns:
    - None. The GeoTIFF is written to disk.
    """
    metadata.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(classification_map.astype(rasterio.uint8), 1)

def vectorize_classification(classification_map, transform, crs, base_name, output_gpkg_path):
    """
    Vectorises a classification map into class polygons.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Converts contiguous regions of equal class code into polygons, excluding code 0, and
    stores for each polygon the class name ('type', from the module-level CLASS_MAP, or
    'Unknown') and the class code ('type_id').

    Parameters:
    - classification_map: numpy.ndarray, class codes of shape (rows, cols).
    - transform: affine.Affine, affine transform of the raster.
    - crs: rasterio.crs.CRS or str, coordinate reference system of the raster.
    - base_name: str, layer name inside the GeoPackage.
    - output_gpkg_path: str, path of the output GeoPackage.

    Returns:
    - None. The GeoPackage is written to disk.
    """
    geometries = []
    values = classification_map.astype(np.uint8)

    for geom, value in shapes(values, mask=(values > 0), transform=transform):
        class_name = CLASS_MAP.get(value, "Unknown")
        geometries.append({
            "geometry": shape(geom),
            "properties": {"type": class_name, "type_id": value}
        })

    gdf = gpd.GeoDataFrame.from_features(geometries, crs=crs)
    gdf.to_file(output_gpkg_path, layer=base_name, driver="GPKG")

def classify_and_vectorize_cubes(
    input_folder,
    output_folder,
    model_path,
    pca_path,
    class_map=CLASS_MAP
):
    """
    Classifies and vectorises every hyperspectral cube in a folder.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Loads the trained CNN and the PCA, and for every GeoTIFF (*.tif*) in the input folder
    predicts the class of each pixel, saves the classification map in the 'geotiff' subfolder
    ('<name>_classified_raster.tiff') and its polygons in the 'gpkg' subfolder
    ('<name>_classified_polygons.gpkg'). Default classes: Ice, Rock_1, Rock_2, Debri Patch, Water, Shadow/Crevasse and No_data.

    Parameters:
    - input_folder: str, folder with the hyperspectral reflectance GeoTIFFs.
    - output_folder: str, root folder for the classified outputs.
    - model_path: str, path to the trained CNN (.h5).
    - pca_path: str, path to the fitted PCA (.pkl).
    - class_map: dict, mapping from class code to class name. Default CLASS_MAP.

    Returns:
    - None. Rasters and GeoPackages are written to disk.

    Raises:
    - FileNotFoundError: if the input folder contains no GeoTIFF.
    """
    # Create subfolders
    output_geotiff = os.path.join(output_folder, "geotiff")
    output_gpkg = os.path.join(output_folder, "gpkg")
    os.makedirs(output_geotiff, exist_ok=True)
    os.makedirs(output_gpkg, exist_ok=True)

    print("Loading model and PCA...")
    model = load_model(model_path)
    pca_model = joblib.load(pca_path)

    raster_files = sorted(glob.glob(os.path.join(input_folder, "*.tif*")))
    if not raster_files:
        raise FileNotFoundError(f"No GeoTIFF files found in {input_folder}")

    for raster_path in raster_files:
        print(f"Processing {raster_path}")
        base_name = os.path.splitext(os.path.basename(raster_path))[0]

        output_raster_path = os.path.join(output_geotiff, f"{base_name}_classified_raster.tiff")
        output_vector_path = os.path.join(output_gpkg, f"{base_name}_classified_polygons.gpkg")

        hyperspectral_data, metadata = load_hyperspectral_image(raster_path)
        prepared_data = prepare_full_dataset(hyperspectral_data, pca_model)

        predictions = model.predict(prepared_data, verbose=0).argmax(axis=-1)
        classification_map = predictions.reshape(metadata['height'], metadata['width'])

        save_classification_map(classification_map, metadata, output_raster_path)

        vectorize_classification(
            classification_map=classification_map,
            transform=metadata['transform'],
            crs=metadata['crs'],
            base_name=base_name,
            output_gpkg_path=output_vector_path
        )

    print("✅ All cubes classified and vectorized.")
