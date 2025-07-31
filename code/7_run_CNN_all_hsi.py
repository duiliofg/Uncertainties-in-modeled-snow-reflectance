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

CLASS_MAP = {1: 'Snow', 2: 'Vegetation', 3: 'Shadows', 4: 'No_data'}

def load_hyperspectral_image(path):
    with rasterio.open(path) as src:
        image_data = src.read()
        metadata = src.meta.copy()
    return image_data, metadata

def prepare_full_dataset(hyperspectral_data, pca_model):
    data = np.transpose(hyperspectral_data, (1, 2, 0))  # (rows, cols, bands)
    reshaped_data = data.reshape(-1, data.shape[2])     # Flatten spatial dimensions
    pca_transformed_data = pca_model.transform(reshaped_data)
    return pca_transformed_data.reshape(-1, 1, 1, pca_transformed_data.shape[1])

def save_classification_map(classification_map, metadata, output_path):
    metadata.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(classification_map.astype(rasterio.uint8), 1)

def vectorize_classification(classification_map, transform, crs, base_name, output_gpkg_path):
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
