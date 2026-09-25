"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@anori.cl
- GitHub: https://github.com/duiliofg

Script Title: Train CNN Model from Polygons across Multiple Hyperspectral Cubes

Description:
This script trains a strict CNN model for hyperspectral classification using polygon-labeled data 
distributed across multiple hypercubes. It saves the trained CNN model and PCA for later classification.

Requirements:
- Python 3.x
- Libraries: rasterio, numpy, geopandas, pandas, matplotlib, seaborn, tensorflow, scikit-learn, imbalanced-learn, joblib
"""

import os
import glob
import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import RandomOverSampler
from matplotlib.colors import LinearSegmentedColormap
print('Loading train_strict_cnn_from_polygons')

# Base Functions
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
        metadata = src.meta
    return image_data, metadata

def extract_training_data(hyperspectral_image, metadata, geodataframe, label_column):
    """
    Extracts labelled spectra from pixels inside training polygons.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Rasterises each polygon of the GeoDataFrame onto the grid of the hyperspectral image and
    collects the full spectrum of every pixel falling inside it, together with the class label
    of the polygon. Polygons are assumed to share the CRS of the raster.

    Parameters:
    - hyperspectral_image: numpy.ndarray, image array of shape (bands, rows, cols).
    - metadata: dict, rasterio metadata providing 'transform', 'height' and 'width'.
    - geodataframe: geopandas.GeoDataFrame, training polygons for this image.
    - label_column: str, column of the GeoDataFrame that holds the class label.

    Returns:
    - X: numpy.ndarray, spectra of shape (n_pixels, bands).
    - y: numpy.ndarray, class label of each pixel, of shape (n_pixels,).
    """
    training_data = []
    labels = []
    transform = metadata['transform']
    for index, feature in geodataframe.iterrows():
        geom = feature.geometry
        mask = rasterio.features.geometry_mask([geom], out_shape=(metadata['height'], metadata['width']),
                                               transform=transform, invert=True)
        pixel_indices = np.argwhere(mask)
        for (y, x) in pixel_indices:
            pixel_value = hyperspectral_image[:, y, x]
            training_data.append(pixel_value)
            labels.append(feature[label_column])
    return np.array(training_data), np.array(labels)

def apply_pca(training_data, n_components):
    """
    Fits a PCA to the training spectra and transforms them.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Reduces the spectral dimensionality of the training samples with a principal component
    analysis fitted on the same samples.

    Parameters:
    - training_data: numpy.ndarray, spectra of shape (n_samples, bands).
    - n_components: int, number of principal components to retain.

    Returns:
    - transformed_data: numpy.ndarray, PCA scores of shape (n_samples, n_components).
    - pca: sklearn.decomposition.PCA, fitted PCA model.
    """
    reshaped_data = training_data.reshape(training_data.shape[0], -1)
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(reshaped_data)
    return transformed_data, pca

def create_strict_cnn(input_shape, num_classes):
    """
    Builds and compiles the CNN used for per-pixel classification.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Each pixel is treated as a 1 x 1 image whose channels are the PCA components. The network
    stacks a 5 x 5 convolution with 32 filters, batch normalisation and dropout of 0.5, a
    3 x 3 convolution with 64 filters, batch normalisation and dropout of 0.7, a dense layer
    of 128 ReLU units and a softmax output. The model is compiled with the Adam optimiser
    (learning rate 0.001), sparse categorical cross-entropy loss and accuracy as metric.

    Parameters:
    - input_shape: tuple, shape of one sample, typically (1, 1, n_components).
    - num_classes: int, number of output classes.

    Returns:
    - model: tensorflow.keras.Model, compiled CNN.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def aggregate_training_from_cubes(gdf, raster_folder, cube_column, label_column):
    """
    Aggregates training samples from several hyperspectral cubes.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    For each cube identifier listed in the polygon layer, finds the first GeoTIFF (*.tiff) in
    the raster folder whose name contains '_<id>_', loads it and extracts the labelled spectra
    of the polygons that belong to that cube. Cubes without a matching file are reported and
    skipped.

    Parameters:
    - gdf: geopandas.GeoDataFrame, training polygons for all cubes.
    - raster_folder: str, folder containing the hyperspectral reflectance GeoTIFFs.
    - cube_column: str, column that identifies the cube each polygon belongs to.
    - label_column: str, column that holds the class label.

    Returns:
    - X: numpy.ndarray, stacked spectra of shape (n_pixels, bands).
    - y: numpy.ndarray, class labels of shape (n_pixels,).

    Raises:
    - ValueError: if no training data could be extracted.
    """
    all_samples = []
    all_labels = []

    cube_ids = gdf[cube_column].unique()
    print(f"Found {len(cube_ids)} unique cubes for training.")

    all_rasters = sorted(glob.glob(os.path.join(raster_folder, "*.tiff")))

    for cube_id in cube_ids:
        matching_files = [f for f in all_rasters if f"_{cube_id}_" in os.path.basename(f)]
        if not matching_files:
            print(f"[WARNING] No file found for cube {cube_id}")
            continue

        raster_path = matching_files[0]
        sub_gdf = gdf[gdf[cube_column] == cube_id]

        hyperspectral_data, metadata = load_hyperspectral_image(raster_path)
        samples, labels = extract_training_data(hyperspectral_data, metadata, sub_gdf, label_column)

        all_samples.append(samples)
        all_labels.append(labels)

    if not all_samples:
        raise ValueError("No training data extracted. Check your file paths and cube matching.")
    
    X = np.vstack(all_samples)
    y = np.concatenate(all_labels)
    return X, y


def save_performance_metrics(y_true, y_pred, output_csv_path, output_plot_path):
    """
    Computes classification metrics and saves them with the confusion matrix.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Computes the macro-averaged precision and F1 score and the confusion matrix of the test
    predictions. The two scores are written to a CSV file and the confusion matrix is saved as
    a 4 x 4 in heatmap with a turquoise colour map at 300 DPI.

    Parameters:
    - y_true: array-like, reference class labels.
    - y_pred: array-like, predicted class labels.
    - output_csv_path: str, path of the CSV file with the metrics.
    - output_plot_path: str, path of the confusion matrix figure.

    Returns:
    - None. The CSV file and the figure are written to disk.
    """
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    # Save metrics
    performance_df = pd.DataFrame([{'Precision': precision, 'F1 Score': f1}])
    performance_df.to_csv(output_csv_path, index=False)

    # Create a custom turquoise colormap
    custom_cmap = LinearSegmentedColormap.from_list("turquoise", ["#E0FFFF", "#40E0D0", "#008080"])

    # Plot Confusion Matrix
    plt.figure(figsize=(4, 4))  # Good for 2x2 paper layout
    sns.set(font_scale=1.2)

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=custom_cmap,
        cbar=False,
        annot_kws={"size": 10}
    )
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    
# Main Function to Call
def train_strict_cnn_from_polygons(
    gdf_path,
    raster_folder,
    output_folder,
    cube_column="cube",
    label_column="type",
    pca_components=20,
    epochs=50
):
    """
    Trains the per-pixel CNN classifier from labelled polygons.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Full training pipeline: reads and cleans the polygon layer, extracts the labelled spectra
    from all referenced cubes, reduces them with PCA, balances the classes with random
    oversampling (random_state=42), splits the data into 70 % training, 15 % validation and
    15 % test, trains the CNN and evaluates it on the test subset. The model is saved as
    'cnn_model.h5', the PCA as 'pca_model.pkl', and the metrics as
    'classification_metrics.csv' and 'confusion_matrix.png' in the output folder.

    Parameters:
    - gdf_path: str, path to the vector file with the training polygons.
    - raster_folder: str, folder containing the hyperspectral reflectance GeoTIFFs.
    - output_folder: str, folder where the model, PCA and metrics are saved.
    - cube_column: str, column that identifies the cube of each polygon. Default 'cube'.
    - label_column: str, column with the integer class label. Default 'type'.
    - pca_components: int, number of principal components. Default 20.
    - epochs: int, number of training epochs. Default 50.

    Returns:
    - model: tensorflow.keras.Model, trained CNN.
    - pca_model: sklearn.decomposition.PCA, fitted PCA.
    - test_results: tuple (X_test, y_test, y_pred) with the test samples, their labels and
      the predictions.
    """
    os.makedirs(output_folder, exist_ok=True)
    gdf = gpd.read_file(gdf_path)
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]

    print("Extracting training data...")
    training_data, labels = aggregate_training_from_cubes(
        gdf=gdf,
        raster_folder=raster_folder,
        cube_column=cube_column,
        label_column=label_column
    )

    print("Applying PCA...")
    training_data_pca, pca_model = apply_pca(training_data, n_components=pca_components)

    print("Balancing data...")
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(training_data_pca, labels)

    print("Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    print("Training CNN...")
    model = create_strict_cnn((1, 1, pca_components), np.max(y_resampled) + 1)
    model.fit(X_train.reshape(-1, 1, 1, pca_components), y_train, epochs=epochs,
              validation_data=(X_val.reshape(-1, 1, 1, pca_components), y_val), verbose=1)

    print("Evaluating model...")
    y_pred = model.predict(X_test.reshape(-1, 1, 1, pca_components)).argmax(axis=-1)

    print("Saving model and performance...")
    model.save(os.path.join(output_folder, 'cnn_model.h5'))
    joblib.dump(pca_model, os.path.join(output_folder, 'pca_model.pkl'))
    save_performance_metrics(
        y_true=y_test,
        y_pred=y_pred,
        output_csv_path=os.path.join(output_folder, 'classification_metrics.csv'),
        output_plot_path=os.path.join(output_folder, 'confusion_matrix.png')
    )
    print("✅ Done.")
    return model, pca_model, (X_test, y_test, y_pred)
