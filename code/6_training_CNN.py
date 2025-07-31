"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@student.montana.edu
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
    with rasterio.open(path) as src:
        image_data = src.read()
        metadata = src.meta
    return image_data, metadata

def extract_training_data(hyperspectral_image, metadata, geodataframe, label_column):
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
    reshaped_data = training_data.reshape(training_data.shape[0], -1)
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(reshaped_data)
    return transformed_data, pca

def create_strict_cnn(input_shape, num_classes):
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
