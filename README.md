# Assessing spaceborne snow reflectance uncertainties using UAV hyperspectral measurements

This repository contains the code associated with the manuscript:

**"Assessing spaceborne snow reflectance uncertainties using UAV hyperspectral measurements"**  
**Duilio Fonseca-Gallardo**, **Eric Sproles**, **Shannon Hamp**, **Joseph Shaw**, **Riley D. Logan**, **Anna K. Schweiger**, **Henna-Reetta Hannula**, and **Roberta Pirazzini**.

---

## Overview

Accurate snow surface reflectance is essential for reliable satellite-based albedo products. This project evaluates Landsat 8/9 surface reflectance products using high-resolution UAV-mounted hyperspectral imagery collected at three sites (Montana, USA and Sodankylä, Finland). We identify systematic biases, particularly an underestimation in Band 6, and propose a scalable validation protocol integrating CNN-based classification, RSR-based band simulation, and reflectance comparison.

---

## Key Contributions

- Field data acquisition with **Pika L** and **Pika IR-L** hyperspectral sensors (400–1700 nm)
- Supervised classification using **CNN architecture (Zehnder, 2022)** to segment snow, vegetation, and shadows
- Simulation of Landsat 8/9 **OLI/OLI-2** bands using **Relative Spectral Response (RSR)**
- Quantification of reflectance discrepancies across snow-covered landscapes
- Python tools for:
  - Radiometric correction
  - Georeferencing
  - Reflectance processing
  - Statistical comparison

---

## Repository structure

```
code/
├── 1_hsi_to_geotiff.py                 # Convert Resonon hyperspectral hypercubes to GeoTIFF
├── 2A_rsr_pika_L.py                    # Simulate Landsat OLI/OLI-2 bands B1-B5 from Pika L hypercubes
├── 2B_rsr_pika_IRL.py                  # Simulate Landsat OLI/OLI-2 band B6 from Pika IR-L hypercubes
├── 3_extract_landsat_grid.py           # Build a Landsat-equivalent vector grid from a reference raster
├── 4_compare_landsat_hsi.py            # Compare hyperspectral-derived and observed Landsat reflectance
├── 5a_compare_plot.py                  # Plot reflectance and differences by classified pixel type
├── 5b_compare_plot_IRL.py              # Plot reflectance and differences for the Pika IR-L sensor
├── 5c_comparte_plot_IRL_paper.py       # Publication version of the Pika IR-L comparison figure
├── 6_training_CNN.py                   # Train the CNN model from polygons across multiple hypercubes
├── 7_run_CNN_all_hsi.py                # Apply the trained CNN to every hypercube
├── 7_run_CNN_all_hsi_greenland.py      # Apply the trained CNN to the Greenland hypercubes
├── 8_reflectance_HSI.py                # Reflectance statistics by surface type over the full spectral range
├── 8_reflectance_HSI_greenland.py      # Reflectance statistics by surface type for the Greenland scenes
├── 9_reflectance_RSR.py                # Reflectance statistics by surface type on the RSR-convolved products
├── 9b_reflectance_band.py              # Per-band reflectance statistics by surface type
├── classification_v2.ipynb             # CNN training and classification
└── workflow_reflectance_generic.ipynb  # Workflow that runs scripts 1 to 9 for each study site
```

---

## Workflow

The notebook `workflow_reflectance_generic.ipynb` runs the scripts sequentially:

1. Convert HSI to GeoTIFF
2. Simulate the Landsat RSR response using Pika L and Pika IR-L hyperspectral data
3. Extract the Landsat grid
4. Compare reflectance (HSI vs Landsat)
5. Generate comparison plots
6. Train the CNN classifier
7. Apply the CNN to all HSI cubes
8. Compute reflectance from HSI based on classified pixels
9. Compute reflectance from RSR-convolved HSI based on classified pixels

Directory paths in the notebook are placeholders (`/path/to/snow_albedo/...`) and must be set to the local data structure.

---

## Requirements

- Python 3.x
- numpy, pandas, scipy, matplotlib, seaborn, tqdm
- rasterio, rioxarray, geopandas, shapely, spectral
- tensorflow, scikit-learn, imbalanced-learn, joblib
