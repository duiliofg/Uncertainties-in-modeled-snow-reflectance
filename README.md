# Assessing spaceborne snow reflectance uncertainties using UAV hyperspectral measurements

This repository contains the code and resources associated with the manuscript:

**"Assessing spaceborne snow reflectance uncertainties using UAV hyperspectral measurements"**  
**Duilio Fonseca-Gallardo**, **Eric Sproles**, **Shannon Hamp**, **Joseph Shaw**, **Riley D. Logan**, **Anna
K. Schweiger**, **Henna-Reetta Hannula**, and **Roberta Pirazzini**.

---

## Overview

Accurate snow surface reflectance is essential for reliable satellite-based albedo products. This project evaluates Landsat 8/9 surface reflectance products using high-resolution UAV-mounted hyperspectral imagery collected at three sites (Montana, USA and Sodankylä, Finland). We identify systematic biases—particularly an underestimation in Band 6—and propose a scalable validation protocol integrating CNN-based classification, RSR-based band simulation, and reflectance comparison.

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



