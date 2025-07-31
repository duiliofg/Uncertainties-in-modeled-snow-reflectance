# Evaluating Uncertainties in Modeled Snow Reflectance Using Multispectral Remote Sensing and UAV-Based Hyperspectral Imaging

This repository contains the code and resources associated with the manuscript:

**"Evaluating Uncertainties in Modeled Snow Reflectance Using Multispectral Remote Sensing and UAV-Based Hyperspectral Imaging"**  
**Duilio Fonseca-Gallardo**, **Eric Sproles**, **Shannon Hamp**, and **Joseph Shaw**

---

## 📌 Overview

Accurate snow surface reflectance is essential for reliable satellite-based albedo products. This project evaluates Landsat 8/9 surface reflectance products using high-resolution UAV-mounted hyperspectral imagery collected at three sites (Montana, USA and Sodankylä, Finland). We identify systematic biases—particularly an underestimation in Band 6—and propose a scalable validation protocol integrating CNN-based classification, RSR-based band simulation, and reflectance comparison.

---

## 🚀 Key Contributions

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

## 📁 Repository Structure

```bash
├── src/                 # Python scripts for preprocessing, RSR simulation, and classification
├── notebooks/           # Jupyter notebooks with visualizations and analysis
├── data/                # Sample input hypercubes or links/instructions for access
├── results/             # Outputs, masks, metrics, and plots
├── README.md            # Project description (you are here)

