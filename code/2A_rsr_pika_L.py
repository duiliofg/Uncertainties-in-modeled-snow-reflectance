# Pika L

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spectral.io import envi
from scipy.interpolate import interp1d
import rasterio
from rasterio.transform import Affine
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

print('Loading simulate_landsat_bands')

# Functions
def extract_crs_and_transform_from_hdr(hdr_path):
    with open(hdr_path, 'r') as f:
        hdr = f.read()
    info = re.search(r"map info = \{(.*?)\}", hdr)
    if not info:
        raise ValueError("Map info not found")
    parts = info.group(1).split(',')
    x0 = float(parts[3])
    y0 = float(parts[4])
    px_x = float(parts[5])
    px_y = float(parts[6])
    zone = int(parts[7])
    hemi = parts[8].strip().lower() + 'ern'
    transform = Affine(px_x, 0, x0, 0, -px_y, y0)
    crs = f"+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m +no_defs"
    return crs, transform

def save_geotiff_multiband(output_path, data, crs, transform):
    data = np.squeeze(data)
    h, w, b = data.shape
    with rasterio.open(output_path, 'w', driver='GTiff',
        height=h, width=w, count=b, dtype='float32',
        crs=crs, transform=transform, nodata=0) as dst:
        for i in range(b):
            dst.write(data[:, :, i].astype('float32'), i + 1)
            
def simulate_landsat_bands(cube_path, rsr_csv_dir, output_root_dir, basename, bands_info, satellite):
    cube = envi.open(cube_path)
    data = cube.load()
    wavelengths = np.array(cube.bands.centers)
    hdr_path = cube_path if cube_path.endswith('.hdr') else cube_path + '.hdr'
    crs, transform = extract_crs_and_transform_from_hdr(hdr_path)

    simulated_bands_rsr = []
    simulated_bands_avg = []

    for band_name, (wl_min, wl_max) in bands_info.items():
        rsr_csv = os.path.join(rsr_csv_dir, f"Landsat{satellite}_{band_name}_RSR_nm.csv")
        rsr_df = pd.read_csv(rsr_csv)
        rsr_interp_func = interp1d(rsr_df['wavelength'], rsr_df['response'], bounds_error=False, fill_value=0)
        rsr_interp = rsr_interp_func(wavelengths)
        rsr_norm = rsr_interp / np.sum(rsr_interp)

        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        subcube = data[:, :, mask]

        # Simulate with RSR
        weighted = subcube * rsr_norm[mask][np.newaxis, np.newaxis, :]
        band_rsr = np.sum(weighted, axis=2)
        simulated_bands_rsr.append(band_rsr[:, :, np.newaxis])

        # Simulate with simple average
        band_avg = np.mean(subcube, axis=2)
        simulated_bands_avg.append(band_avg[:, :, np.newaxis])

    stack_rsr = np.concatenate(simulated_bands_rsr, axis=2)
    stack_avg = np.concatenate(simulated_bands_avg, axis=2)

    # Output paths
    rsr_dir = os.path.join(output_root_dir, f"landsat{satellite}")
    avg_dir = os.path.join(output_root_dir, "avg")
    os.makedirs(rsr_dir, exist_ok=True)
    os.makedirs(avg_dir, exist_ok=True)

    output_path_rsr = os.path.join(rsr_dir, f"{basename}_Landsat{satellite}_RSR.tif")
    output_path_avg = os.path.join(avg_dir, f"{basename}_Landsat{satellite}_AVG.tif")

    save_geotiff_multiband(output_path_rsr, stack_rsr, crs, transform)
    save_geotiff_multiband(output_path_avg, stack_avg, crs, transform)

    print(f"✅ {basename} - Landsat {satellite}")
    print(f"   RSR saved to: {output_path_rsr}")
    print(f"   AVG saved to: {output_path_avg}")
