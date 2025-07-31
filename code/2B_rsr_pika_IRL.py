# Pikra IR-L v2

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

warnings.filterwarnings('ignore', category=DeprecationWarning)

print('Loading extract_crs_and_transform_from_hdr')
# Functions
def extract_crs_and_transform_from_hdr(hdr_path):
    with open(hdr_path, 'r') as f:
        hdr = f.read()
    info = re.search(r'map info = \{(.*?)\}', hdr)
    if not info:
        raise ValueError('Map info not found')
    parts = info.group(1).split(',')
    x0 = float(parts[3])
    y0 = float(parts[4])
    px_x = float(parts[5])
    px_y = float(parts[6])
    zone = int(parts[7])
    hemi = parts[8].strip().lower() + 'ern'
    transform = Affine(px_x, 0, x0, 0, -px_y, y0)
    crs = f'+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m +no_defs'
    return crs, transform

def save_geotiff_multiband(output_path, data, crs, transform):
    # Forzar a tener 3 dimensiones
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
    
    data = np.squeeze(data)
    h, w, b = data.shape
    with rasterio.open(output_path, 'w', driver='GTiff',
        height=h, width=w, count=b, dtype='float32',
        crs=crs, transform=transform, nodata=0) as dst:
        for i in range(b):
            dst.write(data[:, :, i].astype('float32'), i + 1)
            
def simulate_pika_irl_band6(cube_path, rsr_csv_dirs, output_root_dir, basename, satellite):
    cube = envi.open(cube_path)
    data = cube.load()
    wavelengths = np.array(cube.bands.centers)
    hdr_path = cube_path if cube_path.endswith('.hdr') else cube_path + '.hdr'
    crs, transform = extract_crs_and_transform_from_hdr(hdr_path)

    # Read RSR for Landsat 8 or 9 Band 6
    rsr_csv = os.path.join(rsr_csv_dirs, f'Landsat{satellite}_B6_RSR_nm.csv')
    rsr_df = pd.read_csv(rsr_csv)
    rsr_interp_func = interp1d(rsr_df['wavelength'], rsr_df['response'], bounds_error=False, fill_value=0)
    rsr_interp = rsr_interp_func(wavelengths)
    rsr_norm = rsr_interp / np.sum(rsr_interp)

    # Band 6 spectral range
    wl_min, wl_max = 1560, 1660
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    subcube = data[:, :, mask]

    # Apply RSR
    filtered_subcube = subcube * rsr_norm[mask][np.newaxis, np.newaxis, :]
    simulated_rsr = np.sum(filtered_subcube, axis=2)

    # Simple average (no RSR)
    simulated_avg = np.mean(subcube, axis=2)

    # === Define output folders ===
    rsr_dir = os.path.join(output_root_dir, f'landsat{satellite}')
    avg_dir = os.path.join(output_root_dir, 'avg')
    os.makedirs(rsr_dir, exist_ok=True)
    os.makedirs(avg_dir, exist_ok=True)

    output_path_rsr = os.path.join(rsr_dir, f'{basename}_Landsat{satellite}_B6_RSR.tif')
    output_path_avg = os.path.join(avg_dir, f'{basename}_Landsat{satellite}_B6_AVG.tif')

    # === Save RSR GeoTIFF ===
    with rasterio.open(output_path_rsr, 'w', driver='GTiff',
                       height=simulated_rsr.shape[0], width=simulated_rsr.shape[1],
                       count=1, dtype='float32', crs=crs, transform=transform, nodata=0) as dst:
        dst.write(simulated_rsr.astype('float32'), 1)

    # === Save AVG GeoTIFF ===
    with rasterio.open(output_path_avg, 'w', driver='GTiff',
                       height=simulated_avg.shape[0], width=simulated_avg.shape[1],
                       count=1, dtype='float32', crs=crs, transform=transform, nodata=0) as dst:
        dst.write(simulated_avg.astype('float32'), 1)

    print(f'✅ Landsat {satellite} Band 6 processed for {basename}')
    print(f'   - RSR saved to: {output_path_rsr}')
    print(f'   - AVG saved to: {output_path_avg}')
