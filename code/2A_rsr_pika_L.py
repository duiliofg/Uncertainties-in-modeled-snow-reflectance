"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@student.montana.edu
- GitHub: https://github.com/duiliofg

Script Title: Simulate Landsat OLI/OLI-2 bands B1-B5 from Pika L hypercubes

Description:
Convolves the Pika L hyperspectral reflectance with the normalised relative spectral
response (RSR) functions of Landsat 8 OLI and Landsat 9 OLI-2, producing one multiband
GeoTIFF per hypercube with the simulated bands B1 to B5.

Requirements:
- Python 3.x
- Libraries: matplotlib, numpy, pandas, rasterio, scipy, spectral
"""

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
    """
    Extracts the CRS and affine transform from a header file (.hdr).

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@student.montana.edu
    - GitHub: https://github.com/duiliofg

    Description:
    Parses the 'map info' field of the header to recover the UTM zone, hemisphere,
    upper-left origin and pixel size, and builds the corresponding affine transform and a
    WGS84 UTM PROJ string.

    Parameters:
    - hdr_path: str, path to the header file (.hdr).

    Returns:
    - crs: str, PROJ definition of the WGS84 UTM coordinate reference system.
    - transform: affine.Affine, affine transform of the raster grid.

    Raises:
    - ValueError: if the 'map info' field is not found in the header.
    """
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
    """
    Saves a multiband array as a Float32 GeoTIFF.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@student.montana.edu
    - GitHub: https://github.com/duiliofg

    Description:
    Removes singleton dimensions and writes each band of the array to a GeoTIFF with
    NoData set to 0.

    Parameters:
    - output_path: str, path of the output GeoTIFF.
    - data: numpy.ndarray, array of shape (rows, cols, bands).
    - crs: str, coordinate reference system of the raster.
    - transform: affine.Affine, affine transform of the raster.

    Returns:
    - None. The GeoTIFF is written to disk.
    """
    data = np.squeeze(data)
    h, w, b = data.shape
    with rasterio.open(output_path, 'w', driver='GTiff',
        height=h, width=w, count=b, dtype='float32',
        crs=crs, transform=transform, nodata=0) as dst:
        for i in range(b):
            dst.write(data[:, :, i].astype('float32'), i + 1)
            
def simulate_landsat_bands(cube_path, rsr_csv_dir, output_root_dir, basename, bands_info, satellite):
    """
    Simulates Landsat OLI/OLI-2 bands from a Pika L hypercube.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@student.montana.edu
    - GitHub: https://github.com/duiliofg

    Description:
    For each band in bands_info, interpolates the Landsat relative spectral response (RSR)
    to the band centres of the cube, normalises it to unit sum and applies it as weights to
    the channels inside the band window (RSR product). A simple arithmetic mean of the same
    channels is also computed (AVG product). The two stacks are written as multiband GeoTIFFs
    to '<output_root_dir>/landsat<satellite>/<basename>_Landsat<satellite>_RSR.tif' and
    '<output_root_dir>/avg/<basename>_Landsat<satellite>_AVG.tif'.

    Parameters:
    - cube_path: str, path to the hypercube; '.hdr' is appended to locate the header
      when the path does not already end with it.
    - rsr_csv_dir: str, directory with the RSR files 'Landsat<satellite>_<band>_RSR_nm.csv',
      which contain the columns 'wavelength' (nm) and 'response'.
    - output_root_dir: str, root directory for the outputs.
    - basename: str, base name of the output files.
    - bands_info: dict, mapping from band name to (wl_min, wl_max) window in nm.
    - satellite: int or str, Landsat mission number (8 or 9).

    Returns:
    - None. The GeoTIFFs are written to disk.
    """
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
