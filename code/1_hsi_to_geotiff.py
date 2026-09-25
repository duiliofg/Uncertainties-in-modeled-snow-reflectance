"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@anori.cl
- GitHub: https://github.com/duiliofg

Script Title: Convert Resonon hyperspectral hypercubes to GeoTIFF

Description:
Reads hypercubes acquired with the Resonon Pika L and Pika IR-L imagers,
recovers the CRS and affine transform from the companion header, and writes each cube
to a georeferenced multiband GeoTIFF. Includes a batch routine that walks an input
directory for a given sensor and study site.

Requirements:
- Python 3.x
- Libraries: numpy, rasterio, spectral
"""

import os
import re
import numpy as np
import spectral.io.envi as envi
import rasterio
from rasterio.transform import Affine

print('Loading hypercube_to_geotiff')

def hypercube_to_geotiff(cube_path, output_path):
    """
    Converts a hyperspectral hypercube into a georeferenced multiband GeoTIFF.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Loads the hypercube with the spectral library, removes singleton dimensions, reads the
    'map info' field of the header to build the affine transform and the WGS84 UTM
    coordinate reference system, and writes every spectral band to a GeoTIFF with NoData
    set to 0.

    Parameters:
    - cube_path: str, path to the header file (.hdr) of the hypercube.
    - output_path: str, path of the output GeoTIFF.

    Returns:
    - None. The GeoTIFF is written to disk.

    Raises:
    - ValueError: if the 'map info' field is not found in the header.
    """
    # Load the hypercube data
    cube = envi.open(cube_path).load()

    # Remove any unnecessary dimensions (e.g., single-dimensional axes)
    cube = np.squeeze(cube)

    # Read map info from the header file
    with open(cube_path, 'r') as file:
        hdr_content = file.read()

    # Regex to find the map info
    map_info_match = re.search(r"map info = \{(.*?)\}", hdr_content)
    if map_info_match:
        map_info = map_info_match.group(1).split(',')
        x_origin = float(map_info[3].strip())
        y_origin = float(map_info[4].strip())
        pixel_size_x = float(map_info[5].strip())
        pixel_size_y = float(map_info[6].strip())
        zone = int(map_info[7].strip())
        hemisphere = map_info[8].strip().lower() + 'ern'  # converting 'north' to 'northern'
        
        # Create affine transform
        transform = Affine(pixel_size_x, 0, x_origin,
                           0, -pixel_size_y, y_origin)
        
        # Create CRS string
        crs = f"+proj=utm +zone={zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"

        # Write the data to a GeoTIFF file with NoData value set to 0
        with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=cube.shape[0], width=cube.shape[1],
            count=cube.shape[2] if len(cube.shape) == 3 else 1,  # Ensure the count is correct
            dtype=str(cube.dtype),
            crs=crs, transform=transform, nodata=0) as dst:
            if len(cube.shape) == 3:
                for i in range(cube.shape[2]):
                    dst.write(cube[:, :, i], i + 1)  # Writing each band to the GeoTIFF
            else:
                dst.write(cube, 1)  # Writing the single band to the GeoTIFF

        print(f"GeoTIFF file saved successfully: {output_path}")
    else:
        raise ValueError("Map info not found in HDR file")

import os
import re

def process_all_hypercubes(input_dir, output_dir,sensor,place):
    """
    Converts every hypercube of a directory into GeoTIFF.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Iterates over the '.bip.hdr' files of the input directory, extracts the scene number
    that follows the first underscore of the file name and writes each cube to
    '<YYYYMMDD>_<place>_<number>_<sensor>.tiff', where the date is taken from the first eight
    characters of the file name. Cubes that fail are reported and skipped.

    Parameters:
    - input_dir: str, directory containing the hypercubes.
    - output_dir: str, directory for the output GeoTIFFs.
    - sensor: str, sensor label used in the output name (e.g. 'PIL', 'PIRL').
    - place: str, study site label used in the output name.

    Returns:
    - None. The GeoTIFFs are written to disk.
    """
    # List all files in the input directory
    for file_name in os.listdir(input_dir):
        # Check if the file is a .hdr file for a hypercube
        if file_name.endswith('.bip.hdr'):
            # Construct the full path to the hypercube file
            cube_path = os.path.join(input_dir, file_name)
            
            # Extract the unique number from the filename
            match = re.search(r'_([0-9]+)', file_name)
            if match:
                number = match.group(1)
                # Construct the output file name
                output_file_name = f'{file_name[:8]}_{place}_{number}_{sensor}.tiff'
                output_path = os.path.join(output_dir, output_file_name)
                
                # Process the hypercube and save it as GeoTIFF
                try:
                    hypercube_to_geotiff(cube_path, output_path)
                except Exception as e:
                    print(f"Failed to process {cube_path}: {e}")