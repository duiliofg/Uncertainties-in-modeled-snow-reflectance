"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@anori.cl
- GitHub: https://github.com/duiliofg

Script Title: Compare hyperspectral-derived and observed Landsat reflectance

Description:
For every cell of the Landsat-equivalent grid, extracts the hyperspectral pixels falling
inside the cell and the corresponding Landsat Level-2 surface reflectance value, and
computes per-cell statistics, the reflectance difference, RMSE and the propagated error.
Also exports difference GeoTIFFs and the comparison plots.

Produces the per-cell tables that feed the appendix tables of the manuscript.

Requirements:
- Python 3.x
- Libraries: geopandas, matplotlib, numpy, pandas, rasterio, seaborn, shapely
"""

print('Loading multiband_reflectance_differences')

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def plot_reflectance_comparison(df_violin, df_landsat, output_folder):
    """
    Plots hyperspectral-derived and Landsat reflectance by band for each pixel type.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Keeps the 'snow' and 'mixed' pixel types and draws, for each of them, a box plot of
    reflectance by Landsat band grouped by source (Landsat, Hyperspectral-AVG and
    Hyperspectral-RSR). Figures are saved at 300 DPI with transparent background as
    'reflectance_comparison_<pixel_type>.png'.

    Parameters:
    - df_violin: pandas.DataFrame, per-cell hyperspectral values with the columns
      'pixel_type', 'reflectance', 'landsat_band' and 'source'.
    - df_landsat: pandas.DataFrame, per-cell Landsat values with the same columns.
    - output_folder: str, folder where the figures are saved.

    Returns:
    - None. The figures are written to disk and displayed.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Filter only allowed pixel types
    allowed_types = ["snow", "mixed"]
    df_violin = df_violin[df_violin["pixel_type"].isin(allowed_types)]
    df_landsat = df_landsat[df_landsat["pixel_type"].isin(allowed_types)]

    combined_df = pd.concat([df_violin, df_landsat], ignore_index=True)

    # Style and aesthetics
    sns.set_style("white", {'axes.linewidth': 0.8})
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams.update({
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11
    })

    # Color palette
    palette = {
        'Landsat': '#95a3c3',
        'Hyperspectral-AVG': '#66c2a5',
        'Hyperspectral-RSR': '#fc8d62'
    }

    for pixel_type in allowed_types:
        df_pixel = combined_df[combined_df["pixel_type"] == pixel_type]

        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

        sns.boxplot(
            ax=ax,
            x="landsat_band",
            y="reflectance",
            hue="source",
            data=df_pixel,
            palette=palette,
            dodge=True,
            width=0.5,
            fliersize=3,
            linewidth=1,
            boxprops=dict(edgecolor='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='black')
        )

        ax.set_xlabel("Landsat Band")
        ax.set_ylabel("Reflectance")
        ax.grid(False)

        # Format
        ax.tick_params(axis='both', direction='out', length=6, width=1.2, color='black')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_color("black")

        # Maintain 1:1 aspect ratio
        ax.set_box_aspect(1)

        # Legend outside without affecting ratio
        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.legend_.remove()  
        fig.legend(
            handles, labels,
            title="Source",
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            frameon=False
        )

        # Save with transparent background
        filename = f"reflectance_comparison_{pixel_type}.png"
        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path, bbox_inches='tight', transparent=True)
        plt.show()
        print(f"✅ Saved: {output_path}")


def export_difference_geotiff_cropped(masked_hyper, landsat_values, meta, output_path):
    """
    Exports the per-pixel difference between hyperspectral and Landsat reflectance.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Subtracts, band by band, the Landsat cell mean from every hyperspectral pixel of the cell
    and writes the result as a Float32 GeoTIFF. Pixels that are NaN in the hyperspectral data
    are set to 0 (NoData).

    Parameters:
    - masked_hyper: numpy.ndarray, hyperspectral reflectance cropped to the cell, of shape
      (bands, rows, cols).
    - landsat_values: sequence of float, Landsat mean reflectance of the cell for each band.
    - meta: dict, rasterio profile of the output; 'count', 'dtype' and 'nodata' are updated.
    - output_path: str, path of the output GeoTIFF.

    Returns:
    - None. The GeoTIFF is written to disk.
    """
    difference = np.zeros_like(masked_hyper, dtype=np.float32)

    for i in range(masked_hyper.shape[0]):
        hyper_band = masked_hyper[i]
        landsat_val = landsat_values[i]
        band_diff = hyper_band - landsat_val
        band_diff[np.isnan(hyper_band)] = 0
        difference[i] = band_diff

    meta.update({
        "count": masked_hyper.shape[0],
        "dtype": "float32",
        "nodata": 0
    })

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(difference.astype(np.float32))


def calculate_multiband_reflectance_differences(
    gpkg_path,
    geotiff_folder,
    landsat_path,
    output_folder,
    study_site,
    sensor,
    landsat_image,
    landsat_version,  # New parameter
    valid_pixel_tolerance=60
):
    """
    Compares hyperspectral-derived and Landsat reflectance on a Landsat-equivalent grid.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    For every RSR and AVG product of the selected Landsat version and every grid cell that
    intersects it, crops the hyperspectral raster to the cell, masks values outside (0, 1] and
    skips cells whose valid-pixel percentage is below valid_pixel_tolerance. The Landsat mean of
    the cell is computed for each band (values <= 0 masked). For each band, the statistics of
    the hyperspectral pixels are computed together with the difference (Landsat minus
    hyperspectral pixel), its mean and standard deviation (reflectance_error), the RMSE and the
    propagated error sqrt(std_hyper**2 + reflectance_error**2). Results are written to
    'reflectance_stats_by_grid_L<landsat_version>.csv', a difference GeoTIFF is exported per cell
    and file, and the comparison figures are produced with plot_reflectance_comparison.

    Parameters:
    - gpkg_path: str, GeoPackage of the grid with the columns 'id' and 'type' (pixel type).
    - geotiff_folder: str, root folder with the 'landsat<version>' and 'avg' subfolders
      produced by the RSR simulation.
    - landsat_path: str, Landsat GeoTIFF.
    - output_folder: str, folder for the tables, difference rasters and figures.
    - study_site: str, study site label stored in the table.
    - sensor: str, 'pika irl' compares product band 1 with Landsat B6; any other value
      compares the first five bands with Landsat B1 to B5.
    - landsat_image: str, identifier of the Landsat scene stored in the table.
    - landsat_version: int or str, Landsat mission number (8 or 9).
    - valid_pixel_tolerance: float, minimum percentage of valid hyperspectral pixels in a cell.
      Default 60.

    Returns:
    - df_violin: pandas.DataFrame, per-cell mean hyperspectral reflectance with the columns
      'pixel_type', 'reflectance', 'landsat_band', 'landsat_value' and 'source'.
    - df_landsat: pandas.DataFrame, per-cell Landsat reflectance with the columns
      'pixel_type', 'reflectance', 'landsat_band' and 'source' ('Landsat').
    """
    os.makedirs(output_folder, exist_ok=True)

    gdf = gpd.read_file(gpkg_path)
    gdf['geometry'] = gdf['geometry'].apply(lambda x: x if x.is_valid else x.buffer(0))
    gdf.set_index(gdf.id, inplace=True)

    # Correct directories based on Landsat version
    rsr_dir = os.path.join(geotiff_folder, f"landsat{landsat_version}")
    avg_dir = os.path.join(geotiff_folder, "avg")

    rsr_files = glob.glob(os.path.join(rsr_dir, f"*Landsat{landsat_version}*RSR*.tif"))
    avg_files = glob.glob(os.path.join(avg_dir, f"*Landsat{landsat_version}*AVG*.tif"))
    all_files = [(fp, 'RSR') for fp in rsr_files] + [(fp, 'AVG') for fp in avg_files]

    violin_data, summary_stats, landsat_points = [], [], []

    for geotiff_path, hyper_source in all_files:
        with rasterio.open(geotiff_path) as geotiff_src:
            for index, row in gdf.iterrows():
                geom = [row.geometry]
                if not row.geometry.intersects(box(*geotiff_src.bounds)):
                    continue

                try:
                    masked_hyper, hyper_transform = mask(
                        geotiff_src, geom, crop=True, filled=True, nodata=0
                    )
                    masked_hyper = masked_hyper.astype(np.float32)
                    masked_hyper = np.where((masked_hyper <= 0) | (masked_hyper > 1.0), np.nan, masked_hyper)


                    valid_pixels = np.count_nonzero(~np.isnan(masked_hyper))
                    total_pixels = np.prod(masked_hyper.shape[1:])
                    if valid_pixels / total_pixels * 100 < valid_pixel_tolerance:
                        continue

                    if sensor.strip().lower() == "pika irl":
                        hyper_band_indices = [0]
                        landsat_band_indices = [5]
                    else:
                        hyper_band_indices = list(range(min(5, masked_hyper.shape[0])))
                        landsat_band_indices = hyper_band_indices

                    masked_hyper_selected = masked_hyper[hyper_band_indices]

                    with rasterio.open(landsat_path) as landsat_src:
                        landsat_crop, _ = mask(landsat_src, geom, crop=True, filled=True, nodata=0)
                        landsat_crop = landsat_crop.astype(np.float32)
                        landsat_crop = np.where(landsat_crop <= 0, np.nan, landsat_crop)
                        landsat_means = [np.nanmean(landsat_crop[b]) for b in landsat_band_indices]

                    pixel_type = row['type']

                    for i in range(len(hyper_band_indices)):
                        hyper_band = masked_hyper_selected[i]
                        landsat_value = landsat_means[i]
                        band_number = landsat_band_indices[i] + 1

                        valid_mask = (~np.isnan(hyper_band)) & (hyper_band != 0)
                        valid_hyper = hyper_band[valid_mask]

                        if valid_hyper.size == 0:
                            continue

                        difference =  landsat_value - valid_hyper
                        mean_hyper = np.nanmean(valid_hyper)
                        median_hyper = np.nanmedian(valid_hyper)
                        min_hyper = np.nanmin(valid_hyper)
                        max_hyper = np.nanmax(valid_hyper)
                        std_hyper = np.nanstd(valid_hyper)
                        reflectance_error = np.nanstd(difference)
                        rmse = np.sqrt(np.nanmean(difference ** 2))
                        propagated_error = np.sqrt(std_hyper ** 2 + reflectance_error ** 2)

                        summary_stats.append({
                            "study_site": study_site,
                            "sensor": sensor,
                            "landsat_image": landsat_image,
                            "landsat_version": landsat_version,
                            "grid_id": index,
                            "band": band_number,
                            "pixel_type": pixel_type,
                            "mean_hyper": mean_hyper,
                            "median_hyper": median_hyper,
                            "min_hyper": min_hyper,
                            "max_hyper": max_hyper,
                            "std_hyper": std_hyper,
                            "landsat_value": landsat_value,
                            "difference": np.nanmean(difference),
                            "reflectance_error": reflectance_error,
                            "rmse": rmse,
                            "propagated_error": propagated_error,
                            "simulation_type": hyper_source
                        })

                        violin_data.append({
                            "pixel_type": pixel_type,
                            "reflectance": np.nanmean(valid_hyper),
                            "landsat_band": band_number,
                            "landsat_value": landsat_value,
                            "source": f"Hyperspectral-{hyper_source}"
                        })

                        landsat_points.append({
                            "pixel_type": pixel_type,
                            "reflectance": landsat_value,
                            "landsat_band": band_number
                        })

                    output_diff_path = os.path.join(
                        output_folder,
                        f"{os.path.basename(geotiff_path).replace('.tif','')}_grid_{index}_difference.tif"
                    )
                    export_difference_geotiff_cropped(
                        masked_hyper=masked_hyper_selected,
                        landsat_values=landsat_means,
                        meta={
                            'driver': 'GTiff',
                            'height': masked_hyper.shape[1],
                            'width': masked_hyper.shape[2],
                            'count': len(hyper_band_indices),
                            'dtype': 'float32',
                            'crs': geotiff_src.crs,
                            'transform': hyper_transform,
                            'nodata': 0
                        },
                        output_path=output_diff_path
                    )

                except Exception as e:
                    print(f"Error in grid {index}, file {geotiff_path}: {e}")
                    continue

    df_summary = pd.DataFrame(summary_stats)
    csv_path = os.path.join(output_folder, f"reflectance_stats_by_grid_L{landsat_version}.csv")
    df_summary.to_csv(csv_path, index=False)

    df_violin = pd.DataFrame(violin_data)
    df_landsat = pd.DataFrame(landsat_points)
    df_landsat['source'] = 'Landsat'

    plot_reflectance_comparison(df_violin, df_landsat, output_folder)

    return df_violin, df_landsat



