"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@anori.cl
- GitHub: https://github.com/duiliofg

Script Title: Per-band reflectance statistics by surface type

Description:
Computes reflectance statistics band by band and by classified surface type on the
RSR-convolved products. Values are restricted to the physical range (0, 1] and an
interquartile filter (Q25 - 1.5 IQR, Q75 + 1.5 IQR) is applied before computing the mean
and standard deviation. Note that Q25 and Q75 are stored from the unfiltered data, whereas
mean, std, min, max and Q50 are computed on the filtered subset.

Also produces the summary box plot of reflectance by band and surface type.

Requirements:
- Python 3.x
- Libraries: geopandas, matplotlib, numpy, pandas, rioxarray, seaborn, shapely, tqdm
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import rioxarray as rxr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import mapping
from tqdm import tqdm

print('[INFO] Loaded compute_reflectance_statistics_rsr')

color_map_custom = {
    "Snow": "#0072B2",
    "Vegetation": "#009E73",
    "Shadows": "#E69F00"
}

def compute_reflectance_statistics_rsr(
    reflectance_folder,
    gpkg_folder,
    output_csv_folder,
    output_plot_folder,
    summary_csv_path,
    sensor,
    target_classes={"Snow": "#0072B2", "Vegetation": "#009E73", "Shadows": "#E69F00"}
):
    """
    Computes per-band reflectance statistics by surface type on the simulated Landsat bands.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Pairs each RSR-convolved GeoTIFF with the classified GeoPackage that shares its numeric
    identifier. For each surface type present in the 'type' column, clips the raster to the
    class polygons (all_touched=True) and, band by band, keeps values in (0, 1] and applies an
    interquartile filter (Q25 - 1,5 IQR, Q75 + 1,5 IQR). Mean, std, min, max and Q50 are
    computed on the filtered values, whereas Q25 and Q75 are taken from the unfiltered values.
    Writes one CSV per file, a global summary CSV and 'summary_box_plot.png' with the box plots
    of reflectance by band and surface type.

    Parameters:
    - reflectance_folder: str, folder with the RSR-convolved reflectance GeoTIFFs.
    - gpkg_folder: str, folder with the classified GeoPackages.
    - output_csv_folder: str, folder for the per-file statistics CSVs.
    - output_plot_folder: str, folder for the figures.
    - summary_csv_path: str, path of the global summary CSV.
    - sensor: str, 'pika irl' (single band, labelled B6) or 'pika l' (all bands, labelled B1
      to Bn).
    - target_classes: dict, mapping from class name to colour (hex). Default Snow, Vegetation
      and Shadows.

    Returns:
    - df_stats: pandas.DataFrame, statistics by file, band and class.
    - df_values: pandas.DataFrame, every filtered pixel value with the columns
      'Reflectance (%)', 'band', 'class' and 'file'.

    Raises:
    - ValueError: if the sensor is neither 'pika irl' nor 'pika l'.
    """
    os.makedirs(output_csv_folder, exist_ok=True)
    os.makedirs(output_plot_folder, exist_ok=True)

    def extract_id_from_name(filename):
        """
        Extracts the numeric identifier of a file name.

        Author:
        - Duilio Fonseca-Gallardo
        - Montana State University
        - Email: duilio.fonseca@anori.cl
        - GitHub: https://github.com/duiliofg

        Description:
        Returns the first run of digits that follows an underscore, used to pair each
        reflectance raster with its classified GeoPackage.

        Parameters:
        - filename: str, file name without extension.

        Returns:
        - str or None, the identifier, or None if no match is found.
        """
        match = re.search(r'_(\d+)', filename)
        return match.group(1) if match else None

    all_stats = []
    violin_data_all = []

    tiff_paths = glob.glob(os.path.join(reflectance_folder, "*.tif")) + \
                 glob.glob(os.path.join(reflectance_folder, "*.tiff"))
    gpkg_paths = glob.glob(os.path.join(gpkg_folder, "*.gpkg"))

    for tiff_path in tqdm(tiff_paths, desc="Processing reflectance files"):
        base_name = os.path.splitext(os.path.basename(tiff_path))[0]
        id_number = extract_id_from_name(base_name)

        if not id_number:
            continue

        matching_gpkgs = [p for p in gpkg_paths if f"_{id_number}" in os.path.basename(p)]
        if not matching_gpkgs:
            continue
        gpkg_path = matching_gpkgs[0]

        try:
            raster = rxr.open_rasterio(tiff_path, masked=True)
            gdf = gpd.read_file(gpkg_path).to_crs(raster.rio.crs)
        except Exception:
            continue

        if sensor.lower() == "pika irl":
            band_indices = [0]
            band_numbers = [6]
        elif sensor.lower() == "pika l":
            band_indices = list(range(raster.shape[0]))
            band_numbers = list(range(1, raster.shape[0] + 1))
        else:
            raise ValueError(f"Unknown sensor type: {sensor}")

        file_stats = []
        violin_data = []

        present_classes = set(gdf['type'].unique())
        valid_classes = [cls for cls in target_classes if cls in present_classes]

        for cover_class in valid_classes:
            class_geom = gdf[gdf['type'] == cover_class]
            if class_geom.empty:
                continue

            try:
                clipped = raster.rio.clip(class_geom.geometry.apply(mapping), all_touched=True, drop=False)
                mask = clipped.data
            except Exception:
                continue

            for band_idx, landsat_band in zip(band_indices, band_numbers):
                band_data = mask[band_idx]
                valid_data = band_data[~np.isnan(band_data)]
                valid_data = valid_data[(valid_data != 0) & (valid_data <= 1.0)]

                if valid_data.size == 0:
                    continue

                q25 = np.percentile(valid_data, 25)
                q75 = np.percentile(valid_data, 75)
                iqr = q75 - q25
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                filtered = valid_data[(valid_data >= lower) & (valid_data <= upper)]

                if filtered.size == 0:
                    continue

                stats = {
                    "file": base_name,
                    "band": landsat_band,
                    "class": cover_class,
                    "mean": np.mean(filtered),
                    "std": np.std(filtered),
                    "min": np.min(filtered),
                    "max": np.max(filtered),
                    "Q25": q25,
                    "Q50": np.percentile(filtered, 50),
                    "Q75": q75
                }

                file_stats.append(stats)
                all_stats.append(stats)

                for val in filtered:
                    violin_data_all.append({
                        "Reflectance (%)": val,
                        "band": f"B{landsat_band}",
                        "class": cover_class,
                        "file": base_name
                    })

        if file_stats:
            df = pd.DataFrame(file_stats)
            csv_path = os.path.join(output_csv_folder, f"{base_name}_stats.csv")
            df.to_csv(csv_path, index=False)

    if all_stats:
        df_all = pd.DataFrame(all_stats)
        df_all.to_csv(summary_csv_path, index=False)

        if violin_data_all:
            df_violin_all = pd.DataFrame(violin_data_all)
            band_order = sorted(df_violin_all["band"].unique(), key=lambda x: int(x[1:]))
            class_order = list(color_map_custom.keys())

            fig, ax = plt.subplots(figsize=(8, 6))

            width = 0.2
            offsets = {
                "Snow": -width,
                "Vegetation": 0,
                "Shadows": width
            }

            for cls in class_order:
                data_cls = df_violin_all[df_violin_all["class"] == cls]
                positions = [i + offsets[cls] for i in range(len(band_order))]
                sns.boxplot(
                    data=data_cls,
                    x="band",
                    y="Reflectance (%)",
                    order=band_order,
                    ax=ax,
                    positions=positions,
                    width=width * 0.4,
                    color=color_map_custom[cls],
                    boxprops=dict(edgecolor='black', linewidth=1.2),
                    whiskerprops=dict(color='black', linewidth=1.2),
                    capprops=dict(color='black', linewidth=1.2),
                    flierprops=dict(markerfacecolor=color_map_custom[cls], markeredgecolor='black',
                                    marker='o', alpha=0.5, markersize=3),
                    medianprops=dict(color='black', linewidth=1.5)
                )

            ax.set_xticks(range(len(band_order)))
            ax.set_xticklabels(band_order, fontsize=11)
            ax.set_ylabel("Reflectance", fontsize=12)
            ax.set_xlabel("Simulated Band", fontsize=12)

            y_max = df_violin_all["Reflectance (%)"].max()
            y_max = min(np.ceil(y_max * 10) / 10, 1.0)
            ax.set_ylim(0, y_max)

            handles = [plt.Line2D([], [], color=color_map_custom[cls], label=cls) for cls in class_order]
            ax.legend(handles=handles, title="Surface Type", fontsize=10, title_fontsize=11)

            plt.tight_layout()
            plot_path = os.path.join(output_plot_folder, "summary_box_plot.png")
            plt.savefig(plot_path, dpi=300)
            plt.show()

    return pd.DataFrame(all_stats), pd.DataFrame(violin_data_all)
