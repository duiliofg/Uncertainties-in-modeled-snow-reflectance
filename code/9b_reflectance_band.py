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
    os.makedirs(output_csv_folder, exist_ok=True)
    os.makedirs(output_plot_folder, exist_ok=True)

    def extract_id_from_name(filename):
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
