import os
import glob
import re
import rasterio
import rasterio.features
import geopandas as gpd
import pandas as pd
import joblib
import numpy as np
import rioxarray as rxr
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import mapping
from tqdm import tqdm

print('Loading compute_reflectance_statistics')

def compute_reflectance_statistics(
    reflectance_folder,
    gpkg_folder,
    output_csv_folder,
    output_plot_folder,
    summary_csv_path,
    target_classes={"Snow": "#1f77b4", "Vegetation": "#2ca02c", "Shadows": "#d62728"}
):
    os.makedirs(output_csv_folder, exist_ok=True)
    os.makedirs(output_plot_folder, exist_ok=True)

    def extract_id_from_name(filename):
        match = re.search(r'_(\d+)', filename)
        return match.group(1) if match else None

    all_stats = []
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
            raster = rxr.open_rasterio(tiff_path, masked=True).squeeze()
            gdf = gpd.read_file(gpkg_path).to_crs(raster.rio.crs)
        except Exception as e:
            print(f"[WARNING] Could not read raster or GPKG for {base_name}: {e}")
            continue

        bands = raster.shape[0]
        file_stats = []
        class_data_dict = {}
        present_classes = set(gdf['type'].unique())
        valid_classes = [cls for cls in target_classes if cls in present_classes]

        for cover_class in valid_classes:
            class_geom = gdf[gdf['type'] == cover_class]
            if class_geom.empty:
                continue
            try:
                mask = raster.rio.clip(class_geom.geometry.apply(mapping), all_touched=True, drop=False).data
            except Exception as e:
                print(f"[WARNING] Clipping failed for {cover_class} in {base_name}: {e}")
                continue

            class_band = []
            class_mean = []
            class_std = []

            for band_idx in range(bands):
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
                    "band": band_idx + 1,
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
                class_band.append(band_idx + 1)
                class_mean.append(stats["mean"] )
                class_std.append(stats["std"])

            if class_band:
                class_data_dict[cover_class] = {
                    "band": class_band,
                    "mean": class_mean,
                    "std": class_std
                }

        # Save CSV and plot per file
        if file_stats:
            df = pd.DataFrame(file_stats)
            df.to_csv(os.path.join(output_csv_folder, f"{base_name}_stats.csv"), index=False)

            fig, ax = plt.subplots(figsize=(5, 3.5))
            for cls, data in class_data_dict.items():
                color = target_classes[cls]
                bands_arr = np.array(data["band"])
                means = np.array(data["mean"])
                stds = np.array(data["std"])
                if bands_arr.size == means.size == stds.size:
                    ax.plot(bands_arr, means, label=cls, color=color, linewidth=1.5)
                    ax.fill_between(bands_arr, means - stds, means + stds, alpha=0.25, color=color, linewidth=0)

            xtick_step = 25
            xticks = bands_arr[::xtick_step] if len(bands_arr) > xtick_step else bands_arr
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, rotation=0, fontsize=9)

            ax.set_xlabel("Spectral Band", fontsize=10)
            ax.set_ylabel("Reflectance (%)", fontsize=10)
            ax.tick_params(axis='both', labelsize=9)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.legend(fontsize=8, frameon=False, loc='upper right')
            plt.tight_layout(pad=0.5)
            plt.savefig(os.path.join(output_plot_folder, f"{base_name}_mean_std_plot.png"), dpi=300)
            plt.close()

    # Save global summary
    if all_stats:
        df_all = pd.DataFrame(all_stats)
        df_all.to_csv(summary_csv_path, index=False)
        print(f"\n✅ Summary saved: {summary_csv_path}")

        fig, ax = plt.subplots(figsize=(5.5, 4))
        for cls, color in target_classes.items():
            df_cls = df_all[df_all['class'] == cls]
            if df_cls.empty:
                continue

            grouped = df_cls.groupby("band").agg({
                "mean": lambda x: np.mean(x) ,
                "std": lambda x: np.mean(x)
            }).reset_index()

            bands = grouped["band"].values
            means = grouped["mean"].values
            stds = grouped["std"].values

            ax.plot(bands, means, label=cls, color=color, linewidth=1.5)
            ax.fill_between(bands, means - stds, means + stds, alpha=0.25, color=color, linewidth=0)

        xtick_step = 25
        xticks = bands[::xtick_step] if len(bands) > xtick_step else bands
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=0, fontsize=9)

        ax.set_xlabel("Spectral Band", fontsize=10)
        ax.set_ylabel("Reflectance (%)", fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(output_plot_folder, "cumulative_mean_std_plot.png"), dpi=300)
        plt.show()
        print(f"✅ Cumulative plot saved.")
    else:
        print("\n⚠️ No statistics were generated.")
