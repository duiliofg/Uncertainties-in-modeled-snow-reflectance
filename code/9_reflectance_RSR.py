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
from matplotlib.ticker import MaxNLocator

print('[INFO] Loaded compute_reflectance_statistics_rsr')

color_map_custom = {
    "Snow": "#0072B2",        # Azul
    "Vegetation": "#009E73", # Verde
    "Shadows": "#E69F00"      # Naranja
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

    # List files
    tiff_paths = glob.glob(os.path.join(reflectance_folder, "*.tif")) + \
                 glob.glob(os.path.join(reflectance_folder, "*.tiff"))
    gpkg_paths = glob.glob(os.path.join(gpkg_folder, "*.gpkg"))

    print(f"[INFO] Found {len(tiff_paths)} TIFFs and {len(gpkg_paths)} GPKGs")

    for tiff_path in tqdm(tiff_paths, desc="Processing reflectance files"):
        base_name = os.path.splitext(os.path.basename(tiff_path))[0]
        id_number = extract_id_from_name(base_name)
        print(f"\n[DEBUG] Processing: {base_name} | ID: {id_number}")

        if not id_number:
            print(f"[WARNING] No ID found in {base_name}, skipping.")
            continue

        matching_gpkgs = [p for p in gpkg_paths if f"_{id_number}" in os.path.basename(p)]
        if not matching_gpkgs:
            print(f"[WARNING] No matching GPKG for {base_name}")
            continue
        gpkg_path = matching_gpkgs[0]

        try:
            raster = rxr.open_rasterio(tiff_path, masked=True)
            gdf = gpd.read_file(gpkg_path).to_crs(raster.rio.crs)
        except Exception as e:
            print(f"[ERROR] Reading raster/GPKG failed for {base_name}: {e}")
            continue

        print(f"[INFO] Raster shape: {raster.shape}")

        if sensor.lower() == "pika irl":
            band_indices = [0]
            band_numbers = [6]
        elif sensor.lower() == "pika l":
            band_indices = list(range(raster.shape[0]))
            band_numbers = list(range(1, raster.shape[0] + 1))
        else:
            raise ValueError(f"[FATAL] Unknown sensor type: {sensor}")

        file_stats = []
        violin_data = []

        present_classes = set(gdf['type'].unique())
        valid_classes = [cls for cls in target_classes if cls in present_classes]

        print(f"[INFO] Valid classes in {base_name}: {valid_classes}")

        for cover_class in valid_classes:
            class_geom = gdf[gdf['type'] == cover_class]
            if class_geom.empty:
                print(f"[DEBUG] No geometry found for {cover_class}")
                continue

            try:
                clipped = raster.rio.clip(class_geom.geometry.apply(mapping), all_touched=True, drop=False)
                mask = clipped.data
                print(f"[DEBUG] Clipped data shape for {cover_class}: {mask.shape}")
            except Exception as e:
                print(f"[ERROR] Clipping failed for {cover_class} in {base_name}: {e}")
                continue

            for band_idx, landsat_band in zip(band_indices, band_numbers):
                band_data = mask[band_idx]
                valid_data = band_data[~np.isnan(band_data)]
                valid_data = valid_data[valid_data != 0]

                if valid_data.size == 0:
                    print(f"[DEBUG] No valid data for {cover_class}, Band {landsat_band}")
                    continue

                q25 = np.percentile(valid_data, 25)
                q75 = np.percentile(valid_data, 75)
                iqr = q75 - q25
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                filtered = valid_data[(valid_data >= lower) & (valid_data <= upper)]

                if filtered.size == 0:
                    print(f"[DEBUG] All data filtered out as outliers for {cover_class}, Band {landsat_band}")
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

                print(f"[DEBUG] Stats computed for {cover_class}, Band {landsat_band}")
                file_stats.append(stats)
                all_stats.append(stats)

                for val in filtered:
                    violin_data.append({
                        "Reflectance (%)": val * 100,
                        "band": f"B{landsat_band}",
                        "class": cover_class,
                        "file": base_name
                    })
                    violin_data_all.append({
                        "Reflectance (%)": val * 100,
                        "band": f"B{landsat_band}",
                        "class": cover_class,
                        "file": base_name
                    })

        if violin_data:
            df_violin = pd.DataFrame(violin_data)
            plt.figure(figsize=(6, 4))
            sns.violinplot(
                data=df_violin,
                x="class",
                y="Reflectance (%)",
                hue="band",
                palette="Set2",
                inner="quartile"
            )
            plt.title(f"Reflectance per Surface Type\n{base_name}")
            plt.xlabel("Surface Type")
            plt.ylabel("Reflectance (%)")
            plt.legend(title="Band", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            violin_path = os.path.join(output_plot_folder, f"{base_name}_violin_plot.png")
            plt.savefig(violin_path, dpi=300)
            plt.close()
            print(f"[INFO] Violin plot saved: {violin_path}")

        if file_stats:
            df = pd.DataFrame(file_stats)
            csv_path = os.path.join(output_csv_folder, f"{base_name}_stats.csv")
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Stats saved: {csv_path}")

    # Final summary
    
    
    if all_stats:
        df_all = pd.DataFrame(all_stats)
        df_all.to_csv(summary_csv_path, index=False)
        print(f"\n✅ Summary saved: {summary_csv_path}")

        if violin_data_all:
            df_violin_all = pd.DataFrame(violin_data_all)
            plt.figure(figsize=(9, 6))
            sns.violinplot(
                data=df_violin_all,
                x="class",
                y="Reflectance (%)",
                hue="class",
                palette=color_map_custom,
                inner="quartile"
            )
            #plt.title("Reflectance Summary Across All Images")
            plt.xlabel("Surface Type")
            plt.ylabel("Reflectance (%)")
            plt.legend(title="Band", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path = os.path.join(output_plot_folder, "summary_violin_plot.png")
            plt.savefig(plot_path, dpi=300)
            plt.show()
            print(f"[INFO] Summary violin plot saved: {plot_path}")
    else:
        print("\n⚠️ No statistics were generated.")
