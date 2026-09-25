"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@anori.cl
- GitHub: https://github.com/duiliofg

Script Title: Summary tables of reflectance differences by band, simulation type and pixel type

Description:
Reads the per-cell reflectance difference tables (Sec3.2_*.csv) and aggregates them by
band, simulation type (RSR, AVG) and pixel type into the summary tables
Final_Reflectance_Summary_with_Correct_Error_Propagation_<site>.csv.
Inputs stored in percent are converted to fraction, and inputs with the difference
defined as hyperspectral minus Landsat are converted to Landsat minus hyperspectral,
so that all summary tables share the same format.

Requirements:
- Python 3.x
- Libraries: numpy, pandas
"""

import os
import numpy as np
import pandas as pd

# === CONFIGURATION ===
input_folder = "."
output_folder = "."

SITES = {
    "CARC": {
        "files": ["Sec3.2_CARC_PIKAIRL_reflectance_difference_LANDSAT8.csv"],
        "class_column": "pixel_type",
    },
    "ONION": {
        "files": ["Sec3.2_Onion_PIKAL_reflectance_difference_LANDSAT8.csv",
                  "Sec3.2_Onion_PIKAIRL_reflectance_difference_LANDSAT8.csv"],
        "class_column": "class",
    },
    "Sodankyla": {
        "files": ["Sec3.2_Sodankyla_PIKAIRL_reflectance_difference_LANDSAT9.csv"],
        "class_column": "pixel_type",
        # Input in percent and difference as hyperspectral minus Landsat
        "percent_to_fraction": True,
        "flip_difference_sign": True,
    },
}


def summarize_reflectance_differences(df, class_column="pixel_type"):
    """
    Aggregates per-cell reflectance differences into a summary table.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@anori.cl
    - GitHub: https://github.com/duiliofg

    Description:
    Groups the per-cell records by band, simulation type and pixel type. Means are
    computed for the hyperspectral reflectance, the Landsat reflectance and the
    difference. The hyperspectral uncertainty and the reflectance error are aggregated
    as the square root of the mean of their squares, the Landsat dispersion as the
    population standard deviation, and the uncertainty of the difference as the square
    root of the sum of the squared hyperspectral uncertainty and reflectance error.

    Parameters:
    - df: pandas.DataFrame, per-cell table with the columns 'band', 'simulation_type',
      'pixel_type', 'mean_hyper', 'std_hyper', 'landsat_value', 'difference' and
      'reflectance_error'.
    - class_column: str, name given to the pixel type column in the output.

    Returns:
    - pandas.DataFrame, summary table with one row per band, simulation type and pixel type.
    """
    rows = []
    for (band, sim_type, pixel_type), g in df.groupby(["band", "simulation_type", "pixel_type"]):
        std_hyper = np.sqrt(np.mean(g["std_hyper"] ** 2))
        refl_error = np.sqrt(np.mean(g["reflectance_error"] ** 2))
        rows.append({
            "band": band,
            "simulation_type": sim_type,
            class_column: pixel_type,
            "mean_hyper_mean": g["mean_hyper"].mean(),
            "error_propagation_std_hyper": std_hyper,
            "mean_difference": g["difference"].mean(),
            "reflectance_error_mean": refl_error,
            "mean_landsat": g["landsat_value"].mean(),
            "std_landsat": g["landsat_value"].std(ddof=0),
            "error_propagation_difference": np.sqrt(std_hyper ** 2 + refl_error ** 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    for site, cfg in SITES.items():
        df = pd.concat([pd.read_csv(os.path.join(input_folder, f)) for f in cfg["files"]],
                       ignore_index=True)
        if cfg.get("percent_to_fraction"):
            value_cols = ["mean_hyper", "median_hyper", "min_hyper", "max_hyper", "std_hyper",
                          "landsat_value", "difference", "reflectance_error", "rmse",
                          "propagated_error"]
            df[value_cols] = df[value_cols] / 100.0
        if cfg.get("flip_difference_sign"):
            df["difference"] = -df["difference"]
        summary = summarize_reflectance_differences(df, cfg["class_column"])
        out_path = os.path.join(output_folder,
                                f"Final_Reflectance_Summary_with_Correct_Error_Propagation_{site}.csv")
        summary.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
