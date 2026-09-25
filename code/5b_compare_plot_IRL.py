"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@student.montana.edu
- GitHub: https://github.com/duiliofg

Script Title: Plot reflectance and differences for the Pika IR-L sensor

Description:
Variant of the comparison figure restricted to the Pika IR-L sensor and its single
simulated band (B6), with reduced panel layouts.

Requirements:
- Python 3.x
- Libraries: matplotlib, pandas, seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from matplotlib.ticker import FormatStrFormatter

print('Loading plot_reflectance_and_differences_by_pixel_type_IRL_small_variants')

def plot_reflectance_and_differences_by_pixel_type_IRL(
    hs_pika_IRL, ms_landsat_pika_IRL,
    output_folder
):
    """
    Plots small-format reflectance and difference panels for the Pika IR-L sensor.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@student.montana.edu
    - GitHub: https://github.com/duiliofg

    Description:
    Combines the hyperspectral and Landsat per-cell values of Pika IR-L (band B6) and computes
    the difference as Landsat minus hyperspectral reflectance. For the 'snow' and 'mixed' pixel
    types, produces separate reflectance and difference panels of 4,5 cm, each in a 'normal'
    variant with y-axis label and a 'nolabel' variant without label and with the y axis on the
    right. Figures are saved at 300 DPI with transparent background as
    '<pixel_type>_<variable>_<variant>_4cm.png'.

    Parameters:
    - hs_pika_IRL: pandas.DataFrame, hyperspectral per-cell values for Pika IR-L.
    - ms_landsat_pika_IRL: pandas.DataFrame, Landsat per-cell values for Pika IR-L.
    - output_folder: str, folder where the figures are saved.

    Returns:
    - None. The figures are written to disk.
    """
    custom_palette = {
        "Hyperspectral": "#66c2a5",
        "Landsat": "#95a3c3"
    }

    # Assign metadata
    hs_pika_IRL['landsat_band'] = 6
    ms_landsat_pika_IRL['landsat_band'] = 6
    hs_pika_IRL['source'] = 'Hyperspectral'
    ms_landsat_pika_IRL['source'] = 'Landsat'

    # Combine datasets
    combined_df = pd.concat([hs_pika_IRL, ms_landsat_pika_IRL], ignore_index=True)
    combined_df['difference'] = combined_df['landsat_value'] - combined_df['reflectance']

    # Plot style
    sns.set_style('white', {'axes.linewidth': 0.8})
    plt.rcParams.update({
        'xtick.major.size': 6,
        'xtick.major.width': 1.2,
        'ytick.major.size': 6,
        'ytick.major.width': 1.2,
        'xtick.bottom': True,
        'ytick.left': True,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11
    })

    # 4 cm effective width
    width_effective = 4.5 / 2.54
    pad_inches = 0.01
    figsize = (width_effective, width_effective)

    for pixel_type in combined_df['pixel_type'].unique():
        if pixel_type not in ["snow", "mixed"]:
            continue

        subset = combined_df[combined_df['pixel_type'] == pixel_type]
        for variant in ['normal', 'nolabel']:
            for var_type in ['reflectance', 'difference']:
                fig, ax = plt.subplots(figsize=figsize, dpi=300)

                if var_type == 'reflectance':
                    sns.boxplot(
                        ax=ax,
                        x="landsat_band",
                        y="reflectance",
                        hue="source",
                        data=subset,
                        palette=custom_palette,
                        dodge=True,
                        width=0.4,
                        fliersize=3,
                        linewidth=1,
                        boxprops=dict(edgecolor='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        medianprops=dict(color='black')
                    )
                    ax.set_ylabel("Reflectance" if variant == 'normal' else "")
                else:
                    sns.boxplot(
                        ax=ax,
                        x="landsat_band",
                        y="difference",
                        data=subset,
                        color="#f2f5fa",
                        width=0.2,
                        fliersize=3,
                        linewidth=1,
                        boxprops=dict(edgecolor='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        medianprops=dict(color='black')
                    )
                    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
                    ax.set_ylabel("Reflectance Difference" if variant == 'normal' else "")

                ax.set_xlabel("")  # REMOVE xlabel
                ax.grid(False)

                # Right Y-axis if 'nolabel'
                if variant == 'nolabel':
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")

                if ax.get_legend():
                    ax.legend_.remove()

                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("black")
                    spine.set_linewidth(1)

                ax.tick_params(
                    axis='both',
                    which='major',
                    direction='out',
                    color='black',
                    length=6,
                    width=1.2
                )

                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_color("black")

                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Y to 2 decimals

                ax.set_box_aspect(1)

                fname = f"{pixel_type}_{var_type}_{variant}_4cm.png".replace(" ", "_")
                fpath = os.path.join(output_folder, fname)
                plt.savefig(fpath, bbox_inches='tight', pad_inches=pad_inches, transparent=True)
                plt.close()
                print(f"✅ Saved: {fpath}")
