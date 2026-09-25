"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@student.montana.edu
- GitHub: https://github.com/duiliofg

Script Title: Publication version of the Pika IR-L comparison figure

Description:
Generates the separated-panel version of the Pika IR-L reflectance comparison figure
used in the manuscript.

Requirements:
- Python 3.x
- Libraries: matplotlib, pandas, seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_reflectance_and_differences_small_separate(
    hs_df, ms_df, output_folder, site_name, pixel_types=("snow", "mixed")
):
    """
    Plots the publication panels of reflectance and difference for the Pika IR-L sensor.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@student.montana.edu
    - GitHub: https://github.com/duiliofg

    Description:
    Combines the hyperspectral and Landsat per-cell values (band B6), computes the difference
    as Landsat minus hyperspectral reflectance and, for each requested pixel type, saves a
    reflectance box plot and a difference box plot as separate 4,44 x 4,44 in figures at 300 DPI
    ('<site_name>_reflectance_<pixel_type>.png' and '<site_name>_difference_<pixel_type>.png').

    Parameters:
    - hs_df: pandas.DataFrame, hyperspectral per-cell values.
    - ms_df: pandas.DataFrame, Landsat per-cell values.
    - output_folder: str, folder where the figures are saved.
    - site_name: str, study site label used as file prefix.
    - pixel_types: tuple of str, pixel types to plot. Default ('snow', 'mixed').

    Returns:
    - list of str, sorted paths of the two figures of the last pixel type processed.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Metadata
    hs_df['landsat_band'] = 6
    ms_df['landsat_band'] = 6
    hs_df['source'] = 'Hyperspectral'
    ms_df['source'] = 'Landsat'

    combined_df = pd.concat([hs_df, ms_df], ignore_index=True)
    combined_df['difference'] = combined_df['landsat_value'] - combined_df['reflectance']

    # Style
    sns.set_style('white', {'axes.linewidth': 0.8})
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams.update({
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })

    palette = {
        "Hyperspectral": "#66c2a5",
        "Landsat": "#95a3c3"
    }

    for ptype in pixel_types:
        subset = combined_df[combined_df['pixel_type'] == ptype]

        # Plot Reflectance
        fig1, ax1 = plt.subplots(figsize=(4.44, 4.44), dpi=300)
        sns.boxplot(
            ax=ax1,
            x="landsat_band",
            y="reflectance",
            hue="source",
            data=subset,
            palette=palette,
            dodge=True,
            width=0.4,
            fliersize=3,
            linewidth=1,
            boxprops=dict(edgecolor='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='black')
        )
        ax1.set_xlabel("Landsat Band")
        ax1.set_ylabel("Reflectance")
        ax1.grid(False)
        ax1.legend_.remove()
        ax1.set_aspect('equal', adjustable='box')

        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

        ax1.tick_params(axis='both', which='major', direction='out', color='black', length=6, width=1.2)
        for tick_label in ax1.get_xticklabels() + ax1.get_yticklabels():
            tick_label.set_color("black")

        path1 = os.path.join(output_folder, f"{site_name}_reflectance_{ptype}.png")
        plt.tight_layout()
        fig1.savefig(path1, bbox_inches='tight')
        plt.close(fig1)

        # Plot Difference
        fig2, ax2 = plt.subplots(figsize=(4.44, 4.44), dpi=300)
        sns.boxplot(
            ax=ax2,
            x="landsat_band",
            y="difference",
            data=subset,
            color="#B0B0B0",
            fliersize=3,
            width=0.2,
            linewidth=1,
            boxprops=dict(edgecolor='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='black')
        )
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax2.set_xlabel("Landsat Band")
        ax2.set_ylabel("Reflectance Difference")
        ax2.grid(False)
        ax2.set_aspect('equal', adjustable='box')

        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

        ax2.tick_params(axis='both', which='major', direction='out', color='black', length=6, width=1.2)
        for tick_label in ax2.get_xticklabels() + ax2.get_yticklabels():
            tick_label.set_color("black")

        path2 = os.path.join(output_folder, f"{site_name}_difference_{ptype}.png")
        plt.tight_layout()
        fig2.savefig(path2, bbox_inches='tight')
        plt.close(fig2)

    return sorted([path1, path2])

# Example usage (disabled here)
# plot_reflectance_and_differences_small_separate(hs_df, ms_df, output_folder, "CARC")

