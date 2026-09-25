"""
Author:
- Duilio Fonseca-Gallardo
- Montana State University
- Email: duilio.fonseca@student.montana.edu
- GitHub: https://github.com/duiliofg

Script Title: Plot reflectance and differences by classified pixel type

Description:
Builds the comparison figures between hyperspectral-derived and Landsat reflectance,
disaggregated by classified pixel type.

Requirements:
- Python 3.x
- Libraries: matplotlib, pandas, seaborn
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading plot_reflectance_and_differences_by_pixel_type")

def plot_reflectance_and_differences_by_pixel_type(
    hs_pika_L, ms_landsat_pika_L,
    hs_pika_IRL, ms_landsat_pika_IRL,
    output_folder
):
    """
    Plots reflectance and reflectance differences for Pika L and Pika IR-L by pixel type.

    Author:
    - Duilio Fonseca-Gallardo
    - Montana State University
    - Email: duilio.fonseca@student.montana.edu
    - GitHub: https://github.com/duiliofg

    Description:
    Combines the outputs of calculate_multiband_reflectance_differences for both sensors,
    assigns band 6 to the Pika IR-L records and computes the difference as Landsat minus
    hyperspectral reflectance. For each pixel type, builds a two-panel figure with the box plot
    of reflectance by band and source (left) and the box plot of the difference by band with a
    zero reference line (right). Each panel is 3,32 in (8,433 cm) wide. Figures are saved at
    300 DPI with transparent background as 'reflectance_difference_<pixel_type>.png'.

    Parameters:
    - hs_pika_L: pandas.DataFrame, hyperspectral per-cell values for Pika L.
    - ms_landsat_pika_L: pandas.DataFrame, Landsat per-cell values for Pika L.
    - hs_pika_IRL: pandas.DataFrame, hyperspectral per-cell values for Pika IR-L.
    - ms_landsat_pika_IRL: pandas.DataFrame, Landsat per-cell values for Pika IR-L.
    - output_folder: str, folder where the figures are saved.

    Returns:
    - None. The figures are written to disk and displayed.
    """
    # ----------------------- colour palette -----------------------
    palette_src = {"Hyperspectral": "#66c2a5", "Landsat": "#95a3c3"}

    # ----------------------- metadata tweaks ----------------------
    for df in (hs_pika_IRL, ms_landsat_pika_IRL):
        df["landsat_band"] = 6          # IRL data only has one band (B6)

    hs_pika_L["source"]           = "Hyperspectral"
    hs_pika_IRL["source"]         = "Hyperspectral"
    ms_landsat_pika_L["source"]   = "Landsat"
    ms_landsat_pika_IRL["source"] = "Landsat"

    # ----------------------- merge & new column -------------------
    df = pd.concat(
        [hs_pika_L, hs_pika_IRL, ms_landsat_pika_L, ms_landsat_pika_IRL],
        ignore_index=True
    )
    df["difference"] = df["landsat_value"] - df["reflectance"]

    # ----------------------- global matplotlib style --------------
    sns.set_style("white", {"axes.linewidth": .8})
    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams.update({
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "xtick.major.size": 6,
        "xtick.major.width": 1.2,
        "ytick.major.size": 6,
        "ytick.major.width": 1.2,
        "xtick.bottom": True,
        "ytick.left": True
    })

    # ------------- figure-geometry constants ----------------------
    # target data-region width = 8.433 cm = 3.32 in
    axis_w_in  = 3.32
    axis_gap   = 0.30        # inches between the two sub-plots
    side_marg  = 0.40        # left & right outer margins (inches)
    top_bot_m  = 0.60        # top & bottom margins for title / legend

    fig_w = 2 * axis_w_in + axis_gap + 2 * side_marg    # ≈ 8.04 in
    fig_h = axis_w_in + 2 * top_bot_m                   # keep some breathing room

    # ------------- nice titles for each pixel type ---------------
    nice_name = {"snow": "Snow Cover", "mixed": "Mixed Cover", "none": "Non-snow Cover"}

    for p_type in df["pixel_type"].unique():
        sub = df[df["pixel_type"] == p_type]

        # ---------------- figure / axes --------------------------
        fig, axes = plt.subplots(
            ncols=2,
            figsize=(fig_w, fig_h),
            dpi=300,
            constrained_layout=True,
            gridspec_kw={"wspace": axis_gap / fig_w}   # gap expressed as fraction
        )

        # ------------ left panel : reflectance -------------------
        axL = axes[0]
        sns.boxplot(
            ax=axL,
            x="landsat_band",
            y="reflectance",
            hue="source",
            data=sub,
            palette=palette_src,
            dodge=True,
            width=.55,
            fliersize=3,
            linewidth=1,
            boxprops=dict(edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="black")
        )
        axL.set_xlabel("Landsat Band")
        axL.set_ylabel("Reflectance")
        axL.legend_.remove()
        axL.grid(False)

        # ------------ right panel : difference -------------------
        axR = axes[1]
        sns.boxplot(
            ax=axR,
            x="landsat_band",
            y="difference",
            data=sub,
            color="#f2f5fa",
            width=.50,
            fliersize=3,
            linewidth=1,
            boxprops=dict(edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="black")
        )
        axR.set_xlabel("Landsat Band")
        axR.set_ylabel("Reflectance Difference")
        axR.axhline(0, ls="--", lw=1, color="gray")
        axR.grid(False)

        # ------------ aesthetics shared by both axes -------------
        for ax in (axL, axR):
            ax.set_box_aspect(1)           # 1 : 1 data region
            ax.tick_params(axis="both", direction="out", color="black")
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_color("black")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1)

        # ------------ figure title & legend (outside) ------------
        fig.suptitle(nice_name.get(p_type, p_type), fontsize=12)
        handles, labels = axL.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            title="Source",
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, -0.05)
        )

        # ------------ save ---------------------------------------
        os.makedirs(output_folder, exist_ok=True)
        file_name = f"reflectance_difference_{p_type.replace(' ', '_')}.png"
        out_path  = os.path.join(output_folder, file_name)
        plt.savefig(out_path, bbox_inches="tight", transparent=True)
        plt.show()
        print(f"✅ Plot saved: {out_path}")
