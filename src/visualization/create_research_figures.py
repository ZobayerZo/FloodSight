
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# PLOTTING CONFIGURATION
# ---------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save_figure(fig, filename):
    """Save a figure in PNG and PDF formats."""

    png_path = FIGURE_DIR / f"{filename}.png"
    pdf_path = FIGURE_DIR / f"{filename}.pdf"

    fig.savefig(png_path)
    fig.savefig(pdf_path)

    plt.close(fig)

    print(f"Saved: {png_path.name}")
    print(f"Saved: {pdf_path.name}")


# ---------------------------------------------------------
# FIGURE 1: EVENT-WISE MODEL PERFORMANCE
# ---------------------------------------------------------

def plot_event_performance():

    df = pd.read_csv(
        TABLE_DIR / "event_wise_evaluation.csv"
    )

    required_columns = {
        "event",
        "pooled_iou",
        "mean_sample_iou",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Missing required columns in event_wise_evaluation.csv"
        )

    df = df.sort_values(
        "pooled_iou",
        ascending=True
    )

    y = np.arange(len(df))
    height = 0.36

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.barh(
        y - height / 2,
        df["pooled_iou"],
        height,
        label="Pooled IoU"
    )

    ax.barh(
        y + height / 2,
        df["mean_sample_iou"],
        height,
        label="Mean sample IoU"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["event"])

    ax.set_xlabel("Intersection over Union (IoU)")
    ax.set_ylabel("Flood event")

    ax.set_title(
        "FloodSight: Event-wise Segmentation Performance",
        pad=15
    )

    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.35
    )

    ax.set_axisbelow(True)

    fig.tight_layout()

    save_figure(
        fig,
        "figure_01_event_wise_performance"
    )


# ---------------------------------------------------------
# FIGURE 2: SAR BACKSCATTER SEPARABILITY
# ---------------------------------------------------------

def plot_sar_separability():

    df = pd.read_csv(
        TABLE_DIR / "sar_cohens_d_by_event.csv"
    )

    required_columns = {
        "event",
        "vv_abs_d_mean",
        "vh_abs_d_mean",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Missing required columns in sar_cohens_d_by_event.csv"
        )

    df = df.sort_values(
        "vh_abs_d_mean",
        ascending=True
    )

    y = np.arange(len(df))
    height = 0.36

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.barh(
        y - height / 2,
        df["vv_abs_d_mean"],
        height,
        label="VV polarization"
    )

    ax.barh(
        y + height / 2,
        df["vh_abs_d_mean"],
        height,
        label="VH polarization"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["event"])

    ax.set_xlabel("Absolute Cohen's d")
    ax.set_ylabel("Flood event")

    ax.set_title(
        "SAR Backscatter Separability Across Flood Events",
        pad=15
    )

    ax.legend(loc="lower right")

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.35
    )

    ax.set_axisbelow(True)

    fig.tight_layout()

    save_figure(
        fig,
        "figure_02_sar_backscatter_separability"
    )


# ---------------------------------------------------------
# FIGURE 3: SAR SEPARABILITY VS MODEL PERFORMANCE
# ---------------------------------------------------------

def plot_separability_performance():

    df = pd.read_csv(
        TABLE_DIR / "separability_performance_comparison.csv"
    )

    required_columns = {
        "event",
        "pooled_iou",
        "vv_abs_d_mean",
        "vh_abs_d_mean",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Missing required columns in "
            "separability_performance_comparison.csv"
        )

    fig, ax = plt.subplots(figsize=(9, 7))

    vv = df["vv_abs_d_mean"].to_numpy()
    vh = df["vh_abs_d_mean"].to_numpy()
    iou = df["pooled_iou"].to_numpy()

    ax.scatter(
        vv,
        iou,
        label="VV polarization",
        s=65
    )

    ax.scatter(
        vh,
        iou,
        label="VH polarization",
        marker="s",
        s=65
    )

    # Event labels for both polarization measurements
    for _, row in df.iterrows():

        ax.annotate(
            row["event"],
            (row["vv_abs_d_mean"], row["pooled_iou"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            alpha=0.8
        )

    # Exploratory Pearson correlations
    vv_r, vv_p = pearsonr(vv, iou)
    vh_r, vh_p = pearsonr(vh, iou)

    # Linear trend lines (descriptive only)
    for values, label in [
        (vv, "VV trend"),
        (vh, "VH trend")
    ]:

        slope, intercept = np.polyfit(
            values,
            iou,
            1
        )

        x_line = np.linspace(
            values.min(),
            values.max(),
            100
        )

        y_line = slope * x_line + intercept

        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label=label
        )

    correlation_text = (
        f"VV: r = {vv_r:.3f}, p = {vv_p:.3f}\n"
        f"VH: r = {vh_r:.3f}, p = {vh_p:.3f}\n"
        f"Events: {len(df)}"
    )

    ax.text(
        0.03,
        0.97,
        correlation_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85
        },
        fontsize=9
    )

    ax.set_xlabel("Absolute Cohen's d")
    ax.set_ylabel("Pooled IoU")

    ax.set_title(
        "Relationship Between SAR Separability "
        "and Segmentation Performance",
        pad=15
    )

    ax.set_ylim(0, 1)

    ax.grid(
        linestyle="--",
        alpha=0.35
    )

    ax.legend(loc="lower right")

    ax.set_axisbelow(True)

    fig.tight_layout()

    save_figure(
        fig,
        "figure_03_separability_vs_performance"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("FLOODSIGHT - RESEARCH FIGURE GENERATION")
    print("=" * 65)

    plot_event_performance()
    plot_sar_separability()
    plot_separability_performance()

    print("\n" + "=" * 65)
    print("FIGURE GENERATION COMPLETED")
    print("=" * 65)

    print(f"\nOutput directory: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
