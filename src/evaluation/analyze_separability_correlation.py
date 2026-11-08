
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze_correlation():

    metrics_path = (
        PROJECT_ROOT
        / "reports"
        / "tables"
        / "event_wise_evaluation.csv"
    )

    separability_path = (
        PROJECT_ROOT
        / "reports"
        / "tables"
        / "sar_cohens_d_by_event.csv"
    )

    metrics = pd.read_csv(metrics_path)
    separability = pd.read_csv(separability_path)

    merged = metrics.merge(
        separability,
        on="event",
        how="inner",
    )

    if len(merged) < 3:
        raise ValueError(
            "Insufficient events for correlation analysis."
        )

    # Use absolute standardized separation.
    merged["vv_abs_d"] = merged["vv_d_mean"].abs()
    merged["vh_abs_d"] = merged["vh_d_mean"].abs()

    comparisons = {
        "VV Cohen's d vs pooled IoU": "vv_abs_d",
        "VH Cohen's d vs pooled IoU": "vh_abs_d",
    }

    print("\nSAR SEPARABILITY VS MODEL PERFORMANCE")
    print("=" * 75)

    for label, feature in comparisons.items():

        pearson_r, pearson_p = pearsonr(
            merged[feature],
            merged["pooled_iou"],
        )

        spearman_r, spearman_p = spearmanr(
            merged[feature],
            merged["pooled_iou"],
        )

        print(f"\n{label}")
        print(f"Pearson correlation:  {pearson_r:.4f}")
        print(f"Pearson p-value:      {pearson_p:.4f}")
        print(f"Spearman correlation: {spearman_r:.4f}")
        print(f"Spearman p-value:     {spearman_p:.4f}")

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "tables"
        / "separability_performance_comparison.csv"
    )

    merged.to_csv(output_path, index=False)

    print("\nEvents compared:", len(merged))
    print("Saved:", output_path)


if __name__ == "__main__":
    analyze_correlation()
