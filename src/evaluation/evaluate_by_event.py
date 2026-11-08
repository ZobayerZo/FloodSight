
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_by_event(input_csv=None, output_dir=None):
    """Calculate pooled and mean sample metrics by flood event."""

    if input_csv is None:
        input_csv = (
            PROJECT_ROOT
            / "reports"
            / "tables"
            / "test_sample_metrics.csv"
        )

    if output_dir is None:
        output_dir = PROJECT_ROOT / "reports" / "tables"

    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required_columns = [
        "sample",
        "valid_pixels",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "iou",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Extract event name from sample identifier.
    df["event"] = df["sample"].str.rsplit("_", n=1).str[0]

    grouped = df.groupby("event").agg(
        samples=("sample", "count"),
        valid_pixels=("valid_pixels", "sum"),
        tp=("true_positives", "sum"),
        fp=("false_positives", "sum"),
        fn=("false_negatives", "sum"),
        tn=("true_negatives", "sum"),
        mean_sample_iou=("iou", "mean"),
        median_sample_iou=("iou", "median"),
    )

    # Calculate pooled event-level metrics.
    grouped["pooled_iou"] = (
        grouped["tp"]
        / (grouped["tp"] + grouped["fp"] + grouped["fn"])
    ).where(
        (grouped["tp"] + grouped["fp"] + grouped["fn"]) > 0,
        1.0,
    )

    grouped["precision"] = (
        grouped["tp"] / (grouped["tp"] + grouped["fp"])
    ).where(
        (grouped["tp"] + grouped["fp"]) > 0,
        np.nan,
    )

    grouped["recall"] = (
        grouped["tp"] / (grouped["tp"] + grouped["fn"])
    ).where(
        (grouped["tp"] + grouped["fn"]) > 0,
        np.nan,
    )

    grouped["dice"] = (
        2 * grouped["tp"]
        / (2 * grouped["tp"] + grouped["fp"] + grouped["fn"])
    ).where(
        (2 * grouped["tp"] + grouped["fp"] + grouped["fn"]) > 0,
        1.0,
    )

    grouped["accuracy"] = (
        (grouped["tp"] + grouped["tn"])
        / grouped["valid_pixels"]
    ).where(
        grouped["valid_pixels"] > 0,
        np.nan,
    )

    grouped = grouped.reset_index()
    grouped = grouped.sort_values(
        "pooled_iou",
        ascending=False,
    )

    output_path = output_dir / "event_wise_evaluation.csv"
    grouped.to_csv(output_path, index=False)

    print("\nFLOODSIGHT EVENT-WISE EVALUATION")
    print("=" * 90)

    display_columns = [
        "event",
        "samples",
        "pooled_iou",
        "mean_sample_iou",
        "precision",
        "recall",
    ]

    print(
        grouped[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nTotal event groups:", len(grouped))
    print("Saved report:", output_path)

    return grouped


if __name__ == "__main__":
    evaluate_by_event()
