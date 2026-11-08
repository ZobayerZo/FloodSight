
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def cohens_d(water, non_water):
    """Calculate Cohen's d using pooled standard deviation."""

    nw = len(water)
    nn = len(non_water)

    if nw < 2 or nn < 2:
        return np.nan

    pooled_variance = (
        (nw - 1) * np.var(water, ddof=1)
        + (nn - 1) * np.var(non_water, ddof=1)
    ) / (nw + nn - 2)

    if pooled_variance <= 0:
        return np.nan

    return (
        np.mean(water) - np.mean(non_water)
    ) / np.sqrt(pooled_variance)


def analyze_sar_cohens_d():

    manifest = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "interim"
        / "sen1floods11_manifest.csv"
    )

    test_metrics = pd.read_csv(
        PROJECT_ROOT
        / "reports"
        / "tables"
        / "test_sample_metrics.csv"
    )

    test_metrics["event"] = (
        test_metrics["sample"]
        .str.rsplit("_", n=1)
        .str[0]
    )

    manifest["sample"] = manifest["sample"].astype(str)

    manifest = manifest[
        manifest["sample"].isin(test_metrics["sample"])
    ].copy()

    results = []

    for _, row in manifest.iterrows():

        sar_path = (
            PROJECT_ROOT
            / Path(str(row["sar_file"]).replace("\\", "/"))
        )

        label_path = (
            PROJECT_ROOT
            / Path(str(row["label_file"]).replace("\\", "/"))
        )

        with rasterio.open(sar_path) as src:
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32)

        with rasterio.open(label_path) as src:
            labels = src.read(1)

        valid = (
            np.isfinite(vv)
            & np.isfinite(vh)
            & np.isin(labels, [0, 1])
        )

        water = valid & (labels == 1)
        non_water = valid & (labels == 0)

        if water.sum() < 2 or non_water.sum() < 2:
            continue

        results.append({
            "sample": row["sample"],
            "event": row["event"],
            "water_pixels": int(water.sum()),
            "non_water_pixels": int(non_water.sum()),
            "vv_cohens_d": cohens_d(
                vv[water],
                vv[non_water],
            ),
            "vh_cohens_d": cohens_d(
                vh[water],
                vh[non_water],
            ),
        })

    sample_df = pd.DataFrame(results)

    if sample_df.empty:
        raise RuntimeError("No valid samples for Cohen's d analysis.")

    event_df = sample_df.groupby("event").agg(
        samples=("sample", "count"),
        vv_d_mean=("vv_cohens_d", "mean"),
        vv_d_median=("vv_cohens_d", "median"),
        vh_d_mean=("vh_cohens_d", "mean"),
        vh_d_median=("vh_cohens_d", "median"),
    ).reset_index()

    event_df["vv_abs_d_mean"] = event_df["vv_d_mean"].abs()
    event_df["vh_abs_d_mean"] = event_df["vh_d_mean"].abs()

    output_dir = PROJECT_ROOT / "reports" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_df.to_csv(
        output_dir / "sar_cohens_d_by_sample.csv",
        index=False,
    )

    event_df.to_csv(
        output_dir / "sar_cohens_d_by_event.csv",
        index=False,
    )

    print("\nSAR COHEN'S D BY EVENT")
    print("=" * 100)

    print(
        event_df.sort_values(
            "vv_abs_d_mean",
            ascending=False,
        ).round(4).to_string(index=False)
    )

    print("\nSamples analyzed:", len(sample_df))
    print("Events analyzed:", len(event_df))

    print("\nSaved:")
    print("sar_cohens_d_by_sample.csv")
    print("sar_cohens_d_by_event.csv")


if __name__ == "__main__":
    analyze_sar_cohens_d()
