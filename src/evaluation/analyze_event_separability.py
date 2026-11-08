
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze_event_separability():

    manifest_path = (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "sen1floods11_manifest.csv"
    )

    metrics_path = (
        PROJECT_ROOT
        / "reports"
        / "tables"
        / "test_sample_metrics.csv"
    )

    manifest = pd.read_csv(manifest_path)
    test_metrics = pd.read_csv(metrics_path)

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

        if water.sum() == 0 or non_water.sum() == 0:
            continue

        results.append({
            "sample": row["sample"],
            "event": row["event"],
            "water_pixels": int(water.sum()),
            "non_water_pixels": int(non_water.sum()),
            "vv_water_mean": float(vv[water].mean()),
            "vv_nonwater_mean": float(vv[non_water].mean()),
            "vh_water_mean": float(vh[water].mean()),
            "vh_nonwater_mean": float(vh[non_water].mean()),
            "vv_water_median": float(np.median(vv[water])),
            "vv_nonwater_median": float(np.median(vv[non_water])),
            "vh_water_median": float(np.median(vh[water])),
            "vh_nonwater_median": float(np.median(vh[non_water])),
        })

    sample_df = pd.DataFrame(results)

    if sample_df.empty:
        raise RuntimeError(
            "No samples contained both valid water and non-water pixels."
        )

    # Aggregate pixel counts and sample-level statistics by event.
    event_df = sample_df.groupby("event").agg(
        samples=("sample", "count"),
        water_pixels=("water_pixels", "sum"),
        non_water_pixels=("non_water_pixels", "sum"),
        vv_water_mean=("vv_water_mean", "mean"),
        vv_nonwater_mean=("vv_nonwater_mean", "mean"),
        vh_water_mean=("vh_water_mean", "mean"),
        vh_nonwater_mean=("vh_nonwater_mean", "mean"),
        vv_water_median=("vv_water_median", "median"),
        vv_nonwater_median=("vv_nonwater_median", "median"),
        vh_water_median=("vh_water_median", "median"),
        vh_nonwater_median=("vh_nonwater_median", "median"),
    ).reset_index()

    event_df["vv_mean_difference"] = (
        event_df["vv_water_mean"]
        - event_df["vv_nonwater_mean"]
    )

    event_df["vh_mean_difference"] = (
        event_df["vh_water_mean"]
        - event_df["vh_nonwater_mean"]
    )

    output_dir = PROJECT_ROOT / "reports" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_df.to_csv(
        output_dir / "sar_separability_by_sample.csv",
        index=False,
    )

    event_df.to_csv(
        output_dir / "sar_separability_by_event.csv",
        index=False,
    )

    print("\nSAR WATER/NON-WATER SEPARABILITY")
    print("=" * 110)

    columns = [
        "event",
        "samples",
        "vv_mean_difference",
        "vh_mean_difference",
        "water_pixels",
        "non_water_pixels",
    ]

    print(
        event_df[columns].round(3).to_string(index=False)
    )

    print("\nSamples analyzed:", len(sample_df))
    print("Events analyzed:", len(event_df))

    print("\nSaved:")
    print("sar_separability_by_sample.csv")
    print("sar_separability_by_event.csv")


if __name__ == "__main__":
    analyze_event_separability()
