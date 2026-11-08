
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze_event_features():

    manifest_path = (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "sen1floods11_manifest.csv"
    )

    manifest = pd.read_csv(manifest_path)

    test_metrics = pd.read_csv(
        PROJECT_ROOT
        / "reports"
        / "tables"
        / "test_sample_metrics.csv"
    )

    # Keep only official test samples.
    test_metrics["event"] = (
        test_metrics["sample"]
        .str.rsplit("_", n=1)
        .str[0]
    )

    manifest["sample"] = (
        manifest["sample"].astype(str)
    )

    test_samples = test_metrics["sample"].tolist()

    manifest = manifest[
        manifest["sample"].isin(test_samples)
    ].copy()

    results = []

    for _, row in manifest.iterrows():

        sar_path = PROJECT_ROOT / Path(
            str(row["sar_file"]).replace("\\", "/")
        )

        if not sar_path.exists():
            print("Missing SAR file:", sar_path)
            continue

        with rasterio.open(sar_path) as src:

            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32)

        valid = np.isfinite(vv) & np.isfinite(vh)

        if valid.sum() == 0:
            continue

        vv_valid = vv[valid]
        vh_valid = vh[valid]

        results.append({
            "sample": row["sample"],
            "event": row["event"],
            "vv_mean": float(vv_valid.mean()),
            "vv_std": float(vv_valid.std()),
            "vv_median": float(np.median(vv_valid)),
            "vh_mean": float(vh_valid.mean()),
            "vh_std": float(vh_valid.std()),
            "vh_median": float(np.median(vh_valid)),
            "valid_sar_pixels": int(valid.sum()),
        })

    sample_df = pd.DataFrame(results)

    if sample_df.empty:
        raise RuntimeError("No SAR samples were successfully analyzed.")

    event_df = sample_df.groupby("event").agg(
        samples=("sample", "count"),
        vv_mean=("vv_mean", "mean"),
        vv_std=("vv_std", "mean"),
        vv_median=("vv_median", "median"),
        vh_mean=("vh_mean", "mean"),
        vh_std=("vh_std", "mean"),
        vh_median=("vh_median", "median"),
        valid_sar_pixels=("valid_sar_pixels", "sum"),
    ).reset_index()

    output_dir = PROJECT_ROOT / "reports" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_df.to_csv(
        output_dir / "test_sar_features_by_sample.csv",
        index=False,
    )

    event_df.to_csv(
        output_dir / "test_sar_features_by_event.csv",
        index=False,
    )

    print("\nSAR FEATURES BY EVENT")
    print("=" * 90)

    print(
        event_df.round(3).to_string(index=False)
    )

    print("\nSaved:")
    print("test_sar_features_by_sample.csv")
    print("test_sar_features_by_event.csv")


if __name__ == "__main__":
    analyze_event_features()
