from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data/raw/Sen1Floods11/v1.1/data/flood_events/HandLabeled"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data/raw/Sen1Floods11/v1.1/splits/flood_handlabeled"
)

OUTPUT_DIR = PROJECT_ROOT / "data/interim"
OUTPUT_FILE = OUTPUT_DIR / "sen1floods11_manifest.csv"


def inspect_sample(s1_name, label_name, split_name):

    sample = s1_name.replace("_S1Hand.tif", "")
    event = sample.rsplit("_", 1)[0]

    jrc_name = s1_name.replace(
        "_S1Hand.tif",
        "_JRCWaterHand.tif"
    )

    sar_path = DATA_ROOT / "S1Hand" / s1_name
    label_path = DATA_ROOT / "LabelHand" / label_name
    jrc_path = DATA_ROOT / "JRCWaterHand" / jrc_name

    for path in [sar_path, label_path, jrc_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    with rasterio.open(sar_path) as src:
        sar = src.read()

    with rasterio.open(label_path) as src:
        label = src.read(1)

    with rasterio.open(jrc_path) as src:
        jrc = src.read(1)

    if sar.shape[1:] != label.shape or label.shape != jrc.shape:
        raise ValueError(f"Raster dimension mismatch: {sample}")

    sar_valid = np.isfinite(sar).all(axis=0)
    label_valid = label != -1
    effective = sar_valid & label_valid

    return {
        "sample": sample,
        "event": event,
        "split": split_name,
        "sar_file": str(sar_path.relative_to(PROJECT_ROOT)),
        "label_file": str(label_path.relative_to(PROJECT_ROOT)),
        "jrc_file": str(jrc_path.relative_to(PROJECT_ROOT)),
        "height": label.shape[0],
        "width": label.shape[1],
        "sar_valid_pixels": int(sar_valid.sum()),
        "label_valid_pixels": int(label_valid.sum()),
        "invalid_label_pixels": int((label == -1).sum()),
        "water_pixels": int((label == 1).sum()),
        "non_water_pixels": int((label == 0).sum()),
        "effective_pixels": int(effective.sum()),
        "effective_water_pixels": int(((label == 1) & effective).sum()),
        "effective_non_water_pixels": int(((label == 0) & effective).sum()),
        "effective_pixel_pct": float(effective.mean() * 100),
        "usable": bool(effective.any()),
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    split_files = sorted(SPLIT_DIR.glob("*.csv"))

    if not split_files:
        raise FileNotFoundError("No official split CSV files found.")

    for csv_file in split_files:

        split_name = csv_file.stem.replace("flood_", "")
        df = pd.read_csv(csv_file, header=None)

        print(f"Processing {split_name}: {len(df)} samples")

        for _, row in df.iterrows():
            records.append(
                inspect_sample(
                    str(row.iloc[0]),
                    str(row.iloc[1]),
                    split_name
                )
            )

    manifest = pd.DataFrame(records)

    if manifest["sample"].duplicated().any():
        raise ValueError("Duplicate samples detected in manifest.")

    if len(manifest) != 446:
        raise ValueError(
            f"Expected 446 samples, found {len(manifest)}"
        )

    manifest.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("FLOODSIGHT DATASET MANIFEST")
    print("=" * 60)

    print("Total samples:", len(manifest))
    print("Usable samples:", manifest["usable"].sum())
    print("Unusable samples:", (~manifest["usable"]).sum())

    print("\nSplit summary:")

    summary = manifest.groupby("split").agg(
        samples=("sample", "count"),
        usable=("usable", "sum"),
        effective_pixels=("effective_pixels", "sum")
    )

    print(summary.to_string())

    print("\nManifest saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
