
from pathlib import Path

import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT / "data/interim/sen1floods11_manifest.csv"
)

TEST_DIR = PROJECT_ROOT / "data/processed/test_data"


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    rows = []

    for npz_path in sorted(TEST_DIR.glob("*.npz")):
        sample_id = npz_path.stem

        match = manifest[
            manifest["sample"].astype(str).str.strip() == sample_id
        ]

        if match.empty:
            print(f"Manifest entry not found: {sample_id}")
            continue

        row = match.iloc[0]

        sar_path = PROJECT_ROOT / Path(
            str(row["sar_file"]).replace("\\", "/")
        )

        if not sar_path.exists():
            print(f"SAR file not found: {sar_path}")
            continue

        with rasterio.open(sar_path) as src:
            bounds = src.bounds

            rows.append({
                "sample": sample_id,
                "event": row["event"],
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "min_x": bounds.left,
                "min_y": bounds.bottom,
                "max_x": bounds.right,
                "max_y": bounds.top,
                "resolution_x": src.res[0],
                "resolution_y": src.res[1],
            })

    result = pd.DataFrame(rows)

    output_dir = PROJECT_ROOT / "reports/tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "exposure_footprints.csv"
    result.to_csv(output_path, index=False)

    print("\nFLOODSIGHT - EXPOSURE FOOTPRINT INSPECTION")
    print("=" * 55)

    print(f"Samples inspected: {len(result)}")

    if not result.empty:
        print("\nSamples by event:")
        print(result.groupby("event").size().to_string())

        print("\nCoordinate reference systems:")
        print(result["crs"].value_counts().to_string())

        print("\nGeographical coverage by event:")
        print(
            result.groupby("event")[
                ["min_x", "min_y", "max_x", "max_y"]
            ].agg(["min", "max"]).round(5).to_string()
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
