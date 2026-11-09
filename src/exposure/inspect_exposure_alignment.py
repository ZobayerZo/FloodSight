
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "data/interim/sen1floods11_manifest.csv"
TEST_DIR = ROOT / "data/processed/test_data"
POPULATION_DIR = ROOT / "data/external/worldpop/rasters"

SAMPLE = "Ghana_406026"


def main():
    manifest = pd.read_csv(MANIFEST)

    rows = manifest.loc[
        manifest["sample"] == SAMPLE
    ]

    if rows.empty:
        raise ValueError(f"Sample not found: {SAMPLE}")

    row = rows.iloc[0]

    sar_path = ROOT / row["sar_file"]
    npz_path = TEST_DIR / f"{SAMPLE}.npz"
    population_path = POPULATION_DIR / "Ghana_2018_100m.tif"

    print("\n1. ORIGINAL SENTINEL-1 RASTER")
    print("=" * 65)

    with rasterio.open(sar_path) as ds:
        print("File:", sar_path)
        print("CRS:", ds.crs)
        print("Shape:", ds.height, ds.width)
        print("Transform:", ds.transform)
        print("Bounds:", ds.bounds)
        print("Resolution:", ds.res)

        sar_transform = ds.transform
        sar_crs = ds.crs
        sar_shape = (ds.height, ds.width)

    print("\n2. PROCESSED NPZ")
    print("=" * 65)

    with np.load(npz_path) as data:
        for key in data.files:
            array = data[key]
            print(
                f"{key}: shape={array.shape}, "
                f"dtype={array.dtype}"
            )

        print("NPZ contains geographic metadata: No")

    print("\n3. WORLDPOP RASTER")
    print("=" * 65)

    with rasterio.open(population_path) as ds:
        print("File:", population_path)
        print("CRS:", ds.crs)
        print("Shape:", ds.height, ds.width)
        print("Transform:", ds.transform)
        print("Bounds:", ds.bounds)
        print("Resolution:", ds.res)
        print("NoData:", ds.nodata)

        population_transform = ds.transform
        population_crs = ds.crs

    print("\n4. ALIGNMENT CHECK")
    print("=" * 65)

    print("Same CRS:", sar_crs == population_crs)
    print("SAR shape:", sar_shape)
    print("SAR pixel size:", sar_transform.a, sar_transform.e)
    print(
        "WorldPop pixel size:",
        population_transform.a,
        population_transform.e
    )

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
