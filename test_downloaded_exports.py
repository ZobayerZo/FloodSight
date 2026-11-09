
from pathlib import Path
import numpy as np
import rasterio

files = {
    "WorldCover": Path(
        r"D:\ghana_406026_s1hand_tif_worldcover_2021.tif"
    ),
    "Exposed Cropland": Path(
        r"D:\ghana_406026_s1hand_tif_exposed_cropland.tif"
    ),
}

for name, path in files.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        continue

    with rasterio.open(path) as src:
        data = src.read(1)
        valid = src.read_masks(1) > 0

        print("File:", path.name)
        print("CRS:", src.crs)
        print("Dimensions:", src.width, "x", src.height)
        print("Data type:", src.dtypes[0])
        print("NoData:", src.nodata)
        print("Valid pixels:", int(valid.sum()))

        values, counts = np.unique(
            data[valid], return_counts=True
        )

        print("Pixel value distribution:")
        for value, count in zip(values, counts):
            print(f"  Value {value}: {count:,} pixels")

        if name == "Exposed Cropland":
            exposed_count = int(
                np.count_nonzero(data[valid] == 1)
            )

            print("\nEXPOSED CROPLAND PIXELS:", exposed_count)

            if exposed_count == 108:
                print("PASS: Exactly 108 exposed pixels confirmed.")
            else:
                print(
                    f"CHECK: Expected 108, found {exposed_count}."
                )

print("\nVerification completed.")
