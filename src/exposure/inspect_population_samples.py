
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FOOTPRINTS = PROJECT_ROOT / "reports" / "tables" / "exposure_footprints.csv"
RASTER_DIR = PROJECT_ROOT / "data" / "external" / "worldpop" / "rasters"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

SAMPLES = {
    "Ghana_406026": ("Ghana", 2018),
    "Paraguay_511199": ("Paraguay", 2018),
}


def main():
    import pandas as pd

    footprints = pd.read_csv(FOOTPRINTS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, (sample, (event, year)) in zip(axes, SAMPLES.items()):

        row = footprints.loc[
            footprints["sample"] == sample
        ].iloc[0]

        raster_path = (
            RASTER_DIR / f"{event}_{year}_100m.tif"
        )

        with rasterio.open(raster_path) as ds:

            window = from_bounds(
                row["min_x"],
                row["min_y"],
                row["max_x"],
                row["max_y"],
                transform=ds.transform
            ).round_offsets().round_lengths()

            data = ds.read(1, window=window, masked=True)

            valid = data.compressed()

            coverage = (
                100 * valid.size / data.size
                if data.size else 0
            )

            image = data.filled(np.nan)

            ax.imshow(
                image,
                extent=[
                    row["min_x"], row["max_x"],
                    row["min_y"], row["max_y"]
                ],
                origin="upper",
                cmap="viridis"
            )

            ax.set_title(
                f"{sample}\n"
                f"Coverage: {coverage:.2f}%"
            )

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            print(f"\n{sample}")
            print("Raster window shape:", data.shape)
            print("Valid cells:", valid.size)
            print("Total cells:", data.size)
            print("Coverage:", round(coverage, 2), "%")

            if valid.size:
                print("Population sum:", float(valid.sum()))
                print("Minimum:", float(valid.min()))
                print("Maximum:", float(valid.max()))

    fig.suptitle(
        "WorldPop Population Coverage Within Sample Footprints"
    )

    plt.tight_layout()

    output = OUTPUT_DIR / "worldpop_sample_coverage_diagnostic.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")

    print("\nSaved:", output)

    plt.show()


if __name__ == "__main__":
    main()
