
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[2]

FOOTPRINTS = ROOT / "reports/tables/exposure_footprints.csv"
RASTER = ROOT / "data/external/worldpop/rasters/Paraguay_2018_100m.tif"
OUTPUT = ROOT / "reports/figures/paraguay_nodata_diagnostic.png"

SAMPLE = "Paraguay_511199"
BUFFER_DEGREES = 0.02


def main():
    footprints = pd.read_csv(FOOTPRINTS)
    row = footprints.loc[footprints["sample"] == SAMPLE].iloc[0]

    sample_polygon = box(
        row.min_x, row.min_y, row.max_x, row.max_y
    )

    with rasterio.open(RASTER) as ds:

        bounds = (
            row.min_x - BUFFER_DEGREES,
            row.min_y - BUFFER_DEGREES,
            row.max_x + BUFFER_DEGREES,
            row.max_y + BUFFER_DEGREES
        )

        window = from_bounds(*bounds, transform=ds.transform)
        window = window.round_offsets().round_lengths()

        data = ds.read(1, window=window, masked=True)
        transform = ds.window_transform(window)

        valid = ~np.ma.getmaskarray(data)
        values = data.compressed()

        print("Sample:", SAMPLE)
        print("Expanded window shape:", data.shape)
        print("Valid cells in expanded window:", int(valid.sum()))
        print("Masked cells:", int(data.size - valid.sum()))

        if values.size:
            print("Minimum population:", float(values.min()))
            print("Maximum population:", float(values.max()))
            print("Population sum in expanded window:", float(values.sum()))
        else:
            print("No valid population cells in expanded window.")

        fig, ax = plt.subplots(figsize=(9, 8))

        image = ax.imshow(
            data.filled(np.nan),
            extent=(
                transform.c,
                transform.c + data.shape[1] * transform.a,
                transform.f + data.shape[0] * transform.e,
                transform.f
            ),
            origin="upper",
            cmap="viridis"
        )

        x, y = sample_polygon.exterior.xy
        ax.plot(x, y, color="red", linewidth=2, label="Sample footprint")

        fig.colorbar(image, ax=ax, label="Population per cell")
        ax.set_title("Paraguay_511199 — WorldPop NoData Investigation")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
        plt.show()

        print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
