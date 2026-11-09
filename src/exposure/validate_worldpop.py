
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RASTER_DIR = PROJECT_ROOT / "data" / "external" / "worldpop" / "rasters"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "tables" / "worldpop_raster_validation.csv"


def main():
    raster_files = sorted(RASTER_DIR.glob("*.tif"))

    if not raster_files:
        raise FileNotFoundError(
            f"No GeoTIFF files found in {RASTER_DIR}"
        )

    results = []

    for raster_file in raster_files:
        with rasterio.open(raster_file) as dataset:
            population = dataset.read(1, masked=True)
            values = population.compressed()

            if values.size == 0:
                raise ValueError(
                    f"No valid population pixels in {raster_file.name}"
                )

            total_grid_cells = dataset.width * dataset.height

            results.append({
                "event": raster_file.stem,
                "width": dataset.width,
                "height": dataset.height,
                "total_grid_cells": total_grid_cells,
                "valid_cells": int(values.size),
                "masked_cells": int(total_grid_cells - values.size),
                "valid_grid_percentage": round(
                    100 * values.size / total_grid_cells, 2
                ),
                "crs": str(dataset.crs),
                "resolution_x": dataset.res[0],
                "resolution_y": dataset.res[1],
                "zero_population_cells": int(
                    np.count_nonzero(values == 0)
                ),
                "minimum_population": float(values.min()),
                "maximum_population": float(values.max()),
                "mean_population": float(values.mean()),
                "total_population": float(values.sum()),
                "negative_cells": int(np.count_nonzero(values < 0)),
                "nodata": dataset.nodata
            })

    summary = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    display_columns = [
        "event",
        "total_grid_cells",
        "valid_cells",
        "masked_cells",
        "valid_grid_percentage",
        "total_population"
    ]

    print("\nWORLDPOP RASTER VALIDATION")
    print("=" * 95)
    print(summary[display_columns].to_string(index=False))

    print("\nFiles validated:", len(summary))
    print("Total valid cells:", f"{summary['valid_cells'].sum():,}")
    print(
        "Total population across downloaded areas:",
        f"{summary['total_population'].sum():,.2f}"
    )
    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
