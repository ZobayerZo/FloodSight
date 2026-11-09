
from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AREA_DIR = PROJECT_ROOT / "data" / "external" / "worldpop" / "areas"
RASTER_DIR = PROJECT_ROOT / "data" / "external" / "worldpop" / "rasters"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "tables" / "worldpop_coverage_check.csv"


def main():
    results = []

    for raster_path in sorted(RASTER_DIR.glob("*.tif")):
        event = raster_path.stem.rsplit("_", 2)[0]
        geojson_path = AREA_DIR / f"{event}.geojson"

        with geojson_path.open("r", encoding="utf-8") as f:
            feature = json.load(f)

        geometry = feature["geometry"]

        with rasterio.open(raster_path) as ds:
            population = ds.read(1, masked=True)

            # True for raster cells whose centres are inside the footprint.
            inside_footprint = geometry_mask(
                [geometry],
                out_shape=(ds.height, ds.width),
                transform=ds.transform,
                invert=True,
                all_touched=False
            )

            valid_population = ~np.ma.getmaskarray(population)

            footprint_cells = int(inside_footprint.sum())
            valid_inside = int(
                (inside_footprint & valid_population).sum()
            )

            missing_inside = footprint_cells - valid_inside

            values_inside = population.data[
                inside_footprint & valid_population
            ]

            results.append({
                "event": event,
                "footprint_grid_cells": footprint_cells,
                "valid_population_cells": valid_inside,
                "missing_population_cells": missing_inside,
                "valid_coverage_pct": (
                    100 * valid_inside / footprint_cells
                    if footprint_cells else 0
                ),
                "population_sum": float(values_inside.sum())
            })

    summary = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nWORLDPOP FOOTPRINT COVERAGE CHECK")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
