
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from rasterio.windows import from_bounds, Window


ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT / "data" / "interim"
    / "sen1floods11_manifest.csv"
)

FOOTPRINT_PATH = (
    ROOT / "reports" / "tables"
    / "exposure_footprints.csv"
)

OUTPUT_PATH = (
    ROOT / "reports" / "tables"
    / "worldpop_coverage_diagnostic.csv"
)

POPULATION_DIR = (
    ROOT / "data" / "external"
    / "worldpop" / "rasters"
)

MIN_COVERAGE = 0.80

EVENT_YEARS = {
    "India": 2016,
    "Pakistan": 2017,
    "Sri-Lanka": 2017,
    "Ghana": 2018,
    "Mekong": 2018,
    "Nigeria": 2018,
    "Paraguay": 2018,
    "Somalia": 2018,
    "Spain": 2019,
    "USA": 2019,
}


def main():

    manifest = pd.read_csv(MANIFEST_PATH)
    footprints = pd.read_csv(FOOTPRINT_PATH)

    test_samples = manifest[
        (manifest["split"] == "test_data")
        & (manifest["usable"] == True)
    ].copy()

    test_samples = test_samples.merge(
        footprints,
        on=["sample", "event"],
        how="inner",
        validate="one_to_one",
    )

    results = []

    raster_cache = {}

    for _, row in test_samples.iterrows():

        sample = row["sample"]
        event = row["event"]

        year = EVENT_YEARS[event]

        raster_path = (
            POPULATION_DIR
            / f"{event}_{year}_100m.tif"
        )

        if event not in raster_cache:

            if not raster_path.exists():
                print("Missing raster:", raster_path)
                continue

            raster_cache[event] = rasterio.open(
                raster_path
            )

        ds = raster_cache[event]

        window = from_bounds(
            row["min_x"],
            row["min_y"],
            row["max_x"],
            row["max_y"],
            transform=ds.transform,
        )

        full_window = Window(
            0,
            0,
            ds.width,
            ds.height,
        )

        window = window.intersection(
            full_window
        ).round_offsets().round_lengths()

        if window.width <= 0 or window.height <= 0:

            results.append({
                "sample": sample,
                "event": event,
                "coverage_pct": 0.0,
                "valid_cells": 0,
                "total_cells": 0,
                "population_sum": np.nan,
                "status": "NO_RASTER_OVERLAP",
            })

            continue

        population = ds.read(
            1,
            window=window,
            masked=True,
        )

        valid = (
            ~np.ma.getmaskarray(population)
            & np.isfinite(population.filled(np.nan))
            & (population.filled(0) >= 0)
        )

        total_cells = population.size
        valid_cells = int(valid.sum())

        coverage = (
            valid_cells / total_cells
            if total_cells > 0
            else 0.0
        )

        population_sum = (
            float(population.filled(0)[valid].sum())
            if valid_cells > 0
            else np.nan
        )

        if coverage >= MIN_COVERAGE:
            status = "ADEQUATE"
        elif coverage > 0:
            status = "PARTIAL_COVERAGE"
        else:
            status = "NO_POPULATION_DATA"

        results.append({
            "sample": sample,
            "event": event,
            "year": year,
            "coverage_pct": round(coverage * 100, 2),
            "valid_cells": valid_cells,
            "total_cells": total_cells,
            "population_sum": population_sum,
            "status": status,
        })

    for ds in raster_cache.values():
        ds.close()

    result_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nWORLDPOP COVERAGE SUMMARY")
    print("=" * 65)

    print("Samples analyzed:", len(result_df))

    print("\nStatus counts:")
    print(result_df["status"].value_counts().to_string())

    print("\nCoverage statistics:")
    print(
        result_df["coverage_pct"].describe().round(2)
    )

    print("\nCoverage by event:")
    print(
        result_df.groupby("event")["coverage_pct"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .to_string()
    )

    print("\nLowest coverage samples:")
    print(
        result_df.sort_values("coverage_pct")
        [["sample", "event", "coverage_pct", "status"]]
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
