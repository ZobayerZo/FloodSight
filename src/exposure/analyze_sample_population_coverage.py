
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FOOTPRINT_FILE = (
    PROJECT_ROOT / "reports" / "tables" / "exposure_footprints.csv"
)

RASTER_DIR = (
    PROJECT_ROOT / "data" / "external" / "worldpop" / "rasters"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"

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

# Coverage categories for reporting.
MINIMUM_COVERAGE_PCT = 80.0


def main():
    footprints = pd.read_csv(FOOTPRINT_FILE)

    results = []

    # Open each event raster only once.
    for event, year in EVENT_YEARS.items():

        event_samples = footprints[
            footprints["event"] == event
        ]

        if event_samples.empty:
            raise ValueError(f"No footprint samples found for {event}")

        raster_path = RASTER_DIR / f"{event}_{year}_100m.tif"

        if not raster_path.exists():
            raise FileNotFoundError(raster_path)

        with rasterio.open(raster_path) as ds:

            population = ds.read(1, masked=True)
            population_mask = np.ma.getmaskarray(population)

            for _, row in event_samples.iterrows():

                sample_geometry = box(
                    row["min_x"],
                    row["min_y"],
                    row["max_x"],
                    row["max_y"]
                )

                inside = geometry_mask(
                    [mapping(sample_geometry)],
                    out_shape=(ds.height, ds.width),
                    transform=ds.transform,
                    invert=True,
                    all_touched=False
                )

                total_cells = int(inside.sum())

                valid_inside = inside & ~population_mask

                valid_cells = int(valid_inside.sum())
                missing_cells = total_cells - valid_cells

                coverage = (
                    100 * valid_cells / total_cells
                    if total_cells > 0 else 0
                )

                valid_values = population.data[valid_inside]

                population_sum = (
                    float(valid_values.sum())
                    if valid_cells > 0 else np.nan
                )

                if coverage >= MINIMUM_COVERAGE_PCT:
                    status = "Adequate"
                elif valid_cells > 0:
                    status = "Partial"
                else:
                    status = "No coverage"

                results.append({
                    "sample": row["sample"],
                    "event": event,
                    "year": year,
                    "footprint_cells": total_cells,
                    "valid_population_cells": valid_cells,
                    "missing_population_cells": missing_cells,
                    "coverage_pct": round(coverage, 2),
                    "population_sum": population_sum,
                    "coverage_status": status
                })

    summary = pd.DataFrame(results)

    summary = summary.sort_values(
        ["event", "sample"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_output = OUTPUT_DIR / "worldpop_coverage_by_sample.csv"
    event_output = OUTPUT_DIR / "worldpop_coverage_by_event.csv"

    summary.to_csv(sample_output, index=False)

    event_summary = (
        summary.groupby("event")
        .agg(
            samples=("sample", "count"),
            mean_coverage_pct=("coverage_pct", "mean"),
            median_coverage_pct=("coverage_pct", "median"),
            minimum_coverage_pct=("coverage_pct", "min"),
            maximum_coverage_pct=("coverage_pct", "max"),
            adequate_samples=(
                "coverage_status",
                lambda x: (x == "Adequate").sum()
            ),
            partial_samples=(
                "coverage_status",
                lambda x: (x == "Partial").sum()
            ),
            no_coverage_samples=(
                "coverage_status",
                lambda x: (x == "No coverage").sum()
            )
        )
        .reset_index()
    )

    event_summary.to_csv(event_output, index=False)

    print("\nSAMPLE-LEVEL WORLDPOP COVERAGE")
    print("=" * 100)

    print(
        summary[
            [
                "sample",
                "coverage_pct",
                "valid_population_cells",
                "population_sum",
                "coverage_status"
            ]
        ].to_string(index=False)
    )

    print("\nEVENT-LEVEL SUMMARY")
    print("=" * 100)
    print(event_summary.to_string(index=False))

    print("\nOVERALL SUMMARY")
    print("=" * 100)

    print("Total samples:", len(summary))
    print(
        "Adequate coverage:",
        (summary["coverage_status"] == "Adequate").sum()
    )
    print(
        "Partial coverage:",
        (summary["coverage_status"] == "Partial").sum()
    )
    print(
        "No coverage:",
        (summary["coverage_status"] == "No coverage").sum()
    )

    print("\nSaved:")
    print(sample_output)
    print(event_output)


if __name__ == "__main__":
    main()
