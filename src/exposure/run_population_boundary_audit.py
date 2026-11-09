
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds, transform_geom
from rasterio.features import geometry_mask

from shapely.geometry import box, mapping


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT / "data" / "interim"
    / "sen1floods11_manifest.csv"
)

NPZ_DIR = (
    ROOT / "data" / "processed" / "test_data"
)

POPULATION_DIR = (
    ROOT / "data" / "external"
    / "worldpop" / "rasters"
)

OUTPUT_DIR = (
    ROOT / "reports" / "tables"
)

SAMPLE_OUTPUT = (
    OUTPUT_DIR / "population_boundary_audit.csv"
)

EVENT_OUTPUT = (
    OUTPUT_DIR / "population_boundary_audit_by_event.csv"
)

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


# =========================================================
# 2. HELPER FUNCTION
# =========================================================

def get_population_validity(population):

    values = population.filled(np.nan)

    valid = (
        ~np.ma.getmaskarray(population)
        & np.isfinite(values)
        & (values >= 0)
    )

    return valid


# =========================================================
# 3. AUDIT ONE SAMPLE
# =========================================================

def audit_sample(row):

    sample = row["sample"]
    event = row["event"]
    year = EVENT_YEARS[event]

    sar_path = ROOT / row["sar_file"]

    population_path = (
        POPULATION_DIR
        / f"{event}_{year}_100m.tif"
    )

    if not sar_path.exists():
        raise FileNotFoundError(
            f"SAR raster missing: {sar_path}"
        )

    if not population_path.exists():
        raise FileNotFoundError(
            f"Population raster missing: {population_path}"
        )

    # -----------------------------------------------------
    # READ SAR FOOTPRINT
    # -----------------------------------------------------

    with rasterio.open(sar_path) as sar_ds:

        if sar_ds.crs is None:
            raise ValueError("SAR CRS is missing.")

        sar_bounds = sar_ds.bounds
        sar_crs = sar_ds.crs

        sar_geometry = box(
            sar_bounds.left,
            sar_bounds.bottom,
            sar_bounds.right,
            sar_bounds.top,
        )

    # -----------------------------------------------------
    # OPEN WORLDPOP
    # -----------------------------------------------------

    with rasterio.open(population_path) as pop_ds:

        if pop_ds.crs is None:
            raise ValueError("WorldPop CRS is missing.")

        # Transform SAR bounds and footprint geometry
        # when the two datasets have different CRS.

        if sar_crs != pop_ds.crs:

            bounds = transform_bounds(
                sar_crs,
                pop_ds.crs,
                sar_bounds.left,
                sar_bounds.bottom,
                sar_bounds.right,
                sar_bounds.top,
                densify_pts=21,
            )

            pop_geometry = transform_geom(
                sar_crs,
                pop_ds.crs,
                mapping(sar_geometry),
                precision=-1,
            )

        else:

            bounds = (
                sar_bounds.left,
                sar_bounds.bottom,
                sar_bounds.right,
                sar_bounds.top,
            )

            pop_geometry = mapping(sar_geometry)

        # -------------------------------------------------
        # CURRENT ROUNDED WINDOW
        # -------------------------------------------------

        raw_window = from_bounds(
            *bounds,
            transform=pop_ds.transform,
        )

        full_window = Window(
            0,
            0,
            pop_ds.width,
            pop_ds.height,
        )

        try:

            window = raw_window.intersection(
                full_window
            )

        except Exception:

            return {
                "sample": sample,
                "event": event,
                "year": year,
                "status": "NO_RASTER_OVERLAP",
            }

        window = (
            window
            .round_offsets()
            .round_lengths()
        )

        if window.width <= 0 or window.height <= 0:

            return {
                "sample": sample,
                "event": event,
                "year": year,
                "status": "NO_RASTER_OVERLAP",
            }

        population = pop_ds.read(
            1,
            window=window,
            masked=True,
        )

        window_transform = (
            pop_ds.window_transform(window)
        )

        valid_population = get_population_validity(
            population
        )

        # -------------------------------------------------
        # FOOTPRINT INTERSECTION MASK
        # -------------------------------------------------

        intersects_footprint = geometry_mask(
            [pop_geometry],
            out_shape=population.shape,
            transform=window_transform,
            invert=True,
            all_touched=True,
        )

        # -------------------------------------------------
        # CELL-CENTER MASK
        # -------------------------------------------------

        center_inside_footprint = geometry_mask(
            [pop_geometry],
            out_shape=population.shape,
            transform=window_transform,
            invert=True,
            all_touched=False,
        )

        # -------------------------------------------------
        # CURRENT METHOD
        # -------------------------------------------------

        total_window_cells = int(
            population.size
        )

        valid_window_cells = int(
            valid_population.sum()
        )

        current_coverage = (
            100 * valid_window_cells / total_window_cells
            if total_window_cells > 0
            else np.nan
        )

        # -------------------------------------------------
        # INTERSECTION METHOD
        # -------------------------------------------------

        intersection_cells = int(
            intersects_footprint.sum()
        )

        valid_intersection_cells = int(
            (
                valid_population
                & intersects_footprint
            ).sum()
        )

        intersection_coverage = (
            100 * valid_intersection_cells
            / intersection_cells
            if intersection_cells > 0
            else np.nan
        )

        # -------------------------------------------------
        # CELL-CENTER METHOD
        # -------------------------------------------------

        center_cells = int(
            center_inside_footprint.sum()
        )

        valid_center_cells = int(
            (
                valid_population
                & center_inside_footprint
            ).sum()
        )

        center_coverage = (
            100 * valid_center_cells / center_cells
            if center_cells > 0
            else np.nan
        )

        # -------------------------------------------------
        # BOUNDARY DIFFERENCES
        # -------------------------------------------------

        valid_outside_center = (
            valid_population
            & intersects_footprint
            & ~center_inside_footprint
        )

        valid_cells_outside_center = int(
            valid_outside_center.sum()
        )

        boundary_only_cells = (
            intersects_footprint
            & ~center_inside_footprint
        )

        boundary_cell_count = int(
            boundary_only_cells.sum()
        )

        return {
            "sample": sample,
            "event": event,
            "year": year,

            "total_window_cells": total_window_cells,
            "valid_window_cells": valid_window_cells,
            "current_coverage_pct": round(
                current_coverage, 4
            ),

            "intersection_cells": intersection_cells,
            "valid_intersection_cells": valid_intersection_cells,
            "intersection_coverage_pct": round(
                intersection_coverage, 4
            ),

            "center_cells": center_cells,
            "valid_center_cells": valid_center_cells,
            "center_coverage_pct": round(
                center_coverage, 4
            ),

            "boundary_only_cells": boundary_cell_count,
            "valid_cells_outside_center": valid_cells_outside_center,

            "current_minus_intersection_pct": round(
                current_coverage - intersection_coverage, 4
            ),

            "current_minus_center_pct": round(
                current_coverage - center_coverage, 4
            ),

            "status": (
                "NO_POPULATION_DATA"
                if valid_intersection_cells == 0
                else "PROCESSED"
            ),
        }


# =========================================================
# 4. MAIN BATCH AUDIT
# =========================================================

def main():

    print("=" * 70)
    print("FLOODSIGHT: 88-SAMPLE POPULATION BOUNDARY AUDIT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    test_manifest = manifest[
        manifest["split"] == "test_data"
    ].copy()

    processed_samples = {
        path.stem
        for path in NPZ_DIR.glob("*.npz")
    }

    test_manifest = test_manifest[
        test_manifest["sample"].isin(processed_samples)
    ].copy()

    test_manifest = test_manifest.sort_values(
        ["event", "sample"]
    )

    if len(test_manifest) != 88:
        raise ValueError(
            f"Expected 88 samples; found {len(test_manifest)}."
        )

    print("Samples to audit:", len(test_manifest))

    results = []

    for index, (_, row) in enumerate(
        test_manifest.iterrows(),
        start=1,
    ):

        sample = row["sample"]

        print(
            f"\n[{index}/88] Auditing {sample}"
        )

        try:

            result = audit_sample(row)

            print(
                "Current:",
                result.get("current_coverage_pct"),
                "| Intersection:",
                result.get("intersection_coverage_pct"),
                "| Center:",
                result.get("center_coverage_pct"),
                "| Status:",
                result["status"],
            )

        except Exception as exc:

            print("ERROR:", str(exc))

            result = {
                "sample": sample,
                "event": row["event"],
                "year": EVENT_YEARS.get(
                    row["event"], np.nan
                ),
                "status": "PROCESSING_ERROR",
                "error": str(exc),
            }

        results.append(result)

    # -----------------------------------------------------
    # SAVE SAMPLE-LEVEL RESULTS
    # -----------------------------------------------------

    sample_df = pd.DataFrame(results)

    sample_df.to_csv(
        SAMPLE_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # EVENT-LEVEL SUMMARY
    # -----------------------------------------------------

    successful = sample_df[
        sample_df["status"].isin(
            ["PROCESSED", "NO_POPULATION_DATA"]
        )
    ].copy()

    summary = (
        successful.groupby("event")
        .agg(
            samples=("sample", "count"),

            mean_current_coverage=(
                "current_coverage_pct", "mean"
            ),

            mean_intersection_coverage=(
                "intersection_coverage_pct", "mean"
            ),

            mean_center_coverage=(
                "center_coverage_pct", "mean"
            ),

            total_boundary_cells=(
                "boundary_only_cells", "sum"
            ),

            total_valid_cells_outside_center=(
                "valid_cells_outside_center", "sum"
            ),

            samples_with_valid_boundary_cells=(
                "valid_cells_outside_center",
                lambda x: int((x > 0).sum()),
            ),
        )
        .round(4)
        .reset_index()
    )

    summary.to_csv(
        EVENT_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BOUNDARY AUDIT COMPLETE")
    print("=" * 70)

    print("\nStatus counts:")
    print(
        sample_df["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nCoverage differences:")

    if not successful.empty:

        print(
            successful[
                [
                    "current_minus_intersection_pct",
                    "current_minus_center_pct",
                ]
            ].describe().round(4).to_string()
        )

        print(
            "\nSamples with valid population cells "
            "outside the center-based footprint:",
            int(
                (
                    successful[
                        "valid_cells_outside_center"
                    ] > 0
                ).sum()
            ),
        )

        print(
            "Total valid boundary cells outside centers:",
            int(
                successful[
                    "valid_cells_outside_center"
                ].sum()
            ),
        )

    print("\nEvent summary:")
    print(summary.to_string(index=False))

    print("\nOutput files:")
    print(SAMPLE_OUTPUT)
    print(EVENT_OUTPUT)

    print("=" * 70)


if __name__ == "__main__":
    main()
