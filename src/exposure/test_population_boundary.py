
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

SAMPLE = "Spain_6860600"

MANIFEST_PATH = (
    ROOT / "data" / "interim"
    / "sen1floods11_manifest.csv"
)

POPULATION_PATH = (
    ROOT / "data" / "external" / "worldpop"
    / "rasters" / "Spain_2019_100m.tif"
)

# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 65)
    print("FLOODSIGHT POPULATION BOUNDARY DIAGNOSTIC")
    print("=" * 65)

    # -----------------------------------------------------
    # 1. FIND SAR SAMPLE
    # -----------------------------------------------------

    manifest = pd.read_csv(MANIFEST_PATH)

    rows = manifest[manifest["sample"] == SAMPLE]

    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for {SAMPLE}; found {len(rows)}"
        )

    sar_path = ROOT / rows.iloc[0]["sar_file"]

    if not sar_path.exists():
        raise FileNotFoundError(sar_path)

    # -----------------------------------------------------
    # 2. READ SAR FOOTPRINT
    # -----------------------------------------------------

    with rasterio.open(sar_path) as sar_ds:

        sar_bounds = sar_ds.bounds
        sar_crs = sar_ds.crs

        if sar_crs is None:
            raise ValueError("SAR CRS is missing.")

        # Exact rectangular footprint of the SAR raster.
        sar_geometry = box(
            sar_bounds.left,
            sar_bounds.bottom,
            sar_bounds.right,
            sar_bounds.top,
        )

    # -----------------------------------------------------
    # 3. OPEN WORLDPOP
    # -----------------------------------------------------

    with rasterio.open(POPULATION_PATH) as pop_ds:

        if pop_ds.crs is None:
            raise ValueError("WorldPop CRS is missing.")

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

            # Transform the SAR footprint geometry too.
            from rasterio.warp import transform_geom

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
        # 4. REPRODUCE CURRENT ROUNDED WINDOW
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

        window = raw_window.intersection(full_window)

        window = (
            window
            .round_offsets()
            .round_lengths()
        )

        if window.width <= 0 or window.height <= 0:
            raise ValueError("No WorldPop overlap.")

        population = pop_ds.read(
            1,
            window=window,
            masked=True,
        )

        transform = pop_ds.window_transform(window)

        values = population.filled(np.nan)

        valid_population = (
            ~np.ma.getmaskarray(population)
            & np.isfinite(values)
            & (values >= 0)
        )

        # -------------------------------------------------
        # 5. EXACT FOOTPRINT INTERSECTION MASK
        # -------------------------------------------------

        footprint_mask = geometry_mask(
            [pop_geometry],
            out_shape=population.shape,
            transform=transform,
            invert=True,
            all_touched=True,
        )

        # Cells whose pixel area intersects the SAR footprint.
        valid_population_inside = (
            valid_population & footprint_mask
        )

        # Cells entirely inside the footprint.
        footprint_mask_center = geometry_mask(
            [pop_geometry],
            out_shape=population.shape,
            transform=transform,
            invert=True,
            all_touched=False,
        )

        valid_population_center = (
            valid_population & footprint_mask_center
        )

        # -------------------------------------------------
        # 6. COVERAGE COMPARISON
        # -------------------------------------------------

        total_window_cells = population.size

        valid_window_cells = int(
            valid_population.sum()
        )

        intersecting_cells = int(
            footprint_mask.sum()
        )

        valid_intersecting_cells = int(
            valid_population_inside.sum()
        )

        center_inside_cells = int(
            footprint_mask_center.sum()
        )

        valid_center_inside_cells = int(
            valid_population_center.sum()
        )

        old_coverage = (
            100 * valid_window_cells / total_window_cells
        )

        intersection_coverage = (
            100 * valid_intersecting_cells / intersecting_cells
            if intersecting_cells else np.nan
        )

        center_coverage = (
            100 * valid_center_inside_cells / center_inside_cells
            if center_inside_cells else np.nan
        )

        # -------------------------------------------------
        # 7. REPORT
        # -------------------------------------------------

        print("\nSAMPLE INFORMATION")
        print("-" * 65)
        print("Sample:", SAMPLE)
        print("SAR CRS:", sar_crs)
        print("WorldPop CRS:", pop_ds.crs)
        print("WorldPop resolution:", pop_ds.res)

        print("\nCURRENT WINDOW METHOD")
        print("-" * 65)
        print("Total window cells:", total_window_cells)
        print("Valid population cells:", valid_window_cells)
        print("Current population coverage:", round(old_coverage, 2), "%")

        print("\nFOOTPRINT INTERSECTION METHOD")
        print("-" * 65)
        print("Cells intersecting SAR footprint:", intersecting_cells)
        print("Valid population cells in intersection:", valid_intersecting_cells)
        print("Intersection-based coverage:", round(intersection_coverage, 2), "%")

        print("\nCELL-CENTER METHOD")
        print("-" * 65)
        print("Cells with centers inside footprint:", center_inside_cells)
        print("Valid population cells with centers inside:", valid_center_inside_cells)
        print("Center-based coverage:", round(center_coverage, 2), "%")

        print("\nBOUNDARY DIFFERENCE")
        print("-" * 65)
        print(
            "Cells included by all-touched but not center method:",
            intersecting_cells - center_inside_cells,
        )

        print(
            "Valid population cells outside center-based footprint:",
            valid_intersecting_cells - valid_center_inside_cells,
        )

        print("=" * 65)


if __name__ == "__main__":
    main()
