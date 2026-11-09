
from pathlib import Path

import numpy as np
import rasterio

from rasterio.warp import (
    reproject,
    Resampling,
    transform_bounds,
)

from rasterio.windows import from_bounds, Window


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POPULATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "worldpop"
    / "rasters"
)

MIN_SAR_COVERAGE = 0.80
MIN_POPULATION_COVERAGE = 0.80

MIN_WATER_FRACTION = 1e-6
WATER_CELL_THRESHOLD = 0.01


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def reproject_average(
    source,
    source_transform,
    source_crs,
    destination_shape,
    destination_transform,
    destination_crs,
):
    """Reproject a binary mask using average resampling."""

    destination = np.zeros(
        destination_shape,
        dtype=np.float32,
    )

    reproject(
        source=source.astype(np.float32),
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        src_nodata=None,
        dst_nodata=0,
        resampling=Resampling.average,
    )

    return destination


def get_population_window(
    sar_bounds,
    sar_crs,
    population_dataset,
):
    """Find the WorldPop window intersecting the SAR footprint."""

    if sar_crs != population_dataset.crs:

        bounds = transform_bounds(
            sar_crs,
            population_dataset.crs,
            sar_bounds.left,
            sar_bounds.bottom,
            sar_bounds.right,
            sar_bounds.top,
        )

    else:

        bounds = (
            sar_bounds.left,
            sar_bounds.bottom,
            sar_bounds.right,
            sar_bounds.top,
        )

    raw_window = from_bounds(
        *bounds,
        transform=population_dataset.transform,
    )

    full_window = Window(
        0,
        0,
        population_dataset.width,
        population_dataset.height,
    )

    try:
        window = raw_window.intersection(full_window)

    except Exception:
        return None

    window = (
        window
        .round_offsets()
        .round_lengths()
    )

    if window.width <= 0 or window.height <= 0:
        return None

    return window


# =========================================================
# MAIN EXPOSURE FUNCTION
# =========================================================

def calculate_population_exposure(
    predicted_water,
    sar_valid_mask,
    metadata,
    population_path,
):
    """
    Calculate population exposure for one SAR prediction.

    Args:
        predicted_water:
            Binary predicted-water array (H, W).

        sar_valid_mask:
            Boolean SAR validity array (H, W).

        metadata:
            Spatial metadata returned by
            preprocess_sar_for_inference().

        population_path:
            Path to the selected WorldPop raster.

    Returns:
        Dictionary containing exposure estimates,
        spatial coverage metrics and quality status.
    """

    population_path = Path(population_path)

    if not population_path.exists():
        raise FileNotFoundError(
            f"Population raster not found: {population_path}"
        )

    predicted_water = np.asarray(
        predicted_water
    ).astype(bool)

    sar_valid_mask = np.asarray(
        sar_valid_mask
    ).astype(bool)

    if predicted_water.ndim != 2:
        raise ValueError(
            "Predicted water mask must be two-dimensional."
        )

    if predicted_water.shape != sar_valid_mask.shape:
        raise ValueError(
            "Prediction and SAR validity dimensions differ."
        )

    sar_transform = metadata["transform"]
    sar_crs = metadata["crs"]
    sar_bounds = metadata["bounds"]

    if sar_crs is None:
        raise ValueError("SAR raster has no CRS.")

    if not sar_valid_mask.any():
        raise ValueError("No valid SAR pixels.")

    # Invalid SAR pixels cannot contribute to exposure.
    predicted_water = (
        predicted_water & sar_valid_mask
    )

    with rasterio.open(population_path) as pop_ds:

        window = get_population_window(
            sar_bounds,
            sar_crs,
            pop_ds,
        )

        if window is None:
            return {
                "population_grid_cells": 0,
                "valid_population_cells": 0,
                "population_coverage_pct": 0.0,
                "valid_sar_population_cells": 0,
                "sar_coverage_of_population_pct": 0.0,
                "population_sum_valid_overlap": np.nan,
                "estimated_exposed_population": np.nan,
                "predicted_water_cells": 0,
                "status": "NO_RASTER_OVERLAP",
            }

        population = pop_ds.read(
            1,
            window=window,
            masked=True,
        )

        destination_transform = (
            pop_ds.window_transform(window)
        )

        destination_shape = population.shape

        # Reproject SAR validity.
        sar_valid_fraction = reproject_average(
            source=sar_valid_mask,
            source_transform=sar_transform,
            source_crs=sar_crs,
            destination_shape=destination_shape,
            destination_transform=destination_transform,
            destination_crs=pop_ds.crs,
        )

        # Reproject predicted water.
        water_fraction_all = reproject_average(
            source=predicted_water,
            source_transform=sar_transform,
            source_crs=sar_crs,
            destination_shape=destination_shape,
            destination_transform=destination_transform,
            destination_crs=pop_ds.crs,
        )

        # Calculate water fraction conditional on SAR validity.
        water_fraction = np.zeros(
            destination_shape,
            dtype=np.float32,
        )

        has_sar_coverage = sar_valid_fraction > 0

        water_fraction[has_sar_coverage] = (
            water_fraction_all[has_sar_coverage]
            / sar_valid_fraction[has_sar_coverage]
        )

        water_fraction = np.clip(
            water_fraction,
            0.0,
            1.0,
        )

        water_fraction[
            water_fraction < MIN_WATER_FRACTION
        ] = 0.0

        # Identify valid population cells.
        population_values = population.filled(
            np.nan
        ).astype(np.float64)

        valid_population = (
            ~np.ma.getmaskarray(population)
            & np.isfinite(population_values)
            & (population_values >= 0)
        )

        total_grid_cells = int(population.size)

        valid_population_cells = int(
            valid_population.sum()
        )

        population_coverage = (
            valid_population_cells / total_grid_cells
            if total_grid_cells > 0
            else 0.0
        )

        # Apply SAR coverage eligibility threshold.
        valid_overlap = (
            valid_population
            & (sar_valid_fraction >= MIN_SAR_COVERAGE)
        )

        valid_overlap_cells = int(
            valid_overlap.sum()
        )

        sar_population_coverage = (
            valid_overlap_cells / valid_population_cells
            if valid_population_cells > 0
            else 0.0
        )

        # Available population in eligible cells.
        observed_population = float(
            np.where(
                valid_overlap,
                population_values,
                0.0,
            ).sum()
        )

        # Population-weighted water exposure.
        exposed_population_per_cell = (
            population_values * water_fraction
        )

        estimated_exposure = (
            float(
                np.where(
                    valid_overlap,
                    exposed_population_per_cell,
                    0.0,
                ).sum()
            )
            if valid_overlap_cells > 0
            else np.nan
        )

        # Count cells with at least 1% predicted water.
        exposed_cells = (
            valid_overlap
            & (water_fraction >= WATER_CELL_THRESHOLD)
        )

        exposed_cell_count = int(
            exposed_cells.sum()
        )

        # Quality classification.
        if valid_population_cells == 0:

            status = "NO_POPULATION_DATA"

        elif (
            population_coverage >= MIN_POPULATION_COVERAGE
            and sar_population_coverage >= MIN_SAR_COVERAGE
        ):

            status = "ADEQUATE"

        else:

            status = "PARTIAL_COVERAGE"

        return {
            "population_grid_cells": total_grid_cells,
            "valid_population_cells": valid_population_cells,
            "population_coverage_pct": round(
                population_coverage * 100,
                2,
            ),
            "valid_sar_population_cells": valid_overlap_cells,
            "sar_coverage_of_population_pct": round(
                sar_population_coverage * 100,
                2,
            ),
            "population_sum_valid_overlap": round(
                observed_population,
                4,
            ),
            "estimated_exposed_population": (
                round(estimated_exposure, 4)
                if np.isfinite(estimated_exposure)
                else np.nan
            ),
            "predicted_water_cells": exposed_cell_count,
            "status": status,
        }


# =========================================================
# EVENT RASTER SELECTION
# =========================================================

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


def get_population_raster(event):
    """Return the locally available WorldPop raster for an event."""

    if event not in EVENT_YEARS:
        raise ValueError(
            f"Unknown event: {event}"
        )

    year = EVENT_YEARS[event]

    population_path = (
        POPULATION_DIR
        / f"{event}_{year}_100m.tif"
    )

    if not population_path.exists():
        raise FileNotFoundError(
            f"WorldPop raster unavailable: {population_path}"
        )

    return population_path
