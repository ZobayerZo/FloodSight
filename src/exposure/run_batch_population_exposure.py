
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds, Window

from src.evaluation.predict import load_model, predict_sample


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

CHECKPOINT_PATH = (
    ROOT / "models" / "checkpoints"
    / "floodsight_unet_best.pth"
)

OUTPUT_DIR = (
    ROOT / "reports" / "tables"
)

SAMPLE_OUTPUT = (
    OUTPUT_DIR / "sample_level_population_exposure.csv"
)

EVENT_OUTPUT = (
    OUTPUT_DIR / "event_level_population_exposure.csv"
)

QUALITY_OUTPUT = (
    OUTPUT_DIR / "population_exposure_quality_summary.csv"
)

PREDICTION_THRESHOLD = 0.5

MIN_SAR_COVERAGE = 0.80

MIN_POPULATION_COVERAGE = 0.80

MIN_WATER_FRACTION = 1e-6

WATER_CELL_THRESHOLD = 0.01


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
# 2. HELPER FUNCTIONS
# =========================================================

def get_sar_valid_mask(sar_ds):
    """
    Create a SAR validity mask using both VV and VH bands.

    Reference-label validity is deliberately not used here.
    """

    sar_data = sar_ds.read(masked=True)

    finite = np.isfinite(
        sar_data.filled(np.nan)
    ).all(axis=0)

    band_valid = ~np.ma.getmaskarray(
        sar_data
    ).any(axis=0)

    dataset_valid = (
        sar_ds.dataset_mask() > 0
    )

    return (
        finite
        & band_valid
        & dataset_valid
    ).astype(np.uint8)


def reproject_average(
    source,
    source_transform,
    source_crs,
    destination_shape,
    destination_transform,
    destination_crs,
):
    """
    Reproject a mask using average resampling.

    For binary masks, the result approximates the fraction
    of a destination cell covered by source pixels equal to 1.
    """

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


def get_population_window(sar_bounds, sar_crs, pop_ds):
    """
    Find the WorldPop window intersecting a SAR sample.
    """

    if sar_crs != pop_ds.crs:

        bounds = transform_bounds(
            sar_crs,
            pop_ds.crs,
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

        return None

    window = (
        window
        .round_offsets()
        .round_lengths()
    )

    if window.width <= 0 or window.height <= 0:
        return None

    return window


def calculate_population_exposure(
    predicted_water,
    sar_valid_mask,
    sar_transform,
    sar_crs,
    sar_bounds,
    population_path,
):
    """
    Calculate population exposure within one SAR sample.

    Population values are weighted by the estimated fraction
    of each valid WorldPop cell covered by predicted water.
    """

    if not population_path.exists():

        raise FileNotFoundError(
            f"Population raster missing: {population_path}"
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

        # Reproject SAR-validity mask.
        sar_valid_fraction = reproject_average(
            source=sar_valid_mask,
            source_transform=sar_transform,
            source_crs=sar_crs,
            destination_shape=destination_shape,
            destination_transform=destination_transform,
            destination_crs=pop_ds.crs,
        )

        # Reproject predicted-water mask.
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

        has_sar_coverage = (
            sar_valid_fraction > 0
        )

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

        total_grid_cells = int(
            population.size
        )

        valid_population_cells = int(
            valid_population.sum()
        )

        # Population-data coverage.
        population_coverage = (
            valid_population_cells / total_grid_cells
            if total_grid_cells > 0
            else 0.0
        )

        # Identify population cells with sufficient SAR coverage.
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

        # Calculate observed population in eligible cells.
        observed_population = float(
            np.where(
                valid_overlap,
                population_values,
                0.0,
            ).sum()
        )

        # Calculate population-weighted water exposure.
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

        exposed_cells = (
            valid_overlap
            & (water_fraction >= WATER_CELL_THRESHOLD)
        )

        exposed_cell_count = int(
            exposed_cells.sum()
        )

        # -------------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------------

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
                population_coverage * 100, 2
            ),
            "valid_sar_population_cells": valid_overlap_cells,
            "sar_coverage_of_population_pct": round(
                sar_population_coverage * 100, 2
            ),
            "population_sum_valid_overlap": round(
                observed_population, 4
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
# 3. PROCESS ALL TEST SAMPLES
# =========================================================

def main():

    print("=" * 70)
    print("FLOODSIGHT BATCH POPULATION EXPOSURE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # LOAD MANIFEST
    # -----------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    test_manifest = manifest[
        manifest["split"] == "test_data"
    ].copy()

    processed_files = {
        path.stem
        for path in NPZ_DIR.glob("*.npz")
    }

    test_manifest = test_manifest[
        test_manifest["sample"].isin(processed_files)
    ].copy()

    test_manifest = test_manifest.sort_values(
        ["event", "sample"]
    )

    if len(test_manifest) != 88:

        raise ValueError(
            f"Expected 88 processed test samples, "
            f"found {len(test_manifest)}."
        )

    print("Processed test samples:", len(test_manifest))

    # -----------------------------------------------------
    # LOAD MODEL ONCE
    # -----------------------------------------------------

    model, checkpoint, device = load_model(
        CHECKPOINT_PATH
    )

    print("Model device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])

    # -----------------------------------------------------
    # PROCESS SAMPLES
    # -----------------------------------------------------

    results = []

    for index, (_, row) in enumerate(
        test_manifest.iterrows(),
        start=1,
    ):

        sample = row["sample"]
        event = row["event"]

        print(
            f"\n[{index}/88] Processing {sample} ({event})"
        )

        try:

            year = EVENT_YEARS[event]

            npz_path = (
                NPZ_DIR / f"{sample}.npz"
            )

            sar_path = (
                ROOT / row["sar_file"]
            )

            population_path = (
                POPULATION_DIR
                / f"{event}_{year}_100m.tif"
            )

            # ---------------------------------------------
            # LOAD NPZ
            # ---------------------------------------------

            with np.load(npz_path) as data:

                image = data["image"].astype(
                    np.float32
                )

                reference_valid_mask = data[
                    "valid_mask"
                ].astype(np.uint8)

            if not np.isfinite(image).all():

                raise ValueError(
                    "Processed image contains non-finite values."
                )

            # ---------------------------------------------
            # PREDICTION
            # ---------------------------------------------

            probabilities, prediction = predict_sample(
                model,
                image,
                device,
                threshold=PREDICTION_THRESHOLD,
            )

            # Reference mask is retained only for
            # diagnostics, not for exposure calculation.
            if reference_valid_mask.shape != prediction.shape:

                raise ValueError(
                    "Reference mask and prediction shapes differ."
                )

            # ---------------------------------------------
            # ORIGINAL SAR GEOREFERENCING
            # ---------------------------------------------

            with rasterio.open(sar_path) as sar_ds:

                sar_transform = sar_ds.transform
                sar_crs = sar_ds.crs
                sar_bounds = sar_ds.bounds

                if sar_crs is None:

                    raise ValueError(
                        "SAR image has no CRS."
                    )

                if prediction.shape != (
                    sar_ds.height,
                    sar_ds.width,
                ):

                    raise ValueError(
                        "Prediction dimensions do not match SAR raster."
                    )

                sar_valid_mask = get_sar_valid_mask(
                    sar_ds
                )

            sar_valid_pixels = int(
                sar_valid_mask.sum()
            )

            if sar_valid_pixels == 0:

                raise ValueError(
                    "No valid SAR pixels."
                )

            predicted_water = (
                (prediction == 1)
                & (sar_valid_mask == 1)
            ).astype(np.uint8)

            predicted_water_pixels = int(
                predicted_water.sum()
            )

            # ---------------------------------------------
            # POPULATION EXPOSURE
            # ---------------------------------------------

            exposure = calculate_population_exposure(
                predicted_water=predicted_water,
                sar_valid_mask=sar_valid_mask,
                sar_transform=sar_transform,
                sar_crs=sar_crs,
                sar_bounds=sar_bounds,
                population_path=population_path,
            )

            result = {
                "sample": sample,
                "event": event,
                "year": year,
                "sar_valid_pixels": sar_valid_pixels,
                "reference_valid_pixels": int(
                    reference_valid_mask.sum()
                ),
                "sar_valid_but_reference_invalid_pixels": int(
                    (
                        (sar_valid_mask == 1)
                        & (reference_valid_mask == 0)
                    ).sum()
                ),
                "predicted_water_pixels": predicted_water_pixels,
                "predicted_water_pct_sar_valid": round(
                    100 * predicted_water_pixels / sar_valid_pixels,
                    4,
                ),
                **exposure,
                "error": "",
            }

            print(
                "Water pixels:",
                predicted_water_pixels,
                "| Population coverage:",
                exposure["population_coverage_pct"],
                "% | Exposure:",
                exposure["estimated_exposed_population"],
                "| Status:",
                exposure["status"],
            )

        except Exception as exc:

            print("ERROR:", str(exc))

            result = {
                "sample": sample,
                "event": event,
                "year": EVENT_YEARS.get(event, np.nan),
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

    event_results = []

    for event, group in sample_df.groupby(
        "event",
        sort=True,
    ):

        valid_exposure = group[
            group["estimated_exposed_population"].notna()
        ] if "estimated_exposed_population" in group.columns else group.iloc[0:0]

        adequate = group[
            group["status"] == "ADEQUATE"
        ]

        partial = group[
            group["status"] == "PARTIAL_COVERAGE"
        ]

        no_data = group[
            group["status"] == "NO_POPULATION_DATA"
        ]

        event_results.append({
            "event": event,
            "sample_count": len(group),
            "adequate_samples": len(adequate),
            "partial_samples": len(partial),
            "no_population_data_samples": len(no_data),
            "processing_errors": int(
                (group["status"] == "PROCESSING_ERROR").sum()
            ),
            "mean_population_coverage_pct": (
                round(group["population_coverage_pct"].mean(), 2)
                if "population_coverage_pct" in group.columns
                else np.nan
            ),
            "samples_with_exposure_estimate": len(valid_exposure),
            "sum_observed_population": (
                round(
                    valid_exposure[
                        "population_sum_valid_overlap"
                    ].sum(),
                    2,
                )
                if "population_sum_valid_overlap" in valid_exposure
                else np.nan
            ),
            "sum_sample_exposure_estimates": (
                round(
                    valid_exposure[
                        "estimated_exposed_population"
                    ].sum(),
                    2,
                )
                if len(valid_exposure)
                else np.nan
            ),
        })

    event_df = pd.DataFrame(
        event_results
    )

    event_df.to_csv(
        EVENT_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # QUALITY SUMMARY
    # -----------------------------------------------------

    quality_df = (
        sample_df["status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="sample_count")
    )

    quality_df.to_csv(
        QUALITY_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)

    print("Samples processed:", len(sample_df))

    print("\nQuality status:")
    print(
        sample_df["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nMean population coverage:")
    print(
        round(
            sample_df["population_coverage_pct"].mean(),
            2,
        ),
        "%",
    )

    print("\nOutput files:")
    print(SAMPLE_OUTPUT)
    print(EVENT_OUTPUT)
    print(QUALITY_OUTPUT)

    print("\nIMPORTANT:")
    print(
        "Event-level sums are sums of sample-level "
        "observed-area estimates."
    )

    print(
        "They are not deduplicated event-wide population "
        "exposure totals."
    )

    print("=" * 70)


# =========================================================
# 4. ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
