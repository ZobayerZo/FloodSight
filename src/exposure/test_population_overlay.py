
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

SAMPLE = "Ghana_313799"
YEAR = 2018

NPZ_PATH = (
    ROOT / "data" / "processed" / "test_data"
    / f"{SAMPLE}.npz"
)

MANIFEST_PATH = (
    ROOT / "data" / "interim"
    / "sen1floods11_manifest.csv"
)

POPULATION_PATH = (
    ROOT / "data" / "external" / "worldpop"
    / "rasters" / f"Ghana_{YEAR}_100m.tif"
)

CHECKPOINT_PATH = (
    ROOT / "models" / "checkpoints"
    / "floodsight_unet_best.pth"
)

PREDICTION_THRESHOLD = 0.5

# Require at least 80% valid SAR coverage
# inside a WorldPop cell.
MIN_VALID_FRACTION = 0.80

# Ignore negligible floating-point artifacts.
MIN_WATER_FRACTION = 1e-6

# A cell is counted as a predicted-water cell
# if at least 1% of its area is estimated as water.
WATER_CELL_THRESHOLD = 0.01


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def reproject_average(
    source,
    source_transform,
    source_crs,
    destination_shape,
    destination_transform,
    destination_crs,
):
    """
    Reproject a source array onto the WorldPop grid
    using average resampling.

    For binary masks, this approximates the fraction
    of the destination cell covered by source pixels
    whose value is 1.
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


def get_valid_sar_mask(sar_dataset):
    """
    A SAR pixel is considered valid when:

    1. The dataset mask indicates valid data.
    2. Both VV and VH contain finite values.
    3. Neither band is masked.
    """

    sar_data = sar_dataset.read(masked=True)

    finite_bands = np.isfinite(
        sar_data.filled(np.nan)
    ).all(axis=0)

    dataset_valid = (
        sar_dataset.dataset_mask() > 0
    )

    band_valid = ~np.ma.getmaskarray(
        sar_data
    ).any(axis=0)

    sar_valid_mask = (
        dataset_valid
        & band_valid
        & finite_bands
    ).astype(np.uint8)

    return sar_valid_mask


# =========================================================
# 3. MAIN FUNCTION
# =========================================================

def main():

    print("=" * 65)
    print("FLOODSIGHT POPULATION EXPOSURE ASSESSMENT")
    print("=" * 65)

    # -----------------------------------------------------
    # STEP 1: CHECK REQUIRED FILES
    # -----------------------------------------------------

    required_files = [
        NPZ_PATH,
        MANIFEST_PATH,
        POPULATION_PATH,
        CHECKPOINT_PATH,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # -----------------------------------------------------
    # STEP 2: FIND ORIGINAL SAR FILE
    # -----------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    matching_rows = manifest[
        manifest["sample"] == SAMPLE
    ]

    if len(matching_rows) != 1:

        raise ValueError(
            f"Expected exactly one manifest row for {SAMPLE}; "
            f"found {len(matching_rows)}."
        )

    row = matching_rows.iloc[0]

    sar_path = ROOT / row["sar_file"]

    if not sar_path.exists():

        raise FileNotFoundError(
            f"Original SAR image not found: {sar_path}"
        )

    print("\nINPUT FILES")
    print("-" * 65)

    print("Sample:", SAMPLE)
    print("Processed image:", NPZ_PATH)
    print("Original SAR:", sar_path)
    print("Population raster:", POPULATION_PATH)

    # -----------------------------------------------------
    # STEP 3: LOAD PROCESSED IMAGE AND REFERENCE MASK
    # -----------------------------------------------------

    with np.load(NPZ_PATH) as data:

        image = data["image"].astype(
            np.float32
        )

        reference_valid_mask = data[
            "valid_mask"
        ].astype(np.uint8)

    if image.ndim != 3 or image.shape[0] != 2:

        raise ValueError(
            f"Expected image shape (2, H, W); got {image.shape}"
        )

    if reference_valid_mask.shape != image.shape[1:]:

        raise ValueError(
            "Reference mask dimensions do not match image."
        )

    if not np.isfinite(image).all():

        raise ValueError(
            "Processed SAR image contains non-finite values."
        )

    if not np.isin(
        np.unique(reference_valid_mask),
        [0, 1],
    ).all():

        raise ValueError(
            "Reference-validity mask must contain only 0 and 1."
        )

    # -----------------------------------------------------
    # STEP 4: LOAD MODEL AND GENERATE PREDICTION
    # -----------------------------------------------------

    model, checkpoint, device = load_model(
        CHECKPOINT_PATH
    )

    print("\nMODEL INFORMATION")
    print("-" * 65)

    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Prediction threshold:", PREDICTION_THRESHOLD)

    probabilities, prediction = predict_sample(
        model,
        image,
        device,
        threshold=PREDICTION_THRESHOLD,
    )

    if not np.isfinite(probabilities).all():

        raise ValueError(
            "Model probabilities contain non-finite values."
        )

    if prediction.shape != reference_valid_mask.shape:

        raise ValueError(
            "Prediction and reference-mask dimensions differ."
        )

    # -----------------------------------------------------
    # STEP 5: READ ORIGINAL SAR AND CREATE SAR VALID MASK
    # -----------------------------------------------------

    with rasterio.open(sar_path) as sar_ds:

        sar_transform = sar_ds.transform
        sar_crs = sar_ds.crs
        sar_bounds = sar_ds.bounds

        sar_width = sar_ds.width
        sar_height = sar_ds.height

        if sar_crs is None:

            raise ValueError(
                "Original SAR image has no CRS."
            )

        if (sar_height, sar_width) != prediction.shape:

            raise ValueError(
                "Original SAR dimensions do not match prediction."
            )

        if sar_ds.count != 2:

            raise ValueError(
                f"Expected two SAR bands; found {sar_ds.count}."
            )

        sar_valid_mask = get_valid_sar_mask(
            sar_ds
        )

    # The prediction is evaluated wherever the SAR
    # observations themselves are valid.
    predicted_water = (
        (prediction == 1)
        & (sar_valid_mask == 1)
    ).astype(np.uint8)

    # -----------------------------------------------------
    # STEP 6: SOURCE DATA DIAGNOSTICS
    # -----------------------------------------------------

    total_pixels = prediction.size

    sar_valid_pixels = int(
        sar_valid_mask.sum()
    )

    reference_valid_pixels = int(
        reference_valid_mask.sum()
    )

    predicted_water_pixels = int(
        predicted_water.sum()
    )

    print("\nSOURCE DATA DIAGNOSTICS")
    print("-" * 65)

    print("SAR dimensions:", sar_width, "x", sar_height)
    print("SAR CRS:", sar_crs)

    print("Total SAR pixels:", total_pixels)

    print(
        "SAR-valid pixels:",
        sar_valid_pixels
    )

    print(
        "Reference-label-valid pixels:",
        reference_valid_pixels
    )

    print(
        "SAR-valid but reference-invalid pixels:",
        int(
            (
                (sar_valid_mask == 1)
                & (reference_valid_mask == 0)
            ).sum()
        )
    )

    print(
        "Reference-valid but SAR-invalid pixels:",
        int(
            (
                (reference_valid_mask == 1)
                & (sar_valid_mask == 0)
            ).sum()
        )
    )

    print(
        "Predicted water pixels:",
        predicted_water_pixels
    )

    print(
        "Predicted water percentage of SAR-valid area:",
        round(
            100 * predicted_water_pixels / sar_valid_pixels,
            2,
        ) if sar_valid_pixels else "Undefined",
        "%",
    )

    print(
        "Probability minimum:",
        float(probabilities.min())
    )

    print(
        "Probability maximum:",
        float(probabilities.max())
    )

    print(
        "Probability mean:",
        float(probabilities.mean())
    )

    # -----------------------------------------------------
    # STEP 7: OPEN WORLDPOP RASTER
    # -----------------------------------------------------

    with rasterio.open(POPULATION_PATH) as pop_ds:

        if pop_ds.crs is None:

            raise ValueError(
                "WorldPop raster has no CRS."
            )

        # Transform SAR bounds if the two datasets
        # use different coordinate reference systems.
        if sar_crs != pop_ds.crs:

            population_bounds = transform_bounds(
                sar_crs,
                pop_ds.crs,
                sar_bounds.left,
                sar_bounds.bottom,
                sar_bounds.right,
                sar_bounds.top,
            )

        else:

            population_bounds = (
                sar_bounds.left,
                sar_bounds.bottom,
                sar_bounds.right,
                sar_bounds.top,
            )

        raw_window = from_bounds(
            *population_bounds,
            transform=pop_ds.transform,
        )

        full_window = Window(
            col_off=0,
            row_off=0,
            width=pop_ds.width,
            height=pop_ds.height,
        )

        try:

            window = raw_window.intersection(
                full_window
            )

        except Exception as exc:

            raise ValueError(
                "SAR footprint does not overlap WorldPop raster."
            ) from exc

        window = (
            window
            .round_offsets()
            .round_lengths()
        )

        if window.width <= 0 or window.height <= 0:

            raise ValueError(
                "No overlapping WorldPop cells found."
            )

        population = pop_ds.read(
            1,
            window=window,
            masked=True,
        )

        destination_transform = (
            pop_ds.window_transform(window)
        )

        destination_shape = population.shape

        print("\nWORLDPOP INFORMATION")
        print("-" * 65)

        print("CRS:", pop_ds.crs)
        print("Resolution:", pop_ds.res)
        print("Population grid shape:", destination_shape)

        # -------------------------------------------------
        # STEP 8: REPROJECT SAR VALIDITY AND WATER MASKS
        # -------------------------------------------------

        # Average of SAR validity mask.
        # Represents approximate valid SAR coverage
        # within each WorldPop destination cell.
        sar_valid_fraction = reproject_average(
            source=sar_valid_mask,
            source_transform=sar_transform,
            source_crs=sar_crs,
            destination_shape=destination_shape,
            destination_transform=destination_transform,
            destination_crs=pop_ds.crs,
        )

        # Average of predicted-water mask.
        # Invalid SAR pixels are already excluded because
        # predicted_water is zero at those locations.
        water_fraction_all = reproject_average(
            source=predicted_water,
            source_transform=sar_transform,
            source_crs=sar_crs,
            destination_shape=destination_shape,
            destination_transform=destination_transform,
            destination_crs=pop_ds.crs,
        )

        # Calculate water fraction conditional on
        # valid SAR coverage.
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

        # Remove numerical artifacts.
        water_fraction[
            water_fraction < MIN_WATER_FRACTION
        ] = 0.0

        # -------------------------------------------------
        # STEP 9: VALIDATE POPULATION DATA
        # -------------------------------------------------

        population_values = population.filled(
            np.nan
        ).astype(np.float64)

        valid_population = (
            ~np.ma.getmaskarray(population)
            & np.isfinite(population_values)
            & (population_values >= 0)
        )

        # A population cell is eligible only when:
        # 1. Population data is valid.
        # 2. At least 80% of the cell has valid SAR data.
        valid_overlap = (
            valid_population
            & (
                sar_valid_fraction
                >= MIN_VALID_FRACTION
            )
        )

        # -------------------------------------------------
        # STEP 10: CALCULATE EXPOSURE
        # -------------------------------------------------

        exposed_cells = (
            valid_overlap
            & (
                water_fraction
                >= WATER_CELL_THRESHOLD
            )
        )

        # Population count is weighted by estimated
        # water-covered fraction within each cell.
        exposed_population_per_cell = (
            population_values * water_fraction
        )

        estimated_exposure = float(
            np.nansum(
                np.where(
                    valid_overlap,
                    exposed_population_per_cell,
                    np.nan,
                )
            )
        )

        observed_population = float(
            np.where(
                valid_overlap,
                population_values,
                0.0,
            ).sum()
        )

        valid_population_cells = int(
            valid_population.sum()
        )

        valid_overlap_cells = int(
            valid_overlap.sum()
        )

        exposed_cell_count = int(
            exposed_cells.sum()
        )

        # -------------------------------------------------
        # STEP 11: COVERAGE AND FRACTION DIAGNOSTICS
        # -------------------------------------------------

        print("\nSAR COVERAGE DIAGNOSTICS")
        print("-" * 65)

        print(
            "Cells with any SAR coverage:",
            int(has_sar_coverage.sum())
        )

        print(
            "Cells with at least 80% SAR coverage:",
            valid_overlap_cells
        )

        print(
            "SAR coverage minimum:",
            float(sar_valid_fraction.min())
        )

        print(
            "SAR coverage maximum:",
            float(sar_valid_fraction.max())
        )

        print("\nWATER FRACTION DISTRIBUTION")
        print("-" * 65)

        for threshold in [0.01, 0.10, 0.50, 0.99]:

            count = int(
                (
                    valid_overlap
                    & (water_fraction >= threshold)
                ).sum()
            )

            print(
                f"Valid cells with at least "
                f"{threshold * 100:.0f}% water coverage:",
                count,
            )

        print("\nPOPULATION DATA DIAGNOSTICS")
        print("-" * 65)

        print(
            "WorldPop cells with valid population:",
            valid_population_cells
        )

        print(
            "WorldPop cells excluded for missing data:",
            int(
                (~valid_population).sum()
            )
        )

        print(
            "Population cells meeting SAR coverage threshold:",
            valid_overlap_cells
        )

        print(
            "Population cells with at least 1% water coverage:",
            exposed_cell_count
        )

        # -------------------------------------------------
        # STEP 12: FINAL EXPOSURE RESULTS
        # -------------------------------------------------

        exposure_percentage = (
            100 * estimated_exposure / observed_population
            if observed_population > 0
            else np.nan
        )

        print("\nFINAL POPULATION EXPOSURE RESULTS")
        print("=" * 65)

        print("Sample:", SAMPLE)

        print(
            "Total WorldPop cells in sample window:",
            int(population.size)
        )

        print(
            "Valid population cells:",
            valid_population_cells
        )

        print(
            "Valid SAR-population overlap cells:",
            valid_overlap_cells
        )

        print(
            "Predicted-water cells (>=1% coverage):",
            exposed_cell_count
        )

        print(
            "Population in valid overlap:",
            round(observed_population, 2)
        )

        print(
            "Estimated exposed population:",
            round(estimated_exposure, 2)
        )

        print(
            "Exposure fraction of observed population:",
            round(exposure_percentage, 2)
            if np.isfinite(exposure_percentage)
            else "Undefined",
            "%",
        )

        print(
            "Minimum SAR coverage required:",
            f"{MIN_VALID_FRACTION * 100:.0f}%"
        )

        print(
            "Water-cell reporting threshold:",
            f"{WATER_CELL_THRESHOLD * 100:.0f}%"
        )

        print(
            "Exposure calculation:",
            "WorldPop population count × estimated water fraction"
        )

        print("=" * 65)


# =========================================================
# 4. ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
