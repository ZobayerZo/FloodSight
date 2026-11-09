
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds, Window

from src.evaluation.predict import load_model, predict_sample


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

SAMPLE = "Ghana_406026"
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

OUTPUT_PATH = (
    ROOT / "reports" / "figures"
    / f"{SAMPLE}_population_overlay.png"
)

THRESHOLD = 0.5
MIN_VALID_FRACTION = 0.80


# ---------------------------------------------------------
# REPROJECTION HELPER
# ---------------------------------------------------------

def reproject_average(
    source,
    source_transform,
    source_crs,
    destination_shape,
    destination_transform,
    destination_crs,
):

    destination = np.zeros(
        destination_shape,
        dtype=np.float32
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


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    # 1. Load sample information

    manifest = pd.read_csv(MANIFEST_PATH)

    rows = manifest[
        manifest["sample"] == SAMPLE
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected one manifest entry for {SAMPLE}"
        )

    sar_path = ROOT / rows.iloc[0]["sar_file"]

    # 2. Load processed image

    with np.load(NPZ_PATH) as data:

        image = data["image"].astype(np.float32)
        valid_mask = data["valid_mask"].astype(np.uint8)

    # 3. Generate model prediction

    model, checkpoint, device = load_model(
        CHECKPOINT_PATH
    )

    probabilities, prediction = predict_sample(
        model,
        image,
        device,
        threshold=THRESHOLD,
    )

    predicted_water = (
        (prediction == 1) & (valid_mask == 1)
    ).astype(np.uint8)

    print("Sample:", SAMPLE)
    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Predicted water pixels:", int(predicted_water.sum()))

    # 4. Read SAR geographic information

    with rasterio.open(sar_path) as sar_ds:

        sar_transform = sar_ds.transform
        sar_crs = sar_ds.crs
        sar_bounds = sar_ds.bounds

    # 5. Read corresponding WorldPop area

    with rasterio.open(POPULATION_PATH) as pop_ds:

        raw_window = from_bounds(
            sar_bounds.left,
            sar_bounds.bottom,
            sar_bounds.right,
            sar_bounds.top,
            transform=pop_ds.transform,
        )

        full_window = Window(
            col_off=0,
            row_off=0,
            width=pop_ds.width,
            height=pop_ds.height,
        )

        window = raw_window.intersection(full_window)
        window = window.round_offsets().round_lengths()

        population = pop_ds.read(
            1,
            window=window,
            masked=True,
        )

        destination_transform = (
            pop_ds.window_transform(window)
        )

        destination_shape = population.shape

        population_bounds = rasterio.windows.bounds(
            window,
            pop_ds.transform
        )

        # 6. Calculate water and valid fractions

        water_fraction_all = reproject_average(
            predicted_water,
            sar_transform,
            sar_crs,
            destination_shape,
            destination_transform,
            pop_ds.crs,
        )

        valid_fraction = reproject_average(
            valid_mask,
            sar_transform,
            sar_crs,
            destination_shape,
            destination_transform,
            pop_ds.crs,
        )

        water_fraction = np.zeros(
            destination_shape,
            dtype=np.float32
        )

        covered = valid_fraction > 0

        water_fraction[covered] = (
            water_fraction_all[covered]
            / valid_fraction[covered]
        )

        water_fraction = np.clip(
            water_fraction,
            0.0,
            1.0
        )

        water_fraction[water_fraction < 1e-6] = 0.0

        # 7. Calculate population exposure

        population_values = population.filled(0)

        valid_population = (
            ~np.ma.getmaskarray(population)
            & np.isfinite(population_values)
            & (population_values >= 0)
        )

        valid_overlap = (
            valid_population
            & (valid_fraction >= MIN_VALID_FRACTION)
        )

        exposure_per_cell = (
            population_values * water_fraction
        )

        exposure_per_cell[~valid_overlap] = np.nan

        estimated_exposure = float(
            np.nansum(exposure_per_cell)
        )

        # 8. Prepare display arrays

        population_display = np.where(
            valid_overlap,
            population_values,
            np.nan
        )

        water_display = np.where(
            valid_overlap,
            water_fraction,
            np.nan
        )

    # 9. Generate figure

    sar_extent = (
        sar_bounds.left,
        sar_bounds.right,
        sar_bounds.bottom,
        sar_bounds.top,
    )

    pop_extent = (
        population_bounds[0],
        population_bounds[2],
        population_bounds[1],
        population_bounds[3],
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 7),
        constrained_layout=True,
    )

    # Panel 1: SAR prediction

    axes[0].imshow(
        predicted_water,
        extent=sar_extent,
        origin="upper",
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title(
        "Sentinel-1 predicted water",
        fontsize=13
    )

    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # Panel 2: Water fraction

    image2 = axes[1].imshow(
        water_display,
        extent=pop_extent,
        origin="upper",
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Predicted water fraction per WorldPop cell",
        fontsize=13
    )

    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    fig.colorbar(
        image2,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="Water-covered fraction"
    )

    # Panel 3: Estimated exposure

    image3 = axes[2].imshow(
        exposure_per_cell,
        extent=pop_extent,
        origin="upper",
        interpolation="nearest",
        cmap="OrRd",
    )

    axes[2].set_title(
        "Estimated population exposure per cell",
        fontsize=13
    )

    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")

    fig.colorbar(
        image3,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
        label="Estimated population"
    )

    fig.suptitle(
        f"FloodSight: Population Exposure Overlay — {SAMPLE}\n"
        f"Estimated exposure: {estimated_exposure:.2f}",
        fontsize=15,
    )

    # 10. Save figure

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nVISUALIZATION COMPLETE")
    print("=" * 60)
    print("Estimated exposure:", round(estimated_exposure, 2))
    print("Output figure:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
