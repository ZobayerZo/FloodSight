import json
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATS_PATH = PROJECT_ROOT / "configs/normalization_stats.json"


def load_normalization_stats():
    """Load training-only SAR normalization statistics."""

    with open(STATS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def preprocess_sample(sar_path, label_path, stats=None):
    """
    Preprocess one Sentinel-1 SAR sample.

    Returns:
        image: Float32 array with shape (2, H, W).
        target: Float32 binary label array with shape (H, W).
        valid_mask: Boolean mask identifying usable supervised pixels.
    """

    sar_path = Path(sar_path)
    label_path = Path(label_path)

    if stats is None:
        stats = load_normalization_stats()

    with rasterio.open(sar_path) as src:
        sar = src.read().astype(np.float32)

    with rasterio.open(label_path) as src:
        label = src.read(1)

    if sar.shape[1:] != label.shape:
        raise ValueError(
            f"Image and label dimensions do not match: {sar_path.name}"
        )

    # Both SAR bands must be finite.
    sar_valid = np.isfinite(sar).all(axis=0)

    # Only labels 0 and 1 are valid.
    label_valid = (label == 0) | (label == 1)

    # Pixels usable for supervised learning.
    valid_mask = sar_valid & label_valid

    # Standardize VV and VH independently.
    normalized = np.zeros_like(sar, dtype=np.float32)

    for index, band_name in enumerate(["VV", "VH"]):

        mean = stats[band_name]["mean"]
        std = stats[band_name]["std"]

        normalized[index] = np.where(
            sar_valid,
            (sar[index] - mean) / std,
            0.0
        )

    # Ensure no NaN or infinity reaches the model.
    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Binary target: 1 = water, 0 = non-water.
    target = (label == 1).astype(np.float32)

    return normalized, target, valid_mask


def preprocess_sar_for_inference(sar_path, stats=None):
    """
    Preprocess a two-band Sentinel-1 SAR GeoTIFF for application inference.

    Args:
        sar_path: Path to a two-band SAR GeoTIFF.
        stats: Optional training normalization statistics.

    Returns:
        image: Float32 normalized array with shape (2, H, W).
        sar_valid_mask: Boolean mask identifying valid SAR pixels.
        metadata: Spatial metadata needed for geospatial outputs.
    """

    sar_path = Path(sar_path)

    if not sar_path.exists():
        raise FileNotFoundError(
            f"SAR image not found: {sar_path}"
        )

    if stats is None:
        stats = load_normalization_stats()

    with rasterio.open(sar_path) as src:

        if src.count != 2:
            raise ValueError(
                f"Expected exactly 2 SAR bands (VV, VH), "
                f"but found {src.count}."
            )

        sar = src.read(
            masked=True
        ).astype(np.float32)

        metadata = {
            "height": src.height,
            "width": src.width,
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "nodata": src.nodata,
        }

        dataset_valid = src.dataset_mask() > 0

    sar_data = np.asarray(sar.filled(np.nan), dtype=np.float32)

    # Both SAR bands and the dataset mask must be valid.
    sar_valid_mask = (
        np.isfinite(sar_data).all(axis=0)
        & dataset_valid
        & ~np.ma.getmaskarray(sar).any(axis=0)
    )

    normalized = np.zeros_like(
        sar_data,
        dtype=np.float32
    )

    for index, band_name in enumerate(["VV", "VH"]):

        mean = stats[band_name]["mean"]
        std = stats[band_name]["std"]

        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(
                f"Invalid normalization statistics for {band_name}."
            )

        normalized[index] = np.where(
            sar_valid_mask,
            (sar_data[index] - mean) / std,
            0.0
        )

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return normalized, sar_valid_mask, metadata


if __name__ == "__main__":

    manifest_path = PROJECT_ROOT / "data/interim/sen1floods11_manifest.csv"

    import pandas as pd

    manifest = pd.read_csv(manifest_path)

    sample = manifest[
        (manifest["split"] == "train_data") &
        (manifest["usable"] == True)
    ].iloc[0]

    image, target, valid_mask = preprocess_sample(
        PROJECT_ROOT / sample["sar_file"],
        PROJECT_ROOT / sample["label_file"]
    )

    print("PREPROCESSING TEST")
    print("=" * 50)
    print("Sample:", sample["sample"])
    print("Image shape:", image.shape)
    print("Target shape:", target.shape)
    print("Validity mask shape:", valid_mask.shape)
    print("Image dtype:", image.dtype)
    print("Target dtype:", target.dtype)
    print("Valid pixels:", valid_mask.sum())
    print("Water pixels:", target[valid_mask].sum())
    print("Non-water pixels:", (valid_mask & (target == 0)).sum())
    print("NaN values:", np.isnan(image).sum())
    print("Infinite values:", np.isinf(image).sum())

    assert image.shape == (2, 512, 512)
    assert target.shape == (512, 512)
    assert valid_mask.shape == (512, 512)
    assert np.isfinite(image).all()
    assert valid_mask.any()

    print("\nPreprocessing test passed.")
