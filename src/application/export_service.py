
from io import BytesIO

import numpy as np
from rasterio.io import MemoryFile


# =========================================================
# GEOTIFF EXPORT
# =========================================================

def create_geotiff_bytes(
    array,
    metadata,
    valid_mask,
    nodata,
    dtype,
    description,
):
    """
    Convert a NumPy raster into an in-memory GeoTIFF.

    Preserves the original SAR spatial reference,
    affine transform and raster dimensions.
    """

    array = np.asarray(array)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if array.ndim != 2:
        raise ValueError(
            "Export raster must be two-dimensional."
        )

    if array.shape != valid_mask.shape:
        raise ValueError(
            "Raster and validity mask dimensions differ."
        )

    if (
        array.shape[0] != metadata["height"]
        or array.shape[1] != metadata["width"]
    ):
        raise ValueError(
            "Raster dimensions do not match source metadata."
        )

    if metadata["crs"] is None:
        raise ValueError(
            "Cannot export GeoTIFF without a CRS."
        )

    output = np.full(
        array.shape,
        nodata,
        dtype=dtype,
    )

    finite_mask = np.isfinite(array)
    export_mask = valid_mask & finite_mask

    output[export_mask] = array[export_mask].astype(dtype)

    profile = {
        "driver": "GTiff",
        "height": output.shape[0],
        "width": output.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": metadata["crs"],
        "transform": metadata["transform"],
        "nodata": nodata,
        "compress": "deflate",
    }

    with MemoryFile() as memory_file:

        with memory_file.open(**profile) as dataset:

            dataset.write(output, 1)

            dataset.set_band_description(
                1,
                description,
            )

            dataset.update_tags(
                PRODUCT=description,
                SOURCE="FloodSight",
                SPATIAL_REFERENCE="Original SAR image",
            )

        return memory_file.read()


# =========================================================
# FLOOD PREDICTION EXPORT
# =========================================================

def export_prediction_geotiff(prediction_result):
    """
    Export binary flood prediction.

    Values:
        0 = Valid non-water pixel
        1 = Predicted water pixel
        255 = NoData
    """

    prediction = np.asarray(
        prediction_result["prediction"]
    )

    valid_mask = np.asarray(
        prediction_result["sar_valid_mask"]
    ).astype(bool)

    metadata = prediction_result["metadata"]

    return create_geotiff_bytes(
        array=prediction,
        metadata=metadata,
        valid_mask=valid_mask,
        nodata=255,
        dtype="uint8",
        description="Binary flood prediction",
    )


# =========================================================
# WATER PROBABILITY EXPORT
# =========================================================

def export_probability_geotiff(prediction_result):
    """
    Export continuous water probabilities.

    Values:
        0.0 to 1.0 = Water probability
        -9999 = NoData
    """

    probabilities = np.asarray(
        prediction_result["probabilities"]
    )

    valid_mask = np.asarray(
        prediction_result["sar_valid_mask"]
    ).astype(bool)

    metadata = prediction_result["metadata"]

    return create_geotiff_bytes(
        array=probabilities,
        metadata=metadata,
        valid_mask=valid_mask,
        nodata=-9999.0,
        dtype="float32",
        description="Water probability",
    )


# =========================================================
# UNCERTAINTY EXPORT
# =========================================================

def export_uncertainty_geotiff(prediction_result):
    """
    Export MC Dropout predictive variance.

    Values:
        Float32 = Predictive variance
        -9999 = NoData
    """

    uncertainty = prediction_result.get("uncertainty")

    if uncertainty is None:
        raise ValueError(
            "Uncertainty data is unavailable. "
            "Enable MC Dropout during analysis."
        )

    variance = uncertainty.get("predictive_variance")

    if variance is None:
        raise ValueError(
            "Predictive variance is unavailable."
        )

    valid_mask = np.asarray(
        prediction_result["sar_valid_mask"]
    ).astype(bool)

    metadata = prediction_result["metadata"]

    return create_geotiff_bytes(
        array=variance,
        metadata=metadata,
        valid_mask=valid_mask,
        nodata=-9999.0,
        dtype="float32",
        description="MC Dropout predictive variance",
    )


# =========================================================
# LAND-COVER EXPORT
# =========================================================

def export_landcover_geotiff(
    landcover_result,
    prediction_result,
):
    """
    Export ESA WorldCover land-cover classification.

    The land-cover raster is already aligned to the SAR grid
    by LandCoverExposureService.

    Values follow ESA WorldCover class codes:
        10  = Tree cover
        20  = Shrubland
        30  = Grassland
        40  = Cropland
        50  = Built-up
        60  = Bare / sparse vegetation
        70  = Snow and ice
        80  = Permanent water bodies
        90  = Herbaceous wetland
        95  = Mangroves
        100 = Moss and lichen
        0   = NoData / unavailable
    """

    if not landcover_result:
        raise ValueError(
            "Land-cover results are unavailable."
        )

    landcover_raster = landcover_result.get(
        "landcover_raster"
    )

    landcover_valid_mask = landcover_result.get(
        "landcover_valid_mask"
    )

    if landcover_raster is None:
        raise ValueError(
            "Land-cover raster is unavailable."
        )

    if landcover_valid_mask is None:
        raise ValueError(
            "Land-cover validity mask is unavailable."
        )

    metadata = prediction_result["metadata"]

    sar_valid_mask = np.asarray(
        prediction_result["sar_valid_mask"],
        dtype=bool,
    )

    landcover_valid_mask = np.asarray(
        landcover_valid_mask,
        dtype=bool,
    )

    # Export only pixels valid in both the SAR image
    # and the land-cover classification.
    export_valid_mask = (
        sar_valid_mask & landcover_valid_mask
    )

    return create_geotiff_bytes(
        array=landcover_raster,
        metadata=metadata,
        valid_mask=export_valid_mask,
        nodata=0,
        dtype="uint8",
        description="ESA WorldCover 2021 land-cover classification",
    )


# =========================================================
# EXPOSED CROPLAND EXPORT
# =========================================================

def export_exposed_cropland_geotiff(
    landcover_result,
    prediction_result,
):
    """
    Export the spatial overlap between predicted floodwater
    and cropland.

    Values:
        0 = Valid pixel without predicted water-cropland overlap
        1 = Predicted water overlapping cropland
        255 = NoData

    IMPORTANT:
        This represents estimated spatial exposure, not
        confirmed agricultural damage.
    """

    if not landcover_result:
        raise ValueError(
            "Land-cover results are unavailable."
        )

    exposed_cropland_mask = landcover_result.get(
        "exposed_cropland_mask"
    )

    landcover_valid_mask = landcover_result.get(
        "landcover_valid_mask"
    )

    if exposed_cropland_mask is None:
        raise ValueError(
            "Exposed cropland mask is unavailable."
        )

    if landcover_valid_mask is None:
        raise ValueError(
            "Land-cover validity mask is unavailable."
        )

    metadata = prediction_result["metadata"]

    sar_valid_mask = np.asarray(
        prediction_result["sar_valid_mask"],
        dtype=bool,
    )

    landcover_valid_mask = np.asarray(
        landcover_valid_mask,
        dtype=bool,
    )

    exposed_cropland_mask = np.asarray(
        exposed_cropland_mask,
        dtype=bool,
    )

    # A pixel is valid only when both SAR and
    # WorldCover data are valid.
    export_valid_mask = (
        sar_valid_mask & landcover_valid_mask
    )

    return create_geotiff_bytes(
        array=exposed_cropland_mask.astype(np.uint8),
        metadata=metadata,
        valid_mask=export_valid_mask,
        nodata=255,
        dtype="uint8",
        description="Predicted floodwater and cropland overlap",
    )
