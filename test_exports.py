
import numpy as np
import rasterio

from rasterio.io import MemoryFile

from src.application.application_service import FloodSightApplication

from src.application.export_service import (
    export_prediction_geotiff,
    export_probability_geotiff,
    export_uncertainty_geotiff,
    export_landcover_geotiff,
    export_exposed_cropland_geotiff,
)


# =========================================================
# CONFIGURATION
# =========================================================

SAR_PATH = (
    "data/raw/Sen1Floods11/v1.1/data/"
    "flood_events/HandLabeled/S1Hand/"
    "Ghana_406026_S1Hand.tif"
)

EVENT = "Ghana"


# =========================================================
# GEO TIFF VALIDATION
# =========================================================

def inspect_geotiff(
    name,
    geotiff_bytes,
    expected_shape,
    expected_crs,
    expected_transform,
    expected_nodata,
    expected_dtype,
    allowed_values=None,
    value_range=None,
):
    """Inspect an in-memory GeoTIFF and validate its metadata."""

    if not geotiff_bytes:
        raise AssertionError(
            f"{name}: Empty GeoTIFF output."
        )

    with MemoryFile(geotiff_bytes) as memory_file:

        with memory_file.open() as dataset:

            raster = dataset.read(1)

            print(f"\n--- {name} ---")

            print("Dimensions:", dataset.shape)
            print("CRS:", dataset.crs)
            print("Data type:", dataset.dtypes[0])
            print("NoData:", dataset.nodata)
            print("Valid pixels:", dataset.count)

            # 1. Dimensions
            assert dataset.shape == expected_shape, (
                f"{name}: Incorrect dimensions."
            )

            # 2. CRS
            assert dataset.crs == expected_crs, (
                f"{name}: CRS mismatch."
            )

            # 3. Affine transform
            assert dataset.transform == expected_transform, (
                f"{name}: Transform mismatch."
            )

            # 4. NoData
            assert dataset.nodata == expected_nodata, (
                f"{name}: NoData mismatch."
            )

            # 5. Data type
            assert dataset.dtypes[0] == expected_dtype, (
                f"{name}: Data type mismatch."
            )

            # 6. Valid pixel values
            valid_pixels = raster[
                raster != expected_nodata
            ]

            assert valid_pixels.size > 0, (
                f"{name}: No valid pixels found."
            )

            if allowed_values is not None:

                unique_values = np.unique(valid_pixels)

                assert np.isin(
                    unique_values,
                    allowed_values,
                ).all(), (
                    f"{name}: Unexpected pixel values: "
                    f"{unique_values}"
                )

                print("Unique values:", unique_values)

            if value_range is not None:

                minimum, maximum = value_range

                assert (
                    valid_pixels.min() >= minimum
                    and valid_pixels.max() <= maximum
                ), (
                    f"{name}: Values outside expected range."
                )

                print(
                    "Value range:",
                    float(valid_pixels.min()),
                    "to",
                    float(valid_pixels.max()),
                )

            print("Valid pixel count:", valid_pixels.size)
            print("Validation: PASSED")

            return raster


# =========================================================
# MAIN TEST
# =========================================================

def main():

    print("\n==========================================")
    print(" FLOODSIGHT GEOTIFF EXPORT TEST")
    print("==========================================")

    # -----------------------------------------------------
    # 1. Run the complete application pipeline
    # -----------------------------------------------------

    print("\nRunning FloodSight application...")

    application = FloodSightApplication(
        mc_passes=20,
    )

    result = application.analyze(
        sar_path=SAR_PATH,
        event=EVENT,
        include_uncertainty=True,
        original_filename="Ghana_406026_S1Hand.tif",
    )

    if not result["success"]:
        raise RuntimeError(
            "FloodSight application analysis failed."
        )

    prediction_result = result["prediction"]
    landcover_result = result["landcover_exposure"]

    metadata = prediction_result["metadata"]

    expected_shape = (
        metadata["height"],
        metadata["width"],
    )

    expected_crs = metadata["crs"]
    expected_transform = metadata["transform"]

    sar_valid_mask = np.asarray(
        prediction_result["sar_valid_mask"],
        dtype=bool,
    )

    print("\nApplication analysis completed.")

    print("Raster dimensions:", expected_shape)
    print("CRS:", expected_crs)

    # -----------------------------------------------------
    # 2. Generate all five GeoTIFF exports
    # -----------------------------------------------------

    print("\nGenerating GeoTIFF exports...")

    exports = {
        "Flood prediction": (
            export_prediction_geotiff(
                prediction_result
            ),
            255,
            "uint8",
            [0, 1],
            None,
        ),

        "Water probability": (
            export_probability_geotiff(
                prediction_result
            ),
            -9999.0,
            "float32",
            None,
            (0.0, 1.0),
        ),

        "MC Dropout uncertainty": (
            export_uncertainty_geotiff(
                prediction_result
            ),
            -9999.0,
            "float32",
            None,
            (0.0, float("inf")),
        ),

        "ESA WorldCover": (
            export_landcover_geotiff(
                landcover_result,
                prediction_result,
            ),
            0,
            "uint8",
            [
                10, 20, 30, 40, 50, 60,
                70, 80, 90, 95, 100,
            ],
            None,
        ),

        "Exposed cropland": (
            export_exposed_cropland_geotiff(
                landcover_result,
                prediction_result,
            ),
            255,
            "uint8",
            [0, 1],
            None,
        ),
    }

    # -----------------------------------------------------
    # 3. Validate every exported GeoTIFF
    # -----------------------------------------------------

    validated = 0
    exported_rasters = {}

    for name, export_data in exports.items():

        (
            geotiff_bytes,
            nodata,
            dtype,
            allowed_values,
            value_range,
        ) = export_data

        raster = inspect_geotiff(
            name=name,
            geotiff_bytes=geotiff_bytes,
            expected_shape=expected_shape,
            expected_crs=expected_crs,
            expected_transform=expected_transform,
            expected_nodata=nodata,
            expected_dtype=dtype,
            allowed_values=allowed_values,
            value_range=value_range,
        )

        exported_rasters[name] = raster

        validated += 1

    # -----------------------------------------------------
    # 4. Additional spatial consistency checks
    # -----------------------------------------------------

    print("\n==========================================")
    print(" ADDITIONAL CONSISTENCY CHECKS")
    print("==========================================")

    landcover_valid = np.asarray(
        landcover_result["landcover_valid_mask"],
        dtype=bool,
    )

    exposed_mask = np.asarray(
        landcover_result["exposed_cropland_mask"],
        dtype=bool,
    )

    exported_exposed = exported_rasters[
        "Exposed cropland"
    ]

    # Exported exposed-cropland pixels should match
    # the original overlap mask on valid pixels.
    expected_exposed = (
        exposed_mask & landcover_valid & sar_valid_mask
    )

    actual_exposed = (
        exported_exposed == 1
    )

    assert np.array_equal(
        actual_exposed,
        expected_exposed,
    ), "Exposed-cropland export does not match source mask."

    exposed_pixels = int(
        np.count_nonzero(actual_exposed)
    )

    print(
        "Exported exposed-cropland pixels:",
        exposed_pixels,
    )

    # The Ghana sample previously had 108 exposed
    # cropland pixels. Report rather than hard-code
    # this as a required result.
    print(
        "Application exposed-cropland pixels:",
        landcover_result["exposed_cropland_pixels"],
    )

    assert exposed_pixels == int(
        landcover_result["exposed_cropland_pixels"]
    ), "Exposed-cropland pixel count mismatch."

    print("\nSpatial consistency: PASSED")

    # -----------------------------------------------------
    # 5. Final report
    # -----------------------------------------------------

    print("\n==========================================")
    print(" FINAL EXPORT TEST REPORT")
    print("==========================================")

    print(
        f"Successfully validated: {validated}/5 GeoTIFF exports"
    )

    print("CRS preservation: PASSED")
    print("Transform preservation: PASSED")
    print("Raster dimensions: PASSED")
    print("Data types: PASSED")
    print("NoData values: PASSED")
    print("Pixel value validation: PASSED")
    print("Exposed-cropland consistency: PASSED")

    print("\nAll GeoTIFF export tests completed successfully.")


if __name__ == "__main__":
    main()
