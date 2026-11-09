
import numpy as np

from src.application.application_service import FloodSightApplication
from src.application.map_service import (
    create_flood_map,
    create_landcover_overlay,
    create_exposed_cropland_overlay,
)


SAR_PATH = (
    "data/raw/Sen1Floods11/v1.1/data/"
    "flood_events/HandLabeled/S1Hand/"
    "Ghana_406026_S1Hand.tif"
)


def main():

    print("\nRunning FloodSight map integration test...")

    application = FloodSightApplication(
        mc_passes=20,
    )

    result = application.analyze(
        sar_path=SAR_PATH,
        event="Ghana",
        include_uncertainty=True,
        original_filename="Ghana_406026_S1Hand.tif",
    )

    prediction = result["prediction"]
    landcover = result["landcover_exposure"]

    prediction_mask = np.asarray(
        prediction["sar_valid_mask"],
        dtype=bool,
    )

    landcover_raster = landcover.get(
        "landcover_raster"
    )

    landcover_valid = landcover.get(
        "landcover_valid_mask"
    )

    exposed_mask = landcover.get(
        "exposed_cropland_mask"
    )

    if (
        landcover_raster is None
        or landcover_valid is None
        or exposed_mask is None
    ):
        raise RuntimeError(
            "Land-cover arrays are unavailable. "
            "Check the land-cover analysis status."
        )

    landcover_overlay = create_landcover_overlay(
        landcover_raster,
        landcover_valid,
    )

    exposed_overlay = create_exposed_cropland_overlay(
        exposed_mask,
        prediction_mask & landcover_valid,
    )

    print("\nLAND-COVER OVERLAY:")

    print("Shape:", landcover_overlay.shape)

    print(
        "Visible land-cover pixels:",
        int(np.count_nonzero(landcover_overlay[..., 3])),
    )

    print("\nEXPOSED CROPLAND OVERLAY:")

    print("Shape:", exposed_overlay.shape)

    print(
        "Visible exposed pixels:",
        int(np.count_nonzero(exposed_overlay[..., 3])),
    )

    # Verify exposed-cropland pixel count.
    assert (
        np.count_nonzero(exposed_overlay[..., 3])
        == landcover["exposed_cropland_pixels"]
    )

    # Build complete Folium map.
    flood_map = create_flood_map(
        prediction_result=prediction,
        landcover_result=landcover,
    )

    # Inspect the generated Folium HTML.
    map_html = flood_map.get_root().render()

    expected_layers = [
        "Predicted flood extent",
        "Water probability",
        "Predictive uncertainty",
        "Land cover (ESA WorldCover 2021)",
        "Predicted water over cropland",
    ]

    print("\nMAP LAYER CHECK:")

    for layer in expected_layers:

        present = layer in map_html

        print(f"{layer}: {'FOUND' if present else 'MISSING'}")

        assert present, (
            f"Map layer missing: {layer}"
        )

    # Save a local HTML preview.
    output_path = "floodsight_map_test.html"

    flood_map.save(output_path)

    print("\nMap saved:", output_path)

    print("\nAll map integration tests passed.")


if __name__ == "__main__":
    main()
