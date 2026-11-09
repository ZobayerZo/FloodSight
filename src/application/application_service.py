
from pathlib import Path

from src.application.inference_service import (
    FloodSightInference,
    DEFAULT_CHECKPOINT,
)
from src.application.exposure_service import (
    calculate_population_exposure,
    get_population_raster,
    EVENT_YEARS,
)
from src.application.landcover_service import LandCoverExposureService
from src.application.statistics_service import FloodStatisticsService


class FloodSightApplication:
    """Run flood inference and supporting exposure/statistical analyses."""

    def __init__(
        self,
        checkpoint_path=None,
        device=None,
        threshold=0.5,
        mc_passes=20,
        dropout_probability=0.2,
        landcover_service=None,
        statistics_service=None,
    ):
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT

        self.inference = FloodSightInference(
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=threshold,
            mc_passes=mc_passes,
            dropout_probability=dropout_probability,
        )

        self.available_events = list(EVENT_YEARS.keys())

        self.landcover = (
            landcover_service
            if landcover_service is not None
            else LandCoverExposureService()
        )

        self.statistics = (
            statistics_service
            if statistics_service is not None
            else FloodStatisticsService()
        )

    @staticmethod
    def _landcover_error_result(error):
        """Return an explicit status without discarding the main analysis."""
        return {
            "dataset": "ESA WorldCover",
            "dataset_year": 2021,
            "cropland_class_code": 40,
            "cropland_class_name": "Cropland",
            "coverage_status": "ERROR",
            "exposure_status": "ERROR",
            "error": str(error),
            "interpretation": (
                "Land-cover exposure could not be calculated. "
                "Flood prediction and population exposure remain available."
            ),
            "temporal_caveat": (
                "ESA WorldCover 2021 may post-date the selected flood event; "
                "the result must not be interpreted as verified historical "
                "cropland or agricultural damage."
            ),
        }

    @staticmethod
    def _statistics_error_result(error):
        """Return a statistics error without discarding other results."""
        return {
            "status": "ERROR",
            "error": str(error),
            "interpretation": (
                "Advanced flood statistics could not be calculated. "
                "The original flood prediction and exposure results remain "
                "available."
            ),
        }

    def analyze(
        self,
        sar_path,
        event,
        include_uncertainty=True,
        original_filename=None,
    ):
        """Generate flood prediction and supporting analyses."""

        sar_path = Path(sar_path)

        if not sar_path.exists():
            raise FileNotFoundError(
                f"SAR image not found: {sar_path}"
            )

        if event not in self.available_events:
            raise ValueError(
                f"Unsupported event: {event}. "
                f"Available events: {self.available_events}"
            )

        # 1. SAR inference.
        prediction_result = self.inference.predict(
            sar_path=sar_path,
            include_uncertainty=include_uncertainty,
        )

        predicted_water = prediction_result["prediction"] == 1
        sar_valid_mask = prediction_result["sar_valid_mask"]
        metadata = prediction_result["metadata"]

        # 2. Population exposure: existing calculation unchanged.
        population_path = get_population_raster(event)

        exposure_result = calculate_population_exposure(
            predicted_water=predicted_water,
            sar_valid_mask=sar_valid_mask,
            metadata=metadata,
            population_path=population_path,
        )

        # 3. ESA WorldCover cropland overlap.
        try:
            landcover_result = self.landcover.analyze(
                prediction=prediction_result["prediction"],
                sar_valid_mask=sar_valid_mask,
                metadata=metadata,
            )

        except Exception as error:
            landcover_result = self._landcover_error_result(error)

        # 4. Advanced flood statistics.
        try:
            advanced_statistics = self.statistics.analyze(
                prediction_result=prediction_result
            )

            advanced_statistics["status"] = "SUCCESS"

        except Exception as error:
            advanced_statistics = self._statistics_error_result(error)

        # 5. Unified application response.
        result = {
            "success": True,

            "input": {
                "sar_filename": original_filename or sar_path.name,
                "event": event,
                "population_raster": population_path.name,
                "population_year": EVENT_YEARS[event],
                "landcover_dataset": landcover_result.get(
                    "dataset", "ESA WorldCover"
                ),
                "landcover_year": landcover_result.get(
                    "dataset_year", 2021
                ),
            },

            "prediction": prediction_result,

            "exposure": exposure_result,

            "landcover_exposure": landcover_result,

            "advanced_statistics": advanced_statistics,

            "summary": {
                "predicted_water_pixels": (
                    prediction_result["statistics"][
                        "predicted_water_pixels"
                    ]
                ),

                "predicted_water_percentage": (
                    prediction_result["statistics"][
                        "predicted_water_percentage"
                    ]
                ),

                "estimated_exposed_population": (
                    exposure_result[
                        "estimated_exposed_population"
                    ]
                ),

                "population_coverage_pct": (
                    exposure_result[
                        "population_coverage_pct"
                    ]
                ),

                "exposure_status": exposure_result["status"],

                "estimated_exposed_cropland_ha": (
                    landcover_result.get(
                        "estimated_exposed_cropland_ha"
                    )
                ),

                "cropland_exposure_percentage": (
                    landcover_result.get(
                        "cropland_exposure_percentage"
                    )
                ),

                "worldcover_coverage_percentage": (
                    landcover_result.get(
                        "worldcover_coverage_percentage"
                    )
                ),

                "landcover_status": landcover_result.get(
                    "exposure_status", "UNKNOWN"
                ),

                "flood_area_ha": (
                    advanced_statistics.get(
                        "predicted_flood_area", {}
                    ).get("area_ha")
                ),

                "flood_area_km2": (
                    advanced_statistics.get(
                        "predicted_flood_area", {}
                    ).get("area_km2")
                ),

                "flood_coverage_percentage": (
                    advanced_statistics.get(
                        "flood_coverage_percentage"
                    )
                ),

                "statistics_status": advanced_statistics.get(
                    "status", "UNKNOWN"
                ),
            },
        }

        return result


if __name__ == "__main__":

    app = FloodSightApplication(
        mc_passes=20,
    )

    result = app.analyze(
        sar_path=(
            "data/raw/Sen1Floods11/v1.1/data/"
            "flood_events/HandLabeled/S1Hand/"
            "Ghana_406026_S1Hand.tif"
        ),
        event="Ghana",
        include_uncertainty=True,
    )

    print("\n========== FLOODSIGHT APPLICATION ==========")

    print("\nINPUT:")
    for key, value in result["input"].items():
        print(f"{key}: {value}")

    print("\nSUMMARY:")
    for key, value in result["summary"].items():
        print(f"{key}: {value}")

    print("\nADVANCED FLOOD STATISTICS:")
    advanced = result["advanced_statistics"]

    if advanced.get("status") == "SUCCESS":
        print("Status: SUCCESS")

        flood_area = advanced["predicted_flood_area"]

        print(
            "Flood area:",
            f"{flood_area['area_ha']:.4f} hectares"
        )

        print(
            "Flood area:",
            f"{flood_area['area_km2']:.4f} km²"
        )

        print(
            "Flood coverage:",
            f"{advanced['flood_coverage_percentage']:.4f}%"
        )

        print(
            "Mean probability:",
            f"{advanced['mean_water_probability']:.4f}"
        )

        print("\nPROBABILITY DISTRIBUTION:")

        for category, values in advanced[
            "probability_distribution"
        ].items():
            print(
                category,
                f": {values['pixel_count']} pixels",
                f"({values['percentage_of_valid_area']:.4f}%)"
            )

        print("\nCONFIDENCE:")

        for category, values in advanced["confidence"].items():
            if isinstance(values, dict):
                print(
                    category,
                    f": {values['pixel_count']} pixels",
                    f"({values['area_ha']:.4f} ha)"
                )

        print("\nUNCERTAINTY:")

        uncertainty = advanced["uncertainty"]

        print("Available:", uncertainty["available"])

        if uncertainty["available"]:
            print(
                "Mean variance:",
                uncertainty["mean_predictive_variance"]
            )

            for category in ["low", "moderate", "high"]:
                values = uncertainty[category]

                print(
                    category,
                    f": {values['pixel_count']} pixels",
                    f"({values['area_ha']:.4f} ha)"
                )

    else:
        print("Status: ERROR")
        print("Error:", advanced.get("error"))

    print("\nPOPULATION EXPOSURE:")
    for key, value in result["exposure"].items():
        if not hasattr(value, "shape"):
            print(f"{key}: {value}")

    print("\nLAND-COVER EXPOSURE:")
    for key, value in result["landcover_exposure"].items():
        if not hasattr(value, "shape"):
            print(f"{key}: {value}")

    print("\nApplication test completed.")
