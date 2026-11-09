
from __future__ import annotations

from typing import Any

import numpy as np
from pyproj import Geod

try:
    from pyproj import CRS
except ImportError:
    CRS = None


class FloodStatisticsService:
    """
    Calculate geospatial flood statistics from FloodSight inference results.

    Area calculations:
      - Geographic CRS: geodesic pixel area using the WGS84 ellipsoid.
      - Projected CRS: pixel area using the CRS linear-unit conversion.

    Probability statistics cover all valid SAR pixels with finite
    probabilities.

    Confidence statistics cover predicted water pixels only.

    Uncertainty categories are relative to the current scene's
    predictive-variance distribution. They are not calibrated
    probabilities of prediction error.
    """

    def __init__(
        self,
        lower_confidence_threshold: float = 0.60,
        high_confidence_threshold: float = 0.80,
    ):
        if not (
            0.0 < lower_confidence_threshold
            < high_confidence_threshold
            < 1.0
        ):
            raise ValueError(
                "Confidence thresholds must satisfy "
                "0 < lower < high < 1."
            )

        self.lower_confidence_threshold = float(
            lower_confidence_threshold
        )
        self.high_confidence_threshold = float(
            high_confidence_threshold
        )

    @staticmethod
    def _validate_array(name: str, array: Any, expected_shape: tuple):
        array = np.asarray(array)

        if array.shape != expected_shape:
            raise ValueError(
                f"{name} shape {array.shape} does not match "
                f"expected shape {expected_shape}."
            )

        return array

    @staticmethod
    def _get_pixel_areas(metadata: dict, height: int, width: int):
        """
        Return a 2D array of pixel areas in square metres.

        For geographic north-up rasters, geodesic pixel areas are
        calculated separately for each row.

        For projected rasters, affine pixel area is converted into
        square metres using the CRS linear units.
        """

        transform = metadata.get("transform")
        crs_value = metadata.get("crs")

        if transform is None:
            raise ValueError("Raster metadata is missing its transform.")

        if crs_value is None:
            raise ValueError("Raster metadata is missing its CRS.")

        if CRS is None:
            raise ImportError(
                "pyproj is required for flood area calculations."
            )

        crs = CRS.from_user_input(crs_value)

        # Affine determinant accounts for pixel width, height,
        # and rotation in projected coordinate systems.
        affine_area = abs(
            transform.a * transform.e
            - transform.b * transform.d
        )

        if affine_area <= 0:
            raise ValueError("Invalid raster pixel dimensions.")

        if crs.is_projected:
            if not crs.axis_info:
                raise ValueError(
                    "Projected CRS does not provide linear units."
                )

            unit_factor = crs.axis_info[0].unit_conversion_factor

            if unit_factor is None or unit_factor <= 0:
                raise ValueError(
                    "Cannot determine projected CRS linear units."
                )

            pixel_area_m2 = affine_area * unit_factor**2

            return np.full(
                (height, width),
                pixel_area_m2,
                dtype=np.float64,
            )

        if not crs.is_geographic:
            raise ValueError(
                f"Unsupported CRS type for area calculations: {crs}"
            )

        # This implementation supports north-up geographic rasters.
        if not (
            np.isclose(transform.b, 0.0, atol=1e-12)
            and np.isclose(transform.d, 0.0, atol=1e-12)
        ):
            raise ValueError(
                "Rotated geographic rasters are not supported "
                "by the current area calculator."
            )

        geod = Geod(ellps="WGS84")

        # The first pixel column is sufficient for a north-up
        # geographic grid: all columns in the same row have the
        # same longitude and latitude dimensions.
        left = transform.c
        right = transform.c + transform.a

        row_areas = np.zeros(height, dtype=np.float64)

        for row in range(height):
            top = transform.f + row * transform.e
            bottom = transform.f + (row + 1) * transform.e

            longitudes = [
                left,
                right,
                right,
                left,
            ]

            latitudes = [
                top,
                top,
                bottom,
                bottom,
            ]

            area, _ = geod.polygon_area_perimeter(
                longitudes,
                latitudes,
            )

            row_areas[row] = abs(area)

        if not np.isfinite(row_areas).all() or np.any(row_areas <= 0):
            raise ValueError(
                "Unable to calculate valid geographic pixel areas."
            )

        return np.broadcast_to(
            row_areas[:, np.newaxis],
            (height, width),
        ).copy()

    @staticmethod
    def _area_summary(mask, pixel_areas):
        selected_area_m2 = float(
            pixel_areas[np.asarray(mask, dtype=bool)].sum()
        )

        return {
            "area_m2": selected_area_m2,
            "area_ha": selected_area_m2 / 10000.0,
            "area_km2": selected_area_m2 / 1_000_000.0,
            "pixel_count": int(np.count_nonzero(mask)),
        }

    @staticmethod
    def _percentage(part, whole):
        return (
            float(part / whole * 100.0)
            if whole > 0
            else 0.0
        )

    def analyze(self, prediction_result: dict):
        """
        Calculate advanced flood statistics.

        Args:
            prediction_result: Dictionary returned by
                FloodSightInference.predict().

        Returns:
            Dictionary containing statistics and category summaries.
        """

        required_keys = [
            "prediction",
            "probabilities",
            "sar_valid_mask",
            "metadata",
        ]

        missing = [
            key for key in required_keys
            if key not in prediction_result
        ]

        if missing:
            raise ValueError(
                f"Prediction result is missing required keys: {missing}"
            )

        prediction = np.asarray(
            prediction_result["prediction"]
        )

        probabilities = np.asarray(
            prediction_result["probabilities"],
            dtype=np.float64,
        )

        valid_mask = np.asarray(
            prediction_result["sar_valid_mask"],
            dtype=bool,
        )

        if prediction.ndim != 2:
            raise ValueError("Prediction must be a 2D array.")

        expected_shape = prediction.shape

        probabilities = self._validate_array(
            "Probabilities",
            probabilities,
            expected_shape,
        )

        valid_mask = self._validate_array(
            "SAR validity mask",
            valid_mask,
            expected_shape,
        )

        if not valid_mask.any():
            raise ValueError(
                "No valid SAR pixels are available for analysis."
            )

        valid_mask = (
            valid_mask
            & np.isfinite(probabilities)
        )

        if not valid_mask.any():
            raise ValueError(
                "No valid finite probability values are available."
            )

        if np.any(
            (probabilities[valid_mask] < 0.0)
            | (probabilities[valid_mask] > 1.0)
        ):
            raise ValueError(
                "Valid probability values must be between 0 and 1."
            )

        predicted_water = (
            (prediction == 1)
            & valid_mask
        )

        pixel_areas = self._get_pixel_areas(
            prediction_result["metadata"],
            expected_shape[0],
            expected_shape[1],
        )

        # Area statistics.
        total_valid_area = self._area_summary(
            valid_mask,
            pixel_areas,
        )

        flood_area = self._area_summary(
            predicted_water,
            pixel_areas,
        )

        flood_coverage = self._percentage(
            flood_area["area_m2"],
            total_valid_area["area_m2"],
        )

        # Probability distribution over all valid SAR pixels.
        probability_ranges = [
            ("below_50", 0.0, 0.50),
            ("50_to_60", 0.50, 0.60),
            ("60_to_70", 0.60, 0.70),
            ("70_to_80", 0.70, 0.80),
            ("80_to_100", 0.80, 1.0000001),
        ]

        probability_distribution = {}

        for name, lower, upper in probability_ranges:
            selected = (
                valid_mask
                & (probabilities >= lower)
                & (probabilities < upper)
            )

            summary = self._area_summary(
                selected,
                pixel_areas,
            )

            summary["percentage_of_valid_area"] = self._percentage(
                summary["area_m2"],
                total_valid_area["area_m2"],
            )

            probability_distribution[name] = summary

        # Confidence distribution for predicted water only.
        high_confidence_water = (
            predicted_water
            & (
                probabilities
                >= self.high_confidence_threshold
            )
        )

        lower_confidence_water = (
            predicted_water
            & (
                probabilities
                < self.high_confidence_threshold
            )
        )

        confidence_statistics = {
            "threshold": self.high_confidence_threshold,
            "high_confidence_water": self._area_summary(
                high_confidence_water,
                pixel_areas,
            ),
            "lower_confidence_water": self._area_summary(
                lower_confidence_water,
                pixel_areas,
            ),
        }

        # MC Dropout uncertainty statistics.
        uncertainty_result = prediction_result.get("uncertainty")
        uncertainty_statistics = {
            "available": False,
            "method": "MC Dropout predictive variance",
        }

        if uncertainty_result is not None:
            variance = uncertainty_result.get(
                "predictive_variance"
            )

            if variance is not None:
                variance = self._validate_array(
                    "Predictive variance",
                    np.asarray(variance, dtype=np.float64),
                    expected_shape,
                )

                uncertainty_valid = (
                    valid_mask
                    & np.isfinite(variance)
                    & (variance >= 0.0)
                )

                valid_variance = variance[uncertainty_valid]

                if valid_variance.size > 0:
                    q33, q66 = np.quantile(
                        valid_variance,
                        [1.0 / 3.0, 2.0 / 3.0],
                    )

                    low_uncertainty = (
                        uncertainty_valid
                        & (variance <= q33)
                    )

                    moderate_uncertainty = (
                        uncertainty_valid
                        & (variance > q33)
                        & (variance <= q66)
                    )

                    high_uncertainty = (
                        uncertainty_valid
                        & (variance > q66)
                    )

                    uncertainty_statistics = {
                        "available": True,
                        "method": "MC Dropout predictive variance",
                        "mean_predictive_variance": float(
                            valid_variance.mean()
                        ),
                        "maximum_predictive_variance": float(
                            valid_variance.max()
                        ),
                        "minimum_predictive_variance": float(
                            valid_variance.min()
                        ),
                        "low_threshold": float(q33),
                        "high_threshold": float(q66),
                        "low": self._area_summary(
                            low_uncertainty,
                            pixel_areas,
                        ),
                        "moderate": self._area_summary(
                            moderate_uncertainty,
                            pixel_areas,
                        ),
                        "high": self._area_summary(
                            high_uncertainty,
                            pixel_areas,
                        ),
                        "valid_uncertainty_pixels": int(
                            uncertainty_valid.sum()
                        ),
                        "interpretation": (
                            "Relative uncertainty categories based on "
                            "the 33rd and 66th percentiles of predictive "
                            "variance in this scene. These are not "
                            "calibrated probabilities of prediction error."
                        ),
                    }

        statistics = {
            "sample_name": prediction_result.get("sample_name"),
            "total_valid_area": total_valid_area,
            "predicted_flood_area": flood_area,
            "flood_coverage_percentage": flood_coverage,
            "predicted_water_pixels": flood_area["pixel_count"],
            "valid_sar_pixels": total_valid_area["pixel_count"],
            "mean_water_probability": float(
                probabilities[valid_mask].mean()
            ),
            "probability_distribution": probability_distribution,
            "confidence": confidence_statistics,
            "uncertainty": uncertainty_statistics,
            "thresholds": {
                "model_prediction_threshold": prediction_result.get(
                    "statistics", {}
                ).get("threshold"),
                "lower_confidence_threshold": (
                    self.lower_confidence_threshold
                ),
                "high_confidence_threshold": (
                    self.high_confidence_threshold
                ),
            },
            "area_unit": "hectares and square kilometres",
        }

        return statistics
