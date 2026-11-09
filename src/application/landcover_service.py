
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from pyproj import CRS


# ============================================================
# ESA WORLDCOVER CONFIGURATION
# ============================================================

LAND_COVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

CROPLAND_CLASS = 40
WORLD_COVER_NODATA = 0
MINIMUM_COVERAGE = 0.80

WORLDCOVER_FILENAME_PATTERN = (
    "ESA_WorldCover_10m_2021_v200_*_Map.tif"
)


# ============================================================
# LAND-COVER EXPOSURE SERVICE
# ============================================================

class LandCoverExposureService:
    """
    Estimate the spatial overlap between model-predicted water
    and cropland from ESA WorldCover 2021 v200.

    Important:
        - WorldCover class 40 represents cropland.
        - WorldCover class 0 is treated as NoData.
        - Predictions are not verified flood observations.
        - Exposure is an indicative spatial estimate, not
          confirmed agricultural damage.
        - All output rasters are aligned to the input SAR grid.
    """

    def __init__(
        self,
        worldcover_dir="data/external/worldcover",
        minimum_coverage=MINIMUM_COVERAGE,
    ):
        self.worldcover_dir = Path(worldcover_dir)
        self.minimum_coverage = float(minimum_coverage)

        if not 0 < self.minimum_coverage <= 1:
            raise ValueError(
                "minimum_coverage must be greater than 0 "
                "and less than or equal to 1."
            )

        if not self.worldcover_dir.exists():
            raise FileNotFoundError(
                f"WorldCover directory not found: "
                f"{self.worldcover_dir}"
            )

        self.tile_paths = sorted(
            self.worldcover_dir.glob(
                WORLDCOVER_FILENAME_PATTERN
            )
        )

        if not self.tile_paths:
            raise FileNotFoundError(
                "No ESA WorldCover GeoTIFF tiles were found in "
                f"{self.worldcover_dir}."
            )

    # ========================================================
    # METADATA AND GEOGRAPHIC UTILITIES
    # ========================================================

    @staticmethod
    def _get_bounds(metadata):
        """
        Get raster bounds from metadata.

        Supports both rasterio Bounds objects and ordinary
        four-element tuples.
        """

        bounds = metadata.get("bounds")

        if bounds is not None:

            if hasattr(bounds, "left"):
                return (
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top,
                )

            if len(bounds) == 4:
                return tuple(bounds)

        from rasterio.transform import array_bounds

        return array_bounds(
            metadata["height"],
            metadata["width"],
            metadata["transform"],
        )

    def _find_intersecting_tiles(
        self,
        target_bounds,
        target_crs,
    ):
        """
        Identify WorldCover tiles intersecting the SAR footprint.

        Tile bounds are transformed into the SAR CRS when
        necessary before checking intersection.
        """

        matching_tiles = []

        for tile_path in self.tile_paths:

            with rasterio.open(tile_path) as src:

                if src.crs is None:
                    continue

                tile_bounds = tuple(src.bounds)

                source_crs = CRS.from_user_input(src.crs)

                if source_crs != target_crs:

                    tile_bounds = transform_bounds(
                        src.crs,
                        target_crs,
                        *src.bounds,
                        densify_pts=21,
                    )

                intersects = (
                    max(
                        target_bounds[0],
                        tile_bounds[0],
                    )
                    < min(
                        target_bounds[2],
                        tile_bounds[2],
                    )
                    and
                    max(
                        target_bounds[1],
                        tile_bounds[1],
                    )
                    < min(
                        target_bounds[3],
                        tile_bounds[3],
                    )
                )

                if intersects:
                    matching_tiles.append(tile_path)

        return matching_tiles

    # ========================================================
    # BUILD SAR-ALIGNED LAND-COVER GRID
    # ========================================================

    def _build_landcover_grid(
        self,
        metadata,
        target_bounds,
        target_crs,
    ):
        """
        Reproject and resample WorldCover tiles onto the
        exact spatial grid of the input SAR image.

        Nearest-neighbour resampling is used because
        WorldCover contains categorical class values.
        """

        height = int(metadata["height"])
        width = int(metadata["width"])

        target_transform = metadata["transform"]

        landcover = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        matching_tiles = self._find_intersecting_tiles(
            target_bounds,
            target_crs,
        )

        if not matching_tiles:
            raise ValueError(
                "No WorldCover tiles intersect the SAR image "
                "footprint. Land-cover analysis is unavailable "
                "for this image."
            )

        for tile_path in matching_tiles:

            with rasterio.open(tile_path) as src:

                with WarpedVRT(
                    src,
                    crs=target_crs,
                    transform=target_transform,
                    width=width,
                    height=height,
                    resampling=Resampling.nearest,
                    src_nodata=WORLD_COVER_NODATA,
                    nodata=WORLD_COVER_NODATA,
                ) as vrt:

                    tile_data = vrt.read(
                        1,
                        masked=True,
                    )

                    tile_data = (
                        tile_data
                        .filled(WORLD_COVER_NODATA)
                        .astype(np.uint8)
                    )

                    # Fill only pixels that have not already
                    # received valid land-cover information.
                    fill_mask = (
                        (landcover == WORLD_COVER_NODATA)
                        & (tile_data != WORLD_COVER_NODATA)
                    )

                    landcover[fill_mask] = tile_data[fill_mask]

        return landcover, matching_tiles

    # ========================================================
    # GEOGRAPHIC PIXEL AREA CALCULATION
    # ========================================================

    @staticmethod
    def _pixel_areas_by_row(metadata):
        """
        Calculate geodesic pixel area for each raster row.

        This method assumes:
            - Geographic CRS.
            - North-up or south-up raster grid.
            - No affine rotation or shear.

        Geodesic area is preferable to assuming that every
        pixel has an identical area in square metres.
        """

        crs = CRS.from_user_input(metadata["crs"])

        if not crs.is_geographic:
            raise ValueError(
                "Land-cover area calculation currently "
                "requires a geographic SAR CRS."
            )

        transform = metadata["transform"]

        if (
            abs(transform.b) > 1e-12
            or abs(transform.d) > 1e-12
        ):
            raise ValueError(
                "Rotated or sheared SAR grids are not supported "
                "by the current land-cover area calculation."
            )

        geod = crs.get_geod()

        height = int(metadata["height"])

        row_areas = np.zeros(
            height,
            dtype=np.float64,
        )

        for row in range(height):

            corners = [
                transform * (0, row),
                transform * (1, row),
                transform * (1, row + 1),
                transform * (0, row + 1),
            ]

            longitudes = [
                point[0] for point in corners
            ]

            latitudes = [
                point[1] for point in corners
            ]

            area, _ = geod.polygon_area_perimeter(
                longitudes,
                latitudes,
            )

            row_areas[row] = abs(area)

        return row_areas

    @staticmethod
    def _calculate_area_hectares(mask, row_areas):
        """
        Calculate the geodesic area of a boolean raster mask.

        row_areas contains square metres per pixel for each row.
        """

        if mask.ndim != 2:
            raise ValueError(
                "Area mask must be a two-dimensional array."
            )

        if mask.shape[0] != len(row_areas):
            raise ValueError(
                "Area mask height does not match row areas."
            )

        pixel_count_per_row = mask.sum(axis=1)

        area_square_metres = np.sum(
            pixel_count_per_row * row_areas
        )

        return float(area_square_metres / 10000.0)

    # ========================================================
    # MAIN LAND-COVER ANALYSIS
    # ========================================================

    def analyze(
        self,
        prediction,
        sar_valid_mask,
        metadata,
    ):
        """
        Calculate estimated cropland exposure.

        Parameters
        ----------
        prediction:
            Binary model prediction array.
            1 = predicted water
            0 = predicted non-water

        sar_valid_mask:
            Boolean mask identifying valid SAR pixels.

        metadata:
            Raster metadata containing dimensions, CRS,
            transform and bounds.

        Returns
        -------
        dict
            Scalar exposure statistics and aligned raster
            arrays required for map visualization and export.
        """

        prediction = np.asarray(prediction)

        sar_valid_mask = np.asarray(
            sar_valid_mask,
            dtype=bool,
        )

        expected_shape = (
            int(metadata["height"]),
            int(metadata["width"]),
        )

        if prediction.ndim != 2:
            raise ValueError(
                "Prediction must be a two-dimensional array."
            )

        if prediction.shape != expected_shape:
            raise ValueError(
                "Prediction dimensions do not match "
                "the SAR metadata."
            )

        if sar_valid_mask.shape != expected_shape:
            raise ValueError(
                "SAR validity mask dimensions do not match "
                "the SAR metadata."
            )

        if not np.isin(prediction, [0, 1]).all():
            raise ValueError(
                "Prediction must contain only binary values 0 and 1."
            )

        if metadata.get("crs") is None:
            raise ValueError(
                "SAR raster CRS is missing."
            )

        target_crs = CRS.from_user_input(
            metadata["crs"]
        )

        target_bounds = self._get_bounds(metadata)

        # ----------------------------------------------------
        # ALIGN WORLDCOVER WITH SAR
        # ----------------------------------------------------

        landcover, matching_tiles = (
            self._build_landcover_grid(
                metadata=metadata,
                target_bounds=target_bounds,
                target_crs=target_crs,
            )
        )

        if landcover.shape != expected_shape:
            raise ValueError(
                "Aligned WorldCover raster does not match "
                "the SAR image dimensions."
            )

        # ----------------------------------------------------
        # VALIDITY MASKS
        # ----------------------------------------------------

        worldcover_valid_mask = np.isin(
            landcover,
            list(LAND_COVER_CLASSES.keys()),
        )

        analysis_valid_mask = (
            sar_valid_mask
            & worldcover_valid_mask
        )

        cropland_mask = (
            analysis_valid_mask
            & (landcover == CROPLAND_CLASS)
        )

        predicted_water_mask = (
            analysis_valid_mask
            & (prediction == 1)
        )

        exposed_cropland_mask = (
            cropland_mask
            & (prediction == 1)
        )

        # ----------------------------------------------------
        # GEOGRAPHIC AREA CALCULATIONS
        # ----------------------------------------------------

        row_areas = self._pixel_areas_by_row(
            metadata
        )

        total_sar_area_ha = (
            self._calculate_area_hectares(
                sar_valid_mask,
                row_areas,
            )
        )

        covered_area_ha = (
            self._calculate_area_hectares(
                analysis_valid_mask,
                row_areas,
            )
        )

        cropland_area_ha = (
            self._calculate_area_hectares(
                cropland_mask,
                row_areas,
            )
        )

        exposed_cropland_area_ha = (
            self._calculate_area_hectares(
                exposed_cropland_mask,
                row_areas,
            )
        )

        predicted_water_area_ha = (
            self._calculate_area_hectares(
                predicted_water_mask,
                row_areas,
            )
        )

        # ----------------------------------------------------
        # COVERAGE AND EXPOSURE PERCENTAGES
        # ----------------------------------------------------

        worldcover_coverage_percentage = (
            covered_area_ha / total_sar_area_ha * 100
            if total_sar_area_ha > 0
            else 0.0
        )

        cropland_exposure_percentage = (
            exposed_cropland_area_ha
            / cropland_area_ha
            * 100
            if cropland_area_ha > 0
            else 0.0
        )

        water_area_over_cropland_percentage = (
            exposed_cropland_area_ha
            / predicted_water_area_ha
            * 100
            if predicted_water_area_ha > 0
            else 0.0
        )

        # ----------------------------------------------------
        # COVERAGE STATUS
        # ----------------------------------------------------

        if worldcover_coverage_percentage >= (
            self.minimum_coverage * 100
        ):
            coverage_status = "ADEQUATE"

        elif worldcover_coverage_percentage > 0:
            coverage_status = "PARTIAL_COVERAGE"

        else:
            coverage_status = "NO_DATA"

        if cropland_area_ha <= 0:
            exposure_status = "NO_CROPLAND"

        elif coverage_status == "NO_DATA":
            exposure_status = "NO_DATA"

        else:
            exposure_status = coverage_status

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        interpretation = (
            "Estimated cropland area spatially overlapping "
            "the model-predicted water extent. This is an "
            "indicative spatial estimate and does not confirm "
            "agricultural damage."
        )

        temporal_caveat = (
            "ESA WorldCover 2021 v200 represents land cover "
            "around 2021. It may post-date some flood events "
            "in this study. Therefore, estimated cropland "
            "exposure should not be interpreted as verified "
            "historical agricultural damage."
        )

        # ----------------------------------------------------
        # RETURN RESULTS
        # ----------------------------------------------------

        return {
            # Dataset information
            "dataset": "ESA WorldCover 2021 v200",
            "dataset_year": 2021,
            "cropland_class_code": CROPLAND_CLASS,
            "cropland_class_name": LAND_COVER_CLASSES[
                CROPLAND_CLASS
            ],

            # Area statistics
            "total_sar_valid_area_ha": total_sar_area_ha,
            "worldcover_covered_area_ha": covered_area_ha,
            "cropland_area_ha": cropland_area_ha,
            "estimated_exposed_cropland_ha": (
                exposed_cropland_area_ha
            ),
            "predicted_water_area_ha": predicted_water_area_ha,

            # Percentage statistics
            "cropland_exposure_percentage": (
                cropland_exposure_percentage
            ),
            "water_area_over_cropland_percentage": (
                water_area_over_cropland_percentage
            ),
            "worldcover_coverage_percentage": (
                worldcover_coverage_percentage
            ),

            # Pixel statistics
            "cropland_pixels": int(
                cropland_mask.sum()
            ),
            "exposed_cropland_pixels": int(
                exposed_cropland_mask.sum()
            ),
            "valid_landcover_pixels": int(
                analysis_valid_mask.sum()
            ),

            # Coverage and exposure status
            "coverage_status": coverage_status,
            "exposure_status": exposure_status,

            # Source information
            "matching_tiles": [
                path.name for path in matching_tiles
            ],

            # Interpretation and limitations
            "interpretation": interpretation,
            "temporal_caveat": temporal_caveat,

            # Aligned arrays for map and GeoTIFF export.
            # Do not include these arrays in CSV files.
            "landcover_raster": landcover,
            "landcover_valid_mask": analysis_valid_mask,
            "cropland_mask": cropland_mask,
            "exposed_cropland_mask": exposed_cropland_mask,
        }
