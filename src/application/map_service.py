
import base64
from io import BytesIO

import folium
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from folium.plugins import (
    Fullscreen,
    MiniMap,
    MeasureControl,
    MousePosition,
)
from rasterio.warp import transform_bounds


# =========================================================
# VISUALIZATION SETTINGS
# =========================================================

PROBABILITY_MIN = 0.0
PROBABILITY_MAX = 1.0

UNCERTAINTY_MAX = 0.02

FLOOD_COLOR = "#1ED2DC"
CROPLAND_EXPOSURE_COLOR = "#FF00FF"

# ESA WorldCover 2021 v200 class colors.
WORLDCOVER_COLORS = {
    10: (0, 100, 0),        # Tree cover
    20: (255, 187, 34),     # Shrubland
    30: (255, 255, 76),     # Grassland
    40: (240, 150, 255),    # Cropland
    50: (250, 0, 0),        # Built-up
    60: (180, 180, 180),    # Bare / sparse vegetation
    70: (240, 240, 240),    # Snow and ice
    80: (0, 100, 200),      # Permanent water
    90: (0, 150, 160),      # Herbaceous wetland
    95: (0, 207, 117),      # Mangroves
    100: (250, 230, 160),   # Moss and lichen
}

WORLDCOVER_NAMES = {
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


# =========================================================
# GEOGRAPHIC UTILITIES
# =========================================================

def get_geographic_bounds(metadata):
    """Convert source raster bounds to WGS84."""

    bounds = metadata["bounds"]
    source_crs = metadata["crs"]

    if source_crs is None:
        raise ValueError("Raster CRS is missing.")

    west, south, east, north = transform_bounds(
        source_crs,
        "EPSG:4326",
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
        densify_pts=21,
    )

    if not (
        np.isfinite([west, south, east, north]).all()
        and west < east
        and south < north
    ):
        raise ValueError("Invalid geographic raster bounds.")

    return [
        [south, west],
        [north, east],
    ]


def validate_raster_shapes(
    prediction,
    probabilities,
    valid_mask,
    metadata,
    variance=None,
):
    """Validate raster dimensions before visualization."""

    expected_shape = (
        metadata["height"],
        metadata["width"],
    )

    if prediction.ndim != 2:
        raise ValueError("Prediction must be a 2D raster.")

    if prediction.shape != expected_shape:
        raise ValueError(
            "Prediction dimensions do not match metadata."
        )

    if probabilities.shape != expected_shape:
        raise ValueError(
            "Probability dimensions do not match metadata."
        )

    if valid_mask.shape != expected_shape:
        raise ValueError(
            "Validity mask dimensions do not match metadata."
        )

    if variance is not None and variance.shape != expected_shape:
        raise ValueError(
            "Uncertainty dimensions do not match metadata."
        )


def validate_optional_raster(array, expected_shape, name):
    """Validate an optional land-cover raster or mask."""

    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D raster."
        )

    if array.shape != expected_shape:
        raise ValueError(
            f"{name} dimensions do not match SAR metadata."
        )

    return array


# =========================================================
# IMAGE ENCODING
# =========================================================

def rgba_to_base64(rgba):
    """Convert RGBA array to a PNG data URL."""

    image = Image.fromarray(
        rgba.astype(np.uint8),
        mode="RGBA",
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


# =========================================================
# FLOOD EXTENT OVERLAY
# =========================================================

def create_flood_overlay(prediction, valid_mask):
    """Create a transparent binary flood overlay."""

    prediction = np.asarray(prediction)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    rgba = np.zeros(
        (*prediction.shape, 4),
        dtype=np.uint8,
    )

    flood_pixels = (
        (prediction == 1)
        & valid_mask
    )

    rgba[flood_pixels] = [
        30,
        210,
        220,
        235,
    ]

    return rgba


# =========================================================
# PROBABILITY OVERLAY
# =========================================================

def create_probability_overlay(
    probabilities,
    valid_mask,
):
    """Create a fixed-scale probability heatmap."""

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    valid_mask = np.asarray(valid_mask, dtype=bool)

    rgba = np.zeros(
        (*probabilities.shape, 4),
        dtype=np.uint8,
    )

    valid = (
        valid_mask
        & np.isfinite(probabilities)
    )

    if not valid.any():
        return rgba

    normalized = np.clip(
        (probabilities - PROBABILITY_MIN)
        / (PROBABILITY_MAX - PROBABILITY_MIN),
        0,
        1,
    )

    colors = plt.get_cmap("turbo")(normalized)

    rgba[..., :3] = (
        colors[..., :3] * 255
    ).astype(np.uint8)

    alpha = np.clip(
        probabilities * 255,
        0,
        220,
    )

    alpha[probabilities < 0.05] = 0

    rgba[..., 3] = np.where(
        valid,
        alpha,
        0,
    ).astype(np.uint8)

    return rgba


# =========================================================
# UNCERTAINTY OVERLAY
# =========================================================

def create_uncertainty_overlay(
    variance,
    valid_mask,
):
    """Create an MC Dropout variance heatmap."""

    variance = np.asarray(
        variance,
        dtype=np.float32,
    )

    valid_mask = np.asarray(valid_mask, dtype=bool)

    rgba = np.zeros(
        (*variance.shape, 4),
        dtype=np.uint8,
    )

    valid = (
        valid_mask
        & np.isfinite(variance)
    )

    if not valid.any():
        return rgba

    normalized = np.clip(
        variance / UNCERTAINTY_MAX,
        0,
        1,
    )

    colors = plt.get_cmap("magma")(normalized)

    rgba[..., :3] = (
        colors[..., :3] * 255
    ).astype(np.uint8)

    alpha = np.clip(
        normalized * 235,
        0,
        235,
    )

    rgba[..., 3] = np.where(
        valid,
        alpha,
        0,
    ).astype(np.uint8)

    return rgba


# =========================================================
# ESA WORLDCOVER OVERLAY
# =========================================================

def create_landcover_overlay(
    landcover_raster,
    valid_mask,
):
    """
    Create a categorical ESA WorldCover overlay.

    Uses the official WorldCover class codes and
    distinct colors for each land-cover category.

    Invalid and unclassified pixels remain transparent.
    """

    landcover_raster = np.asarray(
        landcover_raster
    )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    if landcover_raster.ndim != 2:
        raise ValueError(
            "Land-cover raster must be two-dimensional."
        )

    if landcover_raster.shape != valid_mask.shape:
        raise ValueError(
            "Land-cover raster and validity mask dimensions differ."
        )

    rgba = np.zeros(
        (*landcover_raster.shape, 4),
        dtype=np.uint8,
    )

    for class_code, color in WORLDCOVER_COLORS.items():

        class_pixels = (
            (landcover_raster == class_code)
            & valid_mask
        )

        rgba[class_pixels, :3] = color
        rgba[class_pixels, 3] = 190

    return rgba


# =========================================================
# EXPOSED CROPLAND OVERLAY
# =========================================================

def create_exposed_cropland_overlay(
    exposed_cropland_mask,
    valid_mask,
):
    """
    Highlight predicted floodwater overlapping cropland.

    Magenta represents estimated spatial exposure,
    not confirmed agricultural damage.
    """

    exposed_cropland_mask = np.asarray(
        exposed_cropland_mask,
        dtype=bool,
    )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    if exposed_cropland_mask.shape != valid_mask.shape:
        raise ValueError(
            "Exposed cropland and validity mask dimensions differ."
        )

    rgba = np.zeros(
        (*exposed_cropland_mask.shape, 4),
        dtype=np.uint8,
    )

    exposed_pixels = (
        exposed_cropland_mask
        & valid_mask
    )

    rgba[exposed_pixels] = [
        255,
        0,
        255,
        245,
    ]

    return rgba


# =========================================================
# MAP LEGEND
# =========================================================

def add_map_legend(flood_map):
    """Add a professional explanatory map legend."""

    legend_html = """
    <div style="
        position:fixed;
        bottom:28px;
        left:28px;
        z-index:9999;
        background:rgba(15,27,43,0.96);
        color:#F1F5F9;
        padding:16px;
        border:1px solid #42566D;
        border-radius:12px;
        width:245px;
        max-height:65vh;
        overflow-y:auto;
        font-family:Arial,sans-serif;
        font-size:11px;
        box-shadow:0 3px 14px rgba(0,0,0,0.35);
    ">

        <div style="
            font-weight:bold;
            font-size:13px;
            margin-bottom:13px;
            color:#45D6CE;
        ">
            FLOODSIGHT MAP LEGEND
        </div>

        <div style="
            display:flex;
            align-items:center;
            margin-bottom:12px;
        ">
            <span style="
                background:#1ED2DC;
                width:16px;
                height:16px;
                margin-right:9px;
                border-radius:3px;
            "></span>
            Predicted water extent
        </div>

        <div style="
            display:flex;
            align-items:center;
            margin-bottom:12px;
        ">
            <span style="
                background:#FF00FF;
                width:16px;
                height:16px;
                margin-right:9px;
                border-radius:3px;
            "></span>
            Water over cropland
        </div>

        <div style="font-weight:bold;margin-bottom:6px;">
            Water probability
        </div>

        <div style="
            height:11px;
            border-radius:4px;
            background:linear-gradient(
                to right,
                #30123b,
                #4662d8,
                #1fa187,
                #f9e721
            );
        "></div>

        <div style="
            display:flex;
            justify-content:space-between;
            color:#B8C5D3;
            margin-top:5px;
        ">
            <span>0.0</span>
            <span>0.5</span>
            <span>1.0</span>
        </div>

        <div style="
            font-weight:bold;
            margin-top:14px;
            margin-bottom:6px;
        ">
            Predictive variance
        </div>

        <div style="
            height:11px;
            border-radius:4px;
            background:linear-gradient(
                to right,
                #100c27,
                #8c2981,
                #de4968,
                #fcfdbf
            );
        "></div>

        <div style="
            display:flex;
            justify-content:space-between;
            color:#B8C5D3;
            margin-top:5px;
        ">
            <span>0.00</span>
            <span>0.01</span>
            <span>0.02+</span>
        </div>

        <div style="
            font-weight:bold;
            margin-top:14px;
            margin-bottom:8px;
        ">
            ESA WorldCover 2021
        </div>

        <div style="
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:7px;
        ">
            <div><span style="color:#006400;">&#9632;</span> Trees</div>
            <div><span style="color:#FFBB22;">&#9632;</span> Shrubland</div>
            <div><span style="color:#FFFF4C;">&#9632;</span> Grassland</div>
            <div><span style="color:#F096FF;">&#9632;</span> Cropland</div>
            <div><span style="color:#FA0000;">&#9632;</span> Built-up</div>
            <div><span style="color:#B4B4B4;">&#9632;</span> Bare / sparse</div>
            <div><span style="color:#F0F0F0;">&#9632;</span> Snow / ice</div>
            <div><span style="color:#0064C8;">&#9632;</span> Water</div>
            <div><span style="color:#0096A0;">&#9632;</span> Wetland</div>
            <div><span style="color:#00CF75;">&#9632;</span> Mangroves</div>
            <div><span style="color:#FAE6A0;">&#9632;</span> Moss / lichen</div>
        </div>

        <div style="
            color:#AAB8C7;
            font-size:10px;
            margin-top:13px;
            line-height:1.6;
        ">
            Overlays show model predictions and
            spatial exposure estimates, not verified
            flood extent or agricultural damage.
            WorldCover represents land cover around 2021.
        </div>
    </div>
    """

    from branca.element import Element

    flood_map.get_root().html.add_child(
        Element(legend_html)
    )


# =========================================================
# MAIN FLOOD MAP
# =========================================================

def create_flood_map(
    prediction_result,
    map_height=650,
    landcover_result=None,
):
    """
    Create the interactive FloodSight geospatial map.

    Default:
        Satellite imagery
        Predicted flood extent

    Optional:
        Water probability
        Predictive uncertainty
        ESA WorldCover 2021
        Predicted water over cropland
    """

    metadata = prediction_result["metadata"]

    prediction = np.asarray(
        prediction_result["prediction"]
    )

    probabilities = np.asarray(
        prediction_result["probabilities"],
        dtype=np.float32,
    )

    valid_mask = np.asarray(
        prediction_result["sar_valid_mask"]
    ).astype(bool)

    uncertainty = prediction_result.get("uncertainty")

    variance = None

    if uncertainty is not None:

        variance_data = uncertainty.get(
            "predictive_variance"
        )

        if variance_data is not None:
            variance = np.asarray(
                variance_data,
                dtype=np.float32,
            )

    validate_raster_shapes(
        prediction,
        probabilities,
        valid_mask,
        metadata,
        variance,
    )

    expected_shape = (
        metadata["height"],
        metadata["width"],
    )

    # -----------------------------------------------------
    # OPTIONAL LAND-COVER DATA
    # -----------------------------------------------------

    landcover_raster = None
    landcover_valid_mask = None
    exposed_cropland_mask = None

    if landcover_result is not None:

        raw_landcover = landcover_result.get(
            "landcover_raster"
        )

        raw_landcover_valid = landcover_result.get(
            "landcover_valid_mask"
        )

        raw_exposed_cropland = landcover_result.get(
            "exposed_cropland_mask"
        )

        # The application can return an ERROR result
        # without raster arrays. In that case, skip
        # agricultural layers and keep the flood map.

        if (
            raw_landcover is not None
            and raw_landcover_valid is not None
        ):

            landcover_raster = validate_optional_raster(
                raw_landcover,
                expected_shape,
                "Land-cover raster",
            )

            landcover_valid_mask = validate_optional_raster(
                raw_landcover_valid,
                expected_shape,
                "Land-cover validity mask",
            ).astype(bool)

            # Do not display WorldCover class 0.
            landcover_valid_mask &= (
                landcover_raster != 0
            )

            if raw_exposed_cropland is not None:

                exposed_cropland_mask = validate_optional_raster(
                    raw_exposed_cropland,
                    expected_shape,
                    "Exposed cropland mask",
                ).astype(bool)

                # Restrict exposure to valid SAR and
                # valid WorldCover pixels.
                exposed_cropland_mask &= (
                    valid_mask & landcover_valid_mask
                )

    image_bounds = get_geographic_bounds(metadata)

    center_lat = (
        image_bounds[0][0] + image_bounds[1][0]
    ) / 2

    center_lon = (
        image_bounds[0][1] + image_bounds[1][1]
    ) / 2

    # -----------------------------------------------------
    # MAP INITIALIZATION
    # -----------------------------------------------------

    flood_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    # -----------------------------------------------------
    # BASEMAPS
    # -----------------------------------------------------

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(flood_map)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite imagery",
        overlay=False,
        control=True,
    ).add_to(flood_map)

    # -----------------------------------------------------
    # FLOOD EXTENT
    # -----------------------------------------------------

    flood_group = folium.FeatureGroup(
        name="Predicted flood extent",
        show=True,
    )

    flood_overlay = create_flood_overlay(
        prediction,
        valid_mask,
    )

    folium.raster_layers.ImageOverlay(
        image=rgba_to_base64(flood_overlay),
        bounds=image_bounds,
        opacity=1.0,
        name="Flood extent raster",
        interactive=True,
        cross_origin=False,
        zindex=3,
    ).add_to(flood_group)

    flood_group.add_to(flood_map)

    # -----------------------------------------------------
    # WATER PROBABILITY
    # -----------------------------------------------------

    probability_group = folium.FeatureGroup(
        name="Water probability",
        show=False,
    )

    probability_overlay = create_probability_overlay(
        probabilities,
        valid_mask,
    )

    folium.raster_layers.ImageOverlay(
        image=rgba_to_base64(probability_overlay),
        bounds=image_bounds,
        opacity=1.0,
        name="Probability raster",
        interactive=True,
        cross_origin=False,
        zindex=4,
    ).add_to(probability_group)

    probability_group.add_to(flood_map)

    # -----------------------------------------------------
    # PREDICTIVE UNCERTAINTY
    # -----------------------------------------------------

    if variance is not None:

        uncertainty_group = folium.FeatureGroup(
            name="Predictive uncertainty",
            show=False,
        )

        uncertainty_overlay = create_uncertainty_overlay(
            variance,
            valid_mask,
        )

        folium.raster_layers.ImageOverlay(
            image=rgba_to_base64(uncertainty_overlay),
            bounds=image_bounds,
            opacity=1.0,
            name="Uncertainty raster",
            interactive=True,
            cross_origin=False,
            zindex=5,
        ).add_to(uncertainty_group)

        uncertainty_group.add_to(flood_map)

    # -----------------------------------------------------
    # ESA WORLDCOVER LAND-COVER LAYER
    # -----------------------------------------------------

    if (
        landcover_raster is not None
        and landcover_valid_mask is not None
    ):

        landcover_group = folium.FeatureGroup(
            name="Land cover (ESA WorldCover 2021)",
            show=False,
        )

        landcover_overlay = create_landcover_overlay(
            landcover_raster,
            landcover_valid_mask,
        )

        folium.raster_layers.ImageOverlay(
            image=rgba_to_base64(landcover_overlay),
            bounds=image_bounds,
            opacity=1.0,
            name="ESA WorldCover classification",
            interactive=True,
            cross_origin=False,
            zindex=6,
        ).add_to(landcover_group)

        landcover_group.add_to(flood_map)

    # -----------------------------------------------------
    # EXPOSED CROPLAND LAYER
    # -----------------------------------------------------

    if exposed_cropland_mask is not None:

        exposed_group = folium.FeatureGroup(
            name="Predicted water over cropland",
            show=False,
        )

        exposed_overlay = create_exposed_cropland_overlay(
            exposed_cropland_mask,
            valid_mask & landcover_valid_mask,
        )

        folium.raster_layers.ImageOverlay(
            image=rgba_to_base64(exposed_overlay),
            bounds=image_bounds,
            opacity=1.0,
            name="Estimated exposed cropland",
            interactive=True,
            cross_origin=False,
            zindex=7,
        ).add_to(exposed_group)

        exposed_group.add_to(flood_map)

    # -----------------------------------------------------
    # SAR FOOTPRINT
    # -----------------------------------------------------

    folium.Rectangle(
        bounds=image_bounds,
        color="#45D6CE",
        weight=2,
        fill=False,
        tooltip="Uploaded SAR image footprint",
    ).add_to(flood_map)

    # -----------------------------------------------------
    # NAVIGATION CONTROLS
    # -----------------------------------------------------

    Fullscreen(
        position="topright",
        title="Expand map",
        title_cancel="Exit fullscreen",
    ).add_to(flood_map)

    MiniMap(
        toggle_display=True,
        position="bottomright",
        zoom_level_offset=-5,
    ).add_to(flood_map)

    # -----------------------------------------------------
    # MEASUREMENT TOOL
    # -----------------------------------------------------

    MeasureControl(
        position="topleft",
        primary_length_unit="kilometers",
        secondary_length_unit="meters",
        primary_area_unit="sqkilometers",
        secondary_area_unit="hectares",
    ).add_to(flood_map)

    # -----------------------------------------------------
    # MOUSE COORDINATES
    # -----------------------------------------------------

    MousePosition(
        position="bottomright",
        separator=" | ",
        prefix="Coordinates:",
        num_digits=5,
    ).add_to(flood_map)

    # -----------------------------------------------------
    # LAYER CONTROLS
    # -----------------------------------------------------

    folium.LayerControl(
        collapsed=False,
        position="topright",
    ).add_to(flood_map)

    add_map_legend(flood_map)

    # -----------------------------------------------------
    # FIT TO SAR FOOTPRINT
    # -----------------------------------------------------

    flood_map.fit_bounds(
        image_bounds,
        padding=(15, 15),
    )

    return flood_map
