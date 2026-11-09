
import os
import csv
import math
import tempfile
from io import StringIO
from collections.abc import Mapping

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from streamlit_folium import st_folium

from src.application.application_service import FloodSightApplication
from src.application.exposure_service import EVENT_YEARS
from src.application.map_service import create_flood_map
from src.application.export_service import (
    export_prediction_geotiff,
    export_probability_geotiff,
    export_uncertainty_geotiff,
    export_landcover_geotiff,
    export_exposed_cropland_geotiff,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FloodSight | Flood Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0B1220;
        color: #E8EEF6;
    }

    [data-testid="stSidebar"] {
        background-color: #101C2B;
        border-right: 1px solid #263649;
    }

    [data-testid="stSidebar"] * {
        color: #DCE7F2;
    }

    .brand {
        font-size: 28px;
        font-weight: 800;
        color: #F1F5F9;
        letter-spacing: -1px;
    }

    .brand-accent {
        color: #45D6CE;
    }

    .subtitle {
        color: #91A4B8;
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: 600;
    }

    .section-label {
        color: #45D6CE;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        color: #F1F5F9;
        letter-spacing: -1.2px;
        margin-bottom: 5px;
    }

    .hero-description {
        color: #9BAFC1;
        font-size: 14px;
        line-height: 1.7;
    }

    .metric-card {
        background: linear-gradient(145deg, #17283B, #122033);
        border: 1px solid #293D52;
        border-radius: 13px;
        padding: 20px;
        min-height: 120px;
    }

    .metric-label {
        color: #9BAFC1;
        font-size: 12px;
        font-weight: 600;
    }

    .metric-value {
        color: #F1F5F9;
        font-size: 27px;
        font-weight: 800;
        margin-top: 10px;
    }

    .metric-caption {
        color: #45D6CE;
        font-size: 11px;
        margin-top: 4px;
    }

    .info-card {
        background-color: #142337;
        border: 1px solid #293D52;
        border-radius: 13px;
        padding: 22px;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #13AFA9, #36CFC5);
        color: #061923;
        border: none;
        border-radius: 9px;
        font-weight: 800;
        min-height: 45px;
        width: 100%;
    }

    div.stButton > button:hover {
        background: #65E1D8;
        color: #061923;
        border: none;
    }

    [data-testid="stFileUploader"] {
        background-color: #142337;
        border: 1px dashed #456078;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stMetric"] {
        background-color: #142337;
        border: 1px solid #293D52;
        border-radius: 12px;
        padding: 14px;
    }

    hr {
        border-color: #293D52;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None


# =========================================================
# GENERAL HELPERS
# =========================================================

def make_filename_slug(value):
    """Convert a value into a safe filename component."""
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value)
    ).strip("_")


def finite_number(value):
    """Return a finite float or None."""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def format_metric(value, decimals=3):
    """Format a scalar metric safely."""
    number = finite_number(value)

    if number is not None:
        return f"{number:,.{decimals}f}"

    if value is None:
        return "Unavailable"

    return str(value)


def is_raster_value(value):
    """Identify values that should not be shown in scalar reports."""
    if isinstance(value, np.ndarray):
        return True

    if isinstance(value, (list, tuple)):
        return any(
            isinstance(item, (np.ndarray, list, tuple))
            for item in value
        )

    return False


# =========================================================
# PROBABILITY DISTRIBUTION
# =========================================================

def calculate_probability_distribution(prediction):
    """
    Calculate probability bins directly from the prediction
    arrays. This does not depend on statistics-service keys.
    """

    probabilities = np.asarray(
        prediction["probabilities"],
        dtype=np.float32,
    )

    valid_mask = np.asarray(
        prediction["sar_valid_mask"]
    ).astype(bool)

    if probabilities.shape != valid_mask.shape:
        raise ValueError(
            "Probability and valid-mask dimensions do not match."
        )

    valid = (
        valid_mask
        & np.isfinite(probabilities)
    )

    values = probabilities[valid]

    if values.size == 0:
        return []

    bins = [
        ("Below 50%", 0.0, 0.5),
        ("50–60%", 0.5, 0.6),
        ("60–70%", 0.6, 0.7),
        ("70–80%", 0.7, 0.8),
        ("80–100%", 0.8, 1.000001),
    ]

    output = []

    for label, lower, upper in bins:
        selected = (
            (values >= lower)
            & (values < upper)
        )

        count = int(selected.sum())

        output.append({
            "Probability range": label,
            "Pixels": count,
            "Percentage": (
                count / values.size * 100
            ),
        })

    return output


def render_probability_chart(prediction, chart_key):
    """Display the water-probability histogram."""

    probabilities = np.asarray(
        prediction["probabilities"],
        dtype=np.float32,
    )

    valid_mask = np.asarray(
        prediction["sar_valid_mask"]
    ).astype(bool)

    valid = (
        valid_mask
        & np.isfinite(probabilities)
    )

    values = probabilities[valid]

    if values.size == 0:
        st.info("No valid probability values are available.")
        return

    fig = go.Figure(
        data=[
            go.Histogram(
                x=values,
                nbinsx=40,
                marker_color="#27C9C0",
                opacity=0.88,
            )
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#142337",
        height=350,
        xaxis_title="Water probability",
        yaxis_title="Number of pixels",
        margin=dict(l=20, r=20, t=30, b=30),
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=chart_key,
    )

    st.caption(
        f"Distribution calculated from {values.size:,} "
        "valid SAR pixels. NoData pixels are excluded."
    )


# =========================================================
# ADVANCED STATISTICS
# =========================================================

def flatten_statistics(data, prefix=""):
    """
    Recursively flatten nested scalar statistics.

    Arrays and raster data are excluded.
    The function does not assume particular nested key names.
    """

    rows = []

    if not isinstance(data, Mapping):
        return rows

    for key, value in data.items():

        full_key = (
            f"{prefix}.{key}"
            if prefix
            else str(key)
        )

        if isinstance(value, Mapping):
            rows.extend(
                flatten_statistics(value, full_key)
            )

        elif is_raster_value(value):
            continue

        elif isinstance(value, np.generic):
            rows.append((full_key, value.item()))

        elif isinstance(value, (str, int, float, bool)) or value is None:
            rows.append((full_key, value))

    return rows


def render_advanced_statistics(statistics, summary, prediction):
    """
    Render advanced statistics using confirmed application
    summary fields and dynamically discovered nested values.
    """

    st.markdown("#### Flood extent statistics")

    status = statistics.get("status", "UNKNOWN")

    flood_area_ha = finite_number(
        summary.get("flood_area_ha")
    )

    flood_area_km2 = finite_number(
        summary.get("flood_area_km2")
    )

    flood_coverage = finite_number(
        summary.get("flood_coverage_percentage")
    )

    predicted_pixels = summary.get(
        "predicted_water_pixels"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Flood area",
            (
                f"{flood_area_ha:,.3f} ha"
                if flood_area_ha is not None
                else "Unavailable"
            ),
        )

    with col2:
        st.metric(
            "Flood area (km²)",
            (
                f"{flood_area_km2:,.4f}"
                if flood_area_km2 is not None
                else "Unavailable"
            ),
        )

    with col3:
        st.metric(
            "Flood coverage",
            (
                f"{flood_coverage:.3f}%"
                if flood_coverage is not None
                else "Unavailable"
            ),
        )

    with col4:
        st.metric(
            "Predicted water pixels",
            (
                f"{int(predicted_pixels):,}"
                if predicted_pixels is not None
                else "Unavailable"
            ),
        )

    if status != "SUCCESS":
        st.warning(
            "Advanced statistics were not completed successfully."
        )

        if statistics.get("error"):
            st.code(str(statistics["error"]))

        return

    st.success("Advanced flood statistics completed successfully.")

    st.markdown("---")
    st.markdown("#### Water probability distribution")

    distribution = calculate_probability_distribution(
        prediction
    )

    if distribution:
        st.dataframe(
            distribution,
            use_container_width=True,
            hide_index=True,
        )

        labels = [
            item["Probability range"]
            for item in distribution
        ]

        counts = [
            item["Pixels"]
            for item in distribution
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=counts,
                    marker_color="#27C9C0",
                )
            ]
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1220",
            plot_bgcolor="#142337",
            height=320,
            xaxis_title="Water probability range",
            yaxis_title="Pixel count",
            margin=dict(l=20, r=20, t=30, b=30),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="advanced_probability_bins",
        )

    st.markdown("---")
    st.markdown("#### Detailed statistics returned by the service")

    scalar_rows = flatten_statistics(statistics)

    scalar_rows = [
        (key, value)
        for key, value in scalar_rows
        if key != "status"
    ]

    if scalar_rows:
        st.dataframe(
            [
                {
                    "Metric": key,
                    "Value": str(value) if value is not None else "Unavailable",
                }
                for key, value in scalar_rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No additional scalar statistics were returned."
        )

    st.caption(
        "The detailed table displays scalar values from the "
        "statistics service. Raster arrays are excluded."
    )


# =========================================================
# UNCERTAINTY
# =========================================================

def render_uncertainty(prediction):
    """Display MC Dropout predictive uncertainty."""

    uncertainty = prediction.get("uncertainty")

    if not uncertainty:
        st.info(
            "Uncertainty analysis was not enabled for this assessment."
        )
        return

    variance_data = uncertainty.get(
        "predictive_variance"
    )

    if variance_data is None:
        st.info("Predictive variance is unavailable.")
        return

    variance = np.asarray(
        variance_data,
        dtype=np.float32,
    )

    valid_mask = np.asarray(
        prediction["sar_valid_mask"]
    ).astype(bool)

    if variance.shape != valid_mask.shape:
        st.warning(
            "Uncertainty dimensions do not match the SAR valid mask."
        )
        return

    valid = (
        valid_mask
        & np.isfinite(variance)
    )

    values = variance[valid]

    if values.size == 0:
        st.warning("No valid uncertainty values are available.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Mean predictive variance",
            f"{values.mean():.6f}",
        )

    with col2:
        st.metric(
            "Maximum predictive variance",
            f"{values.max():.6f}",
        )

    with col3:
        st.metric(
            "Valid uncertainty pixels",
            f"{values.size:,}",
        )

    fig = go.Figure(
        data=[
            go.Histogram(
                x=values,
                nbinsx=40,
                marker_color="#F5BE64",
                opacity=0.88,
            )
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#142337",
        height=320,
        xaxis_title="Predictive variance",
        yaxis_title="Number of pixels",
        margin=dict(l=20, r=20, t=30, b=30),
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="floodsight_uncertainty_distribution",
    )

    st.caption(
        "MC Dropout predictive variance measures variation "
        "between stochastic predictions. It is an uncertainty "
        "proxy, not a calibrated probability of prediction error."
    )


# =========================================================
# EXPOSURE DETAILS
# =========================================================

def render_exposure_details(exposure):

    st.markdown("#### Population exposure details")

    details = [
        ("Population grid cells", exposure.get("population_grid_cells")),
        ("Valid population cells", exposure.get("valid_population_cells")),
        ("Population coverage (%)", exposure.get("population_coverage_pct")),
        ("Valid SAR-population cells", exposure.get("valid_sar_population_cells")),
        ("SAR coverage of population (%)", exposure.get("sar_coverage_of_population_pct")),
        ("Population in valid overlap", exposure.get("population_sum_valid_overlap")),
        ("Estimated exposed population", exposure.get("estimated_exposed_population")),
        ("Predicted water cells", exposure.get("predicted_water_cells")),
        ("Exposure status", exposure.get("status")),
    ]

    for index in range(0, len(details), 3):
        columns = st.columns(3)

        for column, item in zip(
            columns,
            details[index:index + 3],
        ):
            label, value = item

            with column:
                if isinstance(value, (int, float, np.number)):
                    display = format_metric(value, 4)
                else:
                    display = str(value) if value is not None else "Unavailable"

                st.metric(label, display)


def render_landcover_details(landcover):

    st.markdown("#### Agricultural exposure details")

    if not landcover:
        st.info("Agricultural exposure results are unavailable.")
        return

    details = [
        ("Dataset", landcover.get("dataset", "ESA WorldCover")),
        ("Dataset year", landcover.get("dataset_year")),
        ("Cropland class", landcover.get("cropland_class_name", "Cropland")),
        ("Total SAR valid area (ha)", landcover.get("total_sar_valid_area_ha")),
        ("WorldCover covered area (ha)", landcover.get("worldcover_covered_area_ha")),
        ("Cropland area (ha)", landcover.get("cropland_area_ha")),
        ("Predicted water area (ha)", landcover.get("predicted_water_area_ha")),
        ("Estimated exposed cropland (ha)", landcover.get("estimated_exposed_cropland_ha")),
        ("Cropland exposure (%)", landcover.get("cropland_exposure_percentage")),
        ("Water area over cropland (%)", landcover.get("water_area_over_cropland_percentage")),
        ("WorldCover coverage (%)", landcover.get("worldcover_coverage_percentage")),
        ("Cropland pixels", landcover.get("cropland_pixels")),
        ("Exposed cropland pixels", landcover.get("exposed_cropland_pixels")),
        ("Valid land-cover pixels", landcover.get("valid_landcover_pixels")),
        ("Coverage status", landcover.get("coverage_status")),
        ("Exposure status", landcover.get("exposure_status")),
    ]

    for index in range(0, len(details), 3):
        columns = st.columns(3)

        for column, item in zip(
            columns,
            details[index:index + 3],
        ):
            label, value = item

            with column:
                if isinstance(value, (int, float, np.number)):
                    display = format_metric(value, 4)
                else:
                    display = str(value) if value is not None else "Unavailable"

                st.metric(label, display)

    matching_tiles = landcover.get("matching_tiles")

    if matching_tiles:
        st.caption(
            "WorldCover tiles used: "
            + ", ".join(map(str, matching_tiles))
        )


# =========================================================
# CSV EXPORT
# =========================================================

def create_statistics_csv(sections):
    """
    Create a CSV from nested scalar values.

    Excludes NumPy arrays, raster masks and raster data.
    """

    buffer = StringIO()
    writer = csv.writer(buffer)

    writer.writerow([
        "Section",
        "Metric",
        "Value",
    ])

    for section_name, section_data in sections:

        for metric_name, metric_value in flatten_statistics(
            section_data
        ):

            writer.writerow([
                section_name,
                metric_name,
                metric_value,
            ])

    return buffer.getvalue()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            Flood<span class="brand-accent">Sight</span>
        </div>
        <div class="subtitle">
            FLOOD INTELLIGENCE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Flood Analysis",
            "Results",
            "About FloodSight",
        ],
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            background:#17283B;
            border:1px solid #293D52;
            border-radius:10px;
            padding:15px;
        ">
            <div style="color:#45D6CE;font-size:11px;
                        font-weight:700;letter-spacing:1px;">
                MODEL STATUS
            </div>
            <div style="color:#E8EEF6;font-size:14px;
                        font-weight:600;margin-top:8px;">
                U-Net ResNet34
            </div>
            <div style="color:#9BAFC1;font-size:12px;">
                Two-band SAR segmentation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("FloodSight Research Application")
    st.caption("Version 1.2 · Local deployment")

    if st.session_state.analysis_result is not None:

        st.markdown("---")
        st.success("A completed analysis is available.")

        if st.button("Clear current analysis"):
            st.session_state.analysis_result = None
            st.session_state.analysis_error = None
            st.rerun()


# =========================================================
# GLOBAL HEADER
# =========================================================

st.markdown(
    """
    <div class="section-label">
        GEOSPATIAL INTELLIGENCE PLATFORM
    </div>
    <div class="hero-title">
        Flood Monitoring & Exposure Intelligence
    </div>
    <div class="hero-description">
        Transform Sentinel-1 SAR imagery into AI-powered
        flood maps, uncertainty assessments and population
        exposure insights.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.markdown("### Platform Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">SEGMENTATION MODEL</div>
                <div class="metric-value">U-Net</div>
                <div class="metric-caption">ResNet34 encoder</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">SUPPORTED EVENTS</div>
                <div class="metric-value">10</div>
                <div class="metric-caption">Population datasets available</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">INPUT DATA</div>
                <div class="metric-value">SAR</div>
                <div class="metric-caption">Two-band GeoTIFF imagery</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#F1F5F9;">
                    Welcome to FloodSight
                </h3>
                <p style="color:#9BAFC1;line-height:1.8;">
                    FloodSight is an AI-powered flood mapping
                    and exposure assessment platform built
                    using Sentinel-1 SAR imagery and deep learning.
                </p>
                <p style="color:#9BAFC1;line-height:1.8;">
                    It combines semantic segmentation,
                    Monte Carlo Dropout uncertainty estimation,
                    population exposure and agricultural overlap analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#F1F5F9;">
                    Analysis workflow
                </h3>
                <p style="color:#45D6CE;">01 · Upload SAR imagery</p>
                <p style="color:#45D6CE;">02 · Select flood event</p>
                <p style="color:#45D6CE;">03 · Generate flood prediction</p>
                <p style="color:#45D6CE;">04 · Assess population exposure</p>
                <p style="color:#45D6CE;">05 · Assess cropland exposure</p>
                <p style="color:#45D6CE;">06 · Explore and export results</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Start a New Flood Analysis"):
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None
        st.rerun()

    if st.session_state.analysis_result is not None:
        st.info(
            "You already have a completed assessment. "
            "Open Results to explore it."
        )


# =========================================================
# FLOOD ANALYSIS PAGE
# =========================================================

elif page == "Flood Analysis":

    st.markdown("### New Flood Analysis")

    st.write(
        "Upload a compatible two-band Sentinel-1 SAR GeoTIFF "
        "and select the corresponding flood event."
    )

    left, right = st.columns([1.1, 0.9])

    with left:

        st.markdown("#### 01 · Input imagery")

        uploaded_file = st.file_uploader(
            "Upload SAR GeoTIFF",
            type=["tif", "tiff"],
            help=(
                "Upload a two-band Sentinel-1 SAR GeoTIFF "
                "containing VV and VH bands."
            ),
            key="sar_upload",
        )

        if uploaded_file:

            st.success(
                f"File selected: {uploaded_file.name}"
            )

            st.caption(
                f"File size: "
                f"{uploaded_file.size / (1024 * 1024):.2f} MB"
            )

            if uploaded_file.size > 200 * 1024 * 1024:
                st.error(
                    "The maximum supported upload size is 200 MB."
                )

    with right:

        st.markdown("#### 02 · Event configuration")

        event = st.selectbox(
            "Select flood event",
            options=list(EVENT_YEARS.keys()),
            format_func=lambda name: (
                f"{name} · {EVENT_YEARS[name]}"
            ),
        )

        population_filename = (
            f"{event}_{EVENT_YEARS[event]}_100m.tif"
        )

        st.info(
            f"Population dataset: {population_filename}"
        )

        include_uncertainty = st.checkbox(
            "Enable MC Dropout uncertainty analysis",
            value=True,
            help=(
                "Runs multiple stochastic forward passes. "
                "This increases processing time."
            ),
        )

        threshold = st.slider(
            "Flood prediction threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
        )

        st.caption(
            f"Current classification threshold: {threshold:.2f}"
        )

    st.markdown("---")

    can_analyze = (
        uploaded_file is not None
        and uploaded_file.size <= 200 * 1024 * 1024
    )

    if st.button(
        "Generate Flood Assessment",
        disabled=not can_analyze,
        key="generate_flood_assessment",
    ):

        st.session_state.analysis_result = None
        st.session_state.analysis_error = None

        temporary_path = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".tif",
                delete=False,
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temporary_path = temp_file.name

            progress = st.progress(0)
            status_box = st.empty()

            status_box.info("Initializing FloodSight model...")
            progress.progress(15)

            application = FloodSightApplication(
                threshold=threshold,
                mc_passes=20,
                dropout_probability=0.2,
            )

            status_box.info(
                "Processing SAR imagery and generating flood prediction..."
            )
            progress.progress(35)

            result = application.analyze(
                sar_path=temporary_path,
                event=event,
                include_uncertainty=include_uncertainty,
                original_filename=uploaded_file.name,
            )

            progress.progress(100)

            st.session_state.analysis_result = result
            st.session_state.analysis_error = None

            status_box.success(
                "Flood assessment completed successfully."
            )

            st.success(
                "Your assessment is ready. Select Results "
                "from the sidebar to explore the output."
            )

        except Exception as error:

            st.session_state.analysis_error = str(error)
            st.error(f"Analysis failed: {error}")

        finally:

            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)


# =========================================================
# RESULTS PAGE
# =========================================================

elif page == "Results":

    st.markdown("### Flood Assessment Results")

    result = st.session_state.analysis_result

    if result is None:

        st.info(
            "No analysis available yet. "
            "Open Flood Analysis to generate a prediction."
        )

    else:

        summary = result["summary"]
        exposure = result["exposure"]
        prediction = result["prediction"]
        landcover = result.get("landcover_exposure") or {}
        advanced_statistics = result.get("advanced_statistics") or {}
        uncertainty = prediction.get("uncertainty")
        input_info = result["input"]

        # -------------------------------------------------
        # SUMMARY METRICS
        # -------------------------------------------------

        st.markdown("#### Assessment overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Predicted Water Pixels",
                f"{summary['predicted_water_pixels']:,}",
            )

        with col2:
            water_percentage = finite_number(
                summary.get("predicted_water_percentage")
            )

            st.metric(
                "Predicted Water",
                (
                    f"{water_percentage:.2f}%"
                    if water_percentage is not None
                    else "Unavailable"
                ),
            )

        with col3:
            exposed_population = finite_number(
                summary.get("estimated_exposed_population")
            )

            st.metric(
                "Estimated Exposure",
                (
                    f"{exposed_population:,.2f}"
                    if exposed_population is not None
                    else "Unavailable"
                ),
            )

        with col4:
            coverage = finite_number(
                summary.get("population_coverage_pct")
            )

            st.metric(
                "Population Coverage",
                (
                    f"{coverage:.2f}%"
                    if coverage is not None
                    else "Unavailable"
                ),
            )

        # -------------------------------------------------
        # FLOOD AREA SUMMARY
        # -------------------------------------------------

        st.markdown("#### Flood extent overview")

        flood_col1, flood_col2, flood_col3 = st.columns(3)

        flood_area_ha = finite_number(
            summary.get("flood_area_ha")
        )

        flood_area_km2 = finite_number(
            summary.get("flood_area_km2")
        )

        flood_coverage = finite_number(
            summary.get("flood_coverage_percentage")
        )

        with flood_col1:
            st.metric(
                "Predicted flood area",
                (
                    f"{flood_area_ha:,.3f} ha"
                    if flood_area_ha is not None
                    else "Unavailable"
                ),
            )

        with flood_col2:
            st.metric(
                "Predicted flood area",
                (
                    f"{flood_area_km2:,.4f} km²"
                    if flood_area_km2 is not None
                    else "Unavailable"
                ),
            )

        with flood_col3:
            st.metric(
                "Flood coverage",
                (
                    f"{flood_coverage:.3f}%"
                    if flood_coverage is not None
                    else "Unavailable"
                ),
            )

        if advanced_statistics.get("status") == "SUCCESS":
            st.success("Advanced flood statistics available.")
        else:
            st.warning(
                "Advanced statistics are unavailable or incomplete."
            )

            if advanced_statistics.get("error"):
                st.caption(str(advanced_statistics["error"]))

        # -------------------------------------------------
        # AGRICULTURAL SUMMARY
        # -------------------------------------------------

        st.markdown("#### Agricultural exposure overview")

        ag_col1, ag_col2, ag_col3 = st.columns(3)

        exposed_cropland = finite_number(
            summary.get("estimated_exposed_cropland_ha")
        )

        cropland_percentage = finite_number(
            summary.get("cropland_exposure_percentage")
        )

        worldcover_coverage = finite_number(
            summary.get("worldcover_coverage_percentage")
        )

        with ag_col1:
            st.metric(
                "Estimated exposed cropland",
                (
                    f"{exposed_cropland:,.3f} ha"
                    if exposed_cropland is not None
                    else "Unavailable"
                ),
            )

        with ag_col2:
            st.metric(
                "Cropland exposure",
                (
                    f"{cropland_percentage:.3f}%"
                    if cropland_percentage is not None
                    else "Unavailable"
                ),
            )

        with ag_col3:
            st.metric(
                "WorldCover coverage",
                (
                    f"{worldcover_coverage:.2f}%"
                    if worldcover_coverage is not None
                    else "Unavailable"
                ),
            )

        st.caption(
            "Agricultural exposure is the estimated spatial overlap "
            "between model-predicted water and ESA WorldCover cropland. "
            "It does not confirm crop loss or agricultural damage."
        )

        st.markdown("---")

        # -------------------------------------------------
        # ANALYSIS INFORMATION
        # -------------------------------------------------

        st.markdown("#### Analysis information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.write(
                f"**Input image:** {input_info['sar_filename']}"
            )
            st.write(
                f"**Flood event:** {input_info['event']}"
            )

        with info_col2:
            st.write(
                f"**Population raster:** "
                f"{input_info['population_raster']}"
            )
            st.write(
                f"**Population year:** "
                f"{input_info['population_year']}"
            )
            st.write(
                f"**Land-cover dataset:** "
                f"{input_info.get('landcover_dataset', 'ESA WorldCover')}"
            )
            st.write(
                f"**Land-cover year:** "
                f"{input_info.get('landcover_year', 2021)}"
            )

        # -------------------------------------------------
        # POPULATION COVERAGE STATUS
        # -------------------------------------------------

        population_status = exposure.get(
            "status",
            "UNKNOWN",
        )

        if population_status == "ADEQUATE":

            st.success(
                "Population coverage meets the configured "
                "coverage requirements."
            )

        elif population_status == "PARTIAL_COVERAGE":

            st.warning(
                "Partial population coverage: the exposure estimate "
                "represents the eligible observed area, not the "
                "entire SAR footprint."
            )

        else:

            st.warning(
                "Population exposure could not be reliably "
                "estimated for this image."
            )

        st.caption(
            "Population exposure is an estimate based on the overlap "
            "between predicted water and population raster values. "
            "It does not represent verified damage, casualties or "
            "confirmed affected individuals."
        )

        # -------------------------------------------------
        # LAND-COVER COVERAGE STATUS
        # -------------------------------------------------

        landcover_status = landcover.get(
            "exposure_status",
            "UNKNOWN",
        )

        if landcover_status == "ADEQUATE":

            st.success(
                "ESA WorldCover coverage is adequate for this SAR footprint."
            )

        elif landcover_status == "PARTIAL_COVERAGE":

            st.warning(
                "WorldCover coverage is partial. Agricultural exposure "
                "applies only to the covered area."
            )

        elif landcover_status == "ERROR":

            st.warning(
                "Agricultural exposure could not be calculated: "
                + str(landcover.get("error", "Unknown error"))
            )

        else:

            st.info(
                "Agricultural exposure is unavailable or no valid "
                "cropland was identified."
            )

        st.caption(
            "Temporal limitation: ESA WorldCover 2021 may post-date "
            "some flood events. Agricultural overlap is indicative, "
            "not verified historical crop damage."
        )

        # -------------------------------------------------
        # RESULTS TABS
        # -------------------------------------------------

        (
            tab_overview,
            tab_statistics,
            tab_map,
            tab_agriculture,
            tab_uncertainty,
            tab_downloads,
        ) = st.tabs([
            "📊 Overview",
            "📈 Flood Statistics",
            "🗺️ Flood Map",
            "🌾 Agricultural Exposure",
            "🔍 Uncertainty",
            "⬇️ Downloads",
        ])

        # =================================================
        # OVERVIEW TAB
        # =================================================

        with tab_overview:

            st.markdown("#### Flood probability distribution")

            render_probability_chart(
                prediction,
                chart_key="floodsight_probability_distribution",
            )

            st.markdown("---")

            render_exposure_details(exposure)

            st.markdown("---")

            render_landcover_details(landcover)

            st.markdown("---")

            st.markdown("#### Interpretation")

            st.info(
                "The predicted water extent represents the model's "
                "water classification. It is not necessarily "
                "newly flooded area."
            )

            if population_status == "PARTIAL_COVERAGE":
                st.warning(
                    "Because population coverage is partial, the "
                    "estimated exposure should not be interpreted "
                    "as the total population exposed across the "
                    "full SAR footprint."
                )

        # =================================================
        # ADVANCED FLOOD STATISTICS TAB
        # =================================================

        with tab_statistics:

            render_advanced_statistics(
                statistics=advanced_statistics,
                summary=summary,
                prediction=prediction,
            )

            st.markdown("---")

            st.info(
                "Flood area and coverage are spatial estimates derived "
                "from model-predicted water pixels over valid SAR pixels. "
                "They do not independently establish that all predicted "
                "water represents newly inundated land."
            )

        # =================================================
        # FLOOD MAP TAB
        # =================================================

        with tab_map:

            st.markdown("#### Geospatial flood visualization")

            st.caption(
                "Explore predicted water extent, probability, "
                "uncertainty and agricultural overlap."
            )

            try:

                flood_map = create_flood_map(
                    prediction_result=prediction,
                    map_height=650,
                    landcover_result=landcover,
                )

                map_key = (
                    "floodsight_map_"
                    + make_filename_slug(input_info["sar_filename"])
                    + "_"
                    + make_filename_slug(input_info["event"])
                )

                st_folium(
                    flood_map,
                    width=None,
                    height=650,
                    key=map_key,
                )

            except Exception as error:

                st.error(
                    f"Unable to render flood map: {error}"
                )

            st.caption(
                "Use the layer selector to explore predicted water, "
                "probability, uncertainty, ESA WorldCover classes "
                "and predicted water over cropland."
            )

        # =================================================
        # AGRICULTURAL EXPOSURE TAB
        # =================================================

        with tab_agriculture:

            st.markdown("#### Agricultural exposure assessment")

            if landcover.get("exposure_status") == "ERROR":

                st.warning(
                    "Agricultural exposure is unavailable. "
                    + str(landcover.get("error", ""))
                )

            elif landcover.get("exposure_status") == "NO_DATA":

                st.info(
                    "No valid WorldCover data was available "
                    "for this SAR footprint."
                )

            else:

                ag_ha = finite_number(
                    landcover.get("estimated_exposed_cropland_ha")
                )

                ag_pct = finite_number(
                    landcover.get("cropland_exposure_percentage")
                )

                crop_ha = finite_number(
                    landcover.get("cropland_area_ha")
                )

                water_ha = finite_number(
                    landcover.get("predicted_water_area_ha")
                )

                metric_a, metric_b, metric_c, metric_d = st.columns(4)

                metric_a.metric(
                    "Estimated exposed cropland",
                    (
                        f"{ag_ha:,.3f} ha"
                        if ag_ha is not None
                        else "Unavailable"
                    ),
                )

                metric_b.metric(
                    "Total cropland area",
                    (
                        f"{crop_ha:,.2f} ha"
                        if crop_ha is not None
                        else "Unavailable"
                    ),
                )

                metric_c.metric(
                    "Predicted water area",
                    (
                        f"{water_ha:,.2f} ha"
                        if water_ha is not None
                        else "Unavailable"
                    ),
                )

                metric_d.metric(
                    "Cropland exposure",
                    (
                        f"{ag_pct:.3f}%"
                        if ag_pct is not None
                        else "Unavailable"
                    ),
                )

                st.markdown("---")

                render_landcover_details(landcover)

                interpretation = landcover.get("interpretation")

                if interpretation:
                    st.info(interpretation)

                temporal_caveat = landcover.get("temporal_caveat")

                if temporal_caveat:
                    st.warning(temporal_caveat)

                if landcover.get("matching_tiles"):
                    st.caption(
                        "Source tiles: "
                        + ", ".join(
                            map(str, landcover["matching_tiles"])
                        )
                    )

        # =================================================
        # UNCERTAINTY TAB
        # =================================================

        with tab_uncertainty:

            st.markdown("#### Model uncertainty assessment")

            if uncertainty:

                render_uncertainty(prediction)

                st.markdown("---")

                st.info(
                    "MC Dropout runs multiple stochastic model "
                    "forward passes. Predictive variance indicates "
                    "variation between these predictions. It is "
                    "an uncertainty proxy, not a calibrated "
                    "probability that a pixel is incorrect."
                )

            else:

                st.info(
                    "MC Dropout uncertainty analysis was not enabled "
                    "when this assessment was generated."
                )

        # =================================================
        # DOWNLOADS TAB
        # =================================================

        with tab_downloads:

            st.markdown("#### Download analysis products")

            st.caption(
                "Export georeferenced raster products and "
                "analysis statistics for further GIS analysis."
            )

            image_slug = make_filename_slug(
                input_info["sar_filename"]
            )

            # ---------------------------------------------
            # FLOOD RASTER PRODUCTS
            # ---------------------------------------------

            st.markdown("##### Flood raster products")

            download_col1, download_col2, download_col3 = (
                st.columns(3)
            )

            with download_col1:

                st.markdown("**Flood prediction**")

                try:

                    prediction_bytes = export_prediction_geotiff(
                        prediction
                    )

                    st.download_button(
                        label="Download prediction",
                        data=prediction_bytes,
                        file_name=f"{image_slug}_flood_prediction.tif",
                        mime="image/tiff",
                        use_container_width=True,
                        key="download_flood_prediction",
                    )

                    st.caption(
                        "Binary raster: 0 = non-water, "
                        "1 = predicted water, 255 = NoData."
                    )

                except Exception as error:

                    st.error(
                        f"Prediction export failed: {error}"
                    )

            with download_col2:

                st.markdown("**Water probability**")

                try:

                    probability_bytes = export_probability_geotiff(
                        prediction
                    )

                    st.download_button(
                        label="Download probability",
                        data=probability_bytes,
                        file_name=f"{image_slug}_water_probability.tif",
                        mime="image/tiff",
                        use_container_width=True,
                        key="download_water_probability",
                    )

                    st.caption(
                        "Float32 raster containing water probabilities."
                    )

                except Exception as error:

                    st.error(
                        f"Probability export failed: {error}"
                    )

            with download_col3:

                st.markdown("**Prediction uncertainty**")

                if uncertainty:

                    try:

                        uncertainty_bytes = (
                            export_uncertainty_geotiff(prediction)
                        )

                        st.download_button(
                            label="Download uncertainty",
                            data=uncertainty_bytes,
                            file_name=(
                                f"{image_slug}_predictive_uncertainty.tif"
                            ),
                            mime="image/tiff",
                            use_container_width=True,
                            key="download_predictive_uncertainty",
                        )

                        st.caption(
                            "Float32 raster containing MC Dropout "
                            "predictive variance."
                        )

                    except Exception as error:

                        st.error(
                            f"Uncertainty export failed: {error}"
                        )

                else:

                    st.info(
                        "Enable MC Dropout during analysis "
                        "to generate this product."
                    )

            st.markdown("---")

            # ---------------------------------------------
            # AGRICULTURAL RASTER PRODUCTS
            # ---------------------------------------------

            st.markdown("##### Agricultural raster products")

            landcover_available = (
                landcover.get("landcover_raster") is not None
                and landcover.get("landcover_valid_mask") is not None
            )

            exposed_cropland_available = (
                landcover_available
                and landcover.get("exposed_cropland_mask") is not None
            )

            ag_download_col1, ag_download_col2 = st.columns(2)

            with ag_download_col1:

                st.markdown("**ESA WorldCover classification**")

                if landcover_available:

                    try:

                        landcover_bytes = export_landcover_geotiff(
                            landcover,
                            prediction,
                        )

                        st.download_button(
                            label="Download land cover",
                            data=landcover_bytes,
                            file_name=f"{image_slug}_worldcover_2021.tif",
                            mime="image/tiff",
                            use_container_width=True,
                            key="download_worldcover_geotiff",
                        )

                        st.caption(
                            "Categorical ESA WorldCover class codes; "
                            "0 = NoData."
                        )

                    except Exception as error:

                        st.error(
                            f"Land-cover export failed: {error}"
                        )

                else:

                    st.info(
                        "Land-cover raster is unavailable."
                    )

            with ag_download_col2:

                st.markdown("**Predicted water over cropland**")

                if exposed_cropland_available:

                    try:

                        exposed_bytes = (
                            export_exposed_cropland_geotiff(
                                landcover,
                                prediction,
                            )
                        )

                        st.download_button(
                            label="Download exposed cropland",
                            data=exposed_bytes,
                            file_name=f"{image_slug}_exposed_cropland.tif",
                            mime="image/tiff",
                            use_container_width=True,
                            key="download_exposed_cropland_geotiff",
                        )

                        st.caption(
                            "Binary overlap mask: 1 = predicted water "
                            "over cropland, 0 = other valid pixels, "
                            "255 = NoData."
                        )

                    except Exception as error:

                        st.error(
                            f"Exposed-cropland export failed: {error}"
                        )

                else:

                    st.info(
                        "Exposed-cropland raster is unavailable."
                    )

            st.caption(
                "Agricultural overlap is an indicative spatial estimate, "
                "not confirmation of crop loss or agricultural damage."
            )

            st.markdown("---")

            # ---------------------------------------------
            # COMPLETE STATISTICS CSV
            # ---------------------------------------------

            st.markdown("##### Complete assessment statistics")

            csv_data = create_statistics_csv([
                ("Input", input_info),
                ("Summary", summary),
                ("Population exposure", exposure),
                ("Agricultural exposure", landcover),
                ("Advanced flood statistics", advanced_statistics),
            ])

            st.download_button(
                label="Download complete statistics (CSV)",
                data=csv_data,
                file_name=f"{image_slug}_floodsight_statistics.csv",
                mime="text/csv",
                key="download_complete_statistics_csv",
            )

            st.caption(
                "The CSV contains scalar values from the input, "
                "summary, population exposure, agricultural exposure "
                "and advanced statistics. Raster arrays are excluded "
                "and provided separately as GeoTIFF files."
            )

            st.markdown("---")

            st.info(
                "GeoTIFF exports preserve the original SAR image "
                "CRS, affine transform and spatial dimensions. "
                "NoData pixels are retained as NoData."
            )


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About FloodSight":

    st.markdown("### About FloodSight")

    st.write(
        "FloodSight is an AI-powered geospatial intelligence "
        "platform designed to identify water extent from "
        "Sentinel-1 SAR imagery and estimate population exposure."
    )

    st.markdown("---")
    st.markdown("#### Core technologies")

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.markdown("**Deep Learning**")
            st.write("PyTorch")
            st.write("U-Net with ResNet34 encoder")
            st.write("Monte Carlo Dropout")

    with col2:

        with st.container(border=True):

            st.markdown("**Geospatial Processing**")
            st.write("Sentinel-1 SAR VV and VH")
            st.write("Rasterio and GeoPandas")
            st.write("WorldPop population datasets")
            st.write("ESA WorldCover 2021")

    col3, col4 = st.columns(2)

    with col3:

        with st.container(border=True):

            st.markdown("**Visualization**")
            st.write("Streamlit")
            st.write("Plotly")
            st.write("Folium")

    with col4:

        with st.container(border=True):

            st.markdown("**Analysis capabilities**")
            st.write("Flood extent prediction")
            st.write("Advanced flood statistics")
            st.write("Predictive uncertainty estimation")
            st.write("Population exposure assessment")
            st.write("Cropland exposure estimation")

    st.markdown("---")
    st.markdown("#### How FloodSight works")

    workflow = [
        (
            "01",
            "Upload SAR imagery",
            "Provide a compatible two-band GeoTIFF.",
        ),
        (
            "02",
            "Select flood event",
            "Choose the corresponding event and population dataset.",
        ),
        (
            "03",
            "Generate flood prediction",
            "The trained U-Net model estimates water probability.",
        ),
        (
            "04",
            "Estimate population exposure",
            "Predicted water extent is combined with population data.",
        ),
        (
            "05",
            "Estimate agricultural exposure",
            "Predicted water is compared with ESA WorldCover cropland.",
        ),
        (
            "06",
            "Explore results",
            "Inspect maps, statistics, uncertainty and exports.",
        ),
    ]

    for number, title, description in workflow:

        with st.container(border=True):

            st.markdown(f"**{number} · {title}**")
            st.caption(description)

    st.markdown("---")
    st.markdown("#### Important interpretation")

    st.warning(
        """
        **Flood extent:** Predicted water extent is not
        necessarily newly flooded area.

        **Population exposure:** This is an estimate based
        on predicted water and population raster overlap.
        It does not represent verified damage, casualties
        or confirmed affected individuals.

        **Coverage:** Partial population coverage means
        the estimate represents only the eligible observed
        area, not necessarily the entire SAR footprint.

        **Agricultural exposure:** The estimated cropland overlap
        is a spatial estimate, not confirmed agricultural damage.

        **WorldCover timing:** ESA WorldCover 2021 may post-date
        some flood events in this study. It should not be treated
        as verified historical land cover for those events.

        **Uncertainty:** MC Dropout predictive variance is a
        relative uncertainty proxy, not a calibrated probability
        of prediction error.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#72869B;
        font-size:11px;
        padding:10px;
    ">
        FLOODSIGHT · AI-POWERED FLOOD INTELLIGENCE
        <br>
        Research prototype · Local deployment
    </div>
    """,
    unsafe_allow_html=True,
)
