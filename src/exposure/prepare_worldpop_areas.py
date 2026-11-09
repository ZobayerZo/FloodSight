
from pathlib import Path
import json

import pandas as pd
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from pyproj import Geod


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FOOTPRINT_FILE = (
    PROJECT_ROOT / "reports" / "tables" / "exposure_footprints.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "worldpop" / "areas"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

geod = Geod(ellps="WGS84")


def main():
    footprints = pd.read_csv(FOOTPRINT_FILE)

    required_columns = {
        "sample", "event", "crs",
        "min_x", "min_y", "max_x", "max_y"
    }

    missing = required_columns - set(footprints.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if not footprints["crs"].eq("EPSG:4326").all():
        raise ValueError("All footprints must use EPSG:4326.")

    records = []

    for event, group in footprints.groupby("event"):
        sample_polygons = []

        for _, row in group.iterrows():
            polygon = box(
                row["min_x"],
                row["min_y"],
                row["max_x"],
                row["max_y"]
            )
            sample_polygons.append(polygon)

        # Merge overlapping or touching sample footprints.
        merged_geometry = unary_union(sample_polygons)

        if merged_geometry.is_empty or not merged_geometry.is_valid:
            raise ValueError(f"Invalid geometry for {event}")

        if merged_geometry.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(
                f"Unexpected geometry type for {event}: "
                f"{merged_geometry.geom_type}"
            )

        # Calculate geodesic area in square kilometres.
        area_m2, _ = geod.geometry_area_perimeter(merged_geometry)
        area_km2 = abs(area_m2) / 1_000_000

        # Create a GeoJSON Feature.
        feature = {
            "type": "Feature",
            "properties": {
                "event": event,
                "sample_count": len(group),
                "crs": "EPSG:4326"
            },
            "geometry": mapping(merged_geometry)
        }

        output_path = OUTPUT_DIR / f"{event}.geojson"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(feature, f)

        min_x, min_y, max_x, max_y = merged_geometry.bounds

        records.append({
            "event": event,
            "samples": len(group),
            "geometry_type": merged_geometry.geom_type,
            "polygon_parts": (
                len(merged_geometry.geoms)
                if merged_geometry.geom_type == "MultiPolygon"
                else 1
            ),
            "area_km2": area_km2,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "geojson_file": str(output_path.relative_to(PROJECT_ROOT))
        })

        print(
            f"{event:12s} | "
            f"samples={len(group):2d} | "
            f"geometry={merged_geometry.geom_type:12s} | "
            f"area={area_km2:10.2f} km2"
        )

    summary = pd.DataFrame(records)

    summary_path = (
        PROJECT_ROOT
        / "reports"
        / "tables"
        / "worldpop_download_areas.csv"
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("WORLDPOP DOWNLOAD AREAS PREPARED")
    print("=" * 65)
    print("Events:", len(summary))
    print("Samples:", summary["samples"].sum())
    print("Total union area:", round(summary["area_km2"].sum(), 2), "km2")
    print("GeoJSON directory:", OUTPUT_DIR)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
