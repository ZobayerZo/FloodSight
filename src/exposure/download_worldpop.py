
from pathlib import Path
import json
import time

import requests
import yaml
import rasterio
from rasterio.io import MemoryFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = PROJECT_ROOT / "configs" / "exposure.yaml"
AREA_DIR = PROJECT_ROOT / "data" / "external" / "worldpop" / "areas"
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "worldpop" / "rasters"

API_URL = "https://api.worldpop.org/v3/raster"


def main():
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    events = config["population"]["events"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "User-Agent": "FloodSight-Research-Project/1.0"
    })

    results = []

    for event, details in events.items():
        year = int(details["year"])
        geojson_path = AREA_DIR / f"{event}.geojson"
        output_path = OUTPUT_DIR / f"{event}_{year}_100m.tif"

        if not geojson_path.exists():
            raise FileNotFoundError(geojson_path)

        if output_path.exists():
            print(f"SKIP {event}: raster already exists.")
            continue

        with geojson_path.open("r", encoding="utf-8") as f:
            geojson = json.load(f)

        payload = {
            "geojson": geojson,
            "year": year,
            "resolution": "100m",
            "layer": "population"
        }

        print(f"\nDownloading {event} ({year}) ...")

        response = session.post(
            API_URL,
            json=payload,
            timeout=(30, 600)
        )

        if response.status_code != 200:
            print("HTTP status:", response.status_code)
            print("Response:", response.text[:2000])
            response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")

        if "json" in content_type.lower():
            raise RuntimeError(
                f"WorldPop returned JSON instead of a raster for {event}: "
                f"{response.text[:2000]}"
            )

        # Validate the GeoTIFF before saving it.
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                metadata = {
                    "width": dataset.width,
                    "height": dataset.height,
                    "bands": dataset.count,
                    "crs": str(dataset.crs),
                    "dtype": dataset.dtypes[0],
                    "nodata": dataset.nodata,
                    "bounds": str(dataset.bounds),
                    "resolution": str(dataset.res),
                }

                if dataset.count != 1:
                    raise ValueError(
                        f"{event}: expected one population band, "
                        f"received {dataset.count}"
                    )

                if dataset.crs is None:
                    raise ValueError(f"{event}: missing CRS.")

        # Save only after successful validation.
        output_path.write_bytes(response.content)

        results.append({
            "event": event,
            "year": year,
            "file": str(output_path.relative_to(PROJECT_ROOT)),
            "size_mb": round(len(response.content) / (1024 ** 2), 2),
            **metadata
        })

        print("Saved:", output_path)
        print("Metadata:", metadata)
        print("Size:", round(len(response.content) / (1024 ** 2), 2), "MB")

        # Keep requests comfortably below the API rate limit.
        time.sleep(2)

    if results:
        import pandas as pd

        summary_path = (
            PROJECT_ROOT
            / "reports"
            / "tables"
            / "worldpop_download_summary.csv"
        )

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(summary_path, index=False)

        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        print("Successfully downloaded:", len(results))
        print("Summary:", summary_path)

    print("\nDownload process finished.")


if __name__ == "__main__":
    main()
