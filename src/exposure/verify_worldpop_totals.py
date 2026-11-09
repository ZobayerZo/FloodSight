
from pathlib import Path
import json
import time

import requests
import rasterio
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AREA_DIR = PROJECT_ROOT / "data" / "external" / "worldpop" / "areas"
RASTER_DIR = PROJECT_ROOT / "data" / "external" / "worldpop" / "rasters"

OUTPUT_FILE = (
    PROJECT_ROOT / "reports" / "tables" / "worldpop_api_crosscheck.csv"
)

API_URL = "https://api.worldpop.org/v3/population"


def main():
    years = {
        "India": 2016,
        "Pakistan": 2017,
        "Sri-Lanka": 2017,
        "Ghana": 2018,
        "Mekong": 2018,
        "Nigeria": 2018,
        "Paraguay": 2018,
        "Somalia": 2018,
        "Spain": 2019,
        "USA": 2019,
    }

    session = requests.Session()
    results = []

    for event, year in years.items():
        geojson_path = AREA_DIR / f"{event}.geojson"
        raster_path = RASTER_DIR / f"{event}_{year}_100m.tif"

        with geojson_path.open("r", encoding="utf-8") as f:
            geojson = json.load(f)

        payload = {
            "geojson": geojson,
            "year": year
        }

        print(f"Checking {event} ({year}) ...")

        response = session.post(
            API_URL,
            json=payload,
            timeout=(30, 300)
        )
        response.raise_for_status()

        api_data = response.json()

        with rasterio.open(raster_path) as ds:
            population = ds.read(1, masked=True)
            raster_total = float(population.sum())

        api_total = float(api_data["total_population"])

        difference = raster_total - api_total
        difference_pct = (
            abs(difference) / api_total * 100
            if api_total else None
        )

        results.append({
            "event": event,
            "year": year,
            "api_population": api_total,
            "raster_population": raster_total,
            "difference": difference,
            "absolute_difference_pct": difference_pct,
            "api_area_km2": api_data.get("area_km2"),
            "api_density": api_data.get("population_density"),
            "data_source": api_data.get("data_source")
        })

        print(
            f"  API total:    {api_total:,.2f}\n"
            f"  Raster total: {raster_total:,.2f}\n"
            f"  Difference:   {difference:,.2f}"
        )

        time.sleep(2)

    summary = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nWORLDPOP API CROSS-CHECK")
    print("=" * 90)
    print(summary.to_string(index=False))
    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
