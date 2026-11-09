
from pathlib import Path
import time

import pandas as pd
import requests
from shapely.geometry import box, mapping


ROOT = Path(__file__).resolve().parents[2]

FOOTPRINT_FILE = ROOT / "reports" / "tables" / "exposure_footprints.csv"

API_URL = "https://api.worldpop.org/v3/population"

SAMPLES = [
    "Paraguay_511199",
    "Paraguay_280900",
    "Ghana_406026",
]


def main():
    footprints = pd.read_csv(FOOTPRINT_FILE)

    session = requests.Session()
    results = []

    for sample in SAMPLES:
        row = footprints.loc[
            footprints["sample"] == sample
        ].iloc[0]

        polygon = box(
            row["min_x"],
            row["min_y"],
            row["max_x"],
            row["max_y"]
        )

        year = {
            "Paraguay": 2018,
            "Ghana": 2018,
        }[row["event"]]

        geojson = {
            "type": "Feature",
            "properties": {"sample": sample},
            "geometry": mapping(polygon)
        }

        payload = {
            "geojson": geojson,
            "year": year
        }

        print(f"\nQuerying {sample} ({year}) ...")

        try:
            response = session.post(
                API_URL,
                json=payload,
                timeout=(30, 300)
            )

            response.raise_for_status()
            data = response.json()

            print("API response:", data)

            results.append({
                "sample": sample,
                "event": row["event"],
                "year": year,
                "status": "Success",
                "population": data.get("total_population"),
                "area_km2": data.get("area_km2"),
                "population_density": data.get("population_density"),
                "response": str(data)
            })

        except Exception as exc:
            print("Request failed:", str(exc))

            results.append({
                "sample": sample,
                "event": row["event"],
                "year": year,
                "status": "Failed",
                "response": str(exc)
            })

        time.sleep(2)

    output = ROOT / "reports" / "tables" / "worldpop_individual_api_check.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(output, index=False)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
