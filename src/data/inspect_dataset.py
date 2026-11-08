
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

print("=" * 60)
print("FLOODSIGHT - DATASET DIRECTORY INSPECTION")
print("=" * 60)

print(f"\nProject root: {PROJECT_ROOT}")
print(f"Raw data directory: {RAW_DATA_DIR}")

if RAW_DATA_DIR.exists():
    print("\nRaw data directory exists.")

    contents = list(RAW_DATA_DIR.iterdir())

    if contents:
        print("\nCurrent contents:")
        for item in contents:
            print(f" - {item.name}")
    else:
        print("\nRaw data directory is currently empty.")
else:
    print("\nRaw data directory does not exist.")
