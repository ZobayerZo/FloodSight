
from pathlib import Path

from src.application.inference_service import FloodSightInference
from src.application.statistics_service import FloodStatisticsService


PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Sen1Floods11"
    / "v1.1"
    / "data"
    / "flood_events"
    / "HandLabeled"
    / "S1Hand"
    / "Ghana_406026_S1Hand.tif"
)
# If your sample is stored elsewhere, replace SAMPLE_PATH
# with its actual path.

print("=" * 60)
print("FLOODSIGHT ADVANCED STATISTICS TEST")
print("=" * 60)

if not SAMPLE_PATH.exists():
    raise FileNotFoundError(
        f"Sample SAR file not found: {SAMPLE_PATH}\n"
        "Please update SAMPLE_PATH to your actual Ghana_406026 file."
    )

print("\nInitializing inference service...")

inference = FloodSightInference(
    mc_passes=20,
    dropout_probability=0.2,
)

print("Running inference...")

prediction_result = inference.predict(
    SAMPLE_PATH,
    include_uncertainty=True,
)

print("Inference completed.")

statistics_service = FloodStatisticsService()

print("\nCalculating advanced statistics...")

results = statistics_service.analyze(
    prediction_result
)

print("\n" + "=" * 60)
print("FLOOD AREA")
print("=" * 60)

for key, value in results["total_valid_area"].items():
    print(f"  Valid area {key}: {value:,.4f}")

for key, value in results["predicted_flood_area"].items():
    print(f"  Flood area {key}: {value:,.4f}")

print(
    "  Flood coverage:",
    f"{results['flood_coverage_percentage']:.4f}%"
)

print("\n" + "=" * 60)
print("PROBABILITY DISTRIBUTION")
print("=" * 60)

for category, values in results["probability_distribution"].items():
    print(f"\n{category}:")

    for key, value in values.items():
        print(f"  {key}: {value:,.4f}")

print("\n" + "=" * 60)
print("CONFIDENCE STATISTICS")
print("=" * 60)

for category, values in results["confidence"].items():
    if isinstance(values, dict):
        print(f"\n{category}:")

        for key, value in values.items():
            print(f"  {key}: {value:,.4f}")
    else:
        print(f"  {category}: {values}")

print("\n" + "=" * 60)
print("UNCERTAINTY STATISTICS")
print("=" * 60)

for key, value in results["uncertainty"].items():
    if isinstance(value, dict):
        print(f"\n{key}:")

        for subkey, subvalue in value.items():
            print(f"  {subkey}: {subvalue:,.6f}")
    else:
        print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("ADVANCED STATISTICS TEST COMPLETED")
print("=" * 60)
