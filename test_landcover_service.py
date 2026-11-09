
from pathlib import Path
import numpy as np

from src.application.inference_service import FloodSightInference
from src.application.landcover_service import LandCoverExposureService
from src.data.preprocessing import preprocess_sar_for_inference


# --------------------------------------------------
# 1. Input
# --------------------------------------------------

sar_path = Path(
    "data/raw/Sen1Floods11/v1.1/data/flood_events/"
    "HandLabeled/S1Hand/Ghana_406026_S1Hand.tif"
)

checkpoint_path = Path(
    "models/checkpoints/floodsight_unet_best.pth"
)

if not sar_path.exists():
    raise FileNotFoundError(sar_path)

if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)


# --------------------------------------------------
# 2. Load and preprocess SAR
# --------------------------------------------------

image, sar_valid_mask, metadata = preprocess_sar_for_inference(
    sar_path
)


# --------------------------------------------------
# 3. Run FloodSight prediction
# --------------------------------------------------

inference = FloodSightInference(
    checkpoint_path=checkpoint_path,
    mc_passes=20,
    dropout_probability=0.2,
)

result = inference.predict(
    sar_path,
    include_uncertainty=False,
)

prediction = result["prediction"]

# --------------------------------------------------
# 4. Run land-cover exposure analysis
# --------------------------------------------------

service = LandCoverExposureService(
    worldcover_dir="data/external/worldcover",
)

landcover_result = service.analyze(
    prediction=prediction,
    sar_valid_mask=result["sar_valid_mask"],
    metadata=result["metadata"],
)


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("FLOODSIGHT LAND-COVER EXPOSURE TEST")
print("=" * 60)

print("SAR image:", result["sample_name"])

for key, value in landcover_result.items():
    print(f"{key}: {value}")

print("=" * 60)
print("LAND-COVER ANALYSIS COMPLETED")
print("=" * 60)
