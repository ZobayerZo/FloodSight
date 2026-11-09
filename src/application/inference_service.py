
from pathlib import Path

import numpy as np
import torch

from src.data.preprocessing import preprocess_sar_for_inference
from src.evaluation.predict import load_model, predict_sample
from src.uncertainty.mc_dropout import MCDropoutPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "floodsight_unet_best.pth"
)


class FloodSightInference:
    """
    Reusable FloodSight inference service.

    Supports deterministic prediction and optional
    Monte Carlo dropout uncertainty estimation.
    """

    def __init__(
        self,
        checkpoint_path=DEFAULT_CHECKPOINT,
        device=None,
        threshold=0.5,
        mc_passes=20,
        dropout_probability=0.2,
    ):

        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = float(threshold)
        self.mc_passes = int(mc_passes)
        self.dropout_probability = float(dropout_probability)

        if not 0.0 < self.threshold < 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        self.model, self.checkpoint, self.device = load_model(
            checkpoint_path=self.checkpoint_path,
            device=device,
        )

        self.mc_predictor = MCDropoutPredictor(
            model=self.model,
            device=self.device,
            dropout_probability=self.dropout_probability,
            num_passes=self.mc_passes,
            threshold=self.threshold,
        )

    def predict(self, sar_path, include_uncertainty=True):
        """
        Run deterministic prediction and optionally MC dropout.

        Returns deterministic and stochastic predictions,
        spatial metadata, validity masks, and statistics.
        """

        sar_path = Path(sar_path)

        if not sar_path.exists():
            raise FileNotFoundError(
                f"SAR image not found: {sar_path}"
            )

        image, sar_valid_mask, metadata = (
            preprocess_sar_for_inference(sar_path)
        )

        # Original deterministic prediction.
        probabilities, prediction = predict_sample(
            model=self.model,
            image=image,
            device=self.device,
            threshold=self.threshold,
        )

        prediction = prediction.copy()
        probabilities = probabilities.copy()

        prediction[~sar_valid_mask] = 0
        probabilities[~sar_valid_mask] = np.nan

        valid_pixel_count = int(sar_valid_mask.sum())

        predicted_water_pixels = int(prediction.sum())

        predicted_water_percentage = (
            predicted_water_pixels / valid_pixel_count * 100.0
            if valid_pixel_count > 0
            else np.nan
        )

        result = {
            "sample_name": sar_path.stem,
            "probabilities": probabilities,
            "prediction": prediction,
            "sar_valid_mask": sar_valid_mask,
            "metadata": metadata,
            "uncertainty": None,
            "statistics": {
                "valid_sar_pixels": valid_pixel_count,
                "predicted_water_pixels": predicted_water_pixels,
                "predicted_water_percentage": float(
                    predicted_water_percentage
                ),
                "mean_water_probability": (
                    float(np.nanmean(probabilities))
                    if valid_pixel_count > 0
                    else np.nan
                ),
                "threshold": self.threshold,
            },
            "model_info": {
                "architecture": "U-Net",
                "encoder": "ResNet34",
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_epoch": self.checkpoint.get("epoch"),
                "device": str(self.device),
            },
        }

        if include_uncertainty:

            mc_result = self.mc_predictor.predict(
                image=image,
                valid_mask=sar_valid_mask,
            )

            uncertainty_map = mc_result["predictive_variance"]

            mc_prediction = mc_result["prediction"]
            mc_mean_probability = mc_result["mean_probability"]

            result["uncertainty"] = {
                "mean_probability": mc_mean_probability,
                "predictive_variance": uncertainty_map,
                "prediction": mc_prediction,
                "num_passes": mc_result["num_passes"],
                "dropout_probability": self.dropout_probability,
            }

            valid_variance = uncertainty_map[sar_valid_mask]

            result["statistics"].update({
                "mc_predicted_water_pixels": int(
                    mc_prediction.sum()
                ),
                "mc_predicted_water_percentage": (
                    float(mc_prediction.sum() / valid_pixel_count * 100.0)
                    if valid_pixel_count > 0
                    else np.nan
                ),
                "mean_predictive_variance": (
                    float(valid_variance.mean())
                    if valid_variance.size > 0
                    else np.nan
                ),
                "max_predictive_variance": (
                    float(valid_variance.max())
                    if valid_variance.size > 0
                    else np.nan
                ),
                "mc_passes": self.mc_passes,
            })

        return result


if __name__ == "__main__":

    import pandas as pd

    manifest_path = (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "sen1floods11_manifest.csv"
    )

    manifest = pd.read_csv(manifest_path)

    sample = manifest[
        (manifest["split"] == "test_data")
        & (manifest["usable"] == True)
    ].iloc[0]

    sar_path = PROJECT_ROOT / sample["sar_file"]

    print("INITIALIZING FLOODSIGHT INFERENCE SERVICE")
    print("=" * 55)

    service = FloodSightInference()

    print("Device:", service.device)
    print("Checkpoint epoch:", service.checkpoint.get("epoch"))

    print("\nRUNNING DETERMINISTIC + MC INFERENCE")

    result = service.predict(
        sar_path,
        include_uncertainty=True,
    )

    print("Sample:", result["sample_name"])

    print("\nMODEL INFORMATION:")
    for key, value in result["model_info"].items():
        print(f"  {key}: {value}")

    print("\nSTATISTICS:")
    for key, value in result["statistics"].items():
        print(f"  {key}: {value}")

    print("\nUNCERTAINTY MAP:")
    print(
        "  Shape:",
        result["uncertainty"]["predictive_variance"].shape
    )

    print("\nFLOODSIGHT INFERENCE SERVICE TEST PASSED")
