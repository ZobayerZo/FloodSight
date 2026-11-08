
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.predict import load_model, predict_sample
from src.evaluation.metrics import calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_dataset(
    dataset_dir,
    checkpoint_path,
    threshold=0.5,
    output_dir=None,
):
    """Evaluate a trained FloodSight model across a dataset."""

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir else (
        PROJECT_ROOT / "reports" / "tables"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(dataset_dir.glob("*.npz"))

    if not sample_files:
        raise FileNotFoundError(
            f"No NPZ samples found in {dataset_dir}"
        )

    model, checkpoint, device = load_model(checkpoint_path)

    sample_results = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_valid_pixels = 0

    print(f"Evaluation samples: {len(sample_files)}")
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Threshold: {threshold}")
    print("-" * 60)

    for index, sample_file in enumerate(sample_files, start=1):

        data = np.load(sample_file)

        image = data["image"]
        target = data["target"]
        valid_mask = data["valid_mask"]

        probabilities, prediction = predict_sample(
            model,
            image,
            device,
            threshold=threshold,
        )

        metrics = calculate_metrics(
            prediction,
            target,
            valid_mask,
        )

        sample_results.append({
            "sample": sample_file.stem,
            **metrics,
        })

        total_tp += metrics["true_positives"]
        total_fp += metrics["false_positives"]
        total_fn += metrics["false_negatives"]
        total_tn += metrics["true_negatives"]
        total_valid_pixels += metrics["valid_pixels"]

        if index % 10 == 0 or index == len(sample_files):
            print(
                f"Processed {index}/{len(sample_files)} samples"
            )

        data.close()

    # Calculate pooled pixel-level metrics.
    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0 else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0 else 0.0
    )

    iou = (
        total_tp / (total_tp + total_fp + total_fn)
        if total_tp + total_fp + total_fn > 0 else 0.0
    )

    dice = (
        2 * total_tp / (2 * total_tp + total_fp + total_fn)
        if 2 * total_tp + total_fp + total_fn > 0 else 0.0
    )

    accuracy = (
        (total_tp + total_tn) / total_valid_pixels
        if total_valid_pixels > 0 else 0.0
    )

    aggregate_results = {
        "samples_evaluated": len(sample_files),
        "valid_pixels": total_valid_pixels,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "true_negatives": total_tn,
        "iou": iou,
        "dice": dice,
        "f1": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }

    # Save per-sample metrics.
    sample_df = pd.DataFrame(sample_results)

    sample_csv = output_dir / "test_sample_metrics.csv"
    sample_df.to_csv(sample_csv, index=False)

    # Save aggregate metrics.
    aggregate_df = pd.DataFrame([aggregate_results])

    aggregate_csv = output_dir / "test_aggregate_metrics.csv"
    aggregate_df.to_csv(aggregate_csv, index=False)

    print("\n" + "=" * 60)
    print("FLOODSIGHT TEST EVALUATION RESULTS")
    print("=" * 60)

    for key, value in aggregate_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value:,}")

    print("\nSaved files:")
    print(sample_csv)
    print(aggregate_csv)

    return sample_df, aggregate_results


if __name__ == "__main__":

    config_path = PROJECT_ROOT / "configs" / "model.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    evaluate_dataset(
        dataset_dir=PROJECT_ROOT / "data" / "processed" / "test_data",
        checkpoint_path=PROJECT_ROOT / config["model"]["checkpoint"],
        threshold=config["prediction"]["threshold"],
    )
