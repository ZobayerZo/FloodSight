
from pathlib import Path

import numpy as np
import torch

from src.models.unet import build_unet


def load_model(
    checkpoint_path,
    device=None,
    encoder_name="resnet34",
):
    """Load a trained FloodSight U-Net checkpoint."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = build_unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, checkpoint, device


def predict_sample(model, image, device, threshold=0.5):
    """Generate water probabilities and a binary prediction."""

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if image.ndim != 3 or image.shape[0] != 2:
        raise ValueError(
            "Expected image shape (2, height, width)."
        )

    image = image.float().unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image)
        probabilities = torch.sigmoid(logits)[0, 0]

    probabilities = probabilities.cpu().numpy()
    prediction = (probabilities >= threshold).astype(np.uint8)

    return probabilities, prediction


def calculate_iou(prediction, target, valid_mask):
    """Calculate IoU using only valid pixels."""

    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    valid_mask = np.asarray(valid_mask).astype(bool)

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")

    if prediction.shape != valid_mask.shape:
        raise ValueError("Prediction and valid-mask shapes differ.")

    intersection = np.logical_and(
        prediction,
        target,
    ) & valid_mask

    union = np.logical_or(
        prediction,
        target,
    ) & valid_mask

    union_count = np.count_nonzero(union)

    if union_count == 0:
        return 0.0

    return np.count_nonzero(intersection) / union_count
