
import numpy as np


def calculate_metrics(prediction, target, valid_mask):
    """
    Calculate binary segmentation metrics using valid pixels only.

    Empty-mask convention:
    - Both masks empty: IoU and Dice = 1.
    - Undefined precision or recall: NaN.
    """

    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    valid_mask = np.asarray(valid_mask).astype(bool)

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")

    if prediction.shape != valid_mask.shape:
        raise ValueError("Prediction and valid-mask shapes differ.")

    pred = prediction[valid_mask]
    truth = target[valid_mask]

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())

    def safe_divide(numerator, denominator):
        return (
            numerator / denominator
            if denominator > 0
            else np.nan
        )

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn

    # Both prediction and reference are empty.
    if union == 0:
        iou = 1.0
        dice = 1.0
    else:
        iou = tp / union
        dice = (
            2 * tp / dice_denominator
            if dice_denominator > 0
            else np.nan
        )

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )

    return {
        "valid_pixels": int(valid_mask.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "iou": iou,
        "dice": dice,
        "f1": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }
