import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss that ignores invalid pixels."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets, valid_mask):
        """
        Args:
            logits: Model outputs, shape (B, 1, H, W).
            targets: Binary labels, shape (B, 1, H, W).
            valid_mask: Valid pixels (1) and invalid pixels (0).

        Returns:
            Scalar combined loss.
        """

        valid_mask = valid_mask.float()
        targets = targets.float()

        valid_pixels = valid_mask.sum()

        if valid_pixels.item() == 0:
            raise ValueError("Batch contains no valid pixels.")

        # Masked Binary Cross-Entropy.
        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        bce_loss = (bce_map * valid_mask).sum() / valid_pixels

        # Masked Dice Loss.
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities * valid_mask
        targets = targets * valid_mask

        intersection = (probabilities * targets).sum()
        denominator = probabilities.sum() + targets.sum()

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        dice_loss = 1.0 - dice_score

        # Combined loss.
        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss
