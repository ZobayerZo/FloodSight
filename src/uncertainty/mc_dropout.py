
import numpy as np
import torch
import torch.nn as nn


class MCDropoutPredictor:
    """
    Monte Carlo dropout inference for FloodSight U-Net.

    Applies Dropout2d to the final decoder features while
    keeping the original trained model weights unchanged.
    """

    def __init__(
        self,
        model,
        device,
        dropout_probability=0.2,
        num_passes=20,
        threshold=0.5,
    ):

        if not 0.0 < dropout_probability < 1.0:
            raise ValueError(
                "Dropout probability must be between 0 and 1."
            )

        if num_passes < 2:
            raise ValueError(
                "At least two MC passes are required."
            )

        self.model = model
        self.device = device
        self.num_passes = int(num_passes)
        self.threshold = float(threshold)

        if not 0.0 < self.threshold < 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        self.dropout = nn.Dropout2d(
            p=dropout_probability
        ).to(device)

    def predict(self, image, valid_mask=None):

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.ndim != 3 or image.shape[0] != 2:
            raise ValueError(
                "Expected image shape (2, height, width)."
            )

        image = image.float().unsqueeze(0).to(self.device)

        # Keep the original model in evaluation mode.
        self.model.eval()

        probability_samples = []

        with torch.inference_mode():

            for _ in range(self.num_passes):

                features = self.model.encoder(image)

                decoder_features = self.model.decoder(
                    features
                )

                # Apply stochastic dropout after the decoder.
                decoder_features = self.dropout(
                    decoder_features
                )

                logits = self.model.segmentation_head(
                    decoder_features
                )

                probabilities = torch.sigmoid(logits)

                probability_samples.append(
                    probabilities[0, 0].cpu().numpy()
                )

        probability_samples = np.stack(
            probability_samples,
            axis=0
        )

        mean_probability = probability_samples.mean(axis=0)

        predictive_variance = probability_samples.var(
            axis=0,
            ddof=1
        )

        prediction = (
            mean_probability >= self.threshold
        ).astype(np.uint8)

        if valid_mask is not None:

            valid_mask = np.asarray(valid_mask).astype(bool)

            if valid_mask.shape != mean_probability.shape:
                raise ValueError(
                    "Validity mask and prediction dimensions differ."
                )

            prediction[~valid_mask] = 0

            mean_probability = mean_probability.copy()
            predictive_variance = predictive_variance.copy()

            mean_probability[~valid_mask] = np.nan
            predictive_variance[~valid_mask] = np.nan

            return {
                "mean_probability": mean_probability,
                "predictive_variance": predictive_variance,
                "prediction": prediction,
                "num_passes": self.num_passes,
                "threshold": self.threshold,
            }