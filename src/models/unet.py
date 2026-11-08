import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=2,
    classes=1,
):
    """Build a U-Net model for Sentinel-1 SAR water segmentation."""

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )

    return model


if __name__ == "__main__":
    model = build_unet()

    print("FLOODSIGHT U-NET ARCHITECTURE")
    print("=" * 60)
    print("Architecture: U-Net")
    print("Encoder: ResNet34")
    print("Input channels: 2 (VV, VH)")
    print("Output channels: 1")
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
