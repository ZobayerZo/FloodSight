import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import preprocess_sample


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT / "data/interim/sen1floods11_manifest.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "data/processed"


def main():

    manifest = pd.read_csv(MANIFEST_PATH)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = []

    usable_samples = manifest[manifest["usable"] == True].copy()

    print("=" * 65)
    print("FLOODSIGHT - BATCH SAR PREPROCESSING")
    print("=" * 65)

    print("Total samples:", len(manifest))
    print("Samples to process:", len(usable_samples))
    print("Samples excluded:", len(manifest) - len(usable_samples))

    for split_name in manifest["split"].unique():

        split_samples = usable_samples[
            usable_samples["split"] == split_name
        ]

        split_output = OUTPUT_ROOT / split_name
        split_output.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {split_name}")
        print("-" * 50)

        for index, (_, row) in enumerate(
            split_samples.iterrows(), start=1
        ):

            sample_name = row["sample"]

            sar_path = PROJECT_ROOT / row["sar_file"]
            label_path = PROJECT_ROOT / row["label_file"]

            image, target, valid_mask = preprocess_sample(
                sar_path,
                label_path
            )

            # Verify preprocessing results.
            if not np.isfinite(image).all():
                raise ValueError(
                    f"Non-finite input values found: {sample_name}"
                )

            if int(valid_mask.sum()) != int(row["effective_pixels"]):
                raise ValueError(
                    f"Effective pixel mismatch: {sample_name}"
                )

            if not valid_mask.any():
                raise ValueError(
                    f"Unexpected empty supervision: {sample_name}"
                )

            # Save processed sample.
            output_file = split_output / f"{sample_name}.npz"

            np.savez_compressed(
                output_file,
                image=image.astype(np.float32),
                target=target.astype(np.float32),
                valid_mask=valid_mask.astype(np.uint8)
            )

            summary.append({
                "sample": sample_name,
                "event": row["event"],
                "split": split_name,
                "output_file": str(
                    output_file.relative_to(PROJECT_ROOT)
                ),
                "effective_pixels": int(valid_mask.sum()),
                "water_pixels": int(target[valid_mask].sum()),
                "non_water_pixels": int(
                    (valid_mask & (target == 0)).sum()
                ),
                "file_size_bytes": output_file.stat().st_size
            })

            print(
                f"[{index}/{len(split_samples)}] "
                f"{sample_name} - "
                f"{int(valid_mask.sum()):,} valid pixels"
            )

    summary_df = pd.DataFrame(summary)

    summary_path = OUTPUT_ROOT / "preprocessing_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("BATCH PREPROCESSING COMPLETED")
    print("=" * 65)

    print("Processed samples:", len(summary_df))
    print("Expected samples:", len(usable_samples))
    print("Excluded samples:", len(manifest) - len(usable_samples))

    print("\nProcessed samples by split:")
    print(
        summary_df.groupby("split")["sample"]
        .count()
        .to_string()
    )

    print("\nTotal processed storage (MB):")
    print(
        round(summary_df["file_size_bytes"].sum() / (1024 ** 2), 2)
    )

    print("\nSummary saved to:")
    print(summary_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nPREPROCESSING FAILED")
        print(type(error).__name__ + ":", error)
        sys.exit(1)
