# FloodSight — Dataset Quality Assessment

## 1. Dataset Overview

- Dataset: Sen1Floods11 v1.1
- Total hand-labelled samples: 446
- Image dimensions: 512 × 512
- SAR bands: VV and VH
- Label values:
  - -1: Invalid / NoData
  - 0: Non-water
  - 1: Water
- CRS: EPSG:4326
- Represented event groups: 11

## 2. Official Dataset Splits

| Split | Samples |
|---|---:|
| Training | 252 |
| Validation | 89 |
| Test | 90 |
| Bolivia data | 15 |
| Total | 446 |

The original official split assignments are preserved.

## 3. Label Quality

Across all samples:

- Invalid label pixels: 15,932,910
- Non-water pixels: 90,277,709
- Water pixels: 10,705,605
- Samples without water-labelled pixels: 52
- Samples with water-labelled pixels: 394

Zero-water samples are not automatically excluded. They may provide valid negative training examples.

## 4. Invalid Label Coverage

- Mean invalid coverage: 13.63%
- Median invalid coverage: 0.95%
- Samples with no invalid pixels: 83
- Samples with more than 25% invalid pixels: 90
- Samples with more than 50% invalid pixels: 38
- Completely invalid samples: 5

Invalid coverage varies considerably across flood events.

## 5. SAR Data Integrity

All 446 SAR images were verified.

- Dimensions: 512 × 512
- Bands: 2
- Band 1: VV
- Band 2: VH
- Total pixels per band: 116,916,224
- Finite pixels per band: 114,404,519
- Non-finite pixels per band: 2,511,705
- Non-finite proportion: approximately 2.15%
- Raster dimension errors: 0

## 6. SAR and Label Validity Relationship

Among non-finite SAR pixels:

- 2,350,195 (93.57%) overlap invalid label pixels.
- 161,510 (6.43%) overlap valid label pixels.

The 161,510 affected pixels occur in 32 samples.

Therefore, label validity alone cannot determine SAR input validity.

## 7. Effective Supervision

Effective supervision is defined as:

SAR-valid AND label-valid.

A SAR pixel is considered valid when both VV and VH are finite.

| Split | Original samples | Usable samples | Effective pixels |
|---|---:|---:|---:|
| Training | 252 | 251 | 57,279,257 |
| Validation | 89 | 86 | 20,274,154 |
| Test | 90 | 88 | 20,443,959 |
| Bolivia data | 15 | 15 | 2,824,434 |

Six samples have zero effective supervision:

| Split | Sample | Reason |
|---|---|---|
| Test | Ghana_83483 | No valid label pixels |
| Test | Paraguay_34417 | No valid SAR pixels |
| Training | Ghana_26376 | No valid label pixels |
| Validation | Ghana_5079 | No valid label pixels |
| Validation | Ghana_234935 | No valid label pixels |
| Validation | Ghana_277 | No valid label pixels |

These samples will be excluded from supervised training and evaluation, with exclusions documented.

## 8. JRC Permanent-Water Investigation

The JRC permanent-water mask overlaps hand-labelled water differently across samples and events.

Sample-level findings:

- Samples with water-labelled pixels: 394
- Samples without water-labelled pixels: 52
- Mean JRC overlap among samples with water pixels: 15.34%
- Median JRC overlap: 0.64%
- Samples with zero JRC overlap: 172

JRC should be treated as an auxiliary reference rather than universally subtracting it from hand-labelled water masks.

## 9. Preliminary Preprocessing Decisions

1. Preserve original raw TIFF files.
2. Keep official split assignments.
3. Treat -1 labels as ignored pixels.
4. Create a separate SAR-validity mask.
5. Require both VV and VH to be finite.
6. Exclude samples with zero effective supervision.
7. Calculate normalization statistics using training data only.
8. Apply identical normalization parameters to validation and test data.
9. Compare a two-channel SAR baseline against a model with an additional SAR-validity channel.

## 10. Important Research Considerations

- Do not randomly mix samples across official splits.
- Maintain event-aware evaluation.
- Document all exclusions.
- Distinguish hand-labelled water from permanent water.
- Do not interpret all water-labelled pixels as newly inundated floodwater without further analysis.
- Report exposure estimates as model-derived estimates, not actual confirmed impacts.

---

**Status:** Initial dataset integrity and quality assessment completed.

**Next phase:** Design and implement reproducible SAR preprocessing.
