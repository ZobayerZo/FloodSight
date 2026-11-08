
# FloodSight: Uncertainty and Reference Disagreement Case Studies

## 1. Overview

This report documents two selected test-sample investigations from
FloodSight, an uncertainty-aware Sentinel-1 SAR water segmentation project.

The analysis uses the recovered U-Net checkpoint with a post-decoder
MC Dropout2d layer (p=0.20), evaluated using 20 stochastic forward passes.

Predictions use a threshold of 0.5. All segmentation metrics are calculated
over valid SAR and reference-label pixels.

The two case studies investigate different forms of model-reference
disagreement.

## 2. Pakistan_528249: False-positive analysis

### 2.1 Segmentation results

- Valid pixels: 76,982
- TP: 0
- FP: 1,085
- FN: 0
- TN: 75,897
- Error rate: 1.41%
- Reference water coverage: 0.00%
- Predicted water coverage: 1.41%

### 2.2 Spatial concentration

The 1,085 false-positive pixels formed 14 connected components.

- Largest component: 309 pixels (28.48%)
- Largest three components: 820 pixels (75.58%)
- Largest five components: 948 pixels (87.37%)
- Largest ten components: 1,070 pixels (98.62%)

The false positives were therefore highly spatially concentrated.

### 2.3 Uncertainty analysis

Mean predictive variance:

- False positives: 0.014756
- True negatives: 0.001819

At a variance threshold of 0.01:

- High-uncertainty false positives: 558
- Low-uncertainty false positives: 527
- False-positive capture rate: 51.43%
- High-uncertainty true negatives: 1,952
- Low-uncertainty true negatives: 73,945
- High-uncertainty error rate: 22.23%
- Low-uncertainty error rate: 0.71%

This sample demonstrates a strong association between predictive
variance and segmentation error.

The threshold is a diagnostic operating point, not a calibrated
probability-of-error threshold.

## 3. Ghana_866994: False-negative and reference disagreement analysis

### 3.1 Segmentation results

- Valid pixels: 117,511
- TP: 71,547
- FP: 10,509
- FN: 34,424
- TN: 1,031
- Error rate: 38.237%
- Reference water coverage: 90.18%
- Predicted water coverage: 69.83%

### 3.2 Uncertainty by prediction outcome

| Outcome | Mean variance | Mean water probability |
|---|---:|---:|
| TP | 0.012776 | 0.9397 |
| FP | 0.037964 | 0.9122 |
| FN | 0.001149 | 0.0395 |
| TN | 0.001971 | 0.0631 |

The false-negative regions have very low predictive variance.
The model is confidently predicting non-water in many pixels
labelled as water by the hand reference.

### 3.3 False-negative spatial concentration

There were 21 connected false-negative components.

- Largest component: 29,614 pixels (86.03% of all FNs)
- Largest three components: 34,066 pixels (98.96%)
- Largest five components: 34,244 pixels (99.48%)

The largest component had:

- Mean variance: 0.000529
- Mean water probability: 0.0236

### 3.4 SAR backscatter characteristics

| Region | VV mean (dB) | VH mean (dB) |
|---|---:|---:|
| Largest FN | -9.742 | -15.464 |
| Other FN | -9.853 | -17.684 |
| TP | -18.648 | -31.764 |
| TN | -9.682 | -15.801 |

The largest false-negative component has VV and VH backscatter
characteristics much closer to the true-negative regions than to
the correctly detected water regions.

The histograms also show substantial distributional overlap between
the largest FN component and TN pixels.

This is an association, not proof of a particular physical land-cover
condition or cause.

### 3.5 JRC permanent-water comparison

| Region | JRC overlap | Overlap percentage |
|---|---:|---:|
| Largest FN | 5 | 0.02% |
| Other FN | 74 | 1.54% |
| TP | 58,900 | 82.32% |
| TN | 22 | 2.13% |

Total JRC water pixels: 69,450.

JRC coverage over valid pixels: 59.10%.

The largest false-negative component has almost no JRC overlap.
However, JRC represents permanent-water information and is not
a complete flood-event reference.

### 3.6 Hand reference versus JRC

| Category | Pixels |
|---|---:|
| Both water | 58,979 |
| Hand water only | 46,992 |
| JRC water only | 10,471 |
| Both non-water | 1,069 |

Agreement statistics:

- Hand/JRC water overlap (recall against hand water): 55.66%
- JRC precision against hand labels: 84.92%
- Overall agreement: 51.10%

These values demonstrate substantial disagreement between the
two reference sources.

### 3.7 Model performance across reference categories

| Category | Pixels | Predicted water | Predicted non-water | Error rate |
|---|---:|---:|---:|---:|
| Both water | 58,979 | 58,900 | 79 | 0.13% |
| Hand water only | 46,992 | 12,647 | 34,345 | 73.09% |
| JRC water only | 10,471 | 10,449 | 22 | 99.79% |
| Both non-water | 1,069 | 60 | 1,009 | 5.61% |

### 3.8 Uncertainty across reference categories

| Category | Mean variance | Median variance | Mean water probability | Error rate |
|---|---:|---:|---:|---:|
| Both water | 0.013665 | 0.003065 | 0.953514 | 0.13% |
| Hand water only | 0.003143 | 0.000281 | 0.262912 | 73.09% |
| JRC water only | 0.038119 | 0.019536 | 0.912530 | 99.79% |
| Both non-water | 0.001736 | 0.000391 | 0.090418 | 5.61% |

The hand-only water category exhibits substantial error with low
predictive variance. The JRC-only category exhibits substantial
error against the hand reference with elevated predictive variance.

These patterns suggest two distinct types of disagreement:
confident disagreement and uncertain disagreement.

They do not establish which reference source is physically correct.

## 4. Comparative findings

| Characteristic | Pakistan_528249 | Ghana_866994 |
|---|---|---|
| Dominant error | FP | FN |
| Error spatial concentration | Top 3 contain 75.58% | Largest component contains 86.03% |
| Error uncertainty | Elevated | Very low in largest FN |
| Additional reference analysis | Not conducted here | JRC comparison |
| Main observation | Uncertainty can flag some false positives | Confident false negatives can escape uncertainty detection |

## 5. Research implications

The investigations suggest that predictive uncertainty and
reference disagreement should be studied together.

Potential research directions include:

1. Evaluate uncertainty-error association across the complete test set.
2. Investigate confidently incorrect predictions separately from
   high-uncertainty errors.
3. Compare model performance in regions where reference sources agree
   and disagree.
4. Evaluate uncertainty thresholds on validation data rather than
   selecting them from individual test examples.
5. Investigate spatial error clusters and their SAR characteristics.
6. Consider event-level and chip-level statistical uncertainty.

## 6. Limitations

1. These are two selected test-sample case studies, not a complete
   evaluation of all test samples.
2. The MC Dropout model uses the original trained U-Net weights with
   a post-decoder Dropout2d wrapper (p=0.20). It was not fine-tuned
   using the dropout wrapper.
3. The uncertainty analysis uses 20 stochastic predictions.
4. Predictive variance is not a calibrated probability of error.
5. JRC permanent-water information is auxiliary and is not definitive
   flood-event ground truth.
6. Hand-reference/JRC disagreement cannot independently establish
   reference-label errors.
7. The two samples belong to the existing test set. This analysis
   does not establish independent-event generalization.
8. Pixel-level observations are spatially correlated and should not
   be treated as independent statistical samples.
9. The physical explanation for the Ghana SAR signature remains
   unresolved.

## 7. Conclusion

The two investigations demonstrate that uncertainty-aware segmentation
requires more than identifying high-variance predictions.

Pakistan_528249 illustrates a case where elevated predictive variance
is associated with false-positive predictions.

Ghana_866994 illustrates a contrasting case where a large false-negative
region is predicted with very low uncertainty and where substantial
disagreement exists between hand-labelled and JRC reference information.

Together, these cases motivate a broader evaluation of uncertainty,
confident errors, spatial error structure, and reference-source agreement
in SAR-based flood mapping.
