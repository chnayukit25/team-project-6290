# Experiment Results Analysis (First 100 Samples)

## Summary Statistics

**Note:** Plaintext metrics are computed on the first 100 samples to match HE dataset size for direct comparison.

| dataset   | method   |   accuracy(%) |   accuracy_std |   precision |   precision_std |   recall |   recall_std |   f1 |   f1_std |   samples |   infer_mean |   infer_std |
|:----------|:---------|--------------:|---------------:|------------:|----------------:|---------:|-------------:|-----:|---------:|----------:|-------------:|------------:|
| he_a      | HE       |            57 |              0 |         0   |               0 |     0    |            0 | 0    |        0 |       100 |     2.9879   |    0.086848 |
| he_b      | HE       |           100 |              0 |         1   |               0 |     1    |            0 | 1    |        0 |       100 |     2.99575  |    0.06447  |
| he_c      | HE       |            86 |              0 |         0.6 |               0 |     0.79 |            0 | 0.68 |        0 |       100 |     3.00172  |    0.051671 |
| plain_a   | plain    |            57 |              0 |         0   |               0 |     0    |            0 | 0    |        0 |       100 |     0.000777 |    0.00067  |
| plain_b   | plain    |           100 |              0 |         1   |               0 |     1    |            0 | 1    |        0 |       100 |     0.00072  |    0.000289 |
| plain_c   | plain    |            86 |              0 |         0.6 |               0 |     0.79 |            0 | 0.68 |        0 |       100 |     0.000682 |    0.00062  |

## Key Findings

- **Accuracy**: HE average 81.00%, Plain average 81.00%
- **Precision**: HE average 0.533, Plain average 0.533
- **Recall**: HE average 0.596, Plain average 0.596
- **Total Latency**: HE 4.59 ms (encrypt+infer), Plain 0.0007 ms (HE is 6,320x slower)
- **Encryption Overhead**: 1.60 ms per sample (35% of total HE time)
- **HE Inference Only**: 3.00 ms (65% of total HE time)

## Dataset Characteristics

- **he_a.csv**: 100 samples (15 positive, 85 negative)
- **he_b.csv**: 100 samples (100 positive, 0 negative)
- **he_c.csv**: 100 samples (19 positive, 81 negative)
- **plain_a.csv**: 100 samples (15 positive, 85 negative)
- **plain_b.csv**: 100 samples (100 positive, 0 negative)
- **plain_c.csv**: 100 samples (19 positive, 81 negative)

## Visualizations

### Accuracy Comparison (First 100 Samples)
![Accuracy Comparison](accuracy_comparison.png)

### Latency Comparison (Log Scale)
![Latency Comparison](latency_comparison.png)

### Probability Distributions
![Probability Distributions](probability_distributions.png)

### Confusion Matrices
![Confusion Matrices](confusion_matrices.png)

### Precision-Recall Comparison (First 100 Samples)
![Precision-Recall Comparison](precision_recall_comparison.png)

### Presentation Slides
Two summary slides designed for clear presentation of key results:

**Accuracy Comparison Slide**
- Shows identical accuracy between HE and plaintext for all three datasets
- Includes average accuracy values (81% for both)
![Accuracy Slide](accuracy_slide.png)

**Latency Comparison Slide (Log Scale)**
- Uses logarithmic scale to visualize large difference in latency
- Left: Total HE time (encryption + inference) vs plaintext inference time
- Right: Slowdown factor (HE is ~6,320x slower than plaintext)
- Includes breakdown of HE time components (encryption vs inference)
![Latency Slide](latency_slide.png)

## Files
- `analyze.py` - Main analysis script (computes metrics, generates visualizations)
- `create_slides.py` - Script to generate presentation slides (accuracy and latency)
- `summary_statistics.csv` - Summary table of all metrics
- `*.png` - Generated visualizations and slides

## Methodology Notes
- HE datasets have 100 samples each (full HE experiment set).
- Plaintext datasets have 879 samples each; first 100 samples used for direct comparison.
- HE inference includes homomorphic encryption overhead (encryption + inference).
- Plain inference is on unencrypted data.
- Classification threshold is 0.5 (probability > 0.5 predicts class 1).
- Standard deviation columns show zero because metrics are computed on a single sample set (no bootstrapping).
