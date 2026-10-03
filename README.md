# Crop Type Classification Representation Learning

This MATLAB coursework project compares four feature representations, raw standardized features, PCA, RBF kernel PCA, and Laplacian Eigenmaps, for 7-class crop classification using kNN, measuring accuracy, macro F1, and wall-clock cost.

## Key Findings

- PCA (50 components, 98.58% variance) matched raw-feature test accuracy (99.46%) with 3.5x fewer dimensions and the highest macro F1 (0.9941).
- Kernel PCA cost 222x more representation time than PCA (50.21s vs 0.23s) for 0.04 pp lower accuracy.
- Laplacian Eigenmaps cost 92x more than PCA for 1.00 pp lower accuracy, with the largest drop on the minority Broadleaf class (F1 0.972 vs 0.983).
- k=1 was optimal for every representation, indicating locally pure class neighborhoods.
- Oats-Wheat confusion was the dominant error mode across all methods, reflecting spectral similarity rather than a representation failure.

## Dataset

This project uses the UCI Winnipeg Crop Mapping dataset, a fused optical + SAR dataset with 325,834 samples x 174 features x 7 classes. The experiment uses a stratified subsample capped at 10,000 per class, yielding 54,741 samples, then applies a stratified 60/20/20 split (32,845 / 10,949 / 10,947). Z-score standardization is fit on train only.

Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) and place `WinnipegDataset.txt` in the repository root.

## Results

| Representation | Best k | Val Acc (%) | Test Acc (%) | Test Macro F1 | Repr. Time (s) |
|---|---:|---:|---:|---:|---:|
| Raw (standardized) | 1 | 99.28 | 99.46 | 0.9935 | 0.00 |
| PCA (50 PCs) | 1 | 99.27 | 99.46 | 0.9941 | 0.23 |
| Kernel PCA (RBF) | 1 | 99.20 | 99.42 | 0.9932 | 50.21 |
| Laplacian Eigenmaps | 1 | 98.71 | 98.46 | 0.9835 | 20.75 |

## Methods

PCA uses SVD of the mean-centered training matrix and is validated against drtoolbox (|r| = 1.0000 on the first 5 PCs).

Kernel PCA uses an RBF kernel with sigma^2 = 294.59 from the median heuristic. It is fit on a 5,000-point stratified subsample because the full kernel matrix would need about 8.6 GB, and uses Nystrom out-of-sample projection.

Laplacian Eigenmaps uses a k=15 neighbor graph with Gaussian weights and solves the generalized eigenproblem Lv = lambda Dv. sigma^2 = 147.30 is selected by downstream kNN accuracy, and out-of-sample projection uses Gaussian-weighted neighbor interpolation (Bengio et al., 2004).

kNN tunes k per representation over {1, 3, 5, 10, 15, 25} on the validation set, using blocked brute-force Euclidean search.

## Implementation Notes

- Implemented from scratch: Laplacian Eigenmaps, Nystrom out-of-sample projection, Gaussian-weighted interpolation for LE, blocked kNN search and classifier, macro F1, confusion matrices.
- Uses drtoolbox (Laurens van der Maaten): `pca()` for PCA scores and `kernel_pca()` for the kPCA fit embedding.
- Do NOT describe the project as "fully from scratch."

## Figures

![2D embeddings for all representations](figures/fig2_2d_embeddings.png)

2D training-set projections colored by crop class.

![k sensitivity curves](figures/fig3_k_sensitivity.png)

Validation accuracy across k values for each representation.

![Accuracy and macro F1](figures/fig4_accuracy_f1.png)

Test accuracy and macro F1 for the best k per representation.

![Normalized confusion matrices](figures/fig5_all_confusion_matrices.png)

Normalized test-set confusion matrices for all four representations.

![Representation timing](figures/fig7_timing.png)

Wall-clock representation computation time on a log scale.

## How to Run

Use MATLAB R2016b+; this project was developed on R2025b. Download [drtoolbox](https://lvdmaaten.github.io/drtoolbox/) and place the `drtoolbox/` folder in the repository root. Place `WinnipegDataset.txt` in the repository root. Then run:

```matlab
crop_representation_comparison
