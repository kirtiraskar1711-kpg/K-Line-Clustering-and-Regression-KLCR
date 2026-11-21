
# K-Line Clustering and Regression (K-LCR)

## Bivariate Pattern Recognition

This repository provides the research implementation, simulation procedures, datasets, and experimental materials associated with the study of **K-Line Clustering and Regression (K-LCR)** for identifying heterogeneous bivariate linear patterns.

The proposed approach aims to identify groups of observations that follow distinct linear relationships, rather than relying solely on their positional proximity in the predictor-response plane.

## 1. Research Overview

In datasets containing multiple linear relationships, observations may belong to different underlying regression patterns even when their positions overlap. K-LCR is designed to identify such patterns through an iterative procedure combining vertical initialization, ordinary least squares (OLS) regression, and perpendicular-distance-based reassignment.

The present research focuses on **bivariate pattern identification and recognition**. Prediction is outside the scope of the current study and is considered a direction for future research.

## 2. Methodology

The K-LCR procedure consists of the following main stages:

1. **Vertical initialization:** Initialize K vertical lines at equally spaced positions along the predictor-variable range.
2. **Initial assignment:** Assign observations to the nearest initial vertical line using predictor-coordinate distance.
3. **OLS fitting:** Estimate a separate OLS regression line for each non-empty cluster.
4. **Perpendicular-distance reassignment:** Reassign observations according to their minimum perpendicular distance from the fitted regression lines.
5. **Iterative refinement:** Re-estimate the OLS lines and repeat assignment until cluster membership stabilizes.
6. **Optional cluster merging:** Evaluate candidate cluster pairs using the absolute decrease in pooled coefficient of determination (R²).

The merging procedure does not use slope similarity as a merging criterion.

## 3. Experimental Evaluation

The experimental framework considers the following simulated pattern structures:

- Parallel linear patterns
- Crossing linear patterns
- Near-parallel linear patterns
- Change-point patterns
- Three-pattern structures

The study also considers selected real-world bivariate datasets, subject to data-source and redistribution conditions.

The evaluation includes comparisons with relevant line-clustering and conventional clustering approaches, raw versus standardized data, initialization sensitivity, minimum cluster-size sensitivity, and merging-threshold sensitivity.

## 4. Repository Structure

```text
K-Line-Clustering-and-Regression-KLCR/
│
├── README.md
├── CITATION.cff
├── LICENSE
│
├── 01_Algorithm/
├── 02_Simulation_Data/
├── 03_Real_Datasets/
├── 04_Experiments/
├── 05_Results/
│   ├── Tables/
│   └── Figures/
└── 06_Documentation/
```

## 5. Reproducibility

The repository will contain the implementation details, simulation-generation procedures, random seeds, data descriptions, analysis scripts, and outputs required to reproduce the reported experiments.

Software versions, parameter settings, preprocessing procedures, and experimental configurations will be documented alongside the relevant scripts.

## 6. Scope and Future Research

The current study is restricted to bivariate linear pattern recognition. Possible future directions include extending the methodology to multivariate data, improving initialization and cluster selection, investigating robust estimation, and studying the use of identified patterns for prediction.

## 7. Code and Data Availability

The source code, datasets, and reproducibility materials will be made available through this repository as the research package is finalized.

A permanent archived version and DOI will be added following the release of the verified research materials.

## 8. Citation

A formal citation entry will be provided after the associated research article is published or assigned its final bibliographic details.

## 9. Author

**Dr. Kirti Raskar**

Assistant Professor  
MIT Arts, Commerce & Science College  
Alandi (D), Pune, Maharashtra, India

---

**Note:** This repository is being prepared to support transparent and reproducible research on K-Line Clustering and Regression (K-LCR).

