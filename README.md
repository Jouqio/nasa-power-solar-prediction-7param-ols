# Solar PV Power Estimation using OLS and NASA POWER Data

An interpretable and reproducible photovoltaic (PV) power estimation framework using **Ordinary Least Squares (OLS)** and **NASA POWER** satellite meteorological data for Bontang, East Kalimantan, Indonesia.

This project evaluates a 7-parameter OLS model, performs comprehensive statistical diagnostics, and benchmarks classical regression against Machine Learning approaches.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)
![NASA POWER](https://img.shields.io/badge/Data-NASA%20POWER-E03C31?style=flat&logo=nasa&logoColor=white)
![Research](https://img.shields.io/badge/Status-Research-orange?style=flat)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat&logo=creativecommons&logoColor=white)

---

## Overview

This repository contains an interpretable statistical framework for estimating monthly photovoltaic (PV) power output using NASA POWER satellite data in tropical equatorial environments.

The study focuses on:

- OLS implementation using Normal Equations
- Statistical interpretability of PV estimation
- Comprehensive econometric diagnostics
- Benchmark comparison against Machine Learning models
- Reproducible research workflow

The research was conducted using meteorological observations from **Bontang, East Kalimantan, Indonesia**, covering the period **2015–2025**.

---

## Key Features

- Ordinary Least Squares (OLS) implemented from-scratch with NumPy
- 7 meteorological parameters from NASA POWER
- Statistical diagnostic pipeline:
  - AIC / BIC
  - Variance Inflation Factor (VIF)
  - Durbin–Watson
  - Breusch–Godfrey
  - Shapiro–Wilk
  - Box–Cox Transformation
- Benchmark comparison with:
  - Random Forest Regressor
  - Support Vector Regression (SVR)
- Chronological train-test split for time-series consistency
- Fully reproducible analysis workflow

---

## Dataset Information

### Source
NASA POWER (Prediction Of Worldwide Energy Resources)

### Study Area
Bontang, East Kalimantan, Indonesia

### Coordinates
`0.1333°N, 117.50°E`

### Period
January 2015 – December 2025

### Temporal Resolution
Monthly

---

## Meteorological Parameters

| Variable | Description |
|---|---|
| GHI | Global Horizontal Irradiance |
| T2M | Air Temperature at 2 m |
| WS10M | Wind Speed at 10 m |
| PS | Surface Pressure |
| CLOUD_AMT | Cloud Cover |
| IMERG_PRECTOT | Precipitation |
| DNI | Direct Normal Irradiance |

---

## Model Performance

| Dataset | R² | MAPE |
|---|---|---|
| Training | 0.99999770 | 0.0087% |
| Testing | 0.99999622 | 0.0137% |

> The near-perfect R² values originate from the deterministic algebraic structure of the PV target variable rather than purely empirical prediction capability.

---

## Machine Learning Benchmark

| Model | Test R² | MAPE |
|---|---|---|
| OLS | 0.99999622 | 0.0137% |
| Random Forest | 0.98622 | 0.7769% |
| SVR (RBF) | 0.92256 | 1.7127% |

The benchmarking experiment demonstrates that classical interpretable regression can outperform more complex non-parametric models on deterministic datasets.

---

## Statistical Diagnostics

The repository includes a comprehensive 8-step diagnostic protocol:

1. Coefficient significance test
2. AIC/BIC model comparison
3. Multicollinearity analysis (VIF)
4. Durbin–Watson statistic
5. Breusch–Godfrey autocorrelation test
6. Lag residual analysis
7. Shapiro–Wilk normality test
8. Box–Cox transformation analysis

---

## Repository Structure

```text
project-root/
├── data/
│   └── POWER_Point_Monthly_*.csv
│
├── docs/
│   └── Artikel_Utama_Estimasi_PV_Ekuatorial.docx
│
├── outputs/
│   ├── Gambar1_Panel_Utama_4in1.png
│   ├── Gambar2_Autokorelasi_Diagnostik.png
│   ├── Gambar3_AIC_BIC_Comparison.png
│   ├── Gambar4_BoxCox_Normalitas.png
│   ├── Gambar5_Residual_Diagnostic.png
│   ├── Gambar6_Heatmap_Korelasi.png
│   ├── Gambar7_Pipeline_Flowchart.png
│   ├── Gambar8_ML_Comparison.png
│   └── hasil_analisis_lengkap.txt
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── CITATION.cff
```
---

## Cara Pakai

```bash
pip install -r requirements.txt
python main.py
```

Data: [NASA POWER](https://power.larc.nasa.gov/data-access-viewer/) — koordinat `0,1333°N; 117,50°E`, resolusi Monthly, periode 2015–2025.

---
Research Contributions

This work contributes to:

Transparent statistical modeling for PV estimation
Interpretability-focused renewable energy analytics
Diagnostic evaluation of deterministic regression systems
Comparative analysis between OLS and Machine Learning methods
Reproducible open-source scientific workflows
---

Author
---
Syauqi Nuzul Abdi
Department of Informatics Engineering
Sekolah Tinggi Teknologi Bontang
Indonesia

## Citation

If you use this repository in academic work, please cite:

```
nuzulabdisyauqi@gmail.com,
  author = {Syauqi Nuzul Abdi},
  title = {Solar PV Power Estimation using OLS and NASA POWER Data},
  year = {2026},
  url = {https://github.com/syauqinzul/nasa-power-solar-prediction-7param-ols}
```

[CC BY 4.0](LICENSE) — This project is licensed under.
