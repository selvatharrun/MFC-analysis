# 3-Stage Machine Learning Pipeline for MFC Membrane Design

## Overview

The 3-stage pipeline directly aligns with the dataset structure. Academic literature often reports **intermediate membrane properties** (IEC, Water Uptake, Proton Conductivity) while **missing final MFC outputs** (Voltage, Power Density). Breaking the system into stages handles missing targets and models the physical progression from material design to performance.

---

## Machine Learning Architecture

```
 [ Input Features ]  --->  Stage 1 Model  --->  [ Predicted Intermediate Features ]
 (Membrane, Filler,                                (IEC, Water Uptake, Thickness,
  Weight %)                                         Proton Conductivity)
                                                             |
                                                             v
 [ Target Outputs ]  <---  Stage 2 Model  <------------------+
 (Voltage, Power Density)

                                  ^
                                  |
                           Stage 3 Engine
                     (Optimization & Inversion)
```

---

## Stage 1: Membrane Property Prediction

Predict intermediate physical/chemical properties from the base formulation parameters.

| Element | Definition |
|---|---|
| **Inputs (X1)** | Membrane Base Type (Categorical), Filler Material (Categorical), Weight % (0 – 10%) |
| **Targets (Y1)** | Ion Exchange Capacity (IEC), Water Swelling/Uptake (%), Thickness (µm), Proton Conductivity (S/cm) |

### Recommended Algorithms

- **Multi-output XGBoost / LightGBM** — after One-Hot Encoding or Target Encoding of categorical features.
- **Random Forest Regressor** — naturally handles tabular feature interactions and missing values.
- **CatBoost Regressor** — ideal if filler types are diverse text strings.

### Key Data Handling Strategy

Since papers report properties in varying units (e.g. `10^-2 S/cm` vs `S/cm`), **standardise unit magnitudes prior to training**.

---

## Stage 2: MFC Performance Prediction

Predict final electrochemical performance metrics based on input parameters plus the predicted or measured intermediate properties.

| Element | Definition |
|---|---|
| **Inputs (X2)** | Input Features (X1) + Intermediate Features (measured from literature, or output from Stage 1 if unmeasured) |
| **Targets (Y2)** | Voltage (mV), Power Density (mW/m²) |

### Recommended Algorithms

- **XGBoost Regressor / Extra Trees Regressor**
- **Artificial Neural Network (ANN / MLP)** — effective at learning non-linear trade-offs (e.g. higher water uptake increasing conductivity up to a threshold before mechanical degradation occurs).

### Missing Value Management

- Use **Stage 1 predictions** to impute missing intermediate values for rows where papers skipped IEC/Conductivity tests, before feeding them into Stage 2.
- Use **`IterativeImputer` (Bayesian Ridge)** for features missing at random.

---

## Stage 3: Optimal Membrane Identification (Inverse Design)

Identify the best membrane composition and filler weight percentage to **maximize Power Density and Voltage** while maintaining target structural stability.

### Approach Options

| Approach | Description |
|---|---|
| **Grid Search / Candidate Screening** (simplest) | Generate a synthetic grid of input combinations (e.g. SPEEK + 0.5% increments of fillers), run them sequentially through Stage 1 → Stage 2, and rank candidates by predicted Power Density. |
| **Multi-Objective Optimization** (NSGA-II / Genetic Algorithm) | Optimize for maximum Power Density while constraining Thickness and Water Swelling within operational limits. |
| **Bayesian Optimization / Acquisition Functions** | Efficiently search continuous weight percentages to find global performance peaks without exhaustive search. |

---

## Practical Implementation Steps

### 1. Feature Preprocessing

- **Clean filler names** — standardise variants such as `"TiO2"`, `"TiO2-SO3H"`, `"Ag"`, `"S-POSS"`.
- **Convert string ranges** — e.g. `"15–60.5 mW/m²"` / `"15–60"` → clean floats.

### 2. Model Evaluation

- **Split data using `GroupKFold`, grouped by Citation / Literature paper.** This prevents data leakage where different weight percentages from the exact same paper leak into test sets.
- **Evaluate regression** using **R² Score, MAE, and RMSE**.

---

## Summary of Pipeline Stages

| Stage | Purpose | Inputs | Targets |
|---|---|---|---|
| **1** | Membrane property prediction | Membrane, Filler, Weight % | IEC, Water Uptake, Thickness, Conductivity |
| **2** | MFC performance prediction | X1 + intermediate features | Voltage, Power Density |
| **3** | Inverse design / optimization | Candidate formulations | Optimal composition & weight % |
