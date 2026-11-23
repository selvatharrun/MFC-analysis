# MFC Membrane Design — Machine Learning Pipeline

> **Stage 1:** predicting the physical properties of proton-exchange membranes from their recipe.

A machine-learning project that helps design better **microbial fuel cell (MFC)** membranes. The biology question is simple: *"given a membrane recipe, can a computer predict how good the membrane will be before we spend a week making it in the lab?"* This repository contains the first stage of that answer, built and explained for people whose background is biology, not computer science.

---

## Contents

- [The biology in one paragraph](#the-biology-in-one-paragraph)
- [Why machine learning for membranes?](#why-machine-learning-for-membranes)
- [The data](#the-data)
- [The three-stage plan](#the-three-stage-plan)
- [Stage 1, step by step](#stage-1-step-by-step)
- [Every ML concept we used, explained plainly](#every-ml-concept-we-used-explained-plainly)
- [What we found (the honest answer)](#what-we-found-the-honest-answer)
- [Running it yourself](#running-it-yourself)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)
- [Quick-reference glossary](#quick-reference-glossary)

---

## The biology in one paragraph

A **microbial fuel cell (MFC)** uses bacteria to turn organic waste (e.g. wastewater) into electricity. The bacteria oxidise their food at one electrode (the **anode**), releasing electrons and protons. The electrons flow through a wire — that is the electricity — while the protons need to travel through a thin plastic film called a **proton-exchange membrane (PEM)** to reach the other electrode (the **cathode**) and combine with oxygen.

The membrane is the bottleneck: it must let protons through easily, but stop the fuel and oxygen from mixing, and it must survive being wet and swollen for months. The standard material, **Nafion**, is excellent but expensive. **SPEEK** (sulfonated poly-ether-ether-ketone) is a promising cheap alternative. Researchers improve SPEEK by mixing in tiny particles called **fillers** (nanoparticles of TiO₂, SiO₂, graphene oxide, silver, and so on). The art is choosing *which* filler and *how much* — which is exactly where machine learning can help.

| Membrane property we predict | What it means biologically | Units |
|---|---|---|
| **Ion Exchange Capacity (IEC)** | How many charged (sulfonic acid) groups the membrane carries. More sites = more protons can hop through. Measured by titration. | meq/g |
| **Water swelling / uptake** | How much water the membrane absorbs. Water is how protons move, but too much swelling tears the membrane apart. | % |
| **Proton conductivity** | How easily protons actually flow through. The single most important performance number. | S/cm |
| **Thickness** | How thin the film is. Thinner = lower resistance, but weaker. | µm |

The ultimate outputs we care about — the MFC's **voltage (mV)** and **power density (mW/m²)** — are handled in later stages of the pipeline.

---

## Why machine learning for membranes?

The relationship between a recipe and the resulting properties is not something we can write down as a clean equation. It depends on the polymer, the filler, the loading, the casting method, the temperature — many interacting effects. Instead of deriving an equation, machine learning **learns the pattern from examples**: feed it lots of "(recipe → measured properties)" pairs from the literature, and it builds a function that maps new recipes to predicted properties.

The honest caveat, which this project confronts head-on, is that *machine learning can only learn patterns that are actually present in the data*. If each chemistry appears only once, the computer cannot magically invent the missing information — and a big part of Stage 1 is measuring exactly how far the data can take us (see [What we found](#what-we-found-the-honest-answer)).

---

## The data

The dataset was manually curated from **30 peer-reviewed papers** (the citations live in `MFC DATA - DATA COLLECTION.csv`) and contains **100 rows**. Each row is one membrane formulation:

- **3 inputs (features):** membrane base type, filler material, filler weight %.
- **4 targets (what we want to predict):** IEC, water swelling, proton conductivity, thickness.
- **2 downstream targets** (used later): voltage and power density.

The raw data is messy — the way real literature always is:

- Units are inconsistent (`cm` vs `µm` vs `mm` for thickness; `S/cm` vs `10⁻² S/cm` vs `mS/cm` for conductivity).
- Values are written as ranges (`1.94–2.15 meq/g`) or with uncertainty (`283 ± 1.3 mW/m²`).
- Unicode subscripts (`TiO₂`) mix with ASCII (`TiO2`).
- A few rows contain copy-paste errors (a filler name spilled into the membrane column).
- Some papers re-tabulate values that also appear in other papers (duplicates).

Stage 1 therefore spends a lot of effort **cleaning and standardising before any modelling** — because a model is only as good as the data you feed it.

---

## The three-stage plan

The full project (see `MFC_Membrane_ML_Pipeline.md`) has three stages:

| Stage | Question | Inputs → Outputs |
|---|---|---|
| **1** (this repo) | "What properties will this recipe have?" | membrane, filler, weight % → IEC, swelling, conductivity, thickness |
| **2** | "How well will the MFC perform?" | stage-1 features + properties → voltage, power density |
| **3** | "What is the *best* recipe?" | inverse search / optimisation over candidate recipes |

Stage 1 exists because the literature often reports intermediate properties but not final MFC performance, and because properties logically come before performance. This repository currently contains **only Stage 1**.

---

## Stage 1, step by step

The notebook `stage1_membrane_properties.ipynb` (and its plain-script twin `stage1_pipeline.py`) runs through these steps:

1. **Load** the three CSV sheets and align them on their shared `S. No` column.
2. **Audit** missingness and the raw categories before touching anything.
3. **Clean text** — fold Unicode (`TiO₂` → `TiO2`), normalise dashes, fix the `S-POSS`-vs-`SPEEK` copy-paste bug.
4. **Parse numbers** — turn ranges into their midpoint, strip units, extract the first number.
5. **Harmonise units** — thickness → µm, conductivity → S/cm, voltage → mV, power density → mW/m². Every conversion is logged in `outputs/data_quality_report.csv`; ambiguous values are *masked* (set to missing) rather than guessed.
6. **Assign paper IDs** — forward-fill citations so we know which paper each row came from. This matters enormously for honest evaluation (see [data leakage](#data-leakage-and-grouped-cross-validation)).
7. **Flag duplicates** — 6 rows that re-tabulate values seen in earlier papers are excluded from training.
8. **Explore** — distributions, a missing-value map, correlations, and a look at the dose-response (how a property changes with filler loading).
9. **Build features** — one-hot encode the two categorical inputs, impute and scale the weight %.
10. **Train 7 models** per target and evaluate them with **paper-grouped cross-validation**.
11. **Quantify leakage** — deliberately re-run with a naive random split to show how misleading that would be.
12. **Pick a winner per target**, refit on all data, and save it as a `.joblib` file.
13. **Diagnose transfer** — ask whether the model reproduces the *within-paper* dose-response.
14. **Anchored mode** — evaluate a smarter strategy that uses a measured neat-membrane baseline.
15. **Hand off to Stage 2** — predict all four properties for every row, preferring measured values, and write `outputs/stage1_predictions.csv`.

---

## Every ML concept we used, explained plainly

This section is the "translation layer" — it explains the computer-science jargon in terms a biologist can follow.

### Features and targets (inputs and outputs)

- **Features** (a.k.a. inputs, predictors, independent variables, *X*): the things you control or know upfront. Here: *membrane, filler, weight %*.
- **Target** (a.k.a. label, response, dependent variable, *y*): the thing you want to predict. Here: *IEC, swelling, conductivity, thickness*.

Think of it as the "recipe" (features) and the "result you measure afterwards" (target).

### Regression vs classification

- **Regression** predicts a **continuous number** (e.g. "conductivity = 0.015 S/cm"). That is what we do — all four targets are numbers.
- **Classification** predicts a **category** (e.g. "this membrane is either stable or not"). We don't use it here.

### What a "model" is

A model is just a **function**: features in, a number out. The function's shape is flexible (it has "parameters"), and **training** means automatically adjusting those parameters so the function's output matches the measured targets as closely as possible. Different model types have different shapes — some are rigid lines, some are flexible trees (see [the model families](#the-model-families)).

### Training vs testing, and why you must separate them

If you show a model the answers and then ask it to predict those same answers, it will look brilliant — the same way a student looks brilliant if you hand them the exam *with* the answer key. To get an honest score you must hold out data the model has never seen:

- **Training set** — the examples the model learns from.
- **Test set** — the examples you judge it on.

### Cross-validation (k-fold)

With only ~100 rows we cannot afford to throw away half the data as a one-off test set. **Cross-validation** recycles the data cleverly:

1. Split the data into *k* parts ("folds").
2. Train on *k−1* folds, test on the held-out fold.
3. Repeat so that every fold is tested exactly once.
4. Average the scores.

Every row gets predicted once, by a model that never saw it during training — the "out-of-fold" prediction. We use `GroupKFold(5)` (5 folds).

### Data leakage (and grouped cross-validation)

The most important statistical idea in this project.

If rows from the **same paper** appear in both the training and test folds, the model can "cheat": it memorises the paper's baseline (e.g. "this lab's SPEEK always conducts ~0.15 S/cm") and regurgitates it, rather than learning the general chemistry. It looks like the model works, but on a genuinely *new* paper it fails.

**Grouped cross-validation** fixes this by keeping all rows from one paper in a single fold — the test fold always contains *entirely unseen papers*, exactly like the real-world task ("predict the properties of a membrane from a paper we've never read").

We prove the point explicitly: the same models score R² ≈ 0.3–0.5 under a naive random split (leaky), but ≈ 0 under the paper-grouped split (honest). The difference *is* the leakage.

### Evaluation metrics

- **R² (coefficient of determination):** "what fraction of the variation in the target does the model explain?" R² = 1 is perfect; R² = 0 means it is no better than always guessing the average; **negative R² means it is worse than guessing the average.** This is our headline metric.
- **MAE (mean absolute error):** the average size of the error, in the target's real units. Easy to interpret: "on average we're off by 0.28 meq/g."
- **RMSE (root-mean-squared error):** like MAE, but it squares errors first, so big mistakes count disproportionately more. Useful when large errors are especially bad.

### The model families

We train seven models so the results don't depend on one lucky choice:

| Model | One-line intuition |
|---|---|
| **Baseline (mean)** | Always predict the average. Deliberately dumb — the floor every real model must beat. |
| **Ridge (linear)** | Draw the "best straight line" through the data, with a small penalty to keep it from being too wiggly. Simple and often surprisingly good on small data. |
| **Random Forest** | Grow hundreds of *decision trees* (each a chain of "if weight > 5% then…" rules) on random subsets of the data, and average their answers. Robust. |
| **Extra Trees** | Like Random Forest, but trees are grown even more randomly. |
| **Histogram Gradient Boosting** | Build trees one at a time, each new tree fixing the mistakes of the last. |
| **XGBoost** | A faster, popular flavour of gradient boosting. |
| **LightGBM** | Another fast gradient-boosting flavour, good on small datasets. |

The key idea behind all the tree methods: instead of one opinionated rule, combine thousands of simple rules and let them vote. They naturally capture "interactions" (e.g. *TiO₂ helps, but only at low loadings*).

### One-hot encoding (turning words into numbers)

Models do arithmetic, so categorical labels like "TiO₂" or "SPEEK" must become numbers. The trick we use, **one-hot encoding**, turns each category into its own yes/no column:

```
filler = "TiO₂"  →  filler_Ag = 0, filler_TiO2 = 1, filler_GO = 0, ...
```

Crucially, we set `handle_unknown="ignore"` so a future filler the model has never seen won't crash it.

### Missing values and imputation

Real data has gaps (a paper didn't measure conductivity). Options: drop the row (wasteful), or **impute** (fill in a sensible guess). We median-impute the weight % (and keep a flag recording *that* it was missing). For the *targets*, though, we **never impute before training** — each target model trains only on rows that actually have a measured value for it, so we don't train a model to copy a guess.

### Standardisation (scaling)

Features with large numeric ranges can dominate a linear model. **Standard scaling** rescales each numeric feature to mean 0, standard deviation 1, so no single feature bullies the others. (Tree models don't need this, but Ridge does.)

### Log transforms

Conductivity and thickness span *orders of magnitude* (0.0001 to 10 S/cm). A model that adds up errors linearly will be dominated by the big values. Taking **log₁₀** before modelling compresses the range and makes relative errors behave sensibly; we undo it (10^…) at the end. This is why the notebook talks about "log target" and "linear target."

### Outliers (IQR rule)

A value far outside the pack (like a swelling of 146.69 %) can distort a model. We *flag* points beyond the **interquartile range** (below Q1 − 1.5×IQR or above Q3 + 1.5×IQR) and report them in the quality report — but we **never silently delete** them, because an outlier can be a real, important result.

### Permutation importance ("what drives the predictions?")

To ask *why* a model predicts what it does, we scramble one feature at a time and watch how much the model's accuracy drops. A big drop means the model relies on that feature. Result: **filler type and weight % dominate; the membrane base is almost never used** — because almost every row is some flavour of SPEEK, so the model has nothing to learn from it.

### The "anchored mode" (a physicist's trick)

Here's the clever part. Two papers can report the same SPEEK + TiO₂ at 2.5% and get values 3× apart — because their base polymer, casting method and test rig differ. That *offset* swamps the signal. But if you know the **neat membrane (0% filler)** value from the *same* series, the offset cancels and you only have to learn the small *increment* the filler adds. We test this: predict the **delta** from the known baseline, then add it back. See the results below.

---

## What we found (the honest answer)

The single most important conclusion of Stage 1 is a limitation, stated plainly:

> **With only three inputs, a model cannot predict the absolute property value of a brand-new, unseen paper.**

Measured by honest paper-grouped cross-validation:

| Target | Best model | R² (honest) | MAE | Verdict |
|---|---|---|---|---|
| IEC | LightGBM | **+0.12** | 0.28 meq/g | Only target that beats "just guess the average" |
| Water swelling | LightGBM | ≈ 0.00 | 12.5 % | Indistinguishable from guessing the mean |
| Proton conductivity | HistGradientBoosting | −0.25 | 0.0075 S/cm | Worse than guessing the mean |
| Thickness | HistGradientBoosting | −0.12 | 32 µm | Worse than guessing the mean |

That looks bad — and **it is the honest truth**, not a bug. The data *does* contain a real signal: within a single paper, increasing filler loading reliably raises IEC (within-paper correlation r ≈ +0.7) and conductivity (r ≈ +0.5). But that dose-response is a *slope*, and the model cannot transfer the slope across papers because each paper has its own unknown baseline.

The **anchored mode** is the practical fix:

| Target | Anchored R² | Anchored MAE |
|---|---|---|
| Water swelling | **+0.84** | 2.3 % |
| Proton conductivity | **+0.90** | 0.0009 S/cm |
| IEC | +0.06 | 0.41 meq/g |
| Thickness | ≈ 0 | 0.58 µm |

**Interpretation:** most of the variance is the series baseline. If you give the model a measured neat-membrane value from the same series, it can reconstruct swelling and conductivity well. But the *delta itself* is still poorly predicted (delta R² < 0) — so this is bookkeeping around a known baseline, **not** a physically learned dose-response.

**Bottom line for Stage 3:** screen fillers only *within* a membrane series whose neat value you have measured. Do not use Stage 1 to compare unrelated chemistries against each other.

---

## Running it yourself

```bash
# 1. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. Install the pinned dependencies
python -m pip install -r requirements.txt

# 3. Register the kernel so Jupyter can use it (done once)
python -m ipykernel install --user --name=mfc-stage1 --display-name="Python (mfc-stage1)"

# 4. Open the notebook
jupyter notebook stage1_membrane_properties.ipynb
```

The notebook saves its results to `outputs/` and the trained models to `models/`.

> **Note on dependencies:** `requirements.txt` pins exact versions that were tested on Python 3.14. CatBoost is intentionally omitted — see the comments in `requirements.txt`.

---

## Repository layout

```
models/
├── MFC DATA - DATA COLLECTION.csv   # raw data + citations (source of truth for paper IDs)
├── MFC DATA - DATA STRUCTURE.csv    # raw data with unit audit
├── MFC DATA - STAGE1.csv            # numeric sheet used for modelling
├── MFC_Membrane_ML_Pipeline.md      # original 3-stage design document
├── stage1_pipeline.py               # Stage-1 analysis as a plain Python script
├── stage1_membrane_properties.ipynb # Stage-1 analysis as a notebook (the main entry point)
├── requirements.txt                 # pinned dependencies
├── requirements.lock                # lock file
├── models/                          # trained per-target models (.joblib) + selection summary
├── outputs/                         # cleaned data, predictions, metrics, quality report
│   └── figs/                        # generated plots
└── venv/                            # virtual environment (git-ignored)
```

Key generated files:
- `outputs/stage1_clean.csv` — cleaned, unit-harmonised dataset.
- `outputs/stage1_predictions.csv` — the Stage-1 → Stage-2 handoff table (measured vs predicted).
- `outputs/model_metrics.csv` — full per-model, per-target scores.
- `outputs/data_quality_report.csv` — every cleaning decision, row by row.
- `outputs/anchored_mode_metrics.csv` — the anchored-baseline experiment.

---

## Known limitations

- **Small data:** 49–66 usable labels per target across 30 papers. Fold-to-fold spread is large; treat differences below ~0.1 R² as noise.
- **Only three inputs.** The absolute level of a new paper is not identifiable from membrane + filler + weight % alone.
- **The dose-response does not generalise** (within-paper r ≈ 0.7, but out-of-fold slope R² ≈ 0).
- **Anchored mode needs a measured baseline** — it only helps inside a series whose 0% value is known.
- A few rows required unit assumptions (thickness assumed mm for S.No 52–53; three conductivity rows masked as ambiguous) — see the quality report.
- The highest-swelling row (146.69 %, S.No 39) is a single high-leverage point.

**To raise the ceiling,** the model needs descriptors that transfer between papers: degree of sulfonation, base-membrane IEC, filler particle size / surface area, and operating temperature.

---

## Quick-reference glossary

| Term | Plain-English meaning |
|---|---|
| **Feature / input / X** | What you control (recipe). |
| **Target / label / y** | What you predict (property). |
| **Regression** | Predicting a number. |
| **Model** | A function learned from examples. |
| **Training / test** | Learn on one set, judge on a held-out set. |
| **Cross-validation** | Recycle data so every row is tested once by a model that never saw it. |
| **GroupKFold** | Group rows by paper so no paper appears in both train and test. |
| **Data leakage** | Accidentally letting the model see test answers — inflates scores. |
| **R²** | Fraction of variance explained; 0 = guessing the mean, negative = worse. |
| **MAE / RMSE** | Average error in real units (RMSE penalises big errors more). |
| **One-hot encoding** | Turn a category into yes/no columns. |
| **Imputation** | Filling in missing values. |
| **Standard scaling** | Rescale features to mean 0, s.d. 1. |
| **Log transform** | Model in log space when values span orders of magnitude. |
| **Outlier (IQR)** | A value far outside the central spread; flagged, not deleted. |
| **Permutation importance** | Shuffle a feature, see how much accuracy drops. |
| **Anchored mode** | Predict the *delta* from a known baseline rather than the absolute value. |
