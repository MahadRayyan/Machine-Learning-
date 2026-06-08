# 🧹 Missing Value Handling (ML Day 1)

A hands-on data preprocessing notebook that tackles missing value detection and imputation on a messy employee dataset using Python, covering everything from visual null analysis to sklearn-based imputation.

---

## ⚡ Technologies

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn (`SimpleImputer`)
- CSV (raw dataset input)

---

## 🚀 Features

- Loads a raw, messy employee CSV dataset and inspects its shape and structure
- Detects null values column-by-column and calculates missing value percentages
- Visualizes the null value distribution using a **Seaborn heatmap**
- Demonstrates **dropping** columns (`Age`) and rows with null values
- Explores multiple imputation strategies:
  - `fillna()` with a constant value
  - **Backward fill** (`bfill`) and **Forward fill** (`ffill`) along rows and columns
  - **Mode imputation** for categorical (`object`) columns — single and multi-column
  - **Mean imputation** for numerical (`float64`) columns in a loop
- Applies **Scikit-learn's `SimpleImputer`** for ML-ready numeric imputation
- Reconstructs a clean DataFrame from imputed arrays

---

## 🧠 The Process

I started with a messy employee dataset full of missing values scattered across both categorical and numeric columns. The first step was understanding the damage — how many nulls, in which columns, and what percentage of the data they represented.

From there I built up a full toolkit of imputation strategies. For categorical columns like `Performance_Score`, I used mode imputation (most frequent value). For numeric columns, I iterated through `float64` types and filled with the column mean. I also explored fill-forward and fill-backward techniques for time-ordered or sequential data scenarios.

Finally, I plugged in Scikit-learn's `SimpleImputer` to handle numeric imputation the ML-pipeline way — `fit_transform` on the relevant columns, then reconstructed into a clean DataFrame. By the end, `isnull().sum()` returned all zeros.

---

## 🛠 Running the Project

1. Place your CSV file in the project directory and update the path in the notebook's `pd.read_csv()` call

2. Install dependencies:

```
pip install pandas seaborn matplotlib scikit-learn
```

3. Open the notebook:

```
jupyter notebook Machine_learning_Day1.ipynb
```

4. Run all cells top to bottom — each section builds on the previous one

---

## 📁 Dataset

| Column | Type | Notes |
|---|---|---|
| Employee fields | `object` / `float64` | Mixed categorical and numeric |
| `Performance_Score` | `object` | Imputed with mode |
| `Age`, `Salary` | `float64` | Imputed with mean / SimpleImputer |
---
## Preview
<img width="1366" height="513" alt="image" src="https://github.com/user-attachments/assets/a5de84b7-61c5-45ec-b45f-f2d10733f8a6" />

<img width="1366" height="554" alt="image" src="https://github.com/user-attachments/assets/87fa8a37-a3b9-4160-8bb4-5034d8a7a111" />
