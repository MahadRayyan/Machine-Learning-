# 🔢 Employee Dataset — Encoding & Outlier Handling (ML Day 2)

A hands-on data preprocessing notebook covering categorical encoding techniques and outlier detection/removal on the same messy employee dataset, preparing it for ML-ready numerical form.

---

## ⚡ Technologies

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn (`OneHotEncoder`, `LabelEncoder`, `OrdinalEncoder`)
- CSV (raw dataset input)

---

## 🚀 Features

- Loads and inspects the messy employee CSV dataset
- Applies **One-Hot Encoding** (`pd.get_dummies`) on boolean columns like `Remote_Work` — generating binary sub-columns from categorical values
- Uses **Scikit-learn's `OneHotEncoder`** with `drop="first"` to eliminate redundant columns and avoid the dummy variable trap
- Applies **Label Encoding** (`LabelEncoder`) on nominal categorical columns like `Performance_Score` — assigning integer labels alphabetically
- Applies **Ordinal Encoding** (`OrdinalEncoder`) on ordered categorical columns like `Status` — preserving meaningful rank order (Active → Pending → Inactive)
- Detects outliers visually using **Seaborn boxplots** for `Age` and `Salary`
- Removes outliers using the **IQR method** — computing Q1, Q3, and IQR to set upper/lower bounds and filter extreme values
- Removes outliers using the **Z-score method** — both the direct mean ± 3×std approach and the computed Z-score column approach
- Confirms cleaned dataset shape after each filtering step

---

## 🧠 The Process

Day 2 picks up right where Day 1 left off — the dataset is now imputed, but still full of text categories that ML models can't work with. The goal: convert everything to numbers, the right way.

For `Remote_Work` (True/False), One-Hot Encoding made sense — no implied order. I used both `pd.get_dummies` and sklearn's `OneHotEncoder` with `drop="first"` to understand the difference between a quick pandas approach and the ML-pipeline version.

For `Performance_Score` (Average, Excellent, Good, Poor), I used `LabelEncoder` — assigning each category an integer. It's fast, but it's purely nominal here, so no ordering is guaranteed.

For `Status` (Active, Pending, Inactive), `OrdinalEncoder` was the right call — I defined the category order explicitly so the encoding reflects the actual hierarchy, not just alphabetical order.

Then came outlier detection. I plotted boxplots for `Age` and `Salary`, computed IQR-based bounds, and filtered rows falling outside them. I also computed Z-scores manually and through formula to cross-validate — anything beyond ±3 standard deviations was flagged as an outlier. Final dataset shape confirmed no rows were lost (the data happened to have no true outliers by either method).

---

## 🛠 Running the Project

1. Place your CSV file in the project directory and update the path in the notebook's `pd.read_csv()` call

2. Install dependencies:

```
pip install pandas seaborn matplotlib scikit-learn
```

3. Open the notebook:

```
jupyter notebook Machine_Learning_Day2.ipynb
```

4. Run all cells top to bottom — encoding sections precede the outlier detection sections

---

## 📁 Dataset

| Column | Encoding Applied | Method |
|---|---|---|
| `Remote_Work` | One-Hot Encoding | `pd.get_dummies` / `OneHotEncoder(drop="first")` |
| `Performance_Score` | Label Encoding | `LabelEncoder` |
| `Status` | Ordinal Encoding | `OrdinalEncoder(categories=[...])` |
| `Age` | Outlier removal | IQR method |
| `Salary` | Outlier removal | IQR method + Z-score |
