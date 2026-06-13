# ⚖️ Employee Dataset — Feature Scaling, Duplicates & Data Types (ML Day 3)

A hands-on data preprocessing notebook covering feature scaling techniques, duplicate detection and removal, and data type conversion — the final steps to get a dataset fully ML-ready.

---

## ⚡ Technologies

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn (`StandardScaler`, `MinMaxScaler`)
- CSV (raw dataset input)

---

## 🚀 Features

- Loads the employee CSV dataset and checks for remaining null values
- Imputes missing `Salary` values with the column mean before scaling
- Applies **Standardization** (`StandardScaler`) on `Salary` — transforms values so the mean becomes 0 and standard deviation becomes 1 using the formula `Xnew = (Xi - Xmean) / std`
- Verifies standardization output via `df.describe()` — confirms near-zero mean and unit variance on the scaled column
- Applies **Min-Max Normalization** (`MinMaxScaler`) on `Salary` — rescales values to the [0, 1] range using the formula `Xnew = (Xi - Xmin) / (Xmax - Xmin)`
- Verifies normalization output via `df.describe()` — confirms min of 0.0 and max of 1.0 on the scaled column
- Demonstrates **duplicate detection** using `df.duplicated()` and removal using `df.drop_duplicates()`
- Covers **value replacement** with `df.replace()` and **data type conversion** with `.astype()`

---

## 🧠 The Process

By Day 3, the dataset had already been imputed and encoded — but the numeric columns still had wildly different magnitudes. A `Salary` in the tens of thousands would completely dominate an `Age` in the thirties in any distance-based model. That's a problem.

I tackled it two ways. **Standardization** (`StandardScaler`) centers the data around zero with unit variance — it's the go-to when the data roughly follows a normal distribution or you're using algorithms like SVM or linear regression. **Normalization** (`MinMaxScaler`) squashes everything between 0 and 1 — better when you need a fixed range, like for neural networks or KNN.

I applied both to `Salary` side-by-side, then verified the results with `df.describe()` to confirm the math checked out. After that, I wrapped up with the final two cleaning steps: checking for and dropping any exact duplicate rows, and covering how to replace specific values or cast columns to a different data type with `astype()`.

---

## 🛠 Running the Project

1. Place your CSV file in the project directory and update the path in the notebook's `pd.read_csv()` call

2. Install dependencies:

```
pip install pandas seaborn matplotlib scikit-learn
```

3. Open the notebook:

```
jupyter notebook Machine_Learning_Day3.ipynb
```

4. Run all cells top to bottom — scaling sections come before duplicates and type conversion

---

## 📁 Scaling Comparison

| Technique | Scaler | Output Range | Mean After | Use When |
|---|---|---|---|---|
| Standardization | `StandardScaler` | Unbounded (~-3 to 3) | ~0 | Normal distribution, SVM, Linear Regression |
| Normalization | `MinMaxScaler` | [0, 1] | ~0.5 | Fixed range needed, KNN, Neural Networks |
