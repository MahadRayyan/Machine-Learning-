# 🏠 House Price Prediction — Regression Metrics, Lasso & Ridge Regularization (ML Day 6)

A hands-on notebook covering regression cost functions and regularization techniques — comparing plain Linear Regression against Lasso (L1) and Ridge (L2) regularization on a house price dataset.

---

## ⚡ Technologies

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn (`LinearRegression`, `Lasso`, `Ridge`, `StandardScaler`, `train_test_split`)
- CSV (house price dataset — 200 rows)

---

## 🚀 Features

- Loads a house price dataset with features including `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`, and `yr_built`
- Visualizes feature correlations with a **heatmap** to understand relationships before modeling
- Applies **StandardScaler** to normalize all input features before training — critical for regularization to work fairly across features
- Splits data into **80/20 train-test** sets and trains three separate models side-by-side
- Builds a baseline **Linear Regression** model — R² ≈ 0.943
- Builds a **Lasso (L1) Regression** model (`alpha=10`) — applies a penalty that can shrink some coefficients all the way to zero, effectively performing **feature selection** — R² ≈ 0.943
- Builds a **Ridge (L2) Regression** model — applies a penalty that shrinks all coefficients toward zero without eliminating them, reducing model complexity and preventing overfitting — R² ≈ 0.942
- Visualizes coefficient magnitudes for all three models with **bar charts** to compare how each regularization method distributes feature importance

---

## 🧠 The Process

Day 6 started in the physical notebook — working out the theory of regression cost functions (MSE, MAE, RMSE) before touching any code.

**Cost function vs Loss function:** the cost function measures the total error across the entire dataset; the loss function measures error on a single row. In regression, MSE is the standard choice as the loss function because it's differentiable — which is what lets gradient descent find the minimum.

**MSE** (Mean Squared Error) squares the difference between predicted and actual values, which penalizes large errors heavily. It's also known as L2 loss, and it's not robust to outliers.

**MAE** (Mean Absolute Error) takes the mean of absolute differences, making it more robust to outliers — but it's not differentiable at zero, so gradient-based optimization is trickier.

**RMSE** (Root Mean Square Error) is just the square root of MSE — interpretable in the same units as the target variable and differentiable.

In the notebook, I then applied this theory by training three models on a house price dataset. After standardizing all features, I compared Linear Regression (no regularization) against Lasso (L1 — can zero out features entirely) and Ridge (L2 — shrinks all features proportionally). All three landed near a 0.942–0.943 R², but the coefficient bar charts told the real story: Lasso visibly zeroed out some less-important features, while Ridge distributed the shrinkage more evenly across all of them.

---

## 🛠 Running the Project

1. Place your `houseprice_200.csv` file in the project directory and update the path in the notebook's `pd.read_csv()` call

2. Install dependencies:

```
pip install pandas seaborn matplotlib scikit-learn
```

3. Open the notebook:

```
jupyter notebook Machine_Learning_Day6.ipynb
```

4. Run all cells top to bottom — the three models are trained independently and their coefficient charts can be compared side-by-side

---

## 📁 Model Comparison

| Model | Regularization | Key Behavior | R² Score |
|---|---|---|---|
| Linear Regression | None | No penalty on coefficients | ~0.943 |
| Lasso (`alpha=10`) | L1 — shrinks some to zero | Feature selection via sparsity | ~0.943 |
| Ridge (default `alpha=1`) | L2 — shrinks all toward zero | Reduces complexity, keeps all features | ~0.942 |

## 📓 Theory Notes

| Metric | Formula | Differentiable | Outlier Robust |
|---|---|---|---|
| MSE | mean of (y − ŷ)² | ✅ Yes | ❌ No |
| MAE | mean of \|y − ŷ\| | ❌ No | ✅ Yes |
| RMSE | √(mean of (y − ŷ)²) | ✅ Yes | ❌ No |

