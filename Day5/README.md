# 📊 Salary Prediction — Multiple Linear & Polynomial Regression (ML Day 5)

A hands-on notebook covering two regression techniques: Multiple Linear Regression for predicting salary from age and experience, and Polynomial Regression for fitting non-linear, quadratic salary growth data.

---

## ⚡ Technologies

- Python
- Pandas
- Seaborn
- Matplotlib
- NumPy
- Scikit-learn (`LinearRegression`, `PolynomialFeatures`, `train_test_split`)
- CSV (regression dataset, self-generated polynomial dataset)

---

## 🚀 Features

- Loads a 10,000-row salary dataset (`age`, `salary`, `experience`) and confirms no missing values
- Visualizes relationships with side-by-side **scatter plots** (`age` vs `salary`, `experience` vs `salary`)
- Generates a **correlation heatmap** (`sns.heatmap`) to inspect feature relationships before modeling
- Builds a **Multiple Linear Regression** model using two independent variables (`age`, `experience`) to predict `salary`, following the formula `y = m1*x1 + m2*x2 + c`
- Trains and evaluates the model with `train_test_split` and `.score()`, achieving ~97.5% R² on test data
- Makes predictions on new input and manually verifies them using the extracted coefficients and intercept
- Synthesizes a custom **quadratic dataset** (`Salary = 1000 + 50*Level + 20*Level²`) to demonstrate non-linear regression
- Builds a **Polynomial Regression** model using `PolynomialFeatures(degree=2)` to transform the input before fitting a `LinearRegression` model
- Achieves a perfect R² score of 1.0 on the synthetic quadratic data, confirming the model correctly recovers the underlying formula
- Plots the original data against the model's predicted curve to visually confirm the fit

---

## 🧠 The Process

Day 5 moved from single-variable regression into the real-world case: predicting salary using **two** independent variables, age and experience. Before modeling, I checked the relationships visually with scatter plots and a correlation heatmap, then trained a Multiple Linear Regression model using both features together.

The model performed well, hitting roughly 97.5% R² on the test set. To make sure I really understood what the model was doing — not just calling `.predict()` — I extracted the coefficients and intercept directly and reconstructed the formula by hand: `salary = 999.05*age + 2496.81*experience + 20151.64`, confirming it matched the model's output for a new prediction.

The second half of the day tackled a different problem: what happens when the relationship between input and output isn't a straight line? I generated a synthetic dataset where salary grows quadratically with experience level (`1000 + 50*Level + 20*Level²`) — a pattern a standard linear model can't capture well. Using `PolynomialFeatures` to expand the input into `[1, Level, Level²]` before fitting a `LinearRegression` model let the algorithm fit a curve instead of a straight line. The result was a perfect R² of 1.0, and pulling out the coefficients (`0, 50, 20`) and intercept (`~1000`) showed the model had recovered the exact formula used to generate the data.

---

## 🛠 Running the Project

1. Place your `regression.csv` file in the project directory and update the path in the notebook's `pd.read_csv()` call (the polynomial dataset is generated automatically by the notebook itself)

2. Install dependencies:

```
pip install pandas seaborn matplotlib numpy scikit-learn
```

3. Open the notebook:

```
jupyter notebook Machine_Learning_Day5.ipynb
```

4. Run all cells top to bottom — the Multiple Linear Regression section uses `regression.csv`, while the Polynomial Regression section generates and uses its own synthetic dataset

---

## 📁 Summary

| Section | Technique | Tool | Result |
|---|---|---|---|
| Salary prediction | Multiple Linear Regression | `LinearRegression` (2 features) | R² ≈ 0.975 |
| Synthetic quadratic data | Polynomial Regression | `PolynomialFeatures(degree=2)` + `LinearRegression` | R² = 1.0 |

