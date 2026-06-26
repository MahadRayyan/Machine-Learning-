# 🎯 Netflix Subscription Prediction — Logistic Regression (ML Day 8)

A hands-on classification notebook introducing Logistic Regression — covering binomial logistic regression with a single feature, multinomial logistic regression with multiple features, and polynomial feature expansion applied to classification.

---

## ⚡ Technologies

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn (`LogisticRegression`, `PolynomialFeatures`, `train_test_split`)
- CSV (Netflix age dataset — single and two-feature versions)

---

## 🚀 Features

- Introduces **Logistic Regression** for binary classification — predicting a discrete output (yes/no) using the sigmoid function `ŷ = 1 / (1 + e^-x)` with a 0.5 decision threshold
- Loads a 43-row Netflix subscription dataset (`age` → `netflix`) and visualizes the binary class distribution with a **scatter plot**
- Builds a **Binomial Logistic Regression** model (single independent variable: `age`) — achieves ~77.8% accuracy on the test set
- Overlays the model's predicted decision boundary as a **line plot** on top of the original scatter plot to visualize how the sigmoid separates the two classes
- Loads a two-feature Netflix dataset (`age`, `monthly_income` → `netflix`) and visualizes class separation with a **hue-colored scatter plot**
- Builds a **Multinomial Logistic Regression** model (two independent variables) — achieves 80% accuracy on the test set
- Applies **Polynomial Feature Expansion** (`PolynomialFeatures(degree=2)`) to the two-feature dataset, expanding the input space with squared and interaction terms (`x1, x2, x1², x1x2, x2²`) to capture non-linear decision boundaries

---

## 🧠 The Process

Day 8 marks the shift from regression to **classification** — predicting categories instead of continuous values.

The key concept is that Logistic Regression doesn't output a raw number — it passes the linear combination `m1x1 + m2x2 + b` through a **sigmoid function**, squashing the output to a probability between 0 and 1. Anything above 0.5 gets classified as 1, anything below as 0.

I started with the simplest possible case: a single feature (`age`) predicting whether someone subscribes to Netflix. A scatter plot of the 43-row dataset made the separation visually obvious, and after an 80/20 train-test split, the model reached ~77.8% accuracy. Plotting the sigmoid decision boundary over the scatter confirmed the model had learned the right threshold age.

The second dataset added `monthly_income` as a second predictor — making this a **multinomial** (multi-feature, not multi-class) setup. The hue-colored scatter plot showed income and age together separating subscribers from non-subscribers more clearly, and the model bumped up to 80% accuracy as a result.

Finally, I layered in **Polynomial Feature Expansion** — instead of a straight-line decision boundary, expanding the features to degree 2 gives the model the ability to draw curves. The expanded feature set includes the original features, their squares, and interaction terms, which is the same trick from Day 5 (Polynomial Regression) now applied to classification.

---

## 🛠 Running the Project

1. Place `netflix_age_dataset.csv` and `netflix_dataset_2_features.csv` in the project directory and update the paths in the notebook's `pd.read_csv()` calls

2. Install dependencies:

```
pip install pandas seaborn matplotlib scikit-learn
```

3. Open the notebook:

```
jupyter notebook Machine_Learning_Day8.ipynb
```

4. Run all cells top to bottom — binomial section uses the single-feature dataset; multinomial and polynomial sections use the two-feature dataset

---

## 📁 Model Comparison

| Section | Features | Model | Accuracy |
|---|---|---|---|
| Binomial Logistic Regression | `age` (1 feature) | `LogisticRegression` | ~77.8% |
| Multinomial Logistic Regression | `age`, `monthly_income` (2 features) | `LogisticRegression` | 80.0% |
| Polynomial Logistic Regression | `age`, `monthly_income` + degree-2 expansion | `PolynomialFeatures` + `LogisticRegression` | 80.0% |
