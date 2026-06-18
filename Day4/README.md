# 📈 Stock & Placement Data — Function Transformation, Train-Test Split & Linear Regression (ML Day 4)

A hands-on notebook covering function transformation for skewed data, train-test splitting for model evaluation, and an introduction to simple linear regression — moving from preprocessing into the first actual ML model.

---

## ⚡ Technologies

- Python
- Pandas
- Seaborn
- Matplotlib
- NumPy
- Scikit-learn (`FunctionTransformer`, `train_test_split`, `LinearRegression`)
- CSV (multiple datasets — World Stock Prices, Boston-like housing, Placement)

---

## 🚀 Features

- Applies **IQR-based outlier removal** on stock trading `Volume` data before transformation
- Visualizes skewed distributions using **Seaborn's distplot**
- Applies **Function Transformation** (`FunctionTransformer` with `np.log1p`) to convert a right-skewed `Volume` distribution into a more normal shape — since most ML models perform better on normally distributed data
- Compares before/after distributions to confirm the transformation reduced skew
- Performs a **Train-Test Split** on a Boston-like housing dataset (509 rows, 13 features) using `train_test_split` with a 75/25 ratio
- Verifies split shapes to confirm correct partitioning of training and test sets
- Introduces **Simple Linear Regression** — single independent variable (`cgpa`) predicting a single dependent variable (`package`) using the `y = mx + c` formula
- Visualizes the `cgpa` vs `package` relationship with a **scatter plot**
- Fits a `LinearRegression` model, generates predictions for new inputs, and evaluates performance using `.score()`
- Extracts and interprets the model's **coefficient (slope)** and **intercept**, manually verifying predictions using the line equation

---

## 🧠 The Process

Day 4 picks up right where feature scaling left off, but with a twist: instead of just standardizing or normalizing, I explored **function transformation** as a way to fix skewed data. The `Volume` column in a world stock prices dataset was heavily right-skewed with extreme outliers, so after first trimming outliers using IQR, I applied a log transformation (`np.log1p`) and compared the distribution plots before and after. The log-transformed version was visibly closer to a normal distribution, which matters because many ML algorithms assume normally distributed inputs.

From there, the notebook shifts into a new phase: actually building a model. Before fitting anything, I split a Boston-like housing dataset into training and test sets using `train_test_split`, holding back 25% of the data to evaluate the model fairly on unseen data later.

The highlight of the day was building my first **Simple Linear Regression** model on a small placement dataset relating CGPA to salary package. After confirming no missing values and visualizing the linear relationship with a scatter plot, I split the data, trained a `LinearRegression` model, and used `.predict()` to estimate a package for a given CGPA. I then pulled out the model's coefficient and intercept directly and manually recomputed the prediction using `y = mx + c` to confirm the model's internal math matched its `.predict()` output exactly.

---

## 🛠 Running the Project

1. Place your CSV files (`World-Stock-Prices-Dataset.csv`, `Boston_like_509_rows.csv`, `placement.csv`) in the project directory and update the paths in the notebook's `pd.read_csv()` calls

2. Install dependencies:

```
pip install pandas seaborn matplotlib numpy scikit-learn
```

3. Open the notebook:

```
jupyter notebook Machine_Learning_Day4.ipynb
```

4. Run all cells top to bottom — the function transformation section runs independently of the regression section, which uses a separate dataset

---

## 📁 Summary

| Section | Technique | Tool |
|---|---|---|
| Volume distribution | Outlier removal | IQR method |
| Volume distribution | Function Transformation | `FunctionTransformer(func=np.log1p)` |
| Housing dataset | Train-Test Split (75/25) | `train_test_split` |
| Placement dataset | Simple Linear Regression | `LinearRegression` |

