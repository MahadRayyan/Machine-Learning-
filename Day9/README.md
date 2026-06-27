# 🌸 Iris & Netflix — Multiclass Classification, Confusion Matrix & Evaluation Metrics (ML Day 9)

A hands-on classification notebook covering multiclass logistic regression using OvR and Multinomial strategies on the Iris dataset, followed by a deep dive into classification evaluation — confusion matrix, precision, recall, and F1 score on a Netflix subscription dataset.

---

## ⚡ Technologies

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn (`LogisticRegression`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `train_test_split`)
- mlxtend (`plot_decision_regions`)
- CSV (Iris 5000-row dataset, Netflix two-feature dataset)

---

## 🚀 Features

- Loads a 5000-row Iris dataset with four features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) and three target classes (`setosa`, `versicolor`, `virginica`)
- Visualizes all pairwise feature relationships with a **Seaborn pairplot** colored by species
- Trains a **One-vs-Rest (OvR)** Logistic Regression model — fits one binary classifier per class, treating each class against all others — achieves **96.8% accuracy**
- Trains a **Multinomial** Logistic Regression model — models all three classes simultaneously using softmax probabilities — achieves **98.3% accuracy**
- Switches to the Netflix two-feature dataset to explore model evaluation beyond raw accuracy
- Trains a binary `LogisticRegression` model and generates a **Confusion Matrix** — visualized as an annotated heatmap showing true positives, true negatives, false positives, and false negatives
- Computes **Precision** (~82.6%) — of all predicted subscribers, how many actually subscribed
- Computes **Recall** (100%) — of all actual subscribers, how many did the model correctly identify (zero false negatives)
- Computes **F1 Score** (~90.5%) — the harmonic mean of precision and recall, the go-to metric when dataset knowledge is limited
- Visualizes the model's **decision boundary** using `mlxtend`'s `plot_decision_regions` to show how the classifier splits the feature space

---

## 🧠 The Process

Day 9 tackled two separate problems. The first was extending Logistic Regression to handle **three classes** instead of two. The Iris dataset is the classic benchmark for this — four flower measurements, three species. I trained two variants: OvR (One-vs-Rest), which fits a separate binary classifier for each species and picks the most confident one, and Multinomial, which models all three classes at once using softmax. Multinomial edged ahead at 98.3% vs OvR's 96.8%.

The second half addressed a critical blind spot: **accuracy alone doesn't tell the full story**. A model that's 95% accurate can still be completely wrong for the use case if it's making the wrong *kind* of errors.

That's where the confusion matrix comes in. Revisiting the Netflix dataset, the model's confusion matrix showed 17 true negatives, 19 true positives, 4 false positives, and — critically — **0 false negatives**. This means the model never missed a real subscriber (perfect recall of 100%), but occasionally flagged a non-subscriber as one (precision ~82.6%).

The key insight: **Type I errors** (false positives — predicting true when actual is false) and **Type II errors** (false negatives — predicting false when actual is true) matter differently depending on context. In medical diagnosis, a Type II error (missing a real case) is far more dangerous than a Type I error. When you don't know which matters more, optimize for **F1 score** — the harmonic mean of precision and recall — which penalizes extreme imbalances between the two. The Netflix model hit an F1 of ~90.5%.

Finally, `plot_decision_regions` made the decision boundary concrete — showing visually exactly where in the age/income feature space the model switches its prediction from 0 to 1.

---

## 🛠 Running the Project

1. Place `iris_5000_rows.csv` and `netflix_dataset_2_features.csv` in the project directory and update the paths in the notebook's `pd.read_csv()` calls

2. Install dependencies:

```
pip install pandas seaborn matplotlib scikit-learn mlxtend
```

3. Open the notebook:

```
jupyter notebook Machine_Learning_Day9.ipynb
```

4. Run all cells top to bottom — the multiclass section uses the Iris dataset; the confusion matrix and metrics section uses the Netflix dataset

---

## 📁 Model Comparison

| Dataset | Model Strategy | Accuracy |
|---|---|---|
| Iris (3 classes) | OvR Logistic Regression | 96.8% |
| Iris (3 classes) | Multinomial Logistic Regression | 98.3% |
| Netflix (binary) | Logistic Regression | ~90% |

## 📊 Evaluation Metrics (Netflix Model)

| Metric | Formula | Score |
|---|---|---|
| Precision | TP / (TP + FP) | ~82.6% |
| Recall | TP / (TP + FN) | 100.0% |
| F1 Score | 2 × (Precision × Recall) / (Precision + Recall) | ~90.5% |
