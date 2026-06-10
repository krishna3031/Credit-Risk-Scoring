# Credit Risk Assessment

A machine learning project to predict loan default risk using Logistic Regression and Random Forest, with SMOTE for class imbalance handling and SHAP for model explainability.

## Problem Statement

Given historical loan data, predict whether a borrower will **not fully pay** their loan (`not.fully.paid` = 1). This is a binary classification problem with a real-world class imbalance — defaults are rarer than repayments.

## Dataset

**File:** `loan_data.csv`

**Target variable:** `not.fully.paid` (0 = paid, 1 = defaulted)

**Key feature:** `purpose` (categorical, one-hot encoded)

> The dataset is not included in this repo. Place `loan_data.csv` in the root directory before running the notebook.

## Project Workflow

```
Load Data → EDA → Preprocessing → Train/Test Split → Scaling → SMOTE → Model Training → Evaluation → Risk Scoring → SHAP Explainability
```

**Step-by-step:**

1. Load and inspect `loan_data.csv`
2. One-hot encode the `purpose` column
3. Split into features (`X`) and target (`y`)
4. Train/test split (80/20)
5. Scale features with `StandardScaler`
6. Apply SMOTE on training data only to handle class imbalance
7. Train Logistic Regression and Random Forest models
8. Evaluate with Accuracy, Classification Report, ROC-AUC, and Confusion Matrix
9. Assign risk categories (Low / Medium / High) based on default probability
10. Explain model predictions using SHAP beeswarm plots

## Models Used

| Model | Notes |
|---|---|
| Logistic Regression | Baseline model, `max_iter=1000` |
| Random Forest | 100 estimators, used for feature importance and SHAP |

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score (via Classification Report)
- ROC-AUC Score
- Confusion Matrix (heatmap)

## Risk Categorization

Default probabilities from Logistic Regression are bucketed as:

| Probability | Category |
|---|---|
| < 0.30 | Low Risk |
| 0.30 – 0.70 | Medium Risk |
| > 0.70 | High Risk |

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
shap
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap
```

## Usage

```bash
# Clone the repo
git clone https://github.com/your-username/credit-risk-assessment.git
cd credit-risk-assessment

# Install dependencies
pip install -r requirements.txt

# Place loan_data.csv in the root directory, then run
jupyter notebook CreditRiskAssessing.ipynb
```

## Project Structure

```
credit-risk-assessment/
├── CreditRiskAssessing.ipynb   # Main notebook
├── loan_data.csv               # Dataset 
└── README.md
```
