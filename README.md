# Predict Customer Churn

A clean-code implementation of the "Predict Customer Churn" project: identify
credit card customers who are most likely to churn, following software
engineering best practices (modular functions, docstrings, logging, and
automated tests) rather than a single notebook.

## Project Description

The library reads customer data, performs exploratory data analysis (EDA),
encodes categorical features, trains two classifiers — a Random Forest
(with hyperparameter search via `GridSearchCV`) and a Logistic Regression —
and saves the resulting plots, trained models, and a run log to disk.

## Files and Data Description

```
.
├── churn_library.py                     # Core library: import, EDA, encoding,
│                                         #   feature engineering, training
├── churn_script_logging_and_tests.py    # pytest test suite for every function,
│                                         #   also runnable as a plain script
├── generate_data.py                     # Generates the synthetic dataset used
│                                         #   in this submission (data/bank_data.csv)
├── requirements.txt                     # Python dependencies
├── data/
│   └── bank_data.csv                    # Input dataset (customer records)
├── images/
│   ├── eda/                             # Exploratory data analysis plots
│   │   ├── churn_distribution.png
│   │   ├── customer_age_distribution.png
│   │   ├── marital_status_distribution.png
│   │   ├── total_transaction_distribution.png
│   │   └── heatmap.png
│   └── results/                         # Model evaluation plots
│       ├── random_forest_classification_report.png
│       ├── logistic_regression_classification_report.png
│       ├── roc_curve_result.png
│       └── feature_importances.png
├── models/
│   ├── rfc_model.pkl                    # Trained Random Forest classifier
│   └── logistic_model.pkl               # Trained Logistic Regression classifier
└── logs/
    └── churn_library.log                # Log of the most recent test run
```

### About the dataset

This submission ships with `data/bank_data.csv`, a **synthetically generated**
dataset (see `generate_data.py`) that mirrors the column names, ranges, and
structure of the classic Kaggle "Credit Card customers" (BankChurners.csv)
dataset used in the original course project, so the full pipeline runs
end-to-end out of the box.

**If you have the real BankChurners.csv from Kaggle**, just replace
`data/bank_data.csv` with it (same column names are expected) and re-run —
`churn_library.py` does not care which one it reads.

## Running the Files

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional — a dataset is already included) Regenerate the synthetic
   dataset:
   ```bash
   python3 generate_data.py
   ```

3. Run the full pipeline (import → EDA → feature engineering → train
   models). This writes plots to `images/`, trained models to `models/`,
   and appends to `logs/churn_library.log`:
   ```bash
   python3 churn_library.py
   ```

4. Run the test suite, which exercises every function and logs
   SUCCESS/ERROR for each check to `logs/churn_library.log`:
   ```bash
   python3 -m pytest churn_script_logging_and_tests.py -v
   ```
   or, without pytest:
   ```bash
   python3 churn_script_logging_and_tests.py
   ```

5. Inspect results:
   - EDA plots: `images/eda/`
   - Model performance plots (classification reports, ROC curve, feature
     importances): `images/results/`
   - Trained models: `models/rfc_model.pkl`, `models/logistic_model.pkl`
   - Test/run log: `logs/churn_library.log`

## Notes for Reviewers

- No secrets, API keys, or credentials are included anywhere in this
  submission.
- `logs/churn_library.log` in this ZIP reflects the most recent successful
  run of `churn_script_logging_and_tests.py` (5/5 tests passed).
