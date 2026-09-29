"""
churn_library.py

Library of functions to find customers who are likely to churn, following
the Udacity "Predict Customer Churn" clean-code project specification.

Pipeline:
    import_data -> perform_eda -> encoder_helper -> perform_feature_engineering
    -> train_models -> classification_report_image / feature_importance_plot

Author: (your name here)
Date: 2026
"""
import os
import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # noqa: E402  -- must be set before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns

import wandb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_curve, RocCurveDisplay

sns.set()

os.environ["QT_QPA_PLATFORM"] = "offscreen"

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="a",
    format="%(name)s - %(levelname)s - %(message)s",
)

CAT_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

QUANT_COLUMNS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

KEEP_COLS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]


def import_data(pth):
    """
    Returns dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)

    # Drop the two Naive-Bayes helper columns shipped in the raw Kaggle
    # export, if present -- they leak information and are not features.
    nb_cols = [c for c in df.columns if c.startswith("Naive_Bayes_Classifier")]
    if nb_cols:
        df = df.drop(columns=nb_cols)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return df


def perform_eda(df):
    """
    Perform EDA on df and save figures to images/eda folder.

    input:
            df: pandas dataframe
    output:
            None
    """
    os.makedirs("./images/eda", exist_ok=True)

    plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.title("Churn Distribution")
    plt.xlabel("Churn (0 = Existing, 1 = Attrited)")
    plt.ylabel("Count")
    plt.savefig("./images/eda/churn_distribution.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.title("Customer Age Distribution")
    plt.xlabel("Customer Age")
    plt.ylabel("Count")
    plt.savefig("./images/eda/customer_age_distribution.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.title("Marital Status Distribution")
    plt.ylabel("Proportion")
    plt.savefig("./images/eda/marital_status_distribution.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.title("Total Transaction Count Distribution")
    plt.savefig("./images/eda/total_transaction_distribution.png")
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df[QUANT_COLUMNS + ["Churn"]].corr(),
        annot=False,
        cmap="Dark2_r",
        linewidths=2,
    )
    plt.title("Feature Correlation Heatmap")
    plt.savefig("./images/eda/heatmap.png")
    plt.close()

    logging.info("perform_eda: SUCCESS - saved 5 EDA plots to ./images/eda")


def encoder_helper(df, category_lst, response="Churn"):
    """
    Helper function to turn each categorical column into a new column with
    the proportion of churn for each category - name of the new column is
    determined by the response argument.

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
                      be used for naming variables or index y column]
    output:
            df: pandas dataframe with new encoded columns
    """
    df = df.copy()
    for category in category_lst:
        group_means = df.groupby(category)[response].mean()
        df[f"{category}_{response}"] = df[category].map(group_means)

    logging.info(
        "encoder_helper: SUCCESS - encoded %d categorical columns", len(category_lst)
    )
    return df


def perform_feature_engineering(df, response="Churn"):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that
                        could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    df = encoder_helper(df, CAT_COLUMNS, response)

    y = df[response]
    X = pd.DataFrame()
    X[KEEP_COLS] = df[KEEP_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    logging.info(
        "perform_feature_engineering: SUCCESS - X_train %s, X_test %s",
        X_train.shape,
        X_test.shape,
    )
    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    Produces classification report for training and testing results and
    stores the report as an image in images/results folder.

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    os.makedirs("./images/results", exist_ok=True)

    reports = {
        "Random Forest": (y_train, y_train_preds_rf, y_test, y_test_preds_rf),
        "Logistic Regression": (y_train, y_train_preds_lr, y_test, y_test_preds_lr),
    }

    for name, (yt, ytp, yte, ytep) in reports.items():
        plt.rc("figure", figsize=(6, 6))
        plt.text(0.01, 1.05, f"{name} Train", {"fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.6,
            str(classification_report(yt, ytp)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.text(0.01, 0.45, f"{name} Test", {"fontsize": 10}, fontproperties="monospace")
        plt.text(
            0.01, 0.0,
            str(classification_report(yte, ytep)),
            {"fontsize": 10},
            fontproperties="monospace",
        )
        plt.axis("off")
        fname = name.lower().replace(" ", "_")
        plt.savefig(f"./images/results/{fname}_classification_report.png", bbox_inches="tight")
        plt.close()

    logging.info("classification_report_image: SUCCESS - saved classification report images")


def feature_importance_plot(model, X_data, output_pth):
    """
    Creates and stores the feature importances in output_pth.

    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    os.makedirs(os.path.dirname(output_pth), exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 10))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()

    logging.info("feature_importance_plot: SUCCESS - saved to %s", output_pth)


def train_models(X_train, X_test, y_train, y_test):
    """
    Train, store model results (images + scores) and store models.

    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    
    run = wandb.init(
        entity="models-accenture25",
        project="predict-customer-churn",
        config={
            "random_state": 42,
            "logistic_solver": "lbfgs",
            "logistic_max_iter": 3000,
            "cv_folds": 5,
            "rf_n_estimators": [50, 100],
            "rf_max_depth": [4, 5, 100],
            "rf_criterion": ["gini", "entropy"],
        },
    )
    
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./images/results", exist_ok=True)

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    param_grid = {
        "n_estimators": [50, 100],
        "max_features": ["sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    logging.info("train_models: SUCCESS - random forest grid search complete")

    lrc.fit(X_train, y_train)
    logging.info("train_models: SUCCESS - logistic regression fit complete")

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    # ROC curve comparison plot
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(lrc, X_test, y_test, ax=ax)
    RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, X_test, y_test, ax=ax)
    plt.title("ROC Curves")
    plt.savefig("./images/results/roc_curve_result.png")
    plt.close()

    feature_importance_plot(
        cv_rfc.best_estimator_, X_train, "./images/results/feature_importances.png"
    )

    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")
    logging.info("train_models: SUCCESS - saved rfc_model.pkl and logistic_model.pkl")

    run.finish()


if __name__ == "__main__":
    dataframe = import_data("./data/bank_data.csv")
    perform_eda(dataframe)
    X_train_, X_test_, y_train_, y_test_ = perform_feature_engineering(dataframe)
    train_models(X_train_, X_test_, y_train_, y_test_)

