"""
churn_script_logging_and_tests.py

Unit tests for each function in churn_library.py. Every test logs SUCCESS or
ERROR to ./logs/churn_library.log so the log file itself is a record of the
test run, as required by the rubric. Runnable directly with `python3
churn_script_logging_and_tests.py` or with `pytest churn_script_logging_and_tests.py`.

Author: (your name here)
Date: 2026
"""
import os
import logging

import pandas as pd
import pytest

import churn_library as cls

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="a",
    format="%(name)s - %(levelname)s - %(message)s",
)

DATA_PATH = "./data/bank_data.csv"


@pytest.fixture(scope="module")
def raw_dataframe():
    """Dataframe straight out of import_data, shared across tests."""
    return cls.import_data(DATA_PATH)


@pytest.fixture(scope="module")
def engineered_data(raw_dataframe):
    """X_train, X_test, y_train, y_test shared across the later tests."""
    return cls.perform_feature_engineering(raw_dataframe)


def test_import():
    """
    Test import_data: file exists, loads into a non-empty dataframe with the
    Churn column derived correctly.
    """
    try:
        df = cls.import_data(DATA_PATH)
        logging.info("test_import: SUCCESS - file loaded")
    except FileNotFoundError as err:
        logging.error("test_import: FAILED - file not found at %s", DATA_PATH)
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("test_import: FAILED - the file doesn't appear to have rows and columns")
        raise err

    try:
        assert "Churn" in df.columns
        assert set(df["Churn"].unique()).issubset({0, 1})
    except AssertionError as err:
        logging.error("test_import: FAILED - Churn column missing or malformed")
        raise err

    logging.info("test_import: SUCCESS - shape=%s, Churn column present", df.shape)


def test_eda(raw_dataframe):
    """
    Test perform_eda: confirms all expected plot files are written to
    images/eda.
    """
    expected_files = [
        "churn_distribution.png",
        "customer_age_distribution.png",
        "marital_status_distribution.png",
        "total_transaction_distribution.png",
        "heatmap.png",
    ]

    cls.perform_eda(raw_dataframe)

    try:
        for fname in expected_files:
            fpath = os.path.join("./images/eda", fname)
            assert os.path.isfile(fpath), f"missing {fpath}"
        logging.info("test_eda: SUCCESS - all %d EDA images present", len(expected_files))
    except AssertionError as err:
        logging.error("test_eda: FAILED - %s", err)
        raise err


def test_encoder_helper(raw_dataframe):
    """
    Test encoder_helper: new mean-encoded columns are created for every
    category passed in, and values are within [0, 1].
    """
    df = cls.encoder_helper(raw_dataframe, cls.CAT_COLUMNS, "Churn")

    try:
        for category in cls.CAT_COLUMNS:
            new_col = f"{category}_Churn"
            assert new_col in df.columns
            assert df[new_col].between(0, 1).all()
        logging.info(
            "test_encoder_helper: SUCCESS - %d columns encoded correctly",
            len(cls.CAT_COLUMNS),
        )
    except AssertionError as err:
        logging.error("test_encoder_helper: FAILED - %s", err)
        raise err


def test_perform_feature_engineering(engineered_data):
    """
    Test perform_feature_engineering: train/test split has the right shapes
    and no overlap in indices.
    """
    X_train, X_test, y_train, y_test = engineered_data

    try:
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1] == len(cls.KEEP_COLS)
        assert X_train.shape[0] > X_test.shape[0]  # 70/30 split
        assert set(X_train.index).isdisjoint(set(X_test.index))
        logging.info(
            "test_perform_feature_engineering: SUCCESS - X_train %s, X_test %s",
            X_train.shape,
            X_test.shape,
        )
    except AssertionError as err:
        logging.error("test_perform_feature_engineering: FAILED - %s", err)
        raise err


def test_train_models(engineered_data):
    """
    Test train_models: confirms models and result images are written to
    disk.
    """
    X_train, X_test, y_train, y_test = engineered_data
    cls.train_models(X_train, X_test, y_train, y_test)

    expected_models = ["./models/rfc_model.pkl", "./models/logistic_model.pkl"]
    expected_images = [
        "./images/results/random_forest_classification_report.png",
        "./images/results/logistic_regression_classification_report.png",
        "./images/results/roc_curve_result.png",
        "./images/results/feature_importances.png",
    ]

    try:
        for fpath in expected_models:
            assert os.path.isfile(fpath), f"missing {fpath}"
        for fpath in expected_images:
            assert os.path.isfile(fpath), f"missing {fpath}"
        logging.info(
            "test_train_models: SUCCESS - %d models and %d result images present",
            len(expected_models),
            len(expected_images),
        )
    except AssertionError as err:
        logging.error("test_train_models: FAILED - %s", err)
        raise err


if __name__ == "__main__":
    # Allows `python3 churn_script_logging_and_tests.py` to run the full
    # sequence outside of pytest as well.
    df_ = cls.import_data(DATA_PATH)
    test_import()
    test_eda(df_)
    test_encoder_helper(df_)
    data_ = cls.perform_feature_engineering(df_)
    test_perform_feature_engineering(data_)
    test_train_models(data_)
    print("All tests completed. See ./logs/churn_library.log for the run log.")
