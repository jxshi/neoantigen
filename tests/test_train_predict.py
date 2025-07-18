import pandas as pd
from pmhctcr_predictor import model


def test_train_and_predict(tmp_path):
    train_csv = 'tests/data/sample_train.csv'
    model_path = tmp_path / 'model.joblib'
    output_csv = tmp_path / 'pred.csv'

    # Train logistic regression model
    model.train_model(train_csv, model_path, k=1)

    # Run prediction on the same data
    model.predict(train_csv, model_path, output_csv)

    # Load predictions and validate
    df_pred = pd.read_csv(output_csv)
    df_train = pd.read_csv(train_csv)
    assert 'prediction' in df_pred.columns
    assert len(df_pred) == len(df_train)


def test_train_and_predict_svm(tmp_path):
    train_csv = 'tests/data/sample_train.csv'
    model_path = tmp_path / 'svm_model.joblib'
    output_csv = tmp_path / 'svm_pred.csv'

    model.train_model_svm(train_csv, model_path, k=1)
    model.predict(train_csv, model_path, output_csv)

    df_pred = pd.read_csv(output_csv)
    df_train = pd.read_csv(train_csv)
    assert 'prediction' in df_pred.columns
    assert len(df_pred) == len(df_train)


def test_train_and_predict_rf(tmp_path):
    train_csv = 'tests/data/sample_train.csv'
    model_path = tmp_path / 'rf_model.joblib'
    output_csv = tmp_path / 'rf_pred.csv'

    model.train_model_rf(train_csv, model_path, k=1)
    model.predict(train_csv, model_path, output_csv)

    df_pred = pd.read_csv(output_csv)
    df_train = pd.read_csv(train_csv)
    assert 'prediction' in df_pred.columns
    assert len(df_pred) == len(df_train)
