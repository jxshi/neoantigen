import pandas as pd
import pytest

from pmhctcr_predictor import model


def test_build_feature_matrix():
    df = pd.DataFrame({
        "tcr_sequence": ["ACD"],
        "pmhc_sequence": ["EFG"],
        "label": [1],
    })
    X = model.build_feature_matrix(df, k=1)
    assert X.shape[0] == 1
    assert X.shape[1] > 0


def test_build_feature_matrix_invalid_k():
    df = pd.DataFrame({
        "tcr_sequence": ["ACD"],
        "pmhc_sequence": ["EFG"],
        "label": [1],
    })
    with pytest.raises(ValueError):
        model.build_feature_matrix(df, k=0)


def test_train_model_missing_columns(tmp_path):
    df = pd.DataFrame({"tcr_sequence": ["ACD"], "label": [1]})
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)
    model_path = tmp_path / "model.joblib"
    with pytest.raises(ValueError):
        model.train_model(csv, model_path, k=1)


def test_train_model_invalid_k(tmp_path):
    df = pd.DataFrame({
        "tcr_sequence": ["ACD"],
        "pmhc_sequence": ["EFG"],
        "label": [1],
    })
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)
    model_path = tmp_path / "model.joblib"
    with pytest.raises(ValueError):
        model.train_model(csv, model_path, k=0)


def test_predict_missing_columns(tmp_path):
    df = pd.DataFrame({"pmhc_sequence": ["ACD"]})
    csv = tmp_path / "pred.csv"
    df.to_csv(csv, index=False)
    model_path = tmp_path / "model.joblib"
    output = tmp_path / "out.csv"
    with pytest.raises(ValueError):
        model.predict(csv, model_path, output)


def test_train_model_with_evaluation(tmp_path):
    train_csv = "tests/data/sample_train.csv"
    model_path = tmp_path / "model.joblib"

    metrics = model.train_model(
        train_csv,
        model_path,
        k=1,
        test_size=0.5,
        random_state=0,
    )

    assert metrics is not None
    stored = model.load_metrics(model_path)
    assert stored is not None
    assert set(stored.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_train_and_evaluate_returns_metrics(tmp_path):
    train_csv = "tests/data/sample_train.csv"
    model_path = tmp_path / "model.joblib"

    metrics = model.train_and_evaluate(
        train_csv,
        model_path,
        method="logreg",
        k=1,
        test_size=0.5,
        random_state=0,
    )

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
