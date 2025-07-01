import pandas as pd
import pytest
from pmhctcr_predictor.model import build_feature_matrix, train_model, predict

def test_build_feature_matrix():
    df = pd.DataFrame({
        'tcr_sequence': ['ACD'],
        'pmhc_sequence': ['EFG'],
        'label': [1]
    })
    X = build_feature_matrix(df, k=1)
    assert X.shape[0] == 1


def test_build_feature_matrix_invalid_k():
    df = pd.DataFrame({
        "tcr_sequence": ["ACD"],
        "pmhc_sequence": ["EFG"],
        "label": [1],
    })
    with pytest.raises(ValueError):
        build_feature_matrix(df, k=0)


def test_train_model_missing_columns(tmp_path):
    df = pd.DataFrame({"tcr_sequence": ["ACD"], "label": [1]})
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)
    model = tmp_path / "model.joblib"
    with pytest.raises(ValueError):
        train_model(csv, model, k=1)


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
        train_model(csv, model_path, k=0)


def test_predict_missing_columns(tmp_path):
    df = pd.DataFrame({"pmhc_sequence": ["ACD"]})
    csv = tmp_path / "pred.csv"
    df.to_csv(csv, index=False)
    model = tmp_path / "model.joblib"
    output = tmp_path / "out.csv"
    with pytest.raises(ValueError):
        predict(csv, model, output)
