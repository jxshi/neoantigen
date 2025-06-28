import pandas as pd
import pytest
from pmhctcr_predictor.model import build_feature_matrix, train_model

def test_build_feature_matrix():
    df = pd.DataFrame({
        'tcr_sequence': ['ACD'],
        'pmhc_sequence': ['EFG'],
        'label': [1]
    })
    X = build_feature_matrix(df, k=1)
    assert X.shape[0] == 1


def test_train_model_invalid_metric(tmp_path):
    csv = "tests/data/sample_train.csv"
    model_path = tmp_path / "model.joblib"
    with pytest.raises(ValueError):
        train_model(csv, model_path, metric="invalid")
