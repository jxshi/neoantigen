import pandas as pd
from pmhctcr_predictor.model import build_feature_matrix, train_model
from joblib import load

def test_build_feature_matrix():
    df = pd.DataFrame({
        'tcr_sequence': ['ACD'],
        'pmhc_sequence': ['EFG'],
        'label': [1]
    })
    X = build_feature_matrix(df, k=1)
    assert X.shape[0] == 1


def test_train_model_params_and_metric(tmp_path):
    df = pd.DataFrame({
        "tcr_sequence": ["AAA", "CCC"],
        "pmhc_sequence": ["DDD", "EEE"],
        "label": [0, 1],
    })
    train_csv = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"
    df.to_csv(train_csv, index=False)

    score = train_model(
        train_csv,
        model_path,
        k=1,
        C=0.5,
        penalty="l1",
        solver="liblinear",
        metric="accuracy",
    )

    params = load(model_path)
    clf = params["model"]

    assert clf.C == 0.5
    assert clf.penalty == "l1"
    assert clf.solver == "liblinear"
    assert 0.0 <= score <= 1.0


def test_train_model_auc_metric(tmp_path):
    df = pd.DataFrame({
        "tcr_sequence": ["AAA", "AAA", "CCC", "CCC"],
        "pmhc_sequence": ["DDD", "EEE", "DDD", "EEE"],
        "label": [0, 1, 0, 1],
    })
    train_csv = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"
    df.to_csv(train_csv, index=False)

    score = train_model(
        train_csv,
        model_path,
        k=1,
        metric="auc",
    )

    assert 0.0 <= score <= 1.0
