import pandas as pd
from sklearn.linear_model import LogisticRegression
from pmhctcr_predictor.model import build_feature_matrix


def test_logistic_regression_probabilities():
    df = pd.read_csv('tests/data/sample_train.csv')
    X = build_feature_matrix(df, k=1)
    y = df['label']
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    assert probs.dtype.kind in {'f', 'd'}
    assert probs.min() >= 0
    assert probs.max() <= 1
    assert all(0 <= p <= 1 for p in probs)
