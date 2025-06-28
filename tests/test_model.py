import pandas as pd
from pmhctcr_predictor.model import build_feature_matrix

def test_build_feature_matrix():
    df = pd.DataFrame({
        'tcr_sequence': ['ACD'],
        'pmhc_sequence': ['EFG'],
        'label': [1]
    })
    X = build_feature_matrix(df, k=1)
    assert X.shape[0] == 1
