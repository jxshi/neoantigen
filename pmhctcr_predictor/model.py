import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

from .features import all_kmers, kmer_vector


def build_feature_matrix(df, k=2):
    """Create a feature matrix for a dataframe of sequences."""
    kmers = all_kmers(k)
    data = []
    for _, row in df.iterrows():
        tcr_vec = kmer_vector(row['tcr_sequence'], k, kmers)
        pmhc_vec = kmer_vector(row['pmhc_sequence'], k, kmers)
        data.append(np.concatenate([tcr_vec, pmhc_vec]))
    return np.array(data)


def train_model(train_csv, model_path, k=2, metric="accuracy"):
    """Train a logistic regression model and save it to disk.

    Parameters
    ----------
    train_csv : str
        Path to the training CSV file.
    model_path : str
        Where to save the trained model.
    k : int, optional
        k-mer size used for feature extraction.
    metric : str, optional
        Name of the evaluation metric. Only ``"accuracy"`` is supported.

    Raises
    ------
    ValueError
        If ``metric`` is not a supported metric name.
    """

    if metric != "accuracy":
        raise ValueError(f"Unsupported metric: {metric}")

    df = pd.read_csv(train_csv)
    X = build_feature_matrix(df, k)
    y = df["label"]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    dump({"model": clf, "k": k}, model_path)


def predict(predict_csv, model_path, output_csv):
    """Predict interaction probabilities for new pairs."""
    df = pd.read_csv(predict_csv)
    params = load(model_path)
    clf = params['model']
    k = params['k']
    X = build_feature_matrix(df, k)
    probs = clf.predict_proba(X)[:, 1]
    df['prediction'] = probs
    df.to_csv(output_csv, index=False)
