import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
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


def train_model(
    train_csv,
    model_path,
    k=2,
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    metric="accuracy",
):
    """Train a logistic regression model and save it to disk.

    Parameters
    ----------
    train_csv : str
        Path to the training CSV file.
    model_path : str
        Where to store the trained model.
    k : int, optional
        k-mer size for feature extraction.
    C : float, optional
        Inverse regularisation strength passed to ``LogisticRegression``.
    penalty : str, optional
        Penalty used by ``LogisticRegression``.
    solver : str, optional
        Solver for ``LogisticRegression``.
    max_iter : int, optional
        Maximum number of iterations for optimisation.
    metric : {"accuracy", "auc"}, optional
        Metric to compute on the training data and return.

    Returns
    -------
    float
        The requested training metric value.
    """

    df = pd.read_csv(train_csv)
    X = build_feature_matrix(df, k)
    y = df["label"]

    clf = LogisticRegression(
        C=C, penalty=penalty, solver=solver, max_iter=max_iter
    )
    clf.fit(X, y)

    if metric == "accuracy":
        preds = clf.predict(X)
        score = accuracy_score(y, preds)
    elif metric == "auc":
        probs = clf.predict_proba(X)[:, 1]
        score = roc_auc_score(y, probs)
    else:
        raise ValueError("metric must be 'accuracy' or 'auc'")

    dump({"model": clf, "k": k}, model_path)
    return score


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
