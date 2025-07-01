import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

from .esm_features import ESMEmbedder

from .features import all_kmers, kmer_vector


def build_feature_matrix(df, k=2):
    """Create a feature matrix for a dataframe of sequences."""
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    kmers = all_kmers(k)
    data = []
    for _, row in df.iterrows():
        tcr_vec = kmer_vector(row['tcr_sequence'], k, kmers)
        pmhc_vec = kmer_vector(row['pmhc_sequence'], k, kmers)
        data.append(np.concatenate([tcr_vec, pmhc_vec]))
    return np.array(data)


def train_model(train_csv, model_path, k=2):
    """Train a logistic regression model and save it to disk."""
    df = pd.read_csv(train_csv)
    required = {"tcr_sequence", "pmhc_sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing columns: {', '.join(sorted(missing))}"
        )
    X = build_feature_matrix(df, k)
    y = df['label']
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    dump({'model': clf, 'k': k}, model_path)


def train_model_svm(train_csv, model_path, k=2):
    """Train an SVM model and save it to disk."""
    df = pd.read_csv(train_csv)
    required = {"tcr_sequence", "pmhc_sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing columns: {', '.join(sorted(missing))}"
        )
    X = build_feature_matrix(df, k)
    y = df["label"]
    clf = SVC(probability=True)
    clf.fit(X, y)
    dump({"model": clf, "k": k}, model_path)


def train_model_rf(train_csv, model_path, k=2, n_estimators=100):
    """Train a random forest model and save it to disk."""
    df = pd.read_csv(train_csv)
    required = {"tcr_sequence", "pmhc_sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing columns: {', '.join(sorted(missing))}"
        )
    X = build_feature_matrix(df, k)
    y = df["label"]
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X, y)
    dump({"model": clf, "k": k}, model_path)


def train_model_esm(train_csv, model_path, model_name="esm2_t6_8M_UR50D"):
    """Train logistic regression using ESM embeddings."""
    df = pd.read_csv(train_csv)
    required = {"tcr_sequence", "pmhc_sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {', '.join(sorted(missing))}")
    embedder = ESMEmbedder(model_name)
    data = [
        embedder.pair_embedding(row["tcr_sequence"], row["pmhc_sequence"])
        for _, row in df.iterrows()
    ]
    X = np.stack(data)
    y = df["label"]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    dump({"model": clf, "model_name": model_name}, model_path)


def predict(predict_csv, model_path, output_csv):
    """Predict interaction probabilities for new pairs."""
    df = pd.read_csv(predict_csv)
    required = {"tcr_sequence", "pmhc_sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing columns: {', '.join(sorted(missing))}"
        )
    params = load(model_path)
    clf = params['model']
    k = params['k']
    X = build_feature_matrix(df, k)
    probs = clf.predict_proba(X)[:, 1]
    df['prediction'] = probs
    df.to_csv(output_csv, index=False)


def predict_esm(predict_csv, model_path, output_csv):
    """Predict using an ESM-based logistic regression model."""
    df = pd.read_csv(predict_csv)
    required = {"tcr_sequence", "pmhc_sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {', '.join(sorted(missing))}")

    params = load(model_path)
    clf = params["model"]
    model_name = params["model_name"]
    embedder = ESMEmbedder(model_name)
    data = [
        embedder.pair_embedding(row["tcr_sequence"], row["pmhc_sequence"])
        for _, row in df.iterrows()
    ]
    X = np.stack(data)
    probs = clf.predict_proba(X)[:, 1]
    df["prediction"] = probs
    df.to_csv(output_csv, index=False)
