from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .esm_features import ESMEmbedder
from .features import all_kmers, kmer_vector


TRAIN_REQUIRED_COLUMNS = {"tcr_sequence", "pmhc_sequence", "label"}
PREDICT_REQUIRED_COLUMNS = {"tcr_sequence", "pmhc_sequence"}


class ModelArtifactKeys:
    MODEL = "model"
    KMER = "k"
    METHOD = "method"
    METRICS = "metrics"
    MODEL_NAME = "model_name"


@dataclass
class EvaluationResult:
    """Container storing evaluation metrics for a trained model."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


def _ensure_positive_k(k: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _load_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {', '.join(sorted(missing))}")


def _build_pair_vector(
    tcr_sequence: str, pmhc_sequence: str, k: int, kmers: Sequence[str]
) -> np.ndarray:
    tcr_vec = kmer_vector(tcr_sequence, k, kmers)
    pmhc_vec = kmer_vector(pmhc_sequence, k, kmers)
    return np.concatenate([tcr_vec, pmhc_vec])


def build_feature_matrix(df: pd.DataFrame, k: int = 2) -> np.ndarray:
    """Create a feature matrix for a dataframe of sequences."""

    _ensure_positive_k(k)
    kmers = all_kmers(k)
    data = [
        _build_pair_vector(row["tcr_sequence"], row["pmhc_sequence"], k, kmers)
        for _, row in df.iterrows()
    ]
    return np.array(data)


def build_feature_matrix_with_kmers(
    df: pd.DataFrame, k: int, kmers: Optional[Sequence[str]] = None
) -> Tuple[np.ndarray, Sequence[str]]:
    """Return features and the kmers used to build them."""

    _ensure_positive_k(k)
    kmers = list(kmers) if kmers is not None else all_kmers(k)
    data = [
        _build_pair_vector(row["tcr_sequence"], row["pmhc_sequence"], k, kmers)
        for _, row in df.iterrows()
    ]
    return np.array(data), kmers


def _evaluate_predictions(model, X: np.ndarray, y: np.ndarray) -> EvaluationResult:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    if len(np.unique(y)) == 1:
        roc_auc = float("nan")
    else:
        roc_auc = roc_auc_score(y, probs)
    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
    )


def _get_estimator(method: str, **kwargs):
    if method == "logreg":
        return LogisticRegression(max_iter=1000, **kwargs)
    if method == "svm":
        kwargs.setdefault("probability", True)
        return SVC(**kwargs)
    if method == "rf":
        return RandomForestClassifier(**kwargs)
    raise ValueError(f"Unknown method '{method}'")


def _prepare_training_data(train_csv: str, k: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Sequence[str]]:
    df = _load_dataframe(train_csv)
    _validate_columns(df, TRAIN_REQUIRED_COLUMNS)
    X, kmers = build_feature_matrix_with_kmers(df, k)
    y = df["label"].to_numpy()
    return X, y, df, kmers


def _train_and_optionally_evaluate(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    test_size: Optional[float],
    random_state: Optional[int],
) -> Tuple[object, Optional[EvaluationResult]]:
    if test_size is None or test_size == 0:
        estimator.fit(X, y)
        return estimator, None

    unique, counts = np.unique(y, return_counts=True)
    stratify = y if len(unique) > 1 else None
    if stratify is not None and np.min(counts) < 2:
        stratify = None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    estimator.fit(X_train, y_train)
    evaluation = _evaluate_predictions(estimator, X_val, y_val)
    return estimator, evaluation


def _save_model(
    model_path: str,
    estimator,
    k: Optional[int] = None,
    method: Optional[str] = None,
    metrics: Optional[EvaluationResult] = None,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    payload: Dict[str, object] = {ModelArtifactKeys.MODEL: estimator}
    if k is not None:
        payload[ModelArtifactKeys.KMER] = k
    if method is not None:
        payload[ModelArtifactKeys.METHOD] = method
    if metrics is not None:
        payload[ModelArtifactKeys.METRICS] = metrics.to_dict()
    if extra:
        payload.update(extra)
    dump(payload, model_path)


def train_model(
    train_csv: str, model_path: str, k: int = 2, *, test_size: Optional[float] = None, random_state: Optional[int] = None, **kwargs
) -> Optional[Dict[str, float]]:
    """Train a logistic regression model and save it to disk."""

    _ensure_positive_k(k)
    X, y, _, _ = _prepare_training_data(train_csv, k)
    estimator = _get_estimator("logreg", **kwargs)
    estimator, evaluation = _train_and_optionally_evaluate(
        estimator, X, y, test_size, random_state
    )
    _save_model(model_path, estimator, k=k, method="logreg", metrics=evaluation)
    return evaluation.to_dict() if evaluation else None


def train_model_svm(
    train_csv: str,
    model_path: str,
    k: int = 2,
    *,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Optional[Dict[str, float]]:
    """Train an SVM model and save it to disk."""

    _ensure_positive_k(k)
    X, y, _, _ = _prepare_training_data(train_csv, k)
    estimator = _get_estimator("svm", **kwargs)
    estimator, evaluation = _train_and_optionally_evaluate(
        estimator, X, y, test_size, random_state
    )
    _save_model(model_path, estimator, k=k, method="svm", metrics=evaluation)
    return evaluation.to_dict() if evaluation else None


def train_model_rf(
    train_csv: str,
    model_path: str,
    k: int = 2,
    *,
    n_estimators: int = 100,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Optional[Dict[str, float]]:
    """Train a random forest model and save it to disk."""

    _ensure_positive_k(k)
    X, y, _, _ = _prepare_training_data(train_csv, k)
    estimator = _get_estimator("rf", n_estimators=n_estimators, **kwargs)
    estimator, evaluation = _train_and_optionally_evaluate(
        estimator, X, y, test_size, random_state
    )
    _save_model(model_path, estimator, k=k, method="rf", metrics=evaluation)
    return evaluation.to_dict() if evaluation else None


def train_model_esm(
    train_csv: str,
    model_path: str,
    model_name: str = "esm2_t6_8M_UR50D",
    *,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Optional[Dict[str, float]]:
    """Train logistic regression using ESM embeddings."""

    df = _load_dataframe(train_csv)
    _validate_columns(df, TRAIN_REQUIRED_COLUMNS)
    embedder = ESMEmbedder(model_name)
    data = [
        embedder.pair_embedding(row["tcr_sequence"], row["pmhc_sequence"])
        for _, row in df.iterrows()
    ]
    X = np.stack(data)
    y = df["label"].to_numpy()
    estimator = LogisticRegression(max_iter=1000, **kwargs)
    estimator, evaluation = _train_and_optionally_evaluate(
        estimator, X, y, test_size, random_state
    )
    _save_model(
        model_path,
        estimator,
        method="esm",
        metrics=evaluation,
        extra={ModelArtifactKeys.MODEL_NAME: model_name},
    )
    return evaluation.to_dict() if evaluation else None


def train_and_evaluate(
    train_csv: str,
    model_path: str,
    *,
    method: str = "logreg",
    k: int = 2,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    **kwargs,
) -> Dict[str, float]:
    """High-level helper to train any classical model with evaluation."""

    method = method.lower()
    if method == "esm":
        evaluation = train_model_esm(
            train_csv,
            model_path,
            test_size=test_size,
            random_state=random_state,
            **kwargs,
        )
        if evaluation is None:
            raise ValueError("Evaluation requires a non-zero test_size")
        return evaluation

    _ensure_positive_k(k)
    X, y, _, _ = _prepare_training_data(train_csv, k)
    estimator = _get_estimator(method, **kwargs)
    estimator, evaluation = _train_and_optionally_evaluate(
        estimator, X, y, test_size, random_state
    )
    if evaluation is None:
        raise ValueError("Evaluation requires a non-zero test_size")
    _save_model(model_path, estimator, k=k, method=method, metrics=evaluation)
    return evaluation.to_dict()


def _load_model_artifacts(model_path: str) -> Dict[str, object]:
    return load(model_path)


def predict(predict_csv: str, model_path: str, output_csv: str) -> None:
    """Predict interaction probabilities for new pairs."""

    df = _load_dataframe(predict_csv)
    _validate_columns(df, PREDICT_REQUIRED_COLUMNS)

    params = _load_model_artifacts(model_path)
    clf = params[ModelArtifactKeys.MODEL]
    k = params.get(ModelArtifactKeys.KMER)
    if k is None:
        raise ValueError("Model file does not contain k-mer configuration")

    X = build_feature_matrix(df, k)
    probs = clf.predict_proba(X)[:, 1]
    df["prediction"] = probs
    df.to_csv(output_csv, index=False)


def predict_esm(predict_csv: str, model_path: str, output_csv: str) -> None:
    """Predict using an ESM-based logistic regression model."""

    df = _load_dataframe(predict_csv)
    _validate_columns(df, PREDICT_REQUIRED_COLUMNS)

    params = _load_model_artifacts(model_path)
    clf = params[ModelArtifactKeys.MODEL]
    model_name = params[ModelArtifactKeys.MODEL_NAME]
    embedder = ESMEmbedder(model_name)
    data = [
        embedder.pair_embedding(row["tcr_sequence"], row["pmhc_sequence"])
        for _, row in df.iterrows()
    ]
    X = np.stack(data)
    probs = clf.predict_proba(X)[:, 1]
    df["prediction"] = probs
    df.to_csv(output_csv, index=False)


def load_metrics(model_path: str) -> Optional[Dict[str, float]]:
    """Load stored evaluation metrics from a trained model file."""

    params = _load_model_artifacts(model_path)
    metrics = params.get(ModelArtifactKeys.METRICS)
    if metrics is None:
        return None
    return dict(metrics)
