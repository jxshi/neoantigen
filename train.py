#!/usr/bin/env python
import argparse
from pmhctcr_predictor.model import train_model


def main():
    parser = argparse.ArgumentParser(description="Train pMHC-TCR interaction model")
    parser.add_argument("train_csv", help="CSV file with tcr_sequence, pmhc_sequence, label")
    parser.add_argument("model_path", help="Path to save trained model")
    parser.add_argument("--k", type=int, default=2, help="k-mer size")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularisation strength")
    parser.add_argument("--penalty", default="l2", help="Penalty for LogisticRegression")
    parser.add_argument("--solver", default="lbfgs", help="Solver for LogisticRegression")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations")
    parser.add_argument("--metric", choices=["accuracy", "auc"], default="accuracy", help="Metric to report")
    args = parser.parse_args()
    score = train_model(
        args.train_csv,
        args.model_path,
        k=args.k,
        C=args.C,
        penalty=args.penalty,
        solver=args.solver,
        max_iter=args.max_iter,
        metric=args.metric,
    )
    print(f"Training {args.metric}: {score:.4f}")


if __name__ == "__main__":
    main()
