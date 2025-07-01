#!/usr/bin/env python
import argparse
from pmhctcr_predictor import model as logreg_model


def main():
    parser = argparse.ArgumentParser(description="Train pMHC-TCR interaction model")
    parser.add_argument("train_csv", help="CSV file with tcr_sequence, pmhc_sequence, label")
    parser.add_argument("model_path", help="Path to save trained model")
    parser.add_argument("--k", type=int, default=2, help="k-mer size for logistic regression")
    parser.add_argument(
        "--method",
        choices=["logreg", "svm", "rf", "deep", "esm"],
        default="logreg",
        help="Training method",
    )
    args = parser.parse_args()
    if args.method == "logreg":
        logreg_model.train_model(args.train_csv, args.model_path, args.k)
    elif args.method == "svm":
        logreg_model.train_model_svm(args.train_csv, args.model_path, args.k)
    elif args.method == "rf":
        logreg_model.train_model_rf(args.train_csv, args.model_path, args.k)
    elif args.method == "esm":
        logreg_model.train_model_esm(args.train_csv, args.model_path)
    else:
        from pmhctcr_predictor import deep_model
        deep_model.train_model(args.train_csv, args.model_path)


if __name__ == "__main__":
    main()
