#!/usr/bin/env python
import argparse
from pmhctcr_predictor import model as logreg_model
from pmhctcr_predictor import deep_model


def main():
    parser = argparse.ArgumentParser(description="Train pMHC-TCR interaction model")
    parser.add_argument("train_csv", help="CSV file with tcr_sequence, pmhc_sequence, label")
    parser.add_argument("model_path", help="Path to save trained model")
    parser.add_argument("--k", type=int, default=2, help="k-mer size for logistic regression")
    parser.add_argument("--method", choices=["logreg", "deep"], default="logreg", help="Training method")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for the deep model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for the deep model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the deep model")
    args = parser.parse_args()
    if args.method == "logreg":
        logreg_model.train_model(args.train_csv, args.model_path, args.k)
    else:
        deep_model.train_model(
            args.train_csv,
            args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
