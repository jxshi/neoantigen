#!/usr/bin/env python
import argparse
from pmhctcr_predictor.model import train_model


def main():
    parser = argparse.ArgumentParser(description="Train pMHC-TCR interaction model")
    parser.add_argument("train_csv", help="CSV file with tcr_sequence, pmhc_sequence, label")
    parser.add_argument("model_path", help="Path to save trained model")
    parser.add_argument("--k", type=int, default=2, help="k-mer size")
    args = parser.parse_args()
    train_model(args.train_csv, args.model_path, args.k)


if __name__ == "__main__":
    main()
