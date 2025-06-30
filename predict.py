#!/usr/bin/env python
import argparse
from pmhctcr_predictor import model as logreg_model


def main():
    parser = argparse.ArgumentParser(description="Predict pMHC-TCR interaction")
    parser.add_argument("input_csv", help="CSV file with tcr_sequence and pmhc_sequence")
    parser.add_argument("model_path", help="Trained model path")
    parser.add_argument("output_csv", help="Destination for predictions")
    parser.add_argument(
        "--method",
        choices=["logreg", "deep", "esm"],
        default="logreg",
        help="Model type",
    )
    args = parser.parse_args()
    if args.method == "logreg":
        logreg_model.predict(args.input_csv, args.model_path, args.output_csv)
    elif args.method == "esm":
        logreg_model.predict_esm(args.input_csv, args.model_path, args.output_csv)
    else:
        from pmhctcr_predictor import deep_model
        deep_model.predict(args.input_csv, args.model_path, args.output_csv)


if __name__ == "__main__":
    main()
