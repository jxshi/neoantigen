#!/usr/bin/env python
import argparse
from pmhctcr_predictor.model import predict


def main():
    parser = argparse.ArgumentParser(description="Predict pMHC-TCR interaction")
    parser.add_argument("input_csv", help="CSV file with tcr_sequence and pmhc_sequence")
    parser.add_argument("model_path", help="Trained model path")
    parser.add_argument("output_csv", help="Destination for predictions")
    args = parser.parse_args()
    predict(args.input_csv, args.model_path, args.output_csv)


if __name__ == "__main__":
    main()
