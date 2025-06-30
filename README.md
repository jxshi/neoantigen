# Neoantigen Predictor

This repository provides a minimal Python implementation for training and
predicting peptide–MHC (pMHC) and T-cell receptor (TCR) interactions.

The package `pmhctcr_predictor` contains utilities to extract simple k‑mer
features from amino‑acid sequences and train a logistic regression model.
The provided scripts demonstrate how to train a model from a CSV file and
predict interaction probabilities for new sequence pairs.

## Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `torch` (for the optional deep learning model)

You can create a conda environment with all required packages using
`environment.yml`:

```bash
conda env create -f environment.yml
conda activate neoantigen
```

This installs PyTorch and the other dependencies listed above.

Install the dependencies via:

```bash
pip install -r requirements.txt
pip install torch  # required only for --method deep
pip install fair-esm  # required only for --method esm
```

Installing PyTorch may require selecting the appropriate version for your
platform. Refer to the [PyTorch installation guide](https://pytorch.org/) if you
encounter issues.

## Training

Prepare a CSV file containing the columns `tcr_sequence`, `pmhc_sequence` and
`label` (1 for interaction, 0 for no interaction). Then run:

```bash
python train.py train_data.csv model.joblib

# use `--method deep` to train the neural model
# python train.py train_data.csv deep_model.pt --method deep
# use `--method esm` to train with ESM embeddings
# python train.py train_data.csv esm_model.joblib --method esm
```

## Prediction

Given a CSV with `tcr_sequence` and `pmhc_sequence`, predict interaction
probabilities:

```bash
python predict.py pairs.csv model.joblib predictions.csv

# for the neural model use:
# python predict.py pairs.csv deep_model.pt predictions.csv --method deep
# for the ESM model use:
# python predict.py pairs.csv esm_model.joblib predictions.csv --method esm
```

The output file will contain the original columns plus a `prediction` column
with the predicted probability of interaction.

## Running Tests

Install the development dependencies and run the test suite with `pytest`:

```bash
pytest
```
