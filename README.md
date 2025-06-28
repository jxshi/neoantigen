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

Install the dependencies via:

```bash
pip install -r requirements.txt
```

## Training

Prepare a CSV file containing the columns `tcr_sequence`, `pmhc_sequence` and
`label` (1 for interaction, 0 for no interaction). Then run:

```bash
python train.py train_data.csv model.joblib
```

## Prediction

Given a CSV with `tcr_sequence` and `pmhc_sequence`, predict interaction
probabilities:

```bash
python predict.py pairs.csv model.joblib predictions.csv
```

The output file will contain the original columns plus a `prediction` column
with the predicted probability of interaction.
