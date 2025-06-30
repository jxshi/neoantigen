# Neoantigen Predictor

This repository contains utilities for predicting interaction probabilities between peptide--MHC (pMHC) sequences and T--cell receptor (TCR) sequences. The main package `pmhctcr_predictor` implements simple k‑mer features, an optional neural model, and logistic regression on pretrained ESM embeddings. Command line scripts are provided for training models and running predictions.

## Requirements

- Python **3.8** or later
- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
- `pytest` (for running the tests)
- `torch` (required only for the deep model)
- `fair-esm` (required only for the ESM model)

All dependencies are listed in `requirements.txt` and `environment.yml`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-user>/neoantigen.git
cd neoantigen
```

### 2. Using Conda (recommended)

Create a new environment with all required packages, including PyTorch:

```bash
conda env create -f environment.yml
conda activate neoantigen
```

If your `.condarc` file overrides channels, add `--override-channels` to the command.

### 3. Using pip

Alternatively, create a virtual environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install optional extras as needed:

```bash
pip install torch      # for --method deep
pip install fair-esm   # for --method esm
```

### 4. Install package for development

(Optional but recommended for testing.)

```bash
pip install -e .
```

## Training

Prepare a CSV file with the columns `tcr_sequence`, `pmhc_sequence`, and `label` (1 for interacting, 0 for non‑interacting). An example file is available at `tests/data/sample_train.csv`.

Train a logistic regression model using k‑mer features:

```bash
python train.py <train.csv> <model.joblib>
```

Set the k‑mer size (default is 2):

```bash
python train.py <train.csv> <model.joblib> --k 3
```

Use the neural model:

```bash
python train.py <train.csv> <model.pt> --method deep
```

Use logistic regression on ESM embeddings:

```bash
python train.py <train.csv> <esm_model.joblib> --method esm
```

## Prediction

Create a CSV containing `tcr_sequence` and `pmhc_sequence` for each pair you wish to score. Run prediction with the model type that matches how the model was trained.

For the logistic regression model:

```bash
python predict.py <pairs.csv> <model.joblib> <predictions.csv>
```

For the neural model:

```bash
python predict.py <pairs.csv> <model.pt> <predictions.csv> --method deep
```

For the ESM model:

```bash
python predict.py <pairs.csv> <esm_model.joblib> <predictions.csv> --method esm
```

The output CSV will contain the input columns plus a `prediction` column holding the probability that each TCR binds the pMHC.

## Interpreting Results

The `prediction` column in the output file is a probability between 0 and 1.
Values close to **1** indicate a high likelihood that the TCR and pMHC interact,
while values near **0** suggest little to no interaction. Thresholds for calling
an interaction depend on the use case, but a common approach is to treat values
above 0.5 as positive and those below 0.5 as negative. You can sort the output
by the `prediction` column to identify the most promising pairs for further
analysis.

## Running Tests

After installing the package in editable mode, run the full test suite:

```bash
pytest
```

## License

This project is licensed under the [MIT License](LICENSE).
