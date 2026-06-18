# Dielectric-response machine learning

This repository provides a reproducible machine-learning workflow for training and validating models that predict `e_electronic`, `e_ionic`, and `e_total`. It also reports `e_total` obtained by summing the electronic and ionic predictions.

## Install

Create the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate dielectric-ml
```

Alternatively, install the Python dependencies with pip:

```bash
python -m pip install -r requirements.txt
```

## Run

Run training and validation from the repository root:

```bash
python src/train_and_validate.py
```

The command reads the processed data in `data/processed`, writes prediction tables to `outputs`, and writes trained model files to `models`.

For a shorter test run:

```bash
python src/train_and_validate.py --trials 10
```

To regenerate the processed feature files first:

```bash
python src/train_feature_engineering.py
python src/external_validation_feature_engineering.py
```

All paths can be overridden with the command-line options shown by:

```bash
python src/train_and_validate.py --help
```
