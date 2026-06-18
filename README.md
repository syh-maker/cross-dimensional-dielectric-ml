# Cross-dimensional dielectric-constant machine learning

This repository contains the data, code, trained models, and supplementary material for the manuscript:

**Cross-dimensional machine learning of in-plane dielectric constants in two-dimensional materials from three-dimensional van der Waals crystals**

The workflow predicts the in-plane dielectric response of two-dimensional materials. It trains and validates machine-learning models for the electronic, ionic, and total dielectric contributions:

```text
e_electronic
e_ionic
e_total
```

It also evaluates the total dielectric constant obtained by summing the independently predicted electronic and ionic contributions:

```text
e_total_sum = e_electronic_pred + e_ionic_pred
```

The repository includes processed feature data, label data, external-validation data, trained `.joblib` models, prediction results, and supplementary material.

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

Run the full training, holdout testing, and external-validation workflow from the repository root:

```bash
python src/train_and_validate.py
```

The script reads the processed data, trains the models, evaluates the development set, holdout test set, and external-validation set, and writes prediction tables, summary metrics, selected features, hyperparameters, and trained model files.

For a shorter test run:

```bash
python src/train_and_validate.py --trials 10
```

To regenerate the processed feature files before training:

```bash
python src/train_feature_engineering.py
python src/external_validation_feature_engineering.py
```

All input/output paths and main settings can be changed from the command line. To view available options:

```bash
python src/train_and_validate.py --help
```

## Outputs

The main prediction tables and metrics are written to:

```text
outputs/
```

The trained model files are written to:

```text
models/
```

or to the model-output directory specified in the training script.

## Citation

If this repository is used in academic work, please cite the associated manuscript:

```text
Cross-dimensional machine learning of in-plane dielectric constants in two-dimensional materials from three-dimensional van der Waals crystals
```

A complete citation will be added after publication.
