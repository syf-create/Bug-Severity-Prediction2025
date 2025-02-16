# CS-MSFL

## Environment Setup:

For Getting Started:

-   Operating System: The provided artifact is tested on Windows.
-   GPU: It is better to have a GPU for running experiments on GPU otherwise it may take a long time.
-   CPU/RAM: There is no strict minimum on these.
-   Python: Python 3 is required.
-   dependencies: The requirements.txt file is referable.

## data

The `data` folder contains bugs from Defects4tJ and Bugs.jar datasets. This folder contains a preprocessing script that unify bug severity values, scale the source code metrics and create `train`, `val`, and `test` splits.

Running this script using `bash preprocessing.sh` command generates 6 files containing `train`, `val`, `tests` splits in `jsonl` (compatible with CodeBERT experiments) and `csv` (compatible with source code metrics experiments) formats.

Running data preprocessing
    -   `cd BugSeverityPrediction2025-main/experiments/data`
    -   `bash preprocessing.sh`
    -   Copy generated `jsonl` and `csv` files into the dataset folder

## dataset

Files available in the `dataset` folder represent data for the getting started section (small subset of data). For reproducing paper results the generated files in the `data` folder should be copied to the `dataset` folder that is used by the model training scripts.

## models

This folder contains all code and scripts for all of the experiments including classic models, CodeBERT models.

## Running Source Code Metrics Models Experiments

1.  `cd BugSeverityPrediction2025-main/experiments/models/code_metrics`
2.  `bash train_test.sh`
3.  Results are generated in the `log` folder

## Running CS-MSFL Model Experiments

1.  `cd BugSeverityPrediction2025-main/experiments/models/code_representation/codebert`
2.  Set `CodeBERT` as the `model_arch` parameter's value in `train.sh` file
3.  `bash train.sh` for training the model
4.  `bash inference.sh` for evaluating the model with the `test` split
5.  Results are generated in the `log` folder。

## How to run with different config/hyperparameters?

-   You can change/add different hyperparameters/configs in `train.sh` and `inference.sh` files.
