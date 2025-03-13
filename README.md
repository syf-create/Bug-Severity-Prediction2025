# CS-MSFL

## Environment Setup:

For Getting Started:

-   Operating System: The provided artifact is tested on Windows.
-   GPU: It is better to have a GPU for running experiments on GPU otherwise it may take a long time.
-   CPU/RAM: There is no strict minimum on these.
-   Python: Python 3 is required.
-   dependencies: The requirements.txt file is referable.

## dataset

The dataset is derived by preprocessing the defect data from the Defects4J and Bugs.jar datasets, which includes unifying defect severity values and scaling source code metrics. This processed dataset is suitable for model training, validation, and testing.   
## models

This folder contains all code and scripts for all of the experiments including classic models, CodeBERT、MS_CodeBert.

## Running Source Code Metrics Models Experiments

1.  `cd BugSeverityPrediction2025-main/experiments/models/code_metrics`
2.  `bash train_test.sh`
3.  Results are generated in the `log` folder

## Running CS-MSFL Model Experiments

1.  `cd BugSeverityPrediction2025-main/experiments/models/code_representation/codebert`
2.  Set `MS_CodeBert` as the `model_arch` parameter's value in `train.sh` file
3.  `bash train.sh` for training the model
4.  `bash inference.sh` for evaluating the model with the `test` split
5.  Results are generated in the `log` folder。

## Run Ablation Experiments

### Ablation Experiment 1: Evaluating the effect of excluding the MSIF module (with the CWO module included)
1.  `cd BugSeverityPrediction2025-main/experiments/models/code_representation/codebert`
2.  Set `CodeBERT` as the `model_arch` parameter's value in `train.sh` file
3.  `bash train.sh` for training the model
4.  `bash inference.sh` for evaluating the model with the `test` split
5.  Results are generated in the `log` folder。

### Ablation Experiment 2: Evaluating the effect of excluding the CWO module
Method: Edit the main function and comment out or delete the call to WeightRefiner.
1.  `cd BugSeverityPrediction2025-main/experiments/models/code_representation/codebert`
2.  Set `MS_CodeBert` as the `model_arch` parameter's value in `train.sh` file
3.  `bash train.sh` for training the model
4.  `bash inference.sh` for evaluating the model with the `test` split
5.  Results are generated in the `log` folder。

## How to run with different config/hyperparameters?

-   You can change/add different hyperparameters/configs in `train.sh` and `inference.sh` files.
