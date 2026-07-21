# Code and Reproducibility Package

This repository contains the analysis and figure-generation code associated with the manuscript:

**Mine-specific external validation and class-conditional shift diagnosis for hydrochemical machine-learning models in mine water inrush source identification**

## Files

- `main_analysis_pipeline.py`: main machine-learning pipeline for the
  4-algorithm × 7-tuning-strategy experiment, comprising 28 model
  configurations.

- `shap_all_combinations_patch.py`: SHAP compatibility, calculation, and export
  utilities used by the main analysis pipeline.

- `per_mine_analysis.py`: mine-specific external-validation analysis,
  class-conditional Kolmogorov-Smirnov diagnostics, and exploratory
  mine-level shift-performance analysis.

- `target_mine_adaptation.py`: secondary target-mine adaptation experiment for
  Xiqu and Tunlan using disjoint local adaptation and held-out evaluation
  subsets.

- `make_figures_submission.py`: generator for the manuscript and supplementary
  figures based on the processed figure-source data.

- `requirements.txt`: pinned Python package requirements for the analysis
  environment.

## Experimental Design

The primary analysis compares four machine-learning algorithm families with
seven tuning strategies:

- Algorithms: Random Forest, XGBoost, LightGBM, and SVM.
- Tuning strategies: Default, Optuna, GridSearch, PSO, SSA, DE, and GWO.
- Total model configurations: 4 × 7 = 28.
- Repeated locked evaluations: 30 runs per configuration.
- Total locked-evaluation records: 28 × 30 = 840.

## Dataset Design

- Malan training set: 153 samples  
  O/G/T/P = 11/81/44/17.

- Malan internal test set: 39 samples  
  O/G/T/P = 3/21/11/4.

- Independent external-validation set: 755 samples from 12 mines  
  O/G/T/P = 23/526/81/125.

The external-validation dataset is used only for locked descriptive scoring and
post-hoc interpretation after model fitting and hyperparameter selection have
been completed within the Malan training domain. It does not enter
hyperparameter search, preprocessing estimation, threshold selection, or
primary model ranking.

The target-mine adaptation analysis is a separate secondary experiment. It does
not alter the model rankings obtained in the primary locked external-validation
analysis.

## Main Analysis

The main pipeline can be run with:

```bash
python main_analysis_pipeline.py \
    --data_dir ../Input_Data \
    --output_dir ../Recreated_Model_Output \
    --protocol budget_matched_repeated
