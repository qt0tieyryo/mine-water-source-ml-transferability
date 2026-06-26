# Code for "Limited cross-mine transferability of machine learning models for mine water inrush source identification: an independent external validation"

[![DOI](https://zenodo.org/badge/1281273287.svg)](https://doi.org/10.5281/zenodo.20928385)

This repository contains the analysis and figure-generation code for the manuscript:

**Limited cross-mine transferability of machine learning models for mine water inrush source identification: an independent external validation**

The code reproduces the cross-mine external-validation protocol and the SHAP–KS feature-shift diagnostics reported in the paper.

## Files

* `main_analysis_pipeline.py`: Main machine-learning pipeline for the 4 algorithms × 7 tuning strategies experiment (28 configurations).
* `shap_all_combinations_patch.py`: SHAP compatibility and export helper used by the main pipeline.
* `make_figures_submission.py`: Generates the manuscript figures when the processed source-data tables are available locally.
* `requirements.txt`: Direct runtime dependencies.
* `requirements-lock.txt`: Full pinned environment, including transitive dependencies, for strict reproduction.
* `LICENSE`: MIT License.
* `.gitignore`: Prevents accidental upload of restricted data, local outputs, and temporary files.

## Data availability

The input datasets and sample-level feature arrays are not included in this public repository because they contain mine-specific hydrogeological information and are subject to data-provider restrictions. Data access requests should be directed to the corresponding author Bin Xu (jinzigaofeng@126.com) and will be considered upon reasonable request and with permission from the relevant data providers.

When authorized data are available locally, they should be placed in an `Input_Data` directory containing `train_set.xlsx`, `test_set.xlsx`, and `external_validation_set.xlsx`. Processed figure-source files, if available locally, should be placed in a `Figure_Source_Data` directory.

## Reproducing the main analysis

The full model-search workflow can be computationally expensive. To rerun the submitted protocol (with authorized input data placed in a local `Input_Data` directory):

```bash
python main_analysis_pipeline.py --data_dir Input_Data --output_dir Recreated_Model_Output --protocol budget_matched_repeated
```

The external validation set is used only after the model-selection step, for locked descriptive external scoring. It is not used for hyperparameter search or model selection.

## Regenerating figures from source data

The figure-generation script is provided to document and reproduce the plotting workflow. Complete regeneration of all submitted figures requires processed source-data files in a local `Figure_Source_Data` directory. These files are not included in the public repository because some figure inputs are derived from restricted hydrochemical datasets.

```bash
python make_figures_submission.py --source_dir Figure_Source_Data --which all
```

Outputs are written to `Figure_Recreated` and `Figure_Recreated_Supplementary`. The finalized submission copies are named `Fig1`–`Fig11` and `FigS1`–`FigS4`. The figure generator accepts both the original internal export names and the cleaned submission filenames.

## Environment

The analysis was run with the following environment:

* Python 3.14.4
* scikit-learn 1.8.0
* XGBoost 3.2.0
* LightGBM 4.6.0
* Optuna 4.8.0
* SHAP 0.51.0
* NumPy 2.4.4
* pandas 3.0.2
* SciPy 1.17.1
* matplotlib 3.10.9
* seaborn 0.13.2
* openpyxl 3.1.5
* joblib 1.5.3

Install the direct runtime dependencies with:

```bash
pip install -r requirements.txt
```

For an exact reproduction of the reported environment, including all transitive dependencies, use:

```bash
pip install -r requirements-lock.txt
```

The optional `--feature_method boruta` branch requires the external `boruta` package and was not part of the submitted default protocol. PyMuPDF is optional and is used only for PDF vector-quality checks in `make_figures_submission.py`; figure generation still runs when PyMuPDF is unavailable.

## Notes

* Random seeds are set inside the main pipeline for the repeated locked-evaluation protocol.
* SVM features are standardized using training-set parameters only.
* Tree-based models use the original feature scale.
* The external validation set is never used for tuning or feature selection.
* Model outputs are intended to support candidate source ranking and expert review, not to provide stand-alone final source calls.

## Citation

If you use this code, please cite the manuscript and this archived repository:

Gao, D.; Zhao, F.; Xu, B.; Li, S.; Yin, S.; Sun, H.; Cao, S. (2026). Limited cross-mine transferability of machine learning models for mine water inrush source identification: an independent external validation. Manuscript under review.

Archived release: Zenodo, https://doi.org/10.5281/zenodo.20928385

## License

Released under the MIT License. See the `LICENSE` file.
