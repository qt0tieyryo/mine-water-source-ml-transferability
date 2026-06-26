# Code for Cross-mine External Validation and SHAP–KS Diagnostics in Mine Water Inrush Source Identification

This repository contains the analysis and figure-generation code for the manuscript:

**Cross-mine external validation and SHAP–KS diagnostics for transferable machine learning in mine water inrush source identification**

The code reproduces the cross-mine external-validation protocol and the SHAP–KS feature-shift diagnostics reported in the paper.

## Files

* `main_analysis_pipeline.py`: Main machine-learning pipeline for the 4 algorithms × 7 tuning strategies experiment (28 configurations).
* `shap_all_combinations_patch.py`: SHAP compatibility and export helper used by the main pipeline.
* `make_figures_submission.py`: Regenerates the manuscript result figures from the processed source-data tables in `Figure_Source_Data`.
* `requirements.txt`: Direct runtime dependencies.
* `requirements-lock.txt`: Full pinned environment, including transitive dependencies, for strict reproduction.
* `LICENSE`: MIT License.
* `.gitignore`: Prevents accidental upload of restricted data, local outputs, and temporary files.

## Data availability

The hydrochemical datasets used in this study, including the training set, internal test set, and external validation set, were obtained from mine hydrogeological surveys and are subject to data-sharing restrictions. They are not publicly available because they contain mine-specific hydrogeological information and are subject to data-provider restrictions.

The data may be made available from the corresponding authors upon reasonable request and with permission from the relevant data providers.

When the input data are available locally, they should be placed in an `Input_Data` directory containing:

* `train_set.xlsx`: Malan Mine training set, 153 samples.
* `test_set.xlsx`: Malan Mine internal test set, 39 samples.
* `external_validation_set.xlsx`: Jinci Spring Basin external validation set, 903 samples.

Column definitions are documented in `Input_Data/README.md`. The processed source-data tables needed to regenerate the figures are provided in `Figure_Source_Data`.

## Reproducing the main analysis

The full model-search workflow can be computationally expensive. To rerun the submitted protocol:

```bash
python main_analysis_pipeline.py --data_dir Input_Data --output_dir Recreated_Model_Output --protocol budget_matched_repeated
```

The external validation set is used only after the model-selection step, for locked descriptive external scoring. It is not used for hyperparameter search or model selection.

## Regenerating figures from source data

The processed source data used for the submitted figures are in `Figure_Source_Data`. To regenerate the main and supplementary result figures:

```bash
python make_figures_submission.py --source_dir Figure_Source_Data --which all
```

Outputs are written to `Figure_Recreated` and `Figure_Recreated_Supplementary`. The finalized submission copies are named `Fig1`–`Fig11` and `FigS1`–`FigS4`. The figure generator accepts both the original internal export names and the cleaned submission filenames used in `Figure_Source_Data`.

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

Gao, D.; Zhao, F.; Xu, B.; Li, S.; Yin, S.; Sun, H.; Cao, S. (2026). Cross-mine external validation and SHAP–KS diagnostics for transferable machine learning in mine water inrush source identification. Manuscript under review.

Archived release: Zenodo, https://doi.org/10.5281/zenodo.XXXXXXX

## License

Released under the MIT License. See the `LICENSE` file.
