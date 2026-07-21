# Code and Reproducibility Package

Manuscript: **Mine-specific external validation and class-conditional shift diagnosis for hydrochemical machine-learning models in mine water inrush source identification**

Target journal: **Remote Sensing**

## Files

- `main_analysis_pipeline.py`: main 4-algorithm × 7-tuning-strategy pipeline and locked internal/external evaluation.
- `shap_all_combinations_patch.py`: robust multiclass SHAP compatibility and export helper.
- `per_mine_analysis.py`: mine-specific performance, class-conditional KS analysis, and weighting-sensitivity analysis.
- `target_mine_adaptation.py`: secondary supervised target-mine adaptation experiment for Xiqu and Tunlan.
- `make_figures_submission.py`: generator for manuscript Figures 3–8 and Supplementary Figures S1–S6.
- `requirements.txt`: exact Python package versions reported in Supplementary Materials, Table S3.

## Locked dataset design

- Malan training set: 153 samples (O/G/T/P = 11/81/44/17).
- Malan internal test set: 39 samples (O/G/T/P = 3/21/11/4).
- Independent external validation set: 755 samples from 12 mines (O/G/T/P = 23/526/81/125).

The external validation set is used only for locked scoring and post-fit interpretation after preprocessing, hyperparameter search, and model selection have been completed in the Malan training domain. The target-mine adaptation experiment is secondary and does not affect the primary external rankings.

## Reproducibility boundary

The raw hydrochemical workbooks are not distributed publicly because they contain mine-specific hydrogeological information and are subject to data-provider restrictions. Full rerunning of the analysis requires the restricted input workbooks in a local input directory.

Expected default input files:

```text
../Input_Data/train_set.xlsx
../Input_Data/test_set.xlsx
../Input_Data/external_validation_set.xlsx
```

The manuscript and Supplementary Data S1 provide non-sensitive summary results needed to audit the reported performance, domain-shift, class-conditional KS, and target-mine adaptation findings. Complete regeneration of every figure also requires the processed figure-source archive used by `make_figures_submission.py`.

## Typical commands

Main locked analysis:

```bash
python main_analysis_pipeline.py \
  --data_dir ../Input_Data \
  --output_dir ../Recreated_Model_Output \
  --protocol budget_matched_repeated
```

Mine-specific diagnostics:

```bash
python per_mine_analysis.py \
  --data_dir ../Input_Data \
  --output_dir ../Recreated_Model_Output \
  --mine_col auto
```

Target-mine adaptation:

```bash
python target_mine_adaptation.py \
  --data_dir ../Input_Data \
  --output_dir ../Recreated_Model_Output \
  --mines auto \
  --model RF-Default \
  --n_repeats 30 \
  --calib_frac 0.5 \
  --seed 20240101
```

Submission figures:

```bash
python make_figures_submission.py \
  --source_dir ../Figure_Source_Data \
  --which all
```

## Target-mine adaptation safeguards

`target_mine_adaptation.py`:

- uses only the 153-sample Malan training set as the source-domain baseline;
- keeps local adaptation and held-out evaluation subsets disjoint in every repeated split;
- represents class imbalance once through balanced sample weights, with `RF class_weight=None`;
- recomputes balanced sample weights for each model's actual training set;
- enforces requested adaptation sample counts through largest-remainder allocation where feasible; and
- writes split-level records, summaries, and the sample-size sweep used for Supplementary Figure S6 and Tables S11–S13.

## Supplementary-file mapping

- Supplementary Materials: Texts S1–S4, Tables S1–S3, and Figures S1–S6.
- Supplementary Data S1: complete Tables S4–S13.

## Repository release and DOI

Keep the article title unchanged in the manuscript, supplementary files, repository description, README, and archive metadata. Create or update the public release only after the manuscript, supplementary files, code, processed non-sensitive tables, and figure files are frozen. Do not upload restricted raw workbooks unless the data providers approve redistribution.
