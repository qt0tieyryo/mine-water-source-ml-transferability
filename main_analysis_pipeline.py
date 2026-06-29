"""
Mine Water Source Identification - Machine Learning Pipeline

Comparison : 4 algorithms x 7 optimizers (28 combinations)
                 RF / XGBoost / LightGBM / SVM-RBF
                 x Default / Optuna / GridSearch / PSO / SSA / DE / GWO
Validation : Internal test set + locked external cross-mine validation set.
             External validation is loaded only after model selection is
             complete and is reported as post-hoc descriptive evidence of
             domain transfer; it never enters hyperparameter search or
             model ranking (Cawley & Talbot, 2010).
Target : 4-class mine water source identification
                 O = Ordovician limestone, G = goaf water,
                 T = Taiyuan limestone, P = Permian sandstone fissure water

Pipeline structure (controlled by --protocol):
  1. Load training and internal-test Excel files.
  2. Geochemical feature engineering, fixed canonical label encoding, and
     fold-internal feature selection / imputation / scaling.
  3. Algorithm x optimizer search and evaluation under one of two protocols:
       - nested_generalization: repeated stratified nested cross-validation
         (5 outer folds x N outer repeats; inner 5 x M repeats) over all 28
         combinations, followed by a full-training final-evaluation phase
         on the held-out internal test set and the external validation set
         across --final_eval_runs random seeds.
       - budget_matched_repeated (default): a compute-budget-matched
         repeated single-search counterpart, evaluated over the same
         final-evaluation seeds for like-for-like comparison.
  4. Tabular and figure artefacts are written under --output_dir; a JSON
     validation-policy log records the descriptive-only role of the
     external validation set.

Usage:
    python main_analysis_pipeline.py \\
        --data_dir ../Input_Data \\
        --output_dir ../Recreated_Model_Output \\
        --protocol budget_matched_repeated

    python make_figures_submission.py --source_dir ../Figure_Source_Data --which all

Dependencies: numpy, pandas, scikit-learn, lightgbm, xgboost, optuna,
              shap, scipy, matplotlib, seaborn, openpyxl, joblib
"""

# ================================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ================================================================================

import numpy as np
import pandas as pd
# shap.summary_plot() internally calls plt.show() / draws to screen, which
# triggers tkinter's event loop. When called from a non-main thread (joblib
# parallel workers, shap internals) tkinter raises:
# RuntimeError: main thread is not in main loop
# Tcl_AsyncDelete: async handler deleted by the wrong thread ->process crash
# Agg is a purely file-based backend (no GUI window) and is thread-safe.
# Must be set before `import matplotlib.pyplot`  - setting it after has no effect.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import multiprocessing
import random
import json
import time
import joblib
from datetime import datetime
from pathlib import Path
import argparse

# Machine Learning
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, train_test_split,
    ParameterGrid
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    matthews_corrcoef, precision_recall_fscore_support
    # precision_score, recall_score, roc_auc_score omitted (unused in pipeline)
    # roc_curve+auc from scipy are used instead; precision/recall not reported.
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import VarianceThreshold

# Boosting Models (kept as baselines)
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Statistical Tests
from scipy import stats
from scipy.stats import wilcoxon

# Advanced Visualization
import shap
from shap_all_combinations_patch import (
    compute_shap_robust,
    mean_abs_shap,
    generate_shap_all_combinations,
)

# Deep Learning imports omitted  - small training-domain sample size (n=192:
# 153 train + 39 internal test) precludes reliable neural network training.

warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
warnings.filterwarnings('ignore', message='.*n_jobs.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*lbfgs.*', category=UserWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='optuna')
warnings.filterwarnings(
    'ignore',
    message='.*X does not have valid feature names.*',
    category=UserWarning,
    module='sklearn',
)
# shap.summary_plot() / summary_legacy() internally seed numpy's global RNG,
# triggering this warning in numpy >= 2.0. The behaviour is harmless for our
# use case (SHAP jitter in beeswarm plots); the warning is a shap library issue.
warnings.filterwarnings(
    'ignore',
    message='.*NumPy global RNG was seeded.*',
    category=FutureWarning,
)
warnings.filterwarnings(
    'ignore',
    message=r'.*np.random.seed.*',
    category=FutureWarning,
)
warnings.filterwarnings(
    'ignore',
    message=r'.*sklearn\.utils\.parallel\.delayed.*sklearn\.utils\.parallel\.Parallel.*',
    category=UserWarning,
    module='sklearn.utils.parallel',
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ================================================================================
# CONSTANTS
# ================================================================================

SEEDS = [
    42, 123, 456, 789, 2024, 333, 555, 777, 999, 1111,
    1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240,
    11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432, 19456, 20480
]

# Parallelism policy:
# Full-run mode uses all available CPU cores by default. Override at runtime with
# --n_jobs / --gridsearch_n_jobs, or environment variables MINEWATER_N_JOBS and
# MINEWATER_GRIDSEARCH_N_JOBS. Passing -1/all/auto also maps to all CPU cores.
def _available_cpu_count():
    try:
        return max(1, int(multiprocessing.cpu_count()))
    except Exception:
        return max(1, int(os.cpu_count() or 1))


def _normalise_n_jobs(value=None, default=None):
    if value is None or str(value).strip() == '':
        return int(default if default is not None else _available_cpu_count())
    raw = str(value).strip().lower()
    if raw in {'-1', 'all', 'auto', 'max'}:
        return _available_cpu_count()
    try:
        n_jobs = int(raw)
    except ValueError:
        return int(default if default is not None else _available_cpu_count())
    if n_jobs <= 0:
        return _available_cpu_count()
    return max(1, n_jobs)


DETERMINISTIC_N_JOBS = _normalise_n_jobs(
    os.environ.get('MINEWATER_N_JOBS'), _available_cpu_count())
GRIDSEARCH_N_JOBS = _normalise_n_jobs(
    os.environ.get('MINEWATER_GRIDSEARCH_N_JOBS'), DETERMINISTIC_N_JOBS)
SEARCH_MODEL_N_JOBS = _normalise_n_jobs(
    os.environ.get('MINEWATER_SEARCH_N_JOBS'), 1)


def configure_parallelism(model_n_jobs=None, gridsearch_n_jobs=None,
                          search_n_jobs=None):
    """Apply CLI/env thread settings before any estimators are built."""
    global DETERMINISTIC_N_JOBS, GRIDSEARCH_N_JOBS, SEARCH_MODEL_N_JOBS
    if model_n_jobs is not None:
        DETERMINISTIC_N_JOBS = _normalise_n_jobs(model_n_jobs, DETERMINISTIC_N_JOBS)
    if gridsearch_n_jobs is not None:
        GRIDSEARCH_N_JOBS = _normalise_n_jobs(gridsearch_n_jobs, GRIDSEARCH_N_JOBS)
    if search_n_jobs is not None:
        SEARCH_MODEL_N_JOBS = _normalise_n_jobs(search_n_jobs, SEARCH_MODEL_N_JOBS)


def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _float_env(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


# Inner-CV defaults are intentionally centralized so code, logs, audit tables and
# manuscript wording always use the same protocol definition.
INNER_CV_SPLITS_DEFAULT = max(2, _int_env('INNER_CV_SPLITS', 5))
INNER_CV_REPEATS_DEFAULT = max(1, _int_env('INNER_CV_REPEATS', 1))
CV_FAILURE_WARN_RATIO = min(1.0, max(0.0, _float_env('CV_FAILURE_WARN_RATIO', 0.10)))
OPTUNA_N_JOBS = max(1, _int_env('OPTUNA_N_JOBS', 1))

MAX_FEATURES_LUT = ['sqrt', 'log2', None]


def _decode_max_features_idx(idx):
    """Map continuous index in [0,1] to RF max_features categorical choice."""
    idx = float(np.clip(idx if idx is not None else 0.0, 0.0, 1.0))
    return MAX_FEATURES_LUT[min(2, int(idx * 3.0))]


def precompute_sample_weight(y):
    """Vectorized class-balanced sample weights for reproducible CV fitting."""
    y = np.asarray(y).astype(int)
    classes, inverse = np.unique(y, return_inverse=True)
    cls_weights = compute_class_weight('balanced', classes=classes, y=y)
    return np.asarray(cls_weights[inverse], dtype=float)


def _safe_cv_score(value, floor=-1.0):
    """Convert non-finite CV score to a deterministic floor for optimizers."""
    try:
        val = float(value)
    except Exception:
        return float(floor)
    return val if np.isfinite(val) else float(floor)


def _summarize_failed_folds(context, failed_folds, n_total):
    """Warn when CV fold failures are non-trivial and expose first errors."""
    n_failed = int(len(failed_folds or []))
    if n_total <= 0 or n_failed <= 0:
        return
    ratio = n_failed / float(n_total)
    if ratio >= CV_FAILURE_WARN_RATIO:
        preview = '; '.join(
            f"fold={f['fold']}:{f['err_type']}:{f['message']}"
            for f in failed_folds[:3]
        )
        warnings.warn(
            f"[{context}] {n_failed}/{n_total} CV folds failed "
            f"({ratio:.1%}). First errors: {preview}",
            RuntimeWarning,
        )


def _fmt_progress_value(value):
    try:
        if value is None or not np.isfinite(float(value)):
            return 'NA'
        return f"{float(value):.4f}"
    except Exception:
        return 'NA'


def _print_progress(message):
    print(f"[Progress] {message}", flush=True)


def _make_optuna_progress_callback(label, n_trials):
    def _callback(study, trial):
        complete_values = [
            float(t.value) for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        best_value = max(complete_values) if complete_values else None
        done = len(study.trials)
        state = getattr(trial.state, 'name', str(trial.state))
        _print_progress(
            f"{label} | Trial {done}/{int(n_trials)} | state={state} | "
            f"value={_fmt_progress_value(trial.value)} | "
            f"best={_fmt_progress_value(best_value)}")
    return _callback

# Ion column names  - order: [K+, Na+, Ca2+, Mg2+, Cl-, SO42-, HCO3-]
ION_COLS = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
# K+ Na+ Ca2+ Mg2+ Cl- SO42- HCO3-
# pH column - dimensionless, acts as an independent geochemical indicator
# (carbonate system equilibrium, redox proxy). Not used to derive ion ratios.
PH_COL = 'x8'

# Final 8-dim feature set for model training and SHAP interpretation.
# Seven field-measured major ions (mg/L) + pH (dimensionless).  Derived
# ratios and TDS proxies are intentionally excluded: (i) they are strict
# algebraic functions of the seven ions, inflating multicollinearity without
# adding information (Hem, 1985; Appelo & Postma, 2005); (ii) keeping
# interpretation tied to primary measurements improves reproducibility across
# laboratories and reviewers.
CORE_FEATURE_COLS = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

# LightGBM parameter whitelist (kept for LGBM baseline models)
LGBM_CLASSIFIER_PARAMS = {
    'boosting_type', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
    'subsample_for_bin', 'objective', 'class_weight', 'min_split_gain',
    'min_child_weight', 'min_child_samples',
    'subsample', 'subsample_freq', 'colsample_bytree',
    'reg_alpha', 'reg_lambda',
    'random_state', 'n_jobs', 'silent', 'importance_type', 'verbosity',
    'num_class',
}

LGBM_NATIVE_TO_SKLEARN_MAP = {
    'feature_fraction': 'colsample_bytree',
    'bagging_fraction': 'subsample',
    'bagging_freq' : 'subsample_freq',
    'lambda_l1' : 'reg_alpha',
    'lambda_l2' : 'reg_lambda',
}

# ================================================================================
# FEATURE DISPLAY NAMES (SCI-quality ion symbols for figures)
# ================================================================================
# Rules:
#   1. Data layer: do NOT rename columns  - keep x1, x2, etc.
#   2. Plot layer only: apply this mapping at each figure function entry point.
#   3. Rendering: mathtext ('$...$')  - no LaTeX installation required.
#      Set mpl.rcParams['mathtext.fontset'] = 'stix' to enable all ion symbols.
FEATURE_DISPLAY_NAMES = {
    'x1': r'$K^+$',
    'x2': r'$Na^+$',
    'x3': r'$Ca^{2+}$',
    'x4': r'$Mg^{2+}$',
    'x5': r'$Cl^-$',
    'x6': r'$SO_4^{2-}$',
    'x7': r'$HCO_3^-$',
    'x8': 'pH',
}


# ================================================================================
# UTILITY: SCI PUBLICATION STYLE
# ================================================================================
import matplotlib as _mpl
import matplotlib.pyplot as _plt

def _set_publication_style():
    """Unified SCI publication style  - call at the start of every figure function.
    Font fallback chain: Times New Roman ->DejaVu Serif ->STIX ->serif
    mathtext: stix fontset (no LaTeX install required for ion symbols)
    """
    _mpl.rcParams['font.family']           = 'serif'
    _mpl.rcParams['font.serif']            = [
        'Times New Roman', 'DejaVu Serif', 'STIX', 'Liberation Serif', 'serif'
    ]
    _mpl.rcParams['mathtext.fontset']      = 'stix'
    _mpl.rcParams['axes.unicode_minus']    = False
    _mpl.rcParams['axes.titlesize']        = 11
    _mpl.rcParams['axes.labelsize']        = 10
    _mpl.rcParams['xtick.labelsize']       = 9
    _mpl.rcParams['ytick.labelsize']       = 9
    _mpl.rcParams['legend.fontsize']       = 9
    _mpl.rcParams['figure.titlesize']      = 12
    _mpl.rcParams['axes.linewidth']        = 0.8
    _mpl.rcParams['xtick.major.width']    = 0.8
    _mpl.rcParams['ytick.major.width']    = 0.8
    _mpl.rcParams['xtick.major.size']      = 4
    _mpl.rcParams['ytick.major.size']      = 4
    _mpl.rcParams['lines.linewidth']       = 1.5
    _mpl.rcParams['figure.facecolor']      = 'white'
    _mpl.rcParams['axes.facecolor']       = 'white'
    _mpl.rcParams['savefig.facecolor']    = 'white'
    _mpl.rcParams['axes.titlelocation']   = 'center'
    _mpl.rcParams['axes.spines.top']      = False
    _mpl.rcParams['axes.spines.right']    = False


# ================================================================================
# UTILITY: SEED CONTROL
# ================================================================================

def _set_all_seeds(seed=42):
    """Set all random seeds for full reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ================================================================================
# PATH CONFIGURATION
# Prefer package-local input data; use MINEWATER_DATA_DIR or CLI arguments for
# external data locations.
def _default_data_dir():
    env_dir = os.environ.get('MINEWATER_DATA_DIR')
    if env_dir:
        return Path(env_dir).expanduser()
    package_input_dir = Path(__file__).resolve().parents[1] / 'Input_Data'
    if (package_input_dir / 'train_set.xlsx').exists() and (package_input_dir / 'test_set.xlsx').exists():
        return package_input_dir
    return Path(__file__).resolve().parent


def _first_existing_file(base_dir, filenames):
    base_dir = Path(base_dir)
    for filename in filenames:
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    return None


IDE_DEFAULT_TRAIN_PATH = None
IDE_DEFAULT_TEST_PATH = None
IDE_DEFAULT_VAL_PATH = None
IDE_DEFAULT_OUTPUT_DIR = None

# Primary metric used for internal model comparison and for descriptive
# reporting on the external mine. It MUST match the CV search objective
# (f1_macro) used by Optuna, GridSearch, and all population-based optimizers.
# Using a different metric for reporting than for search would introduce an
# optimisation-evaluation mismatch
# (Bartz-Beielstein et al., 2020, "Benchmarking in Optimization: Best
# Practice and Open Issues", arXiv:2007.03488).
GENERALISATION_PRIMARY_METRIC = 'f1_macro'
GENERALISATION_PRIMARY_METRIC_LABEL = {
    'accuracy': 'Accuracy',
    'f1_macro': 'F1-macro',
    'f1_weighted': 'F1-weighted',
}.get(GENERALISATION_PRIMARY_METRIC, GENERALISATION_PRIMARY_METRIC)

SVM_KERNEL_OPTIONS = ['linear', 'rbf']
SVM_CLASS_WEIGHT_OPTIONS = [None, 'balanced']


def _svm_kernel_from_idx(idx):
    """Map a continuous population-search coordinate onto a valid SVM kernel."""
    return SVM_KERNEL_OPTIONS[0] if float(idx) < 0.5 else SVM_KERNEL_OPTIONS[1]


def _svm_class_weight_from_idx(idx):
    """Map a continuous population-search coordinate onto a valid class_weight."""
    return SVM_CLASS_WEIGHT_OPTIONS[0] if float(idx) < 0.5 else SVM_CLASS_WEIGHT_OPTIONS[1]


def _normalize_svm_params(params=None):
    """Normalise SVM params from Default / Optuna / GridSearch / population searches."""
    raw = dict(params or {})
    if 'kernel_idx' in raw and 'kernel' not in raw:
        raw['kernel'] = _svm_kernel_from_idx(raw.pop('kernel_idx'))
    if 'class_weight_idx' in raw and 'class_weight' not in raw:
        raw['class_weight'] = _svm_class_weight_from_idx(raw.pop('class_weight_idx'))
    if 'C_log' in raw and 'C' not in raw:
        raw['C'] = float(10 ** np.clip(raw.pop('C_log'), -1.0, 3.0))
    if 'gamma_log' in raw and 'gamma' not in raw:
        raw['gamma'] = float(10 ** np.clip(raw.pop('gamma_log'), -4.0, 1.0))

    kernel = str(raw.get('kernel', 'rbf')).lower()
    if kernel not in SVM_KERNEL_OPTIONS:
        kernel = 'rbf'

    class_weight = raw.get('class_weight', 'balanced')
    if isinstance(class_weight, str) and class_weight.lower() == 'none':
        class_weight = None
    if class_weight not in SVM_CLASS_WEIGHT_OPTIONS:
        class_weight = 'balanced'

    normalised = {
        'kernel': kernel,
        'C': float(raw.get('C', 10.0)),
        'class_weight': class_weight,
    }
    if kernel == 'rbf':
        normalised['gamma'] = raw.get('gamma', 'scale')
    return normalised


def _build_svm_pipeline(params=None, fit_seed=42):
    """Shared SVM pipeline used by all protocol branches."""
    svm_params = _normalize_svm_params(params)
    svc_kwargs = {
        'kernel': svm_params['kernel'],
        'C': float(svm_params['C']),
        'class_weight': svm_params['class_weight'],
        'probability': True,
        'random_state': fit_seed,
    }
    if svm_params['kernel'] == 'rbf':
        svc_kwargs['gamma'] = svm_params.get('gamma', 'scale')
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(**svc_kwargs)),
    ])

TARGET_COLUMN_CANDIDATES = [
    'y',
    '充水水源',
    '水源',
    '水源类型',
    '类别',
    '标签',
    'label',
    'target',
    'water_source',
    'water_source_label',
    '充水水源',
]


def resolve_target_column(df, preferred=None):
    """Resolve the target column name robustly across Chinese/English variants."""
    if preferred and preferred in df.columns:
        return preferred
    normalized = {str(c).strip().lower(): c for c in df.columns}
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(TARGET_COLUMN_CANDIDATES)
    for cand in candidates:
        if cand in df.columns:
            return cand
        cand_norm = str(cand).strip().lower()
        if cand_norm in normalized:
            return normalized[cand_norm]
    raise KeyError(
        f"Target column not found. Available columns: {list(df.columns)} | "
        f"Accepted target candidates: {TARGET_COLUMN_CANDIDATES}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Mine Water Source Identification  - Algorithm x Optimizer Matrix Comparison'
    )
    parser.set_defaults(skip_full_shap_archive=False)
    parser.add_argument('--data_dir', type=str, required=False, default=None)
    parser.add_argument('--train_path', type=str, required=False, default=None)
    parser.add_argument('--test_path', type=str, required=False, default=None)
    parser.add_argument('--output_dir', type=str, required=False,
                        default=str(Path(__file__).parent / 'output6.28'))
    parser.add_argument('--protocol', type=str,
                        choices=['budget_matched_repeated', 'nested_generalization'],
                        default='budget_matched_repeated',
                        help='Experimental workflow. '
                             '"budget_matched_repeated" compares all 28 combinations under '
                             'locked internal-test/external-validation scoring with much lower '
                             'compute cost (default). '
                             '"nested_generalization" reproduces the original repeated nested-CV workflow.')
    parser.add_argument('--n_trials_rf', type=int, default=24,
                        help='Optuna trials for Random Forest in the budget-matched workflow (default: 24)')
    parser.add_argument('--n_trials_xgb',  type=int, default=24,
                        help='Optuna trials for XGBoost in the budget-matched workflow (default: 24)')
    parser.add_argument('--n_trials_svm',  type=int, default=24,
                        help='Optuna trials for SVM-RBF in the budget-matched workflow (default: 24)')
    parser.add_argument('--n_trials_lgbm', type=int, default=24,
                        help='Optuna trials for LightGBM in the budget-matched workflow (default: 24)')
    parser.add_argument('--phase1_runs', type=int, default=10,
                        help='Independent searches per 28 model combinations before Top-5 selection (default: 10)')
    parser.add_argument('--top5_runs', type=int, default=30,
                        help='Independent searches for each Top-5 model after selection (default: 30)')
    parser.add_argument('--top_k_models', type=int, default=5,
                        help='Number of models selected by training-internal CV for the second stage (default: 5)')
    parser.add_argument('--population_size', type=int, default=6,
                        help='Population size for PSO/SSA/DE/GWO searches (default: 6)')
    parser.add_argument('--population_max_iter', type=int, default=3,
                        help='Iterations for PSO/SSA/DE/GWO searches (default: 3)')
    parser.add_argument('--nested_outer_splits', type=int, default=5,
                        help='Outer folds for repeated nested CV (default: 5)')
    parser.add_argument('--nested_outer_repeats', type=int, default=2,
                        help='Outer repeats for repeated nested CV when --protocol nested_generalization (default: 2)')
    parser.add_argument('--nested_inner_splits', type=int, default=INNER_CV_SPLITS_DEFAULT,
                        help=f'Inner folds for hyperparameter search CV (default: {INNER_CV_SPLITS_DEFAULT})')
    parser.add_argument('--nested_inner_repeats', type=int, default=INNER_CV_REPEATS_DEFAULT,
                        help=f'Inner repeats for hyperparameter search CV (default: {INNER_CV_REPEATS_DEFAULT})')
    parser.add_argument('--final_eval_runs', type=int, default=30,
                        help='Repeated full-training searches per algorithm-optimizer combination (default: 30)')
    parser.add_argument('--skip_shap_contrast', action='store_true',
                        help='Skip post-hoc SHAP contrast analysis on external validation.')
    parser.add_argument('--skip_full_shap_archive', action='store_true',
                        help='Skip full test/validation SHAP data archive for all 28 final models.')
    parser.add_argument('--run_full_shap_archive', dest='skip_full_shap_archive',
                        action='store_false',
                        help='Enable the full test/validation SHAP archive for all 28 final models.')
    parser.add_argument('--shap_svm_explain_size', type=int, default=160,
                        help='Maximum SVM samples explained by KernelSHAP per dataset (default: 160).')
    parser.add_argument('--shap_background_size', type=int, default=48,
                        help='Background samples for KernelSHAP/generic SHAP explainers (default: 48).')
    parser.add_argument('--max_grid_candidates', type=int, default=24,
                        help='Cap on GridSearch candidates per algorithm; candidates are sampled deterministically '
                             'from the full grid when capped (default: 24).')
    parser.add_argument('--n_jobs', type=str, default=None,
                        help='Model-level CPU threads for RF/XGBoost/LightGBM/permutation importance. '
                             'Use -1/all/auto for all cores (default: all cores).')
    parser.add_argument('--gridsearch_n_jobs', type=str, default=None,
                        help='GridSearchCV worker count. Use -1/all/auto for all cores '
                             '(default: same as --n_jobs).')
    parser.add_argument('--search_n_jobs', type=str, default='1',
                        help='Model-level CPU threads used inside hyperparameter-search CV fits. '
                             'Default is 1 to avoid nested thread oversubscription and sklearn/joblib warnings.')
    parser.add_argument('--feature_method', type=str, default='domain_knowledge',
                        choices=['domain_knowledge', 'variance_threshold',
                                 'rf_importance', 'boruta'])
    parser.add_argument('--val_path', type=str, default=None,
                        help='Path to external validation Excel file. '
                             'Overrides IDE_DEFAULT_VAL_PATH when provided.')
    parser.add_argument('--convergence_only', '--convergence-only',
                        dest='convergence_only', action='store_true',
                        help='Only regenerate convergence curves from saved RegenData '
                             '(skip full training pipeline).')
    return parser.parse_args()


def resolve_paths(args):
    """Resolve data/output paths with WSL2-local defaults."""
    project_output_dir = Path(__file__).resolve().parent / 'output6.28'
    default_output_arg = str(Path(__file__).parent / 'output6.28')
    default_data_dir = _default_data_dir()
    output_env = os.environ.get('MINEWATER_OUTPUT_DIR')

    if args.output_dir and args.output_dir != default_output_arg:
        output_dir = Path(args.output_dir).expanduser()
    elif output_env:
        output_dir = Path(output_env).expanduser()
    elif IDE_DEFAULT_OUTPUT_DIR:
        output_dir = Path(IDE_DEFAULT_OUTPUT_DIR).expanduser()
    else:
        output_dir = project_output_dir

    if getattr(args, 'convergence_only', False):
        return None, None, output_dir, None

    if args.train_path and args.test_path:
        train_path = Path(args.train_path).expanduser()
        test_path = Path(args.test_path).expanduser()
    elif args.data_dir:
        data_dir = Path(args.data_dir).expanduser()
        train_path = data_dir / 'train_set.xlsx'
        test_path = data_dir / 'test_set.xlsx'
    elif IDE_DEFAULT_TRAIN_PATH and IDE_DEFAULT_TEST_PATH:
        train_path = Path(IDE_DEFAULT_TRAIN_PATH).expanduser()
        test_path = Path(IDE_DEFAULT_TEST_PATH).expanduser()
    else:
        train_path = default_data_dir / 'train_set.xlsx'
        test_path = default_data_dir / 'test_set.xlsx'

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. "
            "Place train_set.xlsx/test_set.xlsx under ../Input_Data, set MINEWATER_DATA_DIR, "
            "or pass --data_dir with the input-data directory.")
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}. "
            "Place train_set.xlsx/test_set.xlsx under ../Input_Data, set MINEWATER_DATA_DIR, "
            "or pass --data_dir with the input-data directory.")

    if getattr(args, 'val_path', None):
        val_path = Path(args.val_path).expanduser()
    elif IDE_DEFAULT_VAL_PATH:
        val_path = Path(IDE_DEFAULT_VAL_PATH).expanduser()
    else:
        val_path = _first_existing_file(
            train_path.parent,
            ['external_validation_set.xlsx', 'val_set.xlsx', 'validation_set.xlsx',
             'Val\u8fc7CBE.xlsx',
             '\u9a8c\u8bc1\u96c6.xlsx'])
        if val_path is None:
            val_path = train_path.parent / 'external_validation_set.xlsx'

    print(
        f"[Info] Runtime paths:\n"
        f" Train : {train_path}\n"
        f" Test  : {test_path}\n"
        f" Val   : {val_path}\n"
        f" Output: {output_dir}",
        flush=True,
    )

    return train_path, test_path, output_dir, val_path


# ================================================================================
# WATER SOURCE LABEL MAPPING
# ================================================================================

CANONICAL_WATER_SOURCE_ORDER = [
    'Ordovician limestone (O)',
    'Goaf water (G)',
    'Taiyuan limestone (T)',
    'Permian sandstone fissure (P)',
]

CANONICAL_WATER_SOURCE_PATTERNS = {
    'Ordovician limestone (O)': ('ordovician', 'ordovician limestone', '奥灰', '奥陶'),
    'Goaf water (G)': ('goaf', 'goaf water', '老空', '采空', '采空区'),
    'Taiyuan limestone (T)': ('taiyuan', 'taiyuan limestone', '太灰', '太原'),
    'Permian sandstone fissure (P)': ('permian', 'permian sandstone', '砂岩', '二叠', '二叠系'),
}


def canonicalize_water_source_label(label):
    """Collapse English/Chinese aliases onto a single canonical class name."""
    label_str = str(label).strip()
    label_lower = label_str.lower()
    exact_aliases = {
        '2': 'Ordovician limestone (O)',
        '2.0': 'Ordovician limestone (O)',
        '3': 'Goaf water (G)',
        '3.0': 'Goaf water (G)',
        '4': 'Taiyuan limestone (T)',
        '4.0': 'Taiyuan limestone (T)',
        '5': 'Permian sandstone fissure (P)',
        '5.0': 'Permian sandstone fissure (P)',
        'o': 'Ordovician limestone (O)',
        'g': 'Goaf water (G)',
        't': 'Taiyuan limestone (T)',
        'p': 'Permian sandstone fissure (P)',
    }
    if label_lower in exact_aliases:
        return exact_aliases[label_lower]
    for canonical_name, patterns in CANONICAL_WATER_SOURCE_PATTERNS.items():
        if any(pat in label_lower for pat in patterns):
            return canonical_name
    return label_str


def encode_labels_fixed(train_labels, test_labels=None, val_labels=None):
    train_labels_c = np.array(
        [canonicalize_water_source_label(lbl) for lbl in train_labels], dtype=object)
    test_labels_c = (np.array(
        [canonicalize_water_source_label(lbl) for lbl in test_labels], dtype=object)
        if test_labels is not None else None)
    val_labels_c = (np.array(
        [canonicalize_water_source_label(lbl) for lbl in val_labels], dtype=object)
        if val_labels is not None else None)

    all_labels = set(train_labels_c.tolist())
    if test_labels_c is not None:
        all_labels |= set(test_labels_c.tolist())
    if val_labels_c is not None:
        all_labels |= set(val_labels_c.tolist())

    canonical_labels = [lbl for lbl in CANONICAL_WATER_SOURCE_ORDER if lbl in all_labels]
    unknown_labels = sorted(lbl for lbl in all_labels if lbl not in CANONICAL_WATER_SOURCE_ORDER)
    fitted_classes = canonical_labels + unknown_labels

    le = LabelEncoder()
    # LabelEncoder.fit() sorts classes alphabetically.  For paper figures,
    # confusion matrices and SHAP class labels we need the declared geological
    # order above, so assign classes_ directly after canonicalisation.
    le.classes_ = np.array(fitted_classes, dtype=object)

    y_train_enc = le.transform(train_labels_c)
    y_test_enc = le.transform(test_labels_c) if test_labels_c is not None else None
    y_val_enc = le.transform(val_labels_c) if val_labels_c is not None else None

    water_source_mapping = {}
    for idx, cls_name in enumerate(le.classes_):
        if cls_name in CANONICAL_WATER_SOURCE_ORDER:
            water_source_mapping[idx] = cls_name
        else:
            water_source_mapping[idx] = f'Class {cls_name}'

    return le, y_train_enc, y_test_enc, y_val_enc, water_source_mapping


# ================================================================================
# SECTION 2: FEATURE ENGINEERING
# ================================================================================

def create_geochemical_features(df, target_col='充水水源', ion_cols=None):
    """Normalize the eight manuscript-approved model features in-place.

    The submission workflow uses only the eight core hydrochemical inputs
    ``x1``-``x8``. Historical ratio, meq, hardness, Gibbs, and other derived
    diagnostics are intentionally excluded so that feature engineering cannot
    silently expand the training matrix beyond the manuscript scope.
    """
    if ion_cols is None:
        ion_cols = ION_COLS

    for col in list(ion_cols) + [PH_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    return df


def select_core_features(X, y, feature_names=None, method='domain_knowledge', seed=42):
    """Feature selection: domain_knowledge / variance_threshold / rf_importance / boruta."""
    np.random.seed(seed)

    if method == 'domain_knowledge':
        if feature_names is not None:
            available_core = [f for f in CORE_FEATURE_COLS if f in feature_names]
            if len(available_core) == len(CORE_FEATURE_COLS):
                idxs = [feature_names.index(f) for f in available_core]
                report = {
                    'method': 'domain_knowledge',
                    'n_features': len(available_core),
                    'reason': ('7 field-measured major ions '
                               '(K+, Na+, Ca2+, Mg2+, Cl-, SO4^2-, HCO3-) + pH'),
                    'features': available_core
                }
                return X[:, idxs], available_core, report
            # Partial match: fall through with a warning rather than silently
            # returning a wrong feature set.
            missing = [f for f in CORE_FEATURE_COLS if f not in feature_names]
            print(f" [select_core_features] domain_knowledge: missing {missing};"
                  f" falling back to all provided features.")
            return X, list(feature_names), {
                'method': 'domain_knowledge_fallback',
                'n_features': X.shape[1],
                'reason': f'CORE columns missing: {missing}',
                'features': list(feature_names),
            }

    if method == 'variance_threshold':
        selector = VarianceThreshold(threshold=0.1)
        X_selected = selector.fit_transform(X)
        mask = selector.get_support()
        sel_feat = [f for f, m in zip(feature_names, mask) if m] if feature_names else \
                   [f'f{i}' for i in range(X_selected.shape[1])]
        return X_selected, sel_feat, {
            'method': 'variance_threshold', 'n_features': X_selected.shape[1],
            'reason': 'Removes low variance features', 'threshold': 0.1
        }

    if method == 'rf_importance':
        rf = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=DETERMINISTIC_N_JOBS)
        rf.fit(X, y)
        importances = rf.feature_importances_
        threshold = np.mean(importances)
        mask = importances > threshold
        X_selected = X[:, mask]
        sel_feat = [f for f, m in zip(feature_names, mask) if m] if feature_names else \
                   [f'f{i}' for i in range(X_selected.shape[1])]
        return X_selected, sel_feat, {
            'method': 'rf_importance', 'n_features': X_selected.shape[1],
            'reason': 'RF importance > mean threshold'
        }

    try:
        from boruta import BorutaPy
        if method == 'boruta':
            rf = RandomForestClassifier(
                n_estimators=100, random_state=seed,
                n_jobs=DETERMINISTIC_N_JOBS, max_depth=5)
            selector = BorutaPy(rf, n_estimators='auto', random_state=seed, verbose=0)
            selector.fit(X, y)
            X_selected = selector.transform(X)
            mask = selector.support_
            sel_feat = [f for f, m in zip(feature_names, mask) if m] if feature_names else \
                       [f'f{i}' for i in range(X_selected.shape[1])]
            return X_selected, sel_feat, {'method': 'boruta', 'n_features': X_selected.shape[1],
                                           'reason': 'Boruta all-relevant selection'}
    except ImportError:
        pass

    # Final fallback: take the first ``len(CORE_FEATURE_COLS)`` columns so the
    # downstream dimensionality matches the canonical training feature set.
    n_features = min(len(CORE_FEATURE_COLS), X.shape[1])
    sel_feat = feature_names[:n_features] if feature_names else [f'f{i}' for i in range(n_features)]
    return X[:, :n_features], sel_feat, {
        'method': 'fallback', 'n_features': n_features,
        'reason': f'Default first {n_features} features (CORE size)'
    }


# ================================================================================
# SECTION 4: UNIFIED OPTUNA BAYESIAN OPTIMIZATION (SYMMETRIC FRAMEWORK)
# ================================================================================


# ================================================================================
# SECTION 4c: OPTIMIZER COMPARISON  - RF BACKBONE
# ================================================================================


# ================================================================================
# SECTION 5: CONFIDENCE INTERVALS
# ================================================================================

class RobustConfidenceInterval:
    """BCa Bootstrap, Wilson Score, Clopper-Pearson, t-distribution CIs."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def bca_bootstrap_ci(self, y_true, y_pred, metric_func, n_bootstrap=2000, alpha=0.05):
        """BCa Bootstrap CI (Efron & Tibshirani, 1993)."""
        np.random.seed(self.random_state)
        metric_original = metric_func(y_true, y_pred)
        bootstrap_metrics = []
        n_samples = len(y_true)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            try:
                bootstrap_metrics.append(metric_func(y_true[indices], y_pred[indices]))
            except Exception:
                continue
        bootstrap_metrics = np.array(bootstrap_metrics)
        prop_below = np.clip(np.mean(bootstrap_metrics < metric_original), 1e-6, 1-1e-6)
        z0 = stats.norm.ppf(prop_below)
        jackknife = []
        for i in range(n_samples):
            mask = np.ones(n_samples, dtype=bool); mask[i] = False
            try:
                jackknife.append(metric_func(y_true[mask], y_pred[mask]))
            except Exception:
                jackknife.append(metric_original)
        jackknife = np.array(jackknife)
        theta_dot = np.mean(jackknife)
        diff = theta_dot - jackknife
        a = np.sum(diff**3) / (6 * (np.sum(diff**2)**1.5) + 1e-10)
        # BCa endpoints follow the standard form in Efron & Tibshirani (1993),
        # Chapter 14: lower endpoint uses z0+za, upper uses z0-za, where za is
        # the standard-normal alpha/2 quantile.
        za = stats.norm.ppf(1 - alpha/2)  # z_{alpha/2} = 1.96
        denom_lower = 1 - a * (z0 + za)
        denom_upper = 1 - a * (z0 - za)
        if abs(denom_lower) < 1e-6 or abs(denom_upper) < 1e-6:
            ci_lower = np.percentile(bootstrap_metrics, alpha/2 * 100)
            ci_upper = np.percentile(bootstrap_metrics, (1-alpha/2) * 100)
            method = 'BCa Bootstrap (fallback to percentile)'
        else:
            z_lo_arg = z0 + (z0 + za) / (denom_lower + 1e-12)
            z_hi_arg = z0 + (z0 - za) / (denom_upper + 1e-12)
            lower_adj = float(np.clip(stats.norm.cdf(z_lo_arg), 0.001, 0.999))
            upper_adj = float(np.clip(stats.norm.cdf(z_hi_arg), 0.001, 0.999))
            if lower_adj >= upper_adj:
                ci_lower = float(np.percentile(bootstrap_metrics, alpha/2 * 100))
                ci_upper = float(np.percentile(bootstrap_metrics, (1-alpha/2) * 100))
                method = 'BCa Bootstrap (fallback to percentile  - adj_ci inverted)'
            else:
                ci_lower = float(np.percentile(bootstrap_metrics, lower_adj * 100))
                ci_upper = float(np.percentile(bootstrap_metrics, upper_adj * 100))
                method = 'BCa Bootstrap'
        return {'metric': metric_original, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
                'method': method, 'n_bootstrap': n_bootstrap}

    def wilson_score_ci(self, n_success, n_total, alpha=0.05):
        p_hat = n_success / n_total
        z = stats.norm.ppf(1 - alpha/2)
        center = (p_hat + z**2/(2*n_total)) / (1 + z**2/n_total)
        margin = (z * np.sqrt(p_hat*(1-p_hat)/n_total + z**2/(4*n_total**2))
                  / (1 + z**2/n_total))
        return {'proportion': p_hat, 'ci_lower': max(0, center-margin),
                'ci_upper': min(1, center+margin), 'method': 'Wilson Score'}

    def clopper_pearson_ci(self, n_success, n_total, alpha=0.05):
        lower = 0 if n_success == 0 else stats.beta.ppf(
            alpha/2, n_success, n_total-n_success+1)
        upper = 1 if n_success == n_total else stats.beta.ppf(
            1-alpha/2, n_success+1, n_total-n_success)
        return {'proportion': n_success/n_total, 'ci_lower': lower,
                'ci_upper': upper, 'method': 'Clopper-Pearson'}

    def multi_seed_t_ci(self, scores_list, alpha=0.05):
        scores = np.array(scores_list)
        mean = np.mean(scores); std = np.std(scores, ddof=1); n = len(scores)
        t_val = stats.t.ppf(1-alpha/2, df=n-1)
        se = std / np.sqrt(n)
        return {'mean': mean, 'std': std, 'ci_lower': mean-t_val*se,
                'ci_upper': mean+t_val*se, 'method': 't-distribution'}


# ================================================================================
# SECTION 6: BASELINE COMPARISON
# ================================================================================


# ================================================================================
# SECTION 6: UNIFIED 30-SEED BASELINE COMPARISON (ALL 28 COMBINATIONS)
# ================================================================================
#
# REWRITTEN: Single ModelFactory eliminates all model-construction inconsistencies.
# Key fixes:
#   - holdout_acc: model trained on 80% only, evaluated on unseen 20% (no leakage)
#   - No redundant CV inside 30-seed loop (test-set F1 IS the paired observation)
#   - All 28 combos use identical model-building and fitting logic
#   - GridSearch/PSO/SSA/DE/GWO params pre-computed once at seed=42
# ================================================================================

# --- Canonical Default Hyperparameters (single source of truth) ---
_DEFAULT_PARAMS = {
    'RF': {'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 8,
           'min_samples_leaf': 4, 'min_samples_split': 6, 'max_samples': 0.8},
    'XGBoost': {'n_estimators': 300, 'max_depth': 4, 'min_child_weight': 5,
                'gamma': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 0.5, 'reg_lambda': 2.0},
    'LightGBM': {'n_estimators': 300, 'num_leaves': 31, 'max_depth': 6,
                 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8,
                 'subsample_freq': 5, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
                 'min_child_samples': 20},
    'SVM': {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale', 'class_weight': 'balanced'},
}

_ALGORITHMS = ['RF', 'XGBoost', 'LightGBM', 'SVM']
_OPTIMIZERS = ['Default', 'Optuna', 'GridSearch', 'PSO', 'SSA', 'DE', 'GWO']

_GRID_PARAMS = {
    # SYMMETRY NOTE: All grid ranges are aligned with the corresponding
    # bounds_* and Optuna search spaces used by PSO/SSA/DE/GWO and Optuna.
    # This ensures that GridSearch covers the same parameter region as all
    # other optimizers; cross-optimizer comparison is not confounded by
    # search-space coverage differences.
    # CV protocol: RepeatedStratifiedKFold configured by INNER_CV_* constants,
    # identical to Optuna and all population-based optimizers.
    'RF': {
        'n_estimators'     : [100, 200, 400],         # aligned: bounds [100,600]
        'max_depth'        : [4, 6, 8, 10],           # aligned: bounds [3,10]
        'min_samples_split': [5, 10, 20],             # aligned: bounds [5,20]
        'min_samples_leaf' : [3, 5, 8],               # aligned: bounds [3,12]
        'max_features'     : ['sqrt', 'log2', None],  # aligned with Optuna RF
        'max_samples'      : [0.65, 0.75, 0.85],      # aligned: bounds [0.60,0.85]
    },
    'XGBoost': {
        # SYMMETRY FIX: added reg_lambda, min_child_weight, gamma to match
        # Optuna XGBoost and expanded bounds_xgb / _POP_BOUNDS['XGBoost'].
        # Previously had only 5 params  - asymmetric advantage to Optuna/PSO/SSA/DE/GWO.
        # n_estimators is NOT part of the grid: it is treated as an
        # early-stopping upper bound (same convention as Optuna XGBoost and
        # the population optimizers). After the best tunable hparams are
        # selected via GridSearchCV, a short post-search early-stopping
        # run on a held-out 15% slice sets n_estimators := best_iteration.
        'max_depth'        : [3, 5, 7],                # bounds [3,8]
        'learning_rate'    : [0.05, 0.1, 0.2],         # bounds [0.01,0.3]
        'subsample'        : [0.7, 0.9],               # bounds [0.5,1.0]
        'colsample_bytree' : [0.7, 0.9],               # bounds [0.5,1.0]
        'reg_alpha'        : [0.01, 0.1, 1.0],         # log-spaced sample of reg_alpha_log in [-5,2]
        'reg_lambda'       : [0.1, 1.0, 10.0],         # aligned with reg_lambda_log in [-5,2]
        'min_child_weight' : [1, 3, 7],                # aligned with bounds [1,10]
        'gamma'            : [0.0, 0.1, 0.5],          # aligned with bounds [0,1]
    },
    'LightGBM': {
        # SYMMETRY FIX: added reg_alpha/reg_lambda to match Optuna LightGBM
        # and bounds_lgbm (lambda_l1_log/lambda_l2_log).
        # n_estimators is NOT part of the grid: early-stopping upper bound
        # (same convention as Optuna and population optimizers; see XGBoost
        # grid above for rationale).
        'num_leaves'        : [20, 31, 50],            # bounds [15,63]
        'learning_rate'     : [0.05, 0.1],             # bounds [0.01,0.3]
        'min_child_samples' : [10, 20, 30],            # bounds [10,80]
        'reg_alpha'         : [0.01, 0.1, 1.0],        # aligned with lambda_l1_log in [-2,2]
        'reg_lambda'        : [0.1, 1.0, 5.0],         # aligned with lambda_l2_log in [-1,2]
    },
    'SVM': [
        {
            'svm__kernel'      : ['linear'],
            'svm__C'           : [0.1, 1.0, 10.0, 100.0, 1000.0],
            'svm__class_weight': SVM_CLASS_WEIGHT_OPTIONS,
        },
        {
            'svm__kernel'      : ['rbf'],
            'svm__C'           : [0.1, 1.0, 10.0, 100.0, 1000.0],
            'svm__gamma'       : [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0],
            'svm__class_weight': SVM_CLASS_WEIGHT_OPTIONS,
        },
    ],
}
# Alias used by grid_combo_count() for backward compatibility
GS_PARAM_GRID = _GRID_PARAMS

_POP_BOUNDS = {
    'RF': [('n_estimators', 100.0, 600.0), ('max_depth', 3.0, 10.0),
           ('min_samples_split', 5.0, 20.0), ('min_samples_leaf', 3.0, 12.0),
           ('max_samples', 0.60, 0.85), ('max_features_idx', 0.0, 1.0)],
    # SYMMETRY FIX: XGBoost bounds now include reg_lambda_log, min_child_weight,
    # gamma  - matching Optuna XGBoost (8 params) and bounds_xgb in
    # run_full_matrix_comparison. Previously had only 6 params.
    'XGBoost': [('n_estimators', 50.0, 500.0), ('max_depth', 3.0, 8.0),
                ('learning_rate', 0.01, 0.3), ('subsample', 0.5, 1.0),
                ('colsample_bytree', 0.5, 1.0), ('reg_alpha_log', -5.0, 2.0),
                ('reg_lambda_log', -5.0, 2.0),   # NEW: aligned with Optuna & bounds_xgb
                ('min_child_weight', 1.0, 10.0),  # NEW: aligned with Optuna & bounds_xgb
                ('gamma', 0.0, 1.0)],             # NEW: aligned with Optuna & bounds_xgb
    'LightGBM': [('n_estimators', 100.0, 600.0), ('num_leaves', 15.0, 63.0),
                  ('learning_rate', 0.01, 0.3), ('min_child_samples', 10.0, 80.0),
                  ('lambda_l1_log', -2.0, 2.0), ('lambda_l2_log', -1.0, 2.0)],
    # SYMMETRY FIX: SVM bounds now include kernel_idx and class_weight_idx
    # so that PSO/SSA/DE/GWO-SVM searches the same 4-dim space as Optuna-SVM
    # (which tunes C, gamma, kernel, class_weight; see run_optuna L946) and the
    # matrix-level bounds_svm (L1633).  Previously only 2 dims were exposed
    # here, silently restricting population optimizers to RBF + balanced only
    # during the 30-seed evaluation.  Indices <0.5 map to the first option.
    'SVM': [('C_log', -1.0, 3.0), ('gamma_log', -4.0, 1.0),
            ('kernel_idx', 0.0, 1.0),          # <0.5 -> linear, >=0.5 -> rbf
            ('class_weight_idx', 0.0, 1.0)],   # <0.5 -> None, >=0.5 -> balanced
}


def _unified_build_model(algo, params, seed, model_n_jobs=None):
    """SINGLE model factory  - called by ALL paths (Default/Optuna/Grid/PSO/SSA/DE/GWO)."""
    model_n_jobs = _normalise_n_jobs(model_n_jobs, DETERMINISTIC_N_JOBS)
    if algo == 'RF':
        return RandomForestClassifier(
            n_estimators=int(params.get('n_estimators', 200)),
            max_depth=int(params.get('max_depth', 8)),
            min_samples_split=int(params.get('min_samples_split', 6)),
            min_samples_leaf=int(params.get('min_samples_leaf', 4)),
            max_samples=float(params.get('max_samples', 0.8)),
            max_features=params.get('max_features', 'sqrt'),
            class_weight='balanced', random_state=seed,
            n_jobs=model_n_jobs)
    elif algo == 'XGBoost':
        # SYMMETRY FIX: reads all 9 params aligned with expanded _POP_BOUNDS['XGBoost'].
        # reg_lambda, min_child_weight, gamma added to match Optuna & _GRID_PARAMS.
        return XGBClassifier(
            n_estimators=int(params.get('n_estimators', 300)),
            max_depth=int(params.get('max_depth', 4)),
            learning_rate=float(params.get('learning_rate', 0.1)),
            subsample=float(params.get('subsample', 0.8)),
            colsample_bytree=float(params.get('colsample_bytree', 0.8)),
            reg_alpha=float(params.get('reg_alpha',
                            10**float(params.get('reg_alpha_log', -1)))),
            reg_lambda=float(params.get('reg_lambda',
                             10**float(params.get('reg_lambda_log', 0)))),   # NEW
            min_child_weight=int(params.get('min_child_weight', 5)),          # NEW
            gamma=float(params.get('gamma', 0.1)),                            # NEW
            base_score=0.5,
            eval_metric='mlogloss', random_state=seed,
            n_jobs=model_n_jobs, verbosity=0)
    elif algo == 'LightGBM':
        return lgb.LGBMClassifier(
            n_estimators=int(params.get('n_estimators', 300)),
            num_leaves=int(params.get('num_leaves', 31)),
            max_depth=int(params.get('max_depth', 6)),
            learning_rate=float(params.get('learning_rate', 0.05)),
            min_child_samples=int(params.get('min_child_samples', 20)),
            colsample_bytree=float(params.get('colsample_bytree', 0.8)),
            subsample=float(params.get('subsample', 0.8)),
            subsample_freq=int(params.get('subsample_freq', 5)),
            reg_alpha=float(params.get('reg_alpha', 0.5)),
            reg_lambda=float(params.get('reg_lambda', 2.0)),
            deterministic=True,
            force_col_wise=True,
            verbosity=-1, random_state=seed, n_jobs=model_n_jobs)
    elif algo == 'SVM':
        return _build_svm_pipeline(params, seed)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def _unified_fit(model, algo, X_tr, y_tr, X_es=None, y_es=None):
    """Single fit protocol  - handles class weighting and early stopping uniformly.

    Boosting fairness fix
    ---------------------
    For XGBoost and LightGBM, early stopping requires a held-out eval set. A
    naive implementation would either (i) feed the model only ``X_tr\\X_es``
    (~85% of X_tr), shrinking the effective boosting training sample below
    that of RF/SVM, or (ii) leak X_es into training.  We avoid both by a
    two-stage protocol:

      Stage 1. Internally carve a 15% eval slice from ``X_tr`` (when ``X_es``
               is not supplied) or use the caller-supplied ``(X_es, y_es)``,
               fit with early stopping (patience 30) to find
               ``best_iteration``.
      Stage 2. Rebuild the estimator with ``n_estimators=best_iteration`` and
               refit on the **full** ``X_tr`` with no ``eval_set``.

    The refit step equalises the training sample size across all four
    algorithms while preserving the data-driven selection of the number of
    boosting rounds. If the first stage fails or the sample is too small, we
    fall back to a direct fit on ``X_tr``.

    Parameters
    ----------
    model : estimator
        Freshly constructed estimator from ``_unified_build_model``.
    algo : {'RF','XGBoost','LightGBM','SVM'}
    X_tr, y_tr : ndarray
        Full training subset (the caller's 80% split).
    X_es, y_es : ndarray or None
        Optional eval set used only for stage 1 early stopping. If None and
        the algorithm needs early stopping, a 15% slice is carved from X_tr.
    """
    sw_full = precompute_sample_weight(y_tr)

    if algo in ('XGBoost', 'LightGBM'):
        # Stage 1 eval split. If caller supplied (X_es, y_es) we trust it;
        # otherwise we carve a 15% stratified slice for stage-1 only.
        if X_es is None or y_es is None or len(X_es) < 5:
            try:
                X_s1, X_s1_es, y_s1, y_s1_es = train_test_split(
                    X_tr, y_tr, test_size=0.15,
                    stratify=y_tr, random_state=0)
            except ValueError:
                X_s1, X_s1_es, y_s1, y_s1_es = X_tr, None, y_tr, None
        else:
            X_s1, X_s1_es, y_s1, y_s1_es = X_tr, X_es, y_tr, y_es

        if X_s1_es is None:
            # Too few samples for early stopping; direct fit on X_tr.
            model.fit(X_tr, y_tr, sample_weight=sw_full)
            return model

        sw_s1 = precompute_sample_weight(y_s1)

        if algo == 'XGBoost':
            model.set_params(early_stopping_rounds=30)
            model.fit(X_s1, y_s1, sample_weight=sw_s1,
                      eval_set=[(X_s1_es, y_s1_es)], verbose=False)
            best_iter = getattr(model, 'best_iteration', None)
            if best_iter is not None and best_iter > 0:
                final_params = model.get_params()
                final_params['n_estimators'] = int(best_iter) + 1
                final_params['early_stopping_rounds'] = None
                refit = XGBClassifier(**final_params)
                refit.fit(X_tr, y_tr, sample_weight=sw_full, verbose=False)
                model.__dict__.update(refit.__dict__)

        else:  # LightGBM
            model.fit(X_s1, y_s1, sample_weight=sw_s1,
                      eval_set=[(X_s1_es, y_s1_es)],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(-1)])
            best_iter = getattr(model, 'best_iteration_', None)
            if best_iter is not None and best_iter > 0:
                final_params = model.get_params()
                final_params['n_estimators'] = int(best_iter)
                refit = lgb.LGBMClassifier(**final_params)
                refit.fit(X_tr, y_tr, sample_weight=sw_full)
                model.__dict__.update(refit.__dict__)

    else:
        model.fit(X_tr, y_tr)
    return model


def _resolve_params(algo, optimizer, optuna_params, gridsearch_params, pop_params):
    """Resolve hyperparams for a given algo x optimizer combination."""
    if optimizer == 'Default':
        return dict(_DEFAULT_PARAMS[algo])
    elif optimizer == 'Optuna':
        p = dict(optuna_params.get(algo, _DEFAULT_PARAMS[algo]))
        p = {k: v for k, v in p.items() if not str(k).startswith('__')}
        return _normalize_svm_params(p) if algo == 'SVM' else p
    elif optimizer == 'GridSearch':
        p = gridsearch_params.get(algo, {})
        if not p:
            return dict(_DEFAULT_PARAMS[algo])
        # GridSearch may return params with pipeline prefixes (e.g. svm__C)
        # Convert them back for _unified_build_model
        if algo == 'SVM':
            cleaned = {}
            for k, v in p.items():
                if str(k).startswith('__'):
                    continue
                cleaned[k.replace('svm__', '')] = v
            return _normalize_svm_params(cleaned)
        return {k: v for k, v in dict(p).items() if not str(k).startswith('__')}
    elif optimizer in ('PSO', 'SSA', 'DE', 'GWO'):
        raw = pop_params.get(optimizer, {}).get(algo, {})
        if not raw:
            return dict(_DEFAULT_PARAMS[algo])
        # Convert continuous pop-optimizer params to model params
        params = {k: v for k, v in dict(raw).items() if not str(k).startswith('__')}
        if algo == 'SVM':
            return _normalize_svm_params(params)
        if algo == 'RF' and 'max_features_idx' in params:
            idx = params.pop('max_features_idx')
            params['max_features'] = _decode_max_features_idx(idx)
        if 'reg_alpha_log' in params:
            params['reg_alpha'] = float(10 ** params.pop('reg_alpha_log'))
        if 'lambda_l1_log' in params:
            params['reg_alpha'] = float(10 ** params.pop('lambda_l1_log'))
        if 'lambda_l2_log' in params:
            params['reg_lambda'] = float(10 ** params.pop('lambda_l2_log'))
        # Clip to valid ranges
        for k in ('n_estimators', 'max_depth', 'min_samples_split',
                   'min_samples_leaf', 'min_child_samples', 'num_leaves'):
            if k in params:
                params[k] = int(round(params[k]))
        return params
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


# Keep _run_population_for_model (lines 3446-3594) and _make_pop_model_from_bounds (3597-3635)
# as they are called by the pre-computation step above.


# ================================================================================
# SECTION 7: STATISTICAL TESTS
# ================================================================================


# ================================================================================
# SECTION 8: VISUALIZATION
# ================================================================================


def _compute_shap_mean_abs(model, X):
    """Version-robust mean|SHAP| for any supported backbone.

    Delegates to ``compute_shap_robust`` (see
    ``shap_all_combinations_patch``), which handles the
    XGBoost >= 3.1.0 base_score regression (shap/shap#4184),
    the 3-D SHAP array returned by LightGBM under shap >= 0.44,
    and SVM via KernelExplainer.
    """
    shap_list, _X_ex, _backend = compute_shap_robust(model, X)
    return shap_list, mean_abs_shap(shap_list)


# ================================================================================
# PIPER DIAGRAM, ROC, CALIBRATION, VIF (unchanged from v4.4  - domain methods)
# ================================================================================


# ================================================================================
# SECTION 8e: ADDITIONAL FIGURES
# ================================================================================


# ================================================================================
# SECTION 8g: ADDITIONAL FIGURES  - Priority 5 & 6
# Unified publication style: all functions call _set_publication_style() at entry.
# Feature labels use FEATURE_DISPLAY_NAMES mapping at plot-layer only.
# ================================================================================

def _save_figure(fig, path_stem, dpi=600):
    """Unified save: PNG + PDF, path_stem has no extension."""
    fig.savefig(str(path_stem) + '.png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(str(path_stem) + '.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')


def _safe_name_stem(name):
    """Convert a model / figure label into a filesystem-safe stem."""
    out = []
    for ch in str(name):
        out.append(ch if ch.isalnum() else '_')
    stem = ''.join(out).strip('_')
    while '__' in stem:
        stem = stem.replace('__', '_')
    return stem or 'unnamed'


def _predict_label_vector(model, X):
    """Predict labels from sklearn-style estimator or ndarray-like output."""
    raw = model.predict(X)
    return (np.argmax(raw, axis=1).astype(int)
            if hasattr(raw, 'ndim') and raw.ndim == 2
            else np.asarray(raw).astype(int))


# ================================================================================
# SECTION 8f: ADVANCED STATISTICAL & DIAGNOSTIC ANALYSIS
# ================================================================================


def _clean_history(values):
    """Convert a raw history container into a finite float list."""
    out = []
    for v in (values or []):
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out.append(fv)
    return out


def regenerate_convergence_from_cache(output_dir):
    """Regenerate convergence curves from saved RegenData only."""
    try:
        import joblib as _jl
    except Exception as e:
        print(f"[Convergence] joblib unavailable: {e}")
        return False

    regen_dir = output_dir / 'RegenData'
    new_path = regen_dir / 'convergence_records.pkl'
    legacy_path = regen_dir / 'optimizer_results.pkl'

    if new_path.exists():
        recs = _jl.load(new_path)
    elif legacy_path.exists():
        recs = _jl.load(legacy_path)
    else:
        print(f"[Convergence] Cache not found: {new_path}")
        return False

    n_rec = len(recs) if hasattr(recs, '__len__') else 0
    print(f"[Convergence] Loaded {n_rec} records from cache.")
    generate_convergence_figure(recs, output_dir=output_dir)
    return True


def generate_convergence_figure(optimizer_results, output_dir=None):
    """Generate convergence curves for all optimizers with comparable x-axes.

    Raw figure:
      x-axis = actual search budget units recorded by each optimizer
               (grid points / Optuna trials / population candidate evaluations).
    Normalized figure:
      x-axis = relative search progress (%) for shape-only comparison under
               unequal computational budgets.
    """
    _set_publication_style()
    matplotlib.use('Agg')
    if output_dir is None:
        return
    figures_dir = output_dir / 'Figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    valid = []
    for r in (optimizer_results or []):
        hist = _clean_history(r.get('best_history', []))
        if not hist:
            continue
        valid.append({
            'algorithm': str(r.get('algorithm', 'RF')),
            'optimizer': str(r.get('optimizer', 'Unknown')),
            'best_history': hist,
            'x_values': list(r.get('x_values', list(range(1, len(hist) + 1)))),
        })

    if not valid:
        print('  [WARN] No convergence history found. Skip convergence figures.')
        return

    colors = {
        'Optuna': '#1f77b4', 'GridSearch': '#2ca02c',
        'PSO': '#ff7f0e', 'SSA': '#9467bd', 'DE': '#8c564b', 'GWO': '#e377c2'
    }
    markers = {'Optuna': 'o', 'GridSearch': 's', 'PSO': '^', 'SSA': 'D', 'DE': 'v', 'GWO': 'P'}
    opt_order = {'Optuna': 0, 'GridSearch': 1, 'PSO': 2, 'SSA': 3, 'DE': 4, 'GWO': 5}
    algo_pref = ['RF', 'XGBoost', 'LightGBM', 'SVM']

    algo_list = sorted({r['algorithm'] for r in valid},
                       key=lambda x: algo_pref.index(x) if x in algo_pref else 99)

    n_algo = len(algo_list)
    n_cols = 2
    n_rows = int(np.ceil(n_algo / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.6 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i, algo in enumerate(algo_list):
        ax = axes[i]
        rows = [r for r in valid if r['algorithm'] == algo]
        rows.sort(key=lambda r: opt_order.get(r['optimizer'], 99))
        for r in rows:
            opt = r['optimizer']
            h = r['best_history']
            xs = np.asarray(r.get('x_values', np.arange(1, len(h) + 1)), dtype=float)
            ax.plot(xs, h,
                    color=colors.get(opt, 'gray'),
                    marker=markers.get(opt, 'o'),
                    markevery=max(1, len(h) // 12),
                    markersize=4,
                    linewidth=1.8,
                    label=f"{opt} (best={max(h):.4f})")
        ax.set_title(f'{algo} convergence')
        ax.set_xlabel('Search budget (grid points / trials / candidate evaluations)')
        ax.set_ylabel('Best CV Macro-F1 so far')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for j in range(n_algo, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Figure11A Convergence Curves - All Optimizers / All Algorithms', fontsize=12, y=1.01)
    plt.tight_layout()
    _save_figure(fig, figures_dir / 'Figure11A_Convergence_AllAlgo')
    plt.close(fig)

    for algo in algo_list:
        rows = [r for r in valid if r['algorithm'] == algo]
        rows.sort(key=lambda r: opt_order.get(r['optimizer'], 99))

        fig_a, ax_a = plt.subplots(figsize=(9, 5))
        for r in rows:
            opt = r['optimizer']
            h = r['best_history']
            xs = np.asarray(r.get('x_values', np.arange(1, len(h) + 1)), dtype=float)
            ax_a.plot(xs, h,
                      color=colors.get(opt, 'gray'),
                      marker=markers.get(opt, 'o'),
                      markevery=max(1, len(h) // 12),
                      markersize=4,
                      linewidth=1.8,
                      label=f"{opt} (best={max(h):.4f})")
        ax_a.set_title(f'Convergence Curves - {algo}')
        ax_a.set_xlabel('Search budget (grid points / trials / candidate evaluations)')
        ax_a.set_ylabel('Best CV Macro-F1 so far')
        ax_a.grid(True, alpha=0.25)
        ax_a.legend(fontsize=9)
        plt.tight_layout()
        _save_figure(fig_a, figures_dir / f'Figure11B_Convergence_{algo}_AllOptimizers')
        plt.close(fig_a)

        for r in rows:
            opt = r['optimizer']
            h = r['best_history']
            xs = np.asarray(r.get('x_values', np.arange(1, len(h) + 1)), dtype=float)
            fig_s, ax_s = plt.subplots(figsize=(7.2, 4.2))
            ax_s.plot(xs, h,
                      color=colors.get(opt, 'gray'),
                      marker=markers.get(opt, 'o'),
                      markevery=max(1, len(h) // 12),
                      markersize=4,
                      linewidth=1.9)
            ax_s.set_title(f'Convergence - {algo} / {opt}')
            ax_s.set_xlabel('Search budget (grid points / trials / candidate evaluations)')
            ax_s.set_ylabel('Best CV Macro-F1 so far')
            ax_s.grid(True, alpha=0.25)
            plt.tight_layout()
            _save_figure(fig_s, figures_dir / f'Figure11C_Convergence_{algo}_{opt}')
            plt.close(fig_s)

    fig_n, axes_n = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.6 * n_rows))
    axes_n = np.atleast_1d(axes_n).reshape(-1)
    for i, algo in enumerate(algo_list):
        ax_n = axes_n[i]
        rows = [r for r in valid if r['algorithm'] == algo]
        rows.sort(key=lambda r: opt_order.get(r['optimizer'], 99))
        for r in rows:
            opt = r['optimizer']
            h = r['best_history']
            xs_raw = np.asarray(r.get('x_values', np.arange(1, len(h) + 1)), dtype=float)
            if len(xs_raw) == 0:
                continue
            x_end = xs_raw[-1] if xs_raw[-1] > 0 else 1.0
            xs_norm = xs_raw / x_end * 100.0
            ax_n.plot(xs_norm, h,
                      color=colors.get(opt, 'gray'),
                      marker=markers.get(opt, 'o'),
                      markevery=max(1, len(h) // 12),
                      markersize=4,
                      linewidth=1.8,
                      label=f"{opt} (best={max(h):.4f})")
        ax_n.set_title(f'{algo} convergence (normalized)')
        ax_n.set_xlabel('Relative search progress (%)')
        ax_n.set_ylabel('Best CV Macro-F1 so far')
        ax_n.grid(True, alpha=0.25)
        ax_n.legend(fontsize=8)

    for j in range(n_algo, len(axes_n)):
        axes_n[j].set_visible(False)

    fig_n.suptitle('Figure11D Convergence Curves - Normalized Search Progress', fontsize=12, y=1.01)
    plt.tight_layout()
    _save_figure(fig_n, figures_dir / 'Figure11D_Convergence_NormalizedProgress_AllAlgo')
    plt.close(fig_n)

    print(f' Convergence figures saved ({len(valid)} curves).')


# ================================================================================
# SECTION 8g: SUPPLEMENTARY ANALYSES
# ================================================================================


# ================================================================================
# SECTION 9: EXCEL EXPORT
# ================================================================================


# ================================================================================
# SECTION 9b: EXTERNAL VALIDATION
# ================================================================================


def _json_safe(obj):
    """Convert numpy/pandas objects into JSON-safe plain Python values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None:
        return None
    try:
        if not isinstance(obj, (str, bytes)) and pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def _metric_pack_for_predictions(y_true, y_pred):
    """Standard metric pack used by the independent-search protocol."""
    return {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'F1_Macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'F1_Weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'Kappa': float(cohen_kappa_score(y_true, y_pred)),
        'MCC': float(matthews_corrcoef(y_true, y_pred)),
    }


def _clean_model_params(params):
    """Drop metadata keys before estimator construction."""
    return {k: v for k, v in dict(params or {}).items()
            if not str(k).startswith('__')}


def _suggest_nested_params(trial, algo):
    """Optuna search spaces used inside nested CV."""
    if algo == 'RF':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 12),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'max_samples': trial.suggest_float('max_samples', 0.60, 0.85),
        }
    if algo == 'XGBoost':
        return {
            'n_estimators': 500,
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': float(10 ** trial.suggest_float('reg_alpha_log', -5, 2)),
            'reg_lambda': float(10 ** trial.suggest_float('reg_lambda_log', -5, 2)),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        }
    if algo == 'LightGBM':
        return {
            'n_estimators': 600,
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'reg_alpha': float(10 ** trial.suggest_float('lambda_l1_log', -2, 2)),
            'reg_lambda': float(10 ** trial.suggest_float('lambda_l2_log', -1, 2)),
        }
    if algo == 'SVM':
        return {
            'C': trial.suggest_float('C', 0.1, 1000.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 10.0, log=True),
            'kernel': trial.suggest_categorical('kernel', SVM_KERNEL_OPTIONS),
            'class_weight': trial.suggest_categorical('class_weight', SVM_CLASS_WEIGHT_OPTIONS),
        }
    raise ValueError(f"Unknown algorithm: {algo}")


def _optuna_best_params_to_model_params(algo, best_params):
    """Convert Optuna's stored parameter dict into estimator-ready params."""
    p = dict(best_params or {})
    if algo == 'XGBoost':
        if 'reg_alpha_log' in p:
            p['reg_alpha'] = float(10 ** p.pop('reg_alpha_log'))
        if 'reg_lambda_log' in p:
            p['reg_lambda'] = float(10 ** p.pop('reg_lambda_log'))
        p.setdefault('n_estimators', 500)
    elif algo == 'LightGBM':
        if 'lambda_l1_log' in p:
            p['reg_alpha'] = float(10 ** p.pop('lambda_l1_log'))
        if 'lambda_l2_log' in p:
            p['reg_lambda'] = float(10 ** p.pop('lambda_l2_log'))
        p.setdefault('n_estimators', 600)
    elif algo == 'SVM':
        p = _normalize_svm_params(p)
    return p


def _fit_model_with_fold_preprocessing(algo, params, X_train_raw, y_train,
                                       X_eval_raw, all_feature_cols,
                                       feature_method, seed,
                                       model_n_jobs=None):
    """Fit imputer/feature selector/model on training fold only."""
    imputer = SimpleImputer(strategy='median')
    X_train_all = imputer.fit_transform(X_train_raw)
    X_eval_all = imputer.transform(X_eval_raw)
    X_train_sel, feature_cols_fold, _selection_report = select_core_features(
        X_train_all, y_train, feature_names=all_feature_cols,
        method=feature_method, seed=seed)
    selected_idx = [all_feature_cols.index(f) for f in feature_cols_fold]
    X_eval_sel = X_eval_all[:, selected_idx]
    model = _unified_build_model(
        algo, _clean_model_params(params), seed,
        model_n_jobs=model_n_jobs)
    _unified_fit(model, algo, X_train_sel, y_train)
    preproc = {
        'imputer': imputer,
        'feature_cols': feature_cols_fold,
        'selected_idx': selected_idx,
    }
    return model, preproc, X_eval_sel


def _transform_with_fold_preprocessing(X_raw, preproc):
    X_all = preproc['imputer'].transform(X_raw)
    return X_all[:, preproc['selected_idx']]


def _cv_f1_preprocessed(algo, params, X_raw, y, all_feature_cols,
                        feature_method, seed,
                        n_splits=INNER_CV_SPLITS_DEFAULT,
                        n_repeats=INNER_CV_REPEATS_DEFAULT):
    """Inner CV score with imputation/feature selection fit inside each fold."""
    y = np.asarray(y).astype(int)
    cv = RepeatedStratifiedKFold(
                                 n_splits=int(n_splits),
                                 n_repeats=max(1, int(n_repeats)),
                                 random_state=seed)
    scores = []
    failed_folds = []
    n_total = 0
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_raw, y)):
        n_total += 1
        try:
            model, _preproc, X_va = _fit_model_with_fold_preprocessing(
                algo, params, X_raw[tr_idx], y[tr_idx], X_raw[va_idx],
                all_feature_cols, feature_method, seed + fold_i,
                model_n_jobs=SEARCH_MODEL_N_JOBS)
            y_pred = _predict_label_vector(model, X_va)
            scores.append(f1_score(y[va_idx], y_pred, average='macro',
                                   zero_division=0))
        except Exception as exc:
            failed_folds.append({
                'fold': int(fold_i),
                'err_type': type(exc).__name__,
                'message': str(exc)[:160],
            })
    _summarize_failed_folds(
        f"cv_f1_preprocessed/{algo}", failed_folds, n_total)
    return float(np.mean(scores)) if scores else float('nan')


def _grid_candidate_to_model_params(algo, candidate):
    """Convert one ParameterGrid candidate into _unified_build_model params."""
    if algo == 'SVM':
        return _normalize_svm_params(
            {str(k).replace('svm__', ''): v for k, v in candidate.items()})
    params = dict(candidate)
    if algo == 'XGBoost':
        params.setdefault('n_estimators', 500)
    if algo == 'LightGBM':
        params.setdefault('n_estimators', 600)
    return params


def _sample_parameter_grid(algo, max_grid_candidates=0, seed=42):
    """Return a deterministic random subset of the full grid when capped."""
    grid_candidates = list(ParameterGrid(_GRID_PARAMS[algo]))
    full_grid_size = len(grid_candidates)
    if max_grid_candidates and int(max_grid_candidates) > 0 and full_grid_size > int(max_grid_candidates):
        rng = np.random.default_rng(int(seed))
        picked = np.sort(rng.choice(full_grid_size, size=int(max_grid_candidates), replace=False))
        grid_candidates = [grid_candidates[int(i)] for i in picked]
    return grid_candidates, full_grid_size


def _population_search_preprocessed(algo, optimizer, X_raw, y, all_feature_cols,
                                    feature_method, seed,
                                    pop_size=20, max_iter=50,
                                    inner_splits=5, inner_repeats=3,
                                    F=0.8, CR=0.9,
                                    progress_label=None):
    """PSO/SSA/DE/GWO search whose fitness uses fully fold-internal preprocessing."""
    _set_all_seeds(seed)
    bounds = _POP_BOUNDS[algo]
    dim = len(bounds)
    lb = np.array([b[1] for b in bounds], dtype=float)
    ub = np.array([b[2] for b in bounds], dtype=float)
    cache = {}

    def resolve_continuous(pos_vec):
        raw = {bounds[i][0]: pos_vec[i] for i in range(dim)}
        return _resolve_params(algo, optimizer, {}, {}, {optimizer: {algo: raw}})

    def fitness(pos_vec):
        key = '|'.join(f'{bounds[i][0]}={float(pos_vec[i]):.8f}' for i in range(dim))
        if key not in cache:
            params = resolve_continuous(pos_vec)
            score = _cv_f1_preprocessed(
                algo, params, X_raw, y, all_feature_cols, feature_method,
                seed, n_splits=inner_splits, n_repeats=inner_repeats)
            cache[key] = _safe_cv_score(score)
        return cache[key]

    pos = np.random.uniform(lb, ub, (pop_size, dim))
    fit_arr = np.array([fitness(pos[i]) for i in range(pop_size)])
    best_history = [float(np.max(fit_arr))]
    if progress_label:
        _print_progress(
            f"{progress_label} | Iter 0/{int(max_iter)} | "
            f"best={_fmt_progress_value(best_history[-1])} | "
            f"unique_fitness={len(cache)}")

    if optimizer == 'PSO':
        vel = np.random.uniform(-(ub - lb) * 0.1, (ub - lb) * 0.1, (pop_size, dim))
        pbest_pos = pos.copy()
        pbest_val = fit_arr.copy()
        gbest_pos = pos[np.argmax(fit_arr)].copy()
        gbest_val = float(fit_arr.max())
        w_max, w_min, c1, c2 = 0.9, 0.4, 1.5, 1.5
        v_max = 0.5 * (ub - lb)
        for t in range(max_iter):
            w = w_max - (w_max - w_min) * t / max(max_iter - 1, 1)
            for i in range(pop_size):
                fit_arr[i] = fitness(pos[i])
                if fit_arr[i] > pbest_val[i]:
                    pbest_val[i] = fit_arr[i]
                    pbest_pos[i] = pos[i].copy()
                if fit_arr[i] > gbest_val:
                    gbest_val = float(fit_arr[i])
                    gbest_pos = pos[i].copy()
            for i in range(pop_size):
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                vel[i] = (w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i])
                          + c2 * r2 * (gbest_pos - pos[i]))
                vel[i] = np.clip(vel[i], -v_max, v_max)
                pos[i] = np.clip(pos[i] + vel[i], lb, ub)
            best_history.append(float(gbest_val))
        best_pos = gbest_pos

    elif optimizer == 'SSA':
        ST, PD, SD = 0.8, 0.7, 0.1
        n_disc = max(1, int(PD * pop_size))
        n_alert = max(1, int(SD * pop_size))
        L = np.ones(dim)
        for _it in range(1, max_iter + 1):
            order = np.argsort(-fit_arr)
            X_best = pos[order[0]].copy()
            X_worst = pos[order[-1]].copy()
            for rank_i in range(n_disc):
                idx = order[rank_i]
                R2, Q, alpha_val = np.random.rand(), np.random.randn(dim), np.random.rand()
                pos[idx] = (pos[idx] * np.exp(-rank_i / (alpha_val * max_iter + 1e-10))
                            if R2 < ST else pos[idx] + Q * L)
                pos[idx] = np.clip(pos[idx], lb, ub)
            n_followers = pop_size - n_disc - n_alert
            for rank_i in range(n_followers):
                idx = order[n_disc + rank_i]
                Q = np.random.randn(dim)
                A = np.random.choice([-1, 1], size=dim)
                A_plus = A.T / (A @ A.T + 1e-10)
                pos[idx] = (Q * np.exp((X_worst - pos[idx]) / ((rank_i + 1) ** 2 + 1e-10))
                            if rank_i >= n_followers // 2
                            else X_best + np.abs(pos[idx] - X_best) * A_plus * L)
                pos[idx] = np.clip(pos[idx], lb, ub)
            alerters = np.random.choice(pop_size, size=n_alert, replace=False)
            f_global = fit_arr[order[0]]
            for i in alerters:
                K = np.random.uniform(-1, 1, dim)
                beta = np.random.randn(dim)
                denom = fit_arr[i] - fit_arr[order[-1]] + 1e-10
                pos[i] = (X_best + beta * (pos[i] - X_best)
                          if fit_arr[i] > f_global
                          else pos[i] + K * (np.abs(pos[i] - X_worst) / denom))
                pos[i] = np.clip(pos[i], lb, ub)
            for i in range(pop_size):
                fit_arr[i] = fitness(pos[i])
            cur = float(np.max(fit_arr))
            best_history.append(max(best_history[-1], cur))
            if progress_label:
                _print_progress(
                    f"{progress_label} | Iter {_it}/{int(max_iter)} | "
                    f"best={_fmt_progress_value(best_history[-1])} | "
                    f"unique_fitness={len(cache)}")
        best_pos = pos[np.argmax(fit_arr)]

    elif optimizer == 'GWO':
        sorted_idx = np.argsort(-fit_arr)
        X_alpha = pos[sorted_idx[0]].copy()
        f_alpha = float(fit_arr[sorted_idx[0]])
        X_beta = pos[sorted_idx[1]].copy()
        X_delta = pos[sorted_idx[2]].copy()
        for t in range(1, max_iter + 1):
            a = 2.0 - 2.0 * t / max_iter
            for i in range(pop_size):
                new_pos = np.zeros(dim)
                for leader in [X_alpha, X_beta, X_delta]:
                    r1, r2 = np.random.rand(dim), np.random.rand(dim)
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    new_pos += leader - A * np.abs(C * leader - pos[i])
                pos[i] = np.clip(new_pos / 3.0, lb, ub)
                fit_arr[i] = fitness(pos[i])
            sorted_idx = np.argsort(-fit_arr)
            cand = float(fit_arr[sorted_idx[0]])
            if cand > f_alpha:
                f_alpha = cand
                X_alpha = pos[sorted_idx[0]].copy()
            X_beta = pos[sorted_idx[1]].copy()
            X_delta = pos[sorted_idx[2]].copy()
            best_history.append(f_alpha)
            if progress_label:
                _print_progress(
                    f"{progress_label} | Iter {t}/{int(max_iter)} | "
                    f"best={_fmt_progress_value(f_alpha)} | "
                    f"unique_fitness={len(cache)}")
        best_pos = X_alpha

    else:  # DE
        for iter_i in range(1, max_iter + 1):
            for i in range(pop_size):
                cands = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = np.random.choice(cands, size=3, replace=False)
                mutant = np.clip(pos[r1] + F * (pos[r2] - pos[r3]), lb, ub)
                mask = np.random.rand(dim) < CR
                mask[np.random.randint(dim)] = True
                trial = np.where(mask, mutant, pos[i])
                f_trial = fitness(trial)
                if f_trial >= fit_arr[i]:
                    pos[i] = trial
                    fit_arr[i] = f_trial
            best_history.append(max(best_history[-1], float(np.max(fit_arr))))
            if progress_label:
                _print_progress(
                    f"{progress_label} | Iter {iter_i}/{int(max_iter)} | "
                    f"best={_fmt_progress_value(best_history[-1])} | "
                    f"unique_fitness={len(cache)}")
        best_pos = pos[np.argmax(fit_arr)]

    best_params = resolve_continuous(best_pos)
    best_params['__best_history'] = [float(v) for v in best_history]
    best_params['__internal_cv_f1'] = float(max(best_history))
    best_params['__n_fitness_evals_unique'] = int(len(cache))
    best_params['__n_fitness_evals_total'] = int(pop_size + pop_size * max_iter)
    return best_params, float(max(best_history))


def _nested_search_once(algo, optimizer, X_raw, y, all_feature_cols,
                        feature_method, seed, n_trials_by_algo,
                        pop_size=20, max_iter=50,
                        inner_splits=5, inner_repeats=3,
                        max_grid_candidates=0,
                        progress_label=None):
    """One inner hyperparameter search, using only the supplied training fold."""
    _set_all_seeds(seed)
    t0 = time.perf_counter()
    if optimizer == 'Default':
        params = dict(_DEFAULT_PARAMS[algo])
        score = _safe_cv_score(_cv_f1_preprocessed(
            algo, params, X_raw, y, all_feature_cols, feature_method, seed,
            n_splits=inner_splits, n_repeats=inner_repeats))
        if progress_label:
            _print_progress(
                f"{progress_label} | Default CV complete | "
                f"score={_fmt_progress_value(score)}")
        params['__best_history'] = [float(score)]
        params['__n_fitness_evals_unique'] = 1
        params['__n_fitness_evals_total'] = 1
    elif optimizer == 'Optuna':
        n_trials_algo = max(1, int(n_trials_by_algo.get(algo, 100)))
        def objective(trial):
            params_trial = _suggest_nested_params(trial, algo)
            raw_score = _cv_f1_preprocessed(
                algo, params_trial, X_raw, y, all_feature_cols,
                feature_method, seed, n_splits=inner_splits,
                n_repeats=inner_repeats)
            return _safe_cv_score(raw_score)
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=seed, n_startup_trials=min(10, n_trials_algo)))
        callbacks = []
        if progress_label:
            callbacks.append(_make_optuna_progress_callback(
                progress_label, n_trials_algo))
        study.optimize(objective, n_trials=n_trials_algo, show_progress_bar=False,
                       callbacks=callbacks, n_jobs=OPTUNA_N_JOBS,
                       gc_after_trial=True)
        params = _optuna_best_params_to_model_params(algo, study.best_params)
        score = float(study.best_value)
        params['__n_trials'] = len(study.trials)
        trial_values = [
            float(t.value) for t in study.trials
            if getattr(t, 'value', None) is not None
        ]
        params['__best_history'] = [
            float(v) for v in np.maximum.accumulate(trial_values)
        ] if trial_values else [float(score)]
        params['__n_fitness_evals_unique'] = int(len(trial_values))
        params['__n_fitness_evals_total'] = int(len(study.trials))
    elif optimizer == 'GridSearch':
        best_score, best_params = -1.0, None
        history = []
        grid_candidates, full_grid_size = _sample_parameter_grid(
            algo, max_grid_candidates=max_grid_candidates, seed=seed)
        for cand_i, cand in enumerate(grid_candidates, 1):
            params_cand = _grid_candidate_to_model_params(algo, cand)
            sc = _cv_f1_preprocessed(
                algo, params_cand, X_raw, y, all_feature_cols,
                feature_method, seed, n_splits=inner_splits,
                n_repeats=inner_repeats)
            sc = _safe_cv_score(sc)
            history.append(float(sc))
            if sc > best_score:
                best_score, best_params = float(sc), dict(params_cand)
            if progress_label:
                _print_progress(
                    f"{progress_label} | Candidate {cand_i}/{len(grid_candidates)} | "
                    f"score={_fmt_progress_value(sc)} | "
                    f"best={_fmt_progress_value(best_score)}")
        params = best_params or dict(_DEFAULT_PARAMS[algo])
        score = float(best_score)
        params['__best_history'] = [float(v) for v in np.maximum.accumulate(history)]
        params['__n_grid_candidates'] = int(len(history))
        params['__n_fitness_evals_unique'] = int(len(history))
        params['__n_fitness_evals_total'] = int(len(history))
        params['__grid_candidates_sampled_from_full'] = int(full_grid_size)
    elif optimizer in ('PSO', 'SSA', 'DE', 'GWO'):
        params, score = _population_search_preprocessed(
            algo, optimizer, X_raw, y, all_feature_cols, feature_method, seed,
            pop_size=pop_size, max_iter=max_iter,
            inner_splits=inner_splits, inner_repeats=inner_repeats,
            progress_label=progress_label)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    elapsed = time.perf_counter() - t0
    params = dict(params)
    params['__internal_cv_f1'] = float(score)
    params['__time_s'] = float(elapsed)
    if progress_label:
        _print_progress(
            f"{progress_label} | Search complete | "
            f"best={_fmt_progress_value(score)} | time={elapsed:.1f}s")
    return params, float(score), float(elapsed)


def _load_external_validation_raw_for_generalization(
        val_path, all_feature_cols, target_col='y', ion_cols=None, le=None,
        water_source_mapping=None):
    """Return raw external feature matrix; no imputation or selection is fit here."""
    if ion_cols is None:
        ion_cols = ION_COLS
    if val_path is None or not Path(val_path).exists():
        return None, None, {'available': False, 'reason': 'file_not_found'}
    val_df = pd.read_excel(val_path)
    feature_input_cols = list(ion_cols) + [PH_COL]
    expected_cols = feature_input_cols + [target_col]
    first_col = str(val_df.columns[0])
    if first_col.replace('.', '', 1).replace('-', '', 1).isdigit() or ion_cols[0] not in val_df.columns:
        if val_df.shape[1] == len(expected_cols):
            val_df = pd.read_excel(val_path, header=None, names=expected_cols)
        else:
            return None, None, {'available': False, 'reason': 'column_count_mismatch'}
    target_col = resolve_target_column(val_df, target_col)
    val_df_fe = create_geochemical_features(val_df.copy(), target_col, ion_cols)
    X_val_raw = val_df_fe[all_feature_cols].values.astype(float)
    y_val_raw = np.array(
        [canonicalize_water_source_label(lbl) for lbl in val_df_fe[target_col].values],
        dtype=object)
    if le is None:
        return X_val_raw, None, {'available': True, 'n_samples': int(len(X_val_raw))}
    unknown = set(y_val_raw) - set(le.classes_)
    if unknown:
        mask = np.isin(y_val_raw, le.classes_)
        X_val_raw, y_val_raw = X_val_raw[mask], y_val_raw[mask]
    y_val = le.transform(y_val_raw)
    unique_v, counts_v = np.unique(y_val, return_counts=True)
    dist = {
        (water_source_mapping or {}).get(int(u), f'C{u}'): int(c)
        for u, c in zip(unique_v, counts_v)
    }
    return X_val_raw, y_val, {'available': True, 'n_samples': int(len(y_val)),
                              'class_distribution': dist}


def _summarize_generalization_records(records, phase_label):
    df = pd.DataFrame(records)
    if df.empty:
        return df
    metric_cols = [c for c in df.columns
                   if c.endswith('_F1_Macro') or c.endswith('_Accuracy')
                   or c.endswith('_Kappa') or c.endswith('_MCC')
                   or c in ('Inner_CV_F1', 'Train_Internal_CV_F1',
                            'Search_Time_s', 'Generalization_Gap')]
    rows = []
    for model_name, grp in df.groupby('Model'):
        row = {
            'Phase': phase_label,
            'Model': model_name,
            'Algorithm': grp['Algorithm'].iloc[0],
            'Optimizer': grp['Optimizer'].iloc[0],
            'N': int(len(grp)),
        }
        for col in metric_cols:
            vals = pd.to_numeric(grp[col], errors='coerce').dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            row[f'{col}_mean'] = float(np.mean(vals))
            row[f'{col}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f'{col}_median'] = float(np.median(vals))
            row[f'{col}_iqr'] = float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))
            if len(vals) > 1:
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                tval = float(stats.t.ppf(0.975, df=len(vals) - 1))
                row[f'{col}_ci95_low'] = float(np.mean(vals) - tval * se)
                row[f'{col}_ci95_high'] = float(np.mean(vals) + tval * se)
        rows.append(row)
    out = pd.DataFrame(rows)
    if phase_label == 'nested_outer_cv' and 'Outer_F1_Macro_mean' in out.columns:
        sort_col = 'Outer_F1_Macro_mean'
    elif 'Test_F1_Macro_mean' in out.columns:
        # External validation is intentionally not the primary sort key here;
        # validation rankings are reported separately as descriptive diagnostics.
        sort_col = 'Test_F1_Macro_mean'
    else:
        sort_col = None
    if sort_col:
        out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
        out.insert(0, f'Rank_By_{sort_col.replace("_mean", "")}',
                   np.arange(1, len(out) + 1))
    return out


def _outer_rank_stability(outer_records):
    df = pd.DataFrame(outer_records)
    if df.empty:
        return df
    rank_rows = []
    for outer_id, grp in df.groupby('Outer_Fold'):
        tmp = grp[['Model', 'Outer_F1_Macro']].dropna().copy()
        tmp['Rank'] = tmp['Outer_F1_Macro'].rank(method='average', ascending=False)
        tmp['Outer_Fold'] = outer_id
        rank_rows.append(tmp)
    if not rank_rows:
        return pd.DataFrame()
    ranks = pd.concat(rank_rows, ignore_index=True)
    rows = []
    for model_name, grp in ranks.groupby('Model'):
        vals = grp['Rank'].to_numpy(float)
        rows.append({
            'Model': model_name,
            'Mean_Rank': float(np.mean(vals)),
            'Std_Rank': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'Median_Rank': float(np.median(vals)),
            'Top1_Count': int(np.sum(vals == 1)),
            'N_Outer_Folds': int(len(vals)),
        })
    return pd.DataFrame(rows).sort_values(['Mean_Rank', 'Std_Rank', 'Model']).reset_index(drop=True)


def _spearman_record(label_a, values_a, label_b, values_b):
    common = sorted(set(values_a) & set(values_b))
    if len(common) < 3:
        return {'Comparison': f'{label_a}_vs_{label_b}', 'N': len(common),
                'Spearman_Rho': np.nan, 'P_Value': np.nan}
    a = np.array([values_a[k] for k in common], dtype=float)
    b = np.array([values_b[k] for k in common], dtype=float)
    rho, p = stats.spearmanr(a, b)
    return {'Comparison': f'{label_a}_vs_{label_b}', 'N': len(common),
            'Spearman_Rho': float(rho), 'P_Value': float(p)}


def _two_way_variance_decomposition(records, response_col):
    df = pd.DataFrame(records)
    if df.empty or response_col not in df.columns:
        return pd.DataFrame()
    df = df[['Algorithm', 'Optimizer', response_col]].dropna()
    if df.empty:
        return pd.DataFrame()
    y = df[response_col].astype(float)
    grand = float(y.mean())
    ss_total = float(np.sum((y - grand) ** 2))
    if ss_total <= 0:
        return pd.DataFrame([{'Response': response_col, 'Factor': 'Total',
                              'SS': ss_total, 'Eta2': np.nan}])
    mean_a = df.groupby('Algorithm')[response_col].mean()
    mean_o = df.groupby('Optimizer')[response_col].mean()
    mean_cell = df.groupby(['Algorithm', 'Optimizer'])[response_col].mean()
    ss_algo = float(sum(len(df[df['Algorithm'] == a]) * (m - grand) ** 2
                        for a, m in mean_a.items()))
    ss_opt = float(sum(len(df[df['Optimizer'] == o]) * (m - grand) ** 2
                       for o, m in mean_o.items()))
    ss_inter = 0.0
    ss_resid = 0.0
    for (a, o), m_cell in mean_cell.items():
        cell = df[(df['Algorithm'] == a) & (df['Optimizer'] == o)]
        expected_additive = mean_a[a] + mean_o[o] - grand
        ss_inter += float(len(cell) * (m_cell - expected_additive) ** 2)
        ss_resid += float(np.sum((cell[response_col].astype(float) - m_cell) ** 2))
    rows = [
        {'Response': response_col, 'Factor': 'Algorithm', 'SS': ss_algo, 'Eta2': ss_algo / ss_total},
        {'Response': response_col, 'Factor': 'Optimizer', 'SS': ss_opt, 'Eta2': ss_opt / ss_total},
        {'Response': response_col, 'Factor': 'Algorithm_x_Optimizer', 'SS': ss_inter, 'Eta2': ss_inter / ss_total},
        {'Response': response_col, 'Factor': 'Run_Residual', 'SS': ss_resid, 'Eta2': ss_resid / ss_total},
        {'Response': response_col, 'Factor': 'Total', 'SS': ss_total, 'Eta2': 1.0},
    ]
    return pd.DataFrame(rows)


def _write_df(df, path_no_ext):
    if df is None or df.empty:
        return
    df.to_csv(str(path_no_ext) + '.csv', index=False)
    try:
        df.to_excel(str(path_no_ext) + '.xlsx', index=False)
    except Exception:
        pass


def _class_name(class_id, water_source_mapping=None):
    return (water_source_mapping or {}).get(int(class_id), f'Class {int(class_id)}')


def _predict_proba_matrix(model, X, all_classes):
    """Return probabilities aligned to all_classes, or None if unavailable."""
    if model is None or not hasattr(model, 'predict_proba'):
        return None
    try:
        raw = np.asarray(model.predict_proba(X), dtype=float)
    except Exception:
        return None
    if raw.ndim != 2:
        return None
    all_classes = [int(c) for c in all_classes]
    out = np.full((raw.shape[0], len(all_classes)), np.nan, dtype=float)
    model_classes = getattr(model, 'classes_', None)
    if model_classes is None and isinstance(model, Pipeline):
        model_classes = getattr(model, 'classes_', None)
    if model_classes is None and raw.shape[1] == len(all_classes):
        return raw
    if model_classes is None:
        return None
    class_pos = {int(c): i for i, c in enumerate(all_classes)}
    for src_i, cls in enumerate(model_classes):
        cls_i = int(cls)
        if cls_i in class_pos and src_i < raw.shape[1]:
            out[:, class_pos[cls_i]] = raw[:, src_i]
    return out


def _prediction_rows(phase, dataset, model_name, algo, optimizer, y_true, y_pred,
                     y_proba, sample_indices, all_classes, water_source_mapping,
                     run_seed=None, outer_fold=None, outer_seed=None):
    rows = []
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sample_indices = np.asarray(sample_indices if sample_indices is not None
                                else np.arange(len(y_true))).astype(int)
    for i in range(len(y_true)):
        row = {
            'Phase': phase,
            'Dataset': dataset,
            'Model': model_name,
            'Algorithm': algo,
            'Optimizer': optimizer,
            'Run_Seed': None if run_seed is None else int(run_seed),
            'Outer_Fold': None if outer_fold is None else int(outer_fold),
            'Outer_Seed': None if outer_seed is None else int(outer_seed),
            'Sample_Index': int(sample_indices[i]),
            'True_Label': int(y_true[i]),
            'True_Name': _class_name(y_true[i], water_source_mapping),
            'Pred_Label': int(y_pred[i]),
            'Pred_Name': _class_name(y_pred[i], water_source_mapping),
            'Correct': bool(y_true[i] == y_pred[i]),
        }
        if y_proba is not None:
            for pos, class_id in enumerate(all_classes):
                row[f'Prob_Class{int(class_id)}'] = float(y_proba[i, pos])
        rows.append(row)
    return rows


def _search_history_rows(search_phase, algo, optimizer, model_name, params,
                         inner_splits, inner_repeats, run_seed=None,
                         outer_fold=None, outer_seed=None):
    history = params.get('__best_history', None) if isinstance(params, dict) else None
    if history is None or len(history) == 0:
        history = [params.get('__internal_cv_f1', np.nan)] if isinstance(params, dict) else [np.nan]
    rows = []
    n_total = int(params.get('__n_fitness_evals_total', len(history))) if isinstance(params, dict) else len(history)
    n_unique = int(params.get('__n_fitness_evals_unique', len(history))) if isinstance(params, dict) else len(history)
    for step, value in enumerate(history, 1):
        rows.append({
            'Search_Phase': search_phase,
            'Model': model_name,
            'Algorithm': algo,
            'Optimizer': optimizer,
            'Run_Seed': None if run_seed is None else int(run_seed),
            'Outer_Fold': None if outer_fold is None else int(outer_fold),
            'Outer_Seed': None if outer_seed is None else int(outer_seed),
            'Step': int(step),
            'Best_Internal_CV_F1': float(value) if pd.notna(value) else np.nan,
            'Progress': float(step / max(len(history), 1)),
            'Inner_CV_Folds_Per_Eval': int(inner_splits) * int(inner_repeats),
            'Fitness_Evals_Total': n_total,
            'Fitness_Evals_Unique': n_unique,
            'Nominal_Model_Fits_Total': n_total * int(inner_splits) * int(inner_repeats),
        })
    return rows


def _search_budget_row(search_phase, algo, optimizer, model_name, params,
                       search_time, inner_splits, inner_repeats, run_seed=None,
                       outer_fold=None, outer_seed=None):
    n_total = int(params.get('__n_fitness_evals_total', params.get('__n_trials',
                  params.get('__n_grid_candidates', 1)))) if isinstance(params, dict) else 1
    n_unique = int(params.get('__n_fitness_evals_unique', n_total)) if isinstance(params, dict) else n_total
    return {
        'Search_Phase': search_phase,
        'Model': model_name,
        'Algorithm': algo,
        'Optimizer': optimizer,
        'Run_Seed': None if run_seed is None else int(run_seed),
        'Outer_Fold': None if outer_fold is None else int(outer_fold),
        'Outer_Seed': None if outer_seed is None else int(outer_seed),
        'Fitness_Evals_Total': n_total,
        'Fitness_Evals_Unique': n_unique,
        'Inner_CV_Folds_Per_Eval': int(inner_splits) * int(inner_repeats),
        'Nominal_Model_Fits_Total': n_total * int(inner_splits) * int(inner_repeats),
        'Search_Time_s': float(search_time),
    }


def _search_space_audit_table(n_trials_by_algo, pop_size, max_iter,
                              max_grid_candidates, inner_splits, inner_repeats):
    rows = []
    inner_folds = int(inner_splits) * int(inner_repeats)
    for algo in _ALGORITHMS:
        for optimizer in _OPTIMIZERS:
            if optimizer == 'Default':
                evals = 1
                space_note = 'fixed_default_params'
            elif optimizer == 'Optuna':
                evals = int(n_trials_by_algo.get(algo, 100))
                space_note = 'tpe_trials'
            elif optimizer == 'GridSearch':
                full_grid = len(list(ParameterGrid(_GRID_PARAMS[algo])))
                if max_grid_candidates and int(max_grid_candidates) > 0:
                    evals = min(full_grid, int(max_grid_candidates))
                    space_note = f'grid_candidates_capped_from_{full_grid}'
                else:
                    evals = full_grid
                    space_note = 'full_grid'
            else:
                evals = int(pop_size) + int(pop_size) * int(max_iter)
                space_note = f'population_{int(pop_size)}x{int(max_iter)}'
            rows.append({
                'Model': f'{algo}-{optimizer}',
                'Algorithm': algo,
                'Optimizer': optimizer,
                'Planned_Fitness_Evals': int(evals),
                'Inner_CV_Folds_Per_Eval': int(inner_folds),
                'Planned_Model_Fits_Per_Search': int(evals) * int(inner_folds),
                'Search_Space_Note': space_note,
            })
    return pd.DataFrame(rows)


def _make_prediction_diagnostic_tables(pred_df, all_classes, water_source_mapping,
                                       output_dir, stem):
    if pred_df is None or pred_df.empty:
        return {}
    tables_dir = output_dir / 'Tables'
    classes = [int(c) for c in all_classes]
    group_cols = ['Phase', 'Dataset', 'Model', 'Algorithm', 'Optimizer', 'Run_Seed']
    group_cols = [c for c in group_cols if c in pred_df.columns]
    prob_cols = [f'Prob_Class{c}' for c in classes if f'Prob_Class{c}' in pred_df.columns]

    confusion_rows, per_class_rows, roc_rows, auc_rows, calib_rows = [], [], [], [], []
    for keys, grp in pred_df.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        y_true = grp['True_Label'].to_numpy(int)
        y_pred = grp['Pred_Label'].to_numpy(int)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        for i, true_cls in enumerate(classes):
            row_sum = int(cm[i, :].sum())
            for j, pred_cls in enumerate(classes):
                confusion_rows.append({
                    **meta,
                    'True_Label': int(true_cls),
                    'True_Name': _class_name(true_cls, water_source_mapping),
                    'Pred_Label': int(pred_cls),
                    'Pred_Name': _class_name(pred_cls, water_source_mapping),
                    'Count': int(cm[i, j]),
                    'Row_Normalized': float(cm[i, j] / row_sum) if row_sum else np.nan,
                })

        p, r, f, support = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, zero_division=0)
        for pos, cls in enumerate(classes):
            per_class_rows.append({
                **meta,
                'Class_Label': int(cls),
                'Class_Name': _class_name(cls, water_source_mapping),
                'Precision': float(p[pos]),
                'Recall': float(r[pos]),
                'F1': float(f[pos]),
                'Support': int(support[pos]),
            })

        if len(prob_cols) == len(classes):
            try:
                from sklearn.metrics import roc_curve, auc
                from sklearn.preprocessing import label_binarize
                from sklearn.calibration import calibration_curve
                y_bin = label_binarize(y_true, classes=classes)
                y_proba = grp[prob_cols].to_numpy(float)
                for pos, cls in enumerate(classes):
                    if y_bin[:, pos].min() == y_bin[:, pos].max():
                        continue
                    fpr, tpr, thresholds = roc_curve(y_bin[:, pos], y_proba[:, pos])
                    auc_val = float(auc(fpr, tpr))
                    auc_rows.append({
                        **meta,
                        'Class_Label': int(cls),
                        'Class_Name': _class_name(cls, water_source_mapping),
                        'AUC': auc_val,
                    })
                    for point_i in range(len(fpr)):
                        roc_rows.append({
                            **meta,
                            'Class_Label': int(cls),
                            'Class_Name': _class_name(cls, water_source_mapping),
                            'Point': int(point_i),
                            'FPR': float(fpr[point_i]),
                            'TPR': float(tpr[point_i]),
                            'Threshold': float(thresholds[point_i]),
                            'AUC': auc_val,
                        })
                    prob_true, prob_pred = calibration_curve(
                        y_bin[:, pos], y_proba[:, pos], n_bins=10, strategy='uniform')
                    bins = np.linspace(0.0, 1.0, 11)
                    bin_id = np.clip(np.digitize(y_proba[:, pos], bins) - 1, 0, 9)
                    ece = 0.0
                    for b in range(10):
                        mask = bin_id == b
                        if not np.any(mask):
                            continue
                        acc_b = float(y_bin[mask, pos].mean())
                        conf_b = float(y_proba[mask, pos].mean())
                        ece += float(np.abs(acc_b - conf_b) * mask.mean())
                    for point_i in range(len(prob_true)):
                        calib_rows.append({
                            **meta,
                            'Class_Label': int(cls),
                            'Class_Name': _class_name(cls, water_source_mapping),
                            'Point': int(point_i),
                            'Mean_Predicted_Probability': float(prob_pred[point_i]),
                            'Observed_Fraction': float(prob_true[point_i]),
                            'ECE_10bin': float(ece),
                        })
            except Exception as exc:
                auc_rows.append({**meta, 'Class_Label': None,
                                 'Class_Name': '__ERROR__', 'AUC': np.nan,
                                 'Error': str(exc)})

    outputs = {}
    for name, rows in [
            ('Confusion_Matrices_Long', confusion_rows),
            ('PerClass_Metrics_Long', per_class_rows),
            ('ROC_Curves_Long', roc_rows),
            ('ROC_AUC_Long', auc_rows),
            ('Calibration_Curves_Long', calib_rows)]:
        df = pd.DataFrame(rows)
        outputs[name] = df
        _write_df(df, tables_dir / f'{stem}_{name}')
    return outputs


def _write_dataset_archives(output_dir, X_train_raw, y_train, X_test_raw, y_test,
                            X_val_raw, y_val, all_feature_cols,
                            water_source_mapping):
    tables_dir = output_dir / 'Tables'
    regen_dir = output_dir / 'RegenData'
    regen_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'X_train_raw': X_train_raw,
        'y_train': np.asarray(y_train).astype(int),
        'X_test_raw': X_test_raw,
        'y_test': np.asarray(y_test).astype(int),
        'feature_names': np.asarray(all_feature_cols, dtype=str),
    }
    if X_val_raw is not None:
        payload['X_val_raw'] = X_val_raw
    if y_val is not None:
        payload['y_val'] = np.asarray(y_val).astype(int)
    np.savez_compressed(regen_dir / 'nested_raw_arrays.npz', **payload)
    np.savez_compressed(regen_dir / 'feature_arrays.npz', **payload)
    with open(regen_dir / 'nested_feature_columns.json', 'w', encoding='utf-8') as f:
        json.dump({'all_feature_cols': list(all_feature_cols),
                   'core_feature_cols': list(CORE_FEATURE_COLS),
                   'feature_display_names': FEATURE_DISPLAY_NAMES},
                  f, indent=2, ensure_ascii=False)

    split_data = [
        ('train', X_train_raw, y_train),
        ('test', X_test_raw, y_test),
    ]
    if X_val_raw is not None and y_val is not None:
        split_data.append(('external_val', X_val_raw, y_val))

    class_rows, summary_rows, ks_rows = [], [], []
    for split, X, y in split_data:
        y = np.asarray(y).astype(int)
        for cls, count in zip(*np.unique(y, return_counts=True)):
            class_rows.append({
                'Split': split,
                'Class_Label': int(cls),
                'Class_Name': _class_name(cls, water_source_mapping),
                'N': int(count),
                'Percent': float(count / max(len(y), 1) * 100.0),
            })
        X_df = pd.DataFrame(X, columns=all_feature_cols)
        for feature in all_feature_cols:
            vals = pd.to_numeric(X_df[feature], errors='coerce').dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            summary_rows.append({
                'Split': split,
                'Feature': feature,
                'Feature_Display': FEATURE_DISPLAY_NAMES.get(feature, feature),
                'N': int(len(vals)),
                'Mean': float(np.mean(vals)),
                'Std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'Median': float(np.median(vals)),
                'IQR': float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25)),
                'Min': float(np.min(vals)),
                'Max': float(np.max(vals)),
            })

    train_df = pd.DataFrame(X_train_raw, columns=all_feature_cols)
    for split, X, _y in split_data[1:]:
        other_df = pd.DataFrame(X, columns=all_feature_cols)
        for feature in all_feature_cols:
            a = pd.to_numeric(train_df[feature], errors='coerce').dropna().to_numpy(float)
            b = pd.to_numeric(other_df[feature], errors='coerce').dropna().to_numpy(float)
            if len(a) == 0 or len(b) == 0:
                continue
            stat, pval = stats.ks_2samp(a, b)
            ks_rows.append({
                'Reference_Split': 'train',
                'Compared_Split': split,
                'Feature': feature,
                'Feature_Display': FEATURE_DISPLAY_NAMES.get(feature, feature),
                'KS_Statistic': float(stat),
                'P_Value': float(pval),
                'Train_Mean': float(np.mean(a)),
                'Compared_Mean': float(np.mean(b)),
            })

    _write_df(pd.DataFrame(class_rows), tables_dir / 'Dataset_Class_Distribution')
    _write_df(pd.DataFrame(summary_rows), tables_dir / 'Dataset_Feature_Summary')
    _write_df(pd.DataFrame(ks_rows), tables_dir / 'Dataset_DomainShift_KS')


def _generate_full_shap_archive(final_model_registry, X_test_raw, y_test,
                                X_val_raw, y_val, all_classes,
                                water_source_mapping, output_dir,
                                skip=False, svm_explain_size=240,
                                background_size=64):
    """Save raw and classwise SHAP data for all representative final models."""
    if skip or not final_model_registry:
        return {}
    root = output_dir / 'Tables' / 'SHAP_FinalArchive'
    regen_dir = output_dir / 'RegenData'
    root.mkdir(parents=True, exist_ok=True)
    regen_dir.mkdir(parents=True, exist_ok=True)
    class_labels = [_class_name(c, water_source_mapping) for c in all_classes]
    datasets = [('test', X_test_raw, y_test)]
    if X_val_raw is not None and y_val is not None:
        datasets.append(('external_val', X_val_raw, y_val))

    cache = {}
    metadata_rows, mean_rows, classwise_mean_rows = [], [], []
    for dataset_name, X_raw, y_true in datasets:
        ds_dir = root / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        y_true = np.asarray(y_true).astype(int)
        cache[dataset_name] = {}
        for model_name, payload in final_model_registry.items():
            safe = _safe_name_stem(model_name)
            model = payload['model']
            preproc = payload['preproc']
            feature_cols = list(preproc['feature_cols'])
            X_sel = _transform_with_fold_preprocessing(X_raw, preproc)
            try:
                shap_matrix, X_used, backend, shap_meta = compute_shap_robust(
                    model, X_sel, background_size=int(background_size),
                    svm_explain_size=int(svm_explain_size), return_metadata=True)
                mean_abs = mean_abs_shap(shap_matrix)
                sample_idx = np.asarray(shap_meta.get('sample_indices'), dtype=int)
                if sample_idx.size == 0:
                    sample_idx = np.arange(X_used.shape[0], dtype=int)
                y_pred_all = _predict_label_vector(model, X_sel)
                y_proba_all = _predict_proba_matrix(model, X_sel, all_classes)
                raw_df = pd.DataFrame(shap_matrix, columns=feature_cols[:shap_matrix.shape[1]])
                raw_df.insert(0, 'SHAP_Row', np.arange(len(raw_df)))
                raw_df.insert(1, 'Original_Row_Index', sample_idx)
                raw_df.insert(2, 'True_Label', y_true[sample_idx])
                raw_df.insert(3, 'Pred_Label', y_pred_all[sample_idx])
                raw_df.to_csv(ds_dir / f'SHAP_Raw_{safe}.csv', index=False)

                x_used_df = pd.DataFrame(X_used, columns=feature_cols[:X_used.shape[1]])
                x_used_df.insert(0, 'Original_Row_Index', sample_idx)
                x_used_df.insert(1, 'True_Label', y_true[sample_idx])
                x_used_df.insert(2, 'Pred_Label', y_pred_all[sample_idx])
                if y_proba_all is not None:
                    for pos, cls in enumerate(all_classes):
                        x_used_df[f'Prob_Class{int(cls)}'] = y_proba_all[sample_idx, pos]
                x_used_df.to_csv(ds_dir / f'SHAP_Input_XUsed_{safe}.csv', index=False)

                for feat, val in zip(feature_cols[:len(mean_abs)], mean_abs):
                    mean_rows.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Algorithm': payload['Algorithm'],
                        'Optimizer': payload['Optimizer'],
                        'Run_Seed': int(payload['Run_Seed']),
                        'Feature': feat,
                        'Feature_Display': FEATURE_DISPLAY_NAMES.get(feat, feat),
                        'Mean_Abs_SHAP': float(val),
                    })

                classwise_values = shap_meta.get('classwise_values')
                if classwise_values is not None:
                    np.savez_compressed(
                        ds_dir / f'SHAP_Classwise_{safe}.npz',
                        values=classwise_values,
                        sample_indices=sample_idx,
                        feature_names=np.asarray(feature_cols[:classwise_values.shape[2]], dtype=str),
                        class_ids=np.asarray(all_classes, dtype=int),
                        class_labels=np.asarray(class_labels, dtype=str),
                    )
                    for class_pos in range(classwise_values.shape[0]):
                        cls_id = int(all_classes[class_pos]) if class_pos < len(all_classes) else class_pos
                        vals = np.mean(np.abs(classwise_values[class_pos]), axis=0)
                        for feat, val in zip(feature_cols[:len(vals)], vals):
                            classwise_mean_rows.append({
                                'Dataset': dataset_name,
                                'Model': model_name,
                                'Algorithm': payload['Algorithm'],
                                'Optimizer': payload['Optimizer'],
                                'Run_Seed': int(payload['Run_Seed']),
                                'Class_Label': cls_id,
                                'Class_Name': _class_name(cls_id, water_source_mapping),
                                'Feature': feat,
                                'Feature_Display': FEATURE_DISPLAY_NAMES.get(feat, feat),
                                'Mean_Abs_SHAP': float(val),
                            })

                metadata_rows.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Algorithm': payload['Algorithm'],
                    'Optimizer': payload['Optimizer'],
                    'Run_Seed': int(payload['Run_Seed']),
                    'Backend': backend,
                    'N_Input_Samples': int(X_sel.shape[0]),
                    'N_SHAP_Samples': int(shap_matrix.shape[0]),
                    'N_Features': int(shap_matrix.shape[1]),
                    'Feature_Columns_JSON': json.dumps(feature_cols, ensure_ascii=False),
                    'Classwise_SHAP_Available': bool(classwise_values is not None),
                    'Selection_Note': 'representative_full_training_model_post_hoc_only',
                })
                cache[dataset_name][model_name] = {
                    'shap_values': shap_matrix,
                    'x_used': X_used,
                    'sample_indices': sample_idx,
                    'classwise_values': classwise_values,
                    'feature_cols': feature_cols,
                    'backend': backend,
                }
                print(f"  [SHAP Archive] {dataset_name} {model_name}: saved ({backend}).")
            except Exception as exc:
                metadata_rows.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Algorithm': payload['Algorithm'],
                    'Optimizer': payload['Optimizer'],
                    'Run_Seed': int(payload['Run_Seed']),
                    'Backend': '__ERROR__',
                    'N_Input_Samples': int(X_sel.shape[0]),
                    'N_SHAP_Samples': 0,
                    'N_Features': int(X_sel.shape[1]),
                    'Feature_Columns_JSON': json.dumps(feature_cols, ensure_ascii=False),
                    'Classwise_SHAP_Available': False,
                    'Selection_Note': str(exc),
                })
                print(f"  [SHAP Archive] {dataset_name} {model_name}: FAILED ({exc})")

    metadata_df = pd.DataFrame(metadata_rows)
    mean_df = pd.DataFrame(mean_rows)
    classwise_df = pd.DataFrame(classwise_mean_rows)
    _write_df(metadata_df, root / 'SHAP_FinalArchive_Metadata')
    _write_df(mean_df, root / 'SHAP_FinalArchive_MeanAbs')
    _write_df(classwise_df, root / 'SHAP_FinalArchive_Classwise_MeanAbs')
    joblib.dump(cache, regen_dir / 'shap_final_archive.pkl')
    return {
        'metadata': metadata_df,
        'mean_abs': mean_df,
        'classwise_mean_abs': classwise_df,
    }


def _write_nested_regen_bundle(output_dir, payload):
    regen_dir = output_dir / 'RegenData'
    regen_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, regen_dir / 'nested_generalization_payload.pkl')
    for key, filename in [
            ('outer_predictions', 'nested_outer_predictions.pkl'),
            ('final_predictions', 'final_predictions.pkl'),
            ('search_history', 'search_convergence_long.pkl'),
            ('search_history', 'convergence_records.pkl'),
            ('search_budget', 'search_budget_audit.pkl'),
            ('search_space_audit', 'search_space_audit.pkl'),
            ('final_model_registry', 'final_model_registry.pkl')]:
        if key in payload:
            joblib.dump(payload[key], regen_dir / filename)
    manifest = {
        'purpose': 'All data needed to redraw paper figures without re-training.',
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'primary_tables': {
            'nested_outer_raw': 'Tables/NestedCV_Outer_Raw.csv',
            'nested_outer_predictions': 'Tables/NestedCV_Outer_Predictions.csv',
            'final_eval_raw': 'Tables/FinalEval_Test_External_Raw.csv',
            'final_predictions': 'Tables/FinalEval_Test_External_Predictions.csv',
            'search_convergence': 'Tables/Search_Convergence_Long.csv',
            'search_budget': 'Tables/Search_Budget_Audit.csv',
            'search_space_audit': 'Tables/Search_Space_Audit.csv',
            'shap_archive': 'Tables/SHAP_FinalArchive/',
            'raw_arrays': 'RegenData/nested_raw_arrays.npz',
            'payload_pickle': 'RegenData/nested_generalization_payload.pkl',
            'final_model_registry': 'RegenData/final_model_registry.pkl',
            'shap_pickle': 'RegenData/shap_final_archive.pkl',
        },
        'anticipated_figures': [
            'dataset class distribution and feature/domain-shift summaries',
            'nested outer-CV boxplots and rank-stability plots',
            'internal-test vs external-validation ranking/scatter plots',
            'generalization-gap heatmaps and bar charts',
            'algorithm vs optimizer effect-size plots',
            'optimizer convergence curves for all algorithms and optimizers',
            'test/external confusion matrices and per-class metric panels',
            'test/external ROC and calibration curves from saved probabilities',
            'overall and classwise SHAP bar/beeswarm/heatmap plots for test and external validation',
        ],
    }
    with open(regen_dir / 'figure_data_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(_json_safe(manifest), f, indent=2, ensure_ascii=False)
    return manifest


def _read_csv_if_exists(path):
    """Read CSV safely; return empty DataFrame when file is absent/unreadable."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as exc:
        warnings.warn(f"Failed to read CSV '{p}': {exc}", RuntimeWarning)
        return pd.DataFrame()


def _export_sci_figure_table_bundle(output_dir, water_source_mapping=None):
    """Materialize all figure/table redraw datasets and a machine-readable index."""
    output_dir = Path(output_dir)
    tables_dir = output_dir / 'Tables'
    regen_dir = output_dir / 'RegenData'
    tables_dir.mkdir(parents=True, exist_ok=True)
    regen_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []

    # ------------------------------------------------------------------
    # 1) Export per-model convergence curves
    # ------------------------------------------------------------------
    search_conv = _read_csv_if_exists(tables_dir / 'Search_Convergence_Long.csv')
    if not search_conv.empty and 'Model' in search_conv.columns:
        conv_dir = tables_dir / 'ConvergenceCurves'
        conv_dir.mkdir(parents=True, exist_ok=True)
        for model_name, grp in search_conv.groupby('Model'):
            algo, optimizer = (str(model_name).split('-', 1) + ['Unknown'])[:2]
            sort_cols = [c for c in ['Run_Seed', 'Step'] if c in grp.columns]
            out = grp.sort_values(sort_cols).copy() if sort_cols else grp.copy()
            out_path = conv_dir / (
                f"ConvergenceCurve_{_safe_name_stem(algo)}_{_safe_name_stem(optimizer)}.csv")
            out.to_csv(out_path, index=False)
            generated_paths.append(str(out_path))

    # ------------------------------------------------------------------
    # 2) Export external-validation per-class F1 for all 28 models
    # ------------------------------------------------------------------
    per_class_long = _read_csv_if_exists(
        tables_dir / 'FinalEval_Test_External_PerClass_Metrics_Long.csv')
    if not per_class_long.empty:
        if 'Dataset' in per_class_long.columns:
            mask_ext = per_class_long['Dataset'].astype(str).str.lower() == 'external_val'
        else:
            mask_ext = pd.Series(False, index=per_class_long.index)
        ext = per_class_long[mask_ext].copy()
        if not ext.empty:
            if 'Run_Seed' not in ext.columns:
                ext['Run_Seed'] = 0
            agg = ext.groupby(
                ['Model', 'Algorithm', 'Optimizer', 'Class_Label', 'Class_Name'],
                as_index=False).agg(
                    Runs=('Run_Seed', 'nunique'),
                    Precision_mean=('Precision', 'mean'),
                    Recall_mean=('Recall', 'mean'),
                    F1_mean=('F1', 'mean'),
                    Support_mean=('Support', 'mean'))
            agg['F1_rank_within_model'] = agg.groupby('Model')['F1_mean'].rank(
                ascending=False, method='average')
            _write_df(agg, tables_dir / 'PerClassF1_External_AllModels')
            generated_paths.append(str(tables_dir / 'PerClassF1_External_AllModels.csv'))

    # ------------------------------------------------------------------
    # 3) Export SHAP files required by Figure 4-6/4-7/4-8
    # ------------------------------------------------------------------
    shap_mean = _read_csv_if_exists(
        tables_dir / 'SHAP_FinalArchive' / 'SHAP_FinalArchive_MeanAbs.csv')
    if not shap_mean.empty:
        mask_ds = (shap_mean['Dataset'].astype(str).str.lower() == 'external_val'
                   if 'Dataset' in shap_mean.columns else
                   pd.Series(False, index=shap_mean.index))
        mask_algo = (shap_mean['Algorithm'].astype(str) == 'LightGBM'
                     if 'Algorithm' in shap_mean.columns else
                     pd.Series(False, index=shap_mean.index))
        lgb_family = shap_mean[
            mask_ds & mask_algo
        ].copy()
        if not lgb_family.empty:
            if 'Optimizer' not in lgb_family.columns:
                lgb_family['Optimizer'] = lgb_family['Model'].astype(str).str.split('-', n=1).str[-1]
            if 'Feature_Display' not in lgb_family.columns:
                lgb_family['Feature_Display'] = lgb_family['Feature']
            lgb_family['Rank_In_Model'] = lgb_family.groupby('Model')['Mean_Abs_SHAP'].rank(
                ascending=False, method='average')
            long_cols = [c for c in [
                'Dataset', 'Model', 'Algorithm', 'Optimizer', 'Run_Seed',
                'Feature', 'Feature_Display', 'Mean_Abs_SHAP', 'Rank_In_Model'
            ] if c in lgb_family.columns]
            _write_df(lgb_family[long_cols], tables_dir / 'GlobalSHAP_LightGBM-Family_Long')
            piv = lgb_family.pivot_table(
                index=['Feature', 'Feature_Display'],
                columns='Optimizer',
                values='Mean_Abs_SHAP',
                aggfunc='mean').reset_index()
            _write_df(piv, tables_dir / 'GlobalSHAP_LightGBM-Family')
            generated_paths.append(str(tables_dir / 'GlobalSHAP_LightGBM-Family.csv'))

    shap_classwise = _read_csv_if_exists(
        tables_dir / 'SHAP_FinalArchive' / 'SHAP_FinalArchive_Classwise_MeanAbs.csv')
    if not shap_classwise.empty:
        mask_ds = (shap_classwise['Dataset'].astype(str).str.lower() == 'external_val'
                   if 'Dataset' in shap_classwise.columns else
                   pd.Series(False, index=shap_classwise.index))
        mask_model = (shap_classwise['Model'].astype(str) == 'LightGBM-Default'
                      if 'Model' in shap_classwise.columns else
                      pd.Series(False, index=shap_classwise.index))
        cls_df = shap_classwise[
            mask_ds & mask_model
        ].copy()
        if not cls_df.empty:
            if 'Feature_Display' not in cls_df.columns:
                cls_df['Feature_Display'] = cls_df['Feature']
            cls_df['Rank_In_Class'] = cls_df.groupby('Class_Name')['Mean_Abs_SHAP'].rank(
                ascending=False, method='average')
            piv = cls_df.pivot_table(
                index=['Feature', 'Feature_Display'],
                columns='Class_Name',
                values='Mean_Abs_SHAP',
                aggfunc='mean').reset_index()
            _write_df(piv, tables_dir / 'ClassLevelSHAP_LightGBM-Default_external')
            _write_df(cls_df, tables_dir / 'ClassLevelSHAP_LightGBM-Default_external_Long')
            generated_paths.append(str(tables_dir / 'ClassLevelSHAP_LightGBM-Default_external.csv'))

    npz_path = (tables_dir / 'SHAP_FinalArchive' / 'external_val' /
                f"SHAP_Classwise_{_safe_name_stem('LightGBM-Default')}.npz")
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                values = np.asarray(npz['values'], dtype=float)  # (class, sample, feature)
                sample_idx = np.asarray(npz.get('sample_indices', np.arange(values.shape[1])), dtype=int)
                feature_names = np.asarray(npz.get(
                    'feature_names',
                    [f'x{i + 1}' for i in range(values.shape[2])]), dtype=str)
                class_ids = np.asarray(npz.get(
                    'class_ids',
                    list(range(values.shape[0]))), dtype=int)
            # Required tensor format for beeswarm redraw:
            # (n_samples, n_features, n_classes)
            tensor = np.transpose(values, (1, 2, 0))
            npy_out = tables_dir / 'BeeswarmSHAP_LightGBM-Default_external.npy'
            np.save(npy_out, tensor)
            generated_paths.append(str(npy_out))

            meta_rows = [{
                'Tensor_File': str(npy_out.name),
                'Shape': str(tuple(int(v) for v in tensor.shape)),
                'Axis_0': 'sample_index',
                'Axis_1': 'feature',
                'Axis_2': 'class',
                'Class_IDs': json.dumps([int(c) for c in class_ids], ensure_ascii=False),
                'Feature_Names': json.dumps([str(f) for f in feature_names], ensure_ascii=False),
                'Sample_Index_Source': 'SHAP_FinalArchive/external_val npz sample_indices',
                'Sample_Count': int(tensor.shape[0]),
            }]
            _write_df(pd.DataFrame(meta_rows), tables_dir / 'BeeswarmSHAP_LightGBM-Default_external_Metadata')
        except Exception as exc:
            warnings.warn(f"Failed to export beeswarm SHAP tensor: {exc}", RuntimeWarning)

    # ------------------------------------------------------------------
    # 4) Build full table/figure data index (main text + supplement)
    # ------------------------------------------------------------------
    catalog_specs = [
        ('Table 3-1', 'Main-Table', 'RF hyperparameter search space',
         ['Tables/Search_Space_Audit.csv']),
        ('Table 4-1', 'Main-Table', 'RF best hyperparameters under 7 optimizers',
         ['Tables/FinalEval_Test_External_Raw.csv']),
        ('Table 4-2', 'Main-Table', '4x7 internal-test macro-F1 matrix',
         ['Tables/FinalEval_Test_External_Summary.csv']),
        ('Table 4-3', 'Main-Table', '4x7 external-validation macro-F1 matrix',
         ['Tables/FinalEval_Test_External_Summary.csv']),
        ('Table 4-4', 'Main-Table', 'Algorithm-family mean/range on external set',
         ['Tables/FinalEval_Test_External_Summary.csv']),
        ('Table 4-5', 'Main-Table', 'Naive baseline vs representative models',
         ['Tables/FinalEval_Test_External_Summary.csv',
          'Tables/Naive_Baseline_External.csv']),
        ('Table 4-6', 'Main-Table', 'Class distribution across datasets',
         ['Tables/Dataset_Class_Distribution.csv']),
        ('Table 4-7', 'Main-Table', 'Per-class F1 on external validation',
         ['Tables/PerClassF1_External_AllModels.csv']),
        ('Table 4-8', 'Main-Table', 'Eta-squared decomposition (algorithm vs optimizer)',
         ['Tables/Algorithm_vs_Optimizer_EtaSquared.csv']),
        ('Table 4-9', 'Main-Table', 'Domain shift KS + FDR (core features)',
         ['Tables/Dataset_DomainShift_KS.csv']),
        ('Table 4-10', 'Main-Table', 'Global SHAP for LightGBM-Default vs RF-Default',
         ['Tables/SHAP_FinalArchive/SHAP_FinalArchive_MeanAbs.csv']),
        ('Table 4-11', 'Main-Table', 'SHAP x Domain-shift joint analysis',
         ['Tables/SHAP_FinalArchive/SHAP_FinalArchive_MeanAbs.csv',
          'Tables/Dataset_DomainShift_KS.csv']),

        ('Figure 2-1', 'Main-Figure', 'Study area + aquifer schematic',
         ['MANUAL_EXTERNAL_SOURCE']),
        ('Figure 3-1', 'Main-Figure', 'Overall analysis workflow diagram',
         ['MANUAL_EXTERNAL_SOURCE']),
        ('Figure 4-1', 'Main-Figure', 'Convergence trajectories by algorithm/optimizer',
         ['Tables/Search_Convergence_Long.csv']),
        ('Figure 4-2', 'Main-Figure', 'Internal-test 28-model bar + 95% CI',
         ['Tables/FinalEval_Test_External_Summary.csv']),
        ('Figure 4-3', 'Main-Figure', 'External-validation 4x7 heatmap',
         ['Tables/FinalEval_Test_External_Summary.csv']),
        ('Figure 4-4', 'Main-Figure', 'Eta-squared stacked bars',
         ['Tables/Algorithm_vs_Optimizer_EtaSquared.csv']),
        ('Figure 4-5', 'Main-Figure', 'Core-feature distribution shift',
         ['Tables/Dataset_DomainShift_KS.csv', 'RegenData/nested_raw_arrays.npz']),
        ('Figure 4-6', 'Main-Figure', 'LightGBM-family global SHAP heatmap',
         ['Tables/GlobalSHAP_LightGBM-Family.csv']),
        ('Figure 4-7', 'Main-Figure', 'LightGBM-Default external class-level beeswarm SHAP',
         ['Tables/BeeswarmSHAP_LightGBM-Default_external.npy']),
        ('Figure 4-8', 'Main-Figure', 'LightGBM-Default class-level mean(|SHAP|) heatmap',
         ['Tables/ClassLevelSHAP_LightGBM-Default_external.csv']),

        ('Table S1', 'Supp-Table', 'Search budget audit',
         ['Tables/Search_Budget_Audit_Summary.csv']),
        ('Table S2', 'Supp-Table', 'Search space audit',
         ['Tables/Search_Space_Audit.csv']),
        ('Table S3', 'Supp-Table', 'Full 28-model summary with CI/IQR',
         ['Tables/FinalEval_Test_External_Summary.csv']),
        ('Table S4', 'Supp-Table', 'Ranking comparison across stages',
         ['Tables/Generalization_Ranking_Comparison.csv']),
        ('Table S5', 'Supp-Table', 'Spearman ranking consistency',
         ['Tables/Generalization_Ranking_Spearman.csv']),
        ('Table S6', 'Supp-Table', 'Internal-test combined ranking',
         ['Tables/SCI_Test_Combined_Ranking.csv']),
        ('Table S7', 'Supp-Table', 'External-validation combined ranking',
         ['Tables/SCI_Validation_Combined_Ranking.csv']),
        ('Table S8', 'Supp-Table', 'Complete domain-shift table',
         ['Tables/Dataset_DomainShift_KS.csv']),

        ('Figure S1', 'Supp-Figure', 'External vs internal macro-F1 scatter',
         ['Tables/FinalEval_Test_External_Summary.csv',
          'Tables/Generalization_Ranking_Spearman.csv']),
        ('Figure S2', 'Supp-Figure', 'Search wall-clock + eval audit',
         ['Tables/Search_Budget_Audit_Summary.csv']),
        ('Figure S3', 'Supp-Figure', 'Correlation matrices (train vs external)',
         ['RegenData/nested_raw_arrays.npz']),
        ('Figure S4', 'Supp-Figure', 'Cross-model SHAP rank migration',
         ['Tables/SHAP_Generalization_Contrast.csv']),
    ]

    catalog_rows = []
    for item_id, item_type, title, rel_files in catalog_specs:
        missing = []
        for rel in rel_files:
            if str(rel).startswith('MANUAL_'):
                continue
            if not (output_dir / rel).exists():
                missing.append(rel)
        auto_ready = (len(missing) == 0 and
                      not any(str(rel).startswith('MANUAL_') for rel in rel_files))
        catalog_rows.append({
            'Item_ID': item_id,
            'Item_Type': item_type,
            'Title': title,
            'Data_Files': '; '.join(rel_files),
            'Auto_Ready': bool(auto_ready),
            'Missing_Files': '; '.join(missing),
            'Needs_Manual_Source': bool(
                any(str(rel).startswith('MANUAL_') for rel in rel_files)),
        })

    catalog_df = pd.DataFrame(catalog_rows)
    _write_df(catalog_df, tables_dir / 'SCI_FigureTable_Data_Index')
    generated_paths.append(str(tables_dir / 'SCI_FigureTable_Data_Index.csv'))

    bundle_manifest = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'output_dir': str(output_dir),
        'n_catalog_items': int(len(catalog_rows)),
        'n_auto_ready_items': int(catalog_df['Auto_Ready'].sum()) if not catalog_df.empty else 0,
        'generated_files': generated_paths,
    }
    with open(regen_dir / 'sci_figure_table_bundle_manifest.json',
              'w', encoding='utf-8') as f:
        json.dump(_json_safe(bundle_manifest), f, indent=2, ensure_ascii=False)

    return {
        'catalog': catalog_df,
        'generated_files': generated_paths,
        'manifest': bundle_manifest,
    }


def _checkpoint_nested_progress(output_dir, label, outer_records,
                                outer_prediction_records,
                                outer_fold_index_records, final_records,
                                final_prediction_records,
                                search_history_records,
                                search_budget_records):
    """Lightweight progress snapshot for very long formal runs."""
    ckpt_dir = output_dir / 'Checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    safe_label = _safe_name_stem(label)
    payload = {
        'label': label,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'outer_records': outer_records,
        'outer_predictions': outer_prediction_records,
        'outer_fold_indices': outer_fold_index_records,
        'final_records': final_records,
        'final_predictions': final_prediction_records,
        'search_history': search_history_records,
        'search_budget': search_budget_records,
    }
    joblib.dump(payload, ckpt_dir / f'{safe_label}.pkl')
    joblib.dump(payload, ckpt_dir / 'latest_checkpoint.pkl')
    for name, records in [
            ('outer_records', outer_records),
            ('outer_predictions', outer_prediction_records),
            ('outer_fold_indices', outer_fold_index_records),
            ('final_records', final_records),
            ('final_predictions', final_prediction_records),
            ('search_history', search_history_records),
            ('search_budget', search_budget_records)]:
        if records:
            pd.DataFrame(records).to_csv(
                ckpt_dir / f'latest_{name}.csv', index=False)
    with open(ckpt_dir / 'latest_checkpoint_manifest.json',
              'w', encoding='utf-8') as f:
        json.dump(_json_safe({
            'label': label,
            'timestamp': payload['timestamp'],
            'counts': {k: len(v) for k, v in payload.items()
                       if isinstance(v, list)},
        }), f, indent=2, ensure_ascii=False)


def _make_generalization_analysis_tables(output_dir, outer_summary,
                                         rank_stability, final_summary,
                                         final_records):
    tables_dir = output_dir / 'Tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cv_map = (dict(zip(outer_summary['Model'], outer_summary['Outer_F1_Macro_mean']))
              if not outer_summary.empty and 'Outer_F1_Macro_mean' in outer_summary.columns else {})
    test_map = (dict(zip(final_summary['Model'], final_summary['Test_F1_Macro_mean']))
                if not final_summary.empty and 'Test_F1_Macro_mean' in final_summary.columns else {})
    val_map = (dict(zip(final_summary['Model'], final_summary['Val_F1_Macro_mean']))
               if not final_summary.empty and 'Val_F1_Macro_mean' in final_summary.columns else {})
    gap_map = (dict(zip(final_summary['Model'], final_summary['Generalization_Gap_mean']))
               if not final_summary.empty and 'Generalization_Gap_mean' in final_summary.columns else {})
    all_models = sorted(set(cv_map) | set(test_map) | set(val_map))
    for model in all_models:
        rows.append({
            'Model': model,
            'OuterCV_F1_Macro': cv_map.get(model, np.nan),
            'InternalTest_F1_Macro': test_map.get(model, np.nan),
            'ExternalVal_F1_Macro': val_map.get(model, np.nan),
            'Val_minus_Test_F1_Macro': gap_map.get(model, np.nan),
        })
    comparison = pd.DataFrame(rows)
    for col in ['OuterCV_F1_Macro', 'InternalTest_F1_Macro', 'ExternalVal_F1_Macro']:
        if col in comparison.columns:
            comparison[f'Rank_{col}'] = comparison[col].rank(ascending=False, method='average')
    _write_df(comparison, tables_dir / 'Generalization_Ranking_Comparison')

    corr_rows = [
        _spearman_record('OuterCV', cv_map, 'InternalTest', test_map),
        _spearman_record('OuterCV', cv_map, 'ExternalVal', val_map),
        _spearman_record('InternalTest', test_map, 'ExternalVal', val_map),
    ]
    corr_df = pd.DataFrame(corr_rows)
    _write_df(corr_df, tables_dir / 'Generalization_Ranking_Spearman')

    effects = []
    for response in ['Val_F1_Macro', 'Generalization_Gap', 'Test_F1_Macro']:
        eff = _two_way_variance_decomposition(final_records, response)
        if not eff.empty:
            effects.append(eff)
    effect_df = pd.concat(effects, ignore_index=True) if effects else pd.DataFrame()
    _write_df(effect_df, tables_dir / 'Algorithm_vs_Optimizer_EtaSquared')
    return comparison, corr_df, effect_df


def _make_budget_generalization_analysis_tables(output_dir, final_summary,
                                                final_records):
    """Compare internal CV, internal test, and external validation rankings."""
    tables_dir = output_dir / 'Tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    train_cv_map = (dict(zip(final_summary['Model'], final_summary['Train_Internal_CV_F1_mean']))
                    if not final_summary.empty and 'Train_Internal_CV_F1_mean' in final_summary.columns else {})
    test_map = (dict(zip(final_summary['Model'], final_summary['Test_F1_Macro_mean']))
                if not final_summary.empty and 'Test_F1_Macro_mean' in final_summary.columns else {})
    val_map = (dict(zip(final_summary['Model'], final_summary['Val_F1_Macro_mean']))
               if not final_summary.empty and 'Val_F1_Macro_mean' in final_summary.columns else {})
    gap_map = (dict(zip(final_summary['Model'], final_summary['Generalization_Gap_mean']))
               if not final_summary.empty and 'Generalization_Gap_mean' in final_summary.columns else {})

    all_models = sorted(set(train_cv_map) | set(test_map) | set(val_map))
    rows = []
    for model in all_models:
        rows.append({
            'Model': model,
            'TrainInternalCV_F1_Macro': train_cv_map.get(model, np.nan),
            'InternalTest_F1_Macro': test_map.get(model, np.nan),
            'ExternalVal_F1_Macro': val_map.get(model, np.nan),
            'Val_minus_Test_F1_Macro': gap_map.get(model, np.nan),
        })
    comparison = pd.DataFrame(rows)
    for col in ['TrainInternalCV_F1_Macro', 'InternalTest_F1_Macro', 'ExternalVal_F1_Macro']:
        if col in comparison.columns:
            comparison[f'Rank_{col}'] = comparison[col].rank(ascending=False, method='average')
    _write_df(comparison, tables_dir / 'Generalization_Ranking_Comparison')

    corr_rows = [
        _spearman_record('TrainInternalCV', train_cv_map, 'InternalTest', test_map),
        _spearman_record('TrainInternalCV', train_cv_map, 'ExternalVal', val_map),
        _spearman_record('InternalTest', test_map, 'ExternalVal', val_map),
    ]
    corr_df = pd.DataFrame(corr_rows)
    _write_df(corr_df, tables_dir / 'Generalization_Ranking_Spearman')

    effects = []
    for response in ['Val_F1_Macro', 'Generalization_Gap', 'Test_F1_Macro']:
        eff = _two_way_variance_decomposition(final_records, response)
        if not eff.empty:
            effects.append(eff)
    effect_df = pd.concat(effects, ignore_index=True) if effects else pd.DataFrame()
    _write_df(effect_df, tables_dir / 'Algorithm_vs_Optimizer_EtaSquared')
    return comparison, corr_df, effect_df


def _make_combined_ranking_table(final_summary, dataset_prefix):
    """Build a multi-metric weighted ranking table for one dataset.

    Combines macro-F1, Accuracy, Cohen's kappa and MCC by averaging their
    per-metric ranks; lower combined rank = better overall performance.
    Used for Tables S6 (internal-test) and S7 (external-validation) so
    that reviewers have a single-number summary that does not depend on
    macro-F1 alone.
    """
    if final_summary is None or final_summary.empty:
        return pd.DataFrame()
    metric_specs = [(f'{dataset_prefix}_F1_Macro_mean', 'F1_Macro'),
                    (f'{dataset_prefix}_Accuracy_mean', 'Accuracy'),
                    (f'{dataset_prefix}_Kappa_mean',    'Kappa'),
                    (f'{dataset_prefix}_MCC_mean',      'MCC')]
    available = [(col, short) for col, short in metric_specs
                 if col in final_summary.columns]
    if not available:
        return pd.DataFrame()
    keep = ['Model', 'Algorithm', 'Optimizer'] + [c for c, _ in available]
    out = final_summary[keep].copy()
    rank_cols = []
    for col, short in available:
        rcol = f'Rank_{short}'
        out[rcol] = out[col].rank(ascending=False, method='average')
        rank_cols.append(rcol)
    out['Combined_Mean_Rank'] = out[rank_cols].mean(axis=1)
    out['Combined_Final_Rank'] = out['Combined_Mean_Rank'].rank(
        ascending=True, method='average')
    return out.sort_values('Combined_Final_Rank').reset_index(drop=True)


def _make_supplementary_ranking_and_baseline_tables(
        output_dir, final_summary, y_train, all_classes, water_source_mapping):
    """Produce Tables S6/S7 (multi-metric combined rankings) and the naive
    majority-class baseline reference (Table 4-5 supporting data).

    The naive baseline is the constant predictor that always returns the
    most frequent training-set class; its expected accuracy on a held-out
    set equals the prior of the majority class, with kappa and MCC equal
    to zero by construction.  This row is reported as a zero-information
    reference against which the 28 representative models can be contrasted.
    """
    if final_summary is None or final_summary.empty:
        return None
    tables_dir = output_dir / 'Tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    test_rank = _make_combined_ranking_table(final_summary, 'Test')
    if not test_rank.empty:
        _write_df(test_rank, tables_dir / 'SCI_Test_Combined_Ranking')

    val_rank = _make_combined_ranking_table(final_summary, 'Val')
    if not val_rank.empty:
        _write_df(val_rank, tables_dir / 'SCI_Validation_Combined_Ranking')

    y_train_arr = np.asarray(y_train).astype(int)
    if y_train_arr.size == 0:
        return {'test_combined_ranking': test_rank, 'val_combined_ranking': val_rank}
    classes_arr = np.asarray(all_classes).astype(int)
    counts = np.array([int((y_train_arr == c).sum()) for c in classes_arr])
    priors = counts / max(counts.sum(), 1)
    majority_idx = int(np.argmax(counts))
    majority_class = int(classes_arr[majority_idx])
    majority_name = _class_name(majority_class, water_source_mapping)
    rows = [{
        'Reference': 'Majority-Class Naive Baseline',
        'Predicted_Class_Label': majority_class,
        'Predicted_Class_Name': majority_name,
        'Class_Prior_Probability': float(priors[majority_idx]),
        'Note': ('Constant predictor using the most frequent training-set '
                 'class; expected accuracy on a held-out set equals the '
                 'majority-class prior, with kappa = 0 and MCC = 0.'),
    }]
    for cls, p in zip(classes_arr, priors):
        rows.append({
            'Reference': 'Class_Prior',
            'Predicted_Class_Label': int(cls),
            'Predicted_Class_Name': _class_name(int(cls), water_source_mapping),
            'Class_Prior_Probability': float(p),
            'Note': '',
        })
    _write_df(pd.DataFrame(rows), tables_dir / 'Naive_Baseline_External')

    return {
        'test_combined_ranking': test_rank,
        'val_combined_ranking': val_rank,
        'majority_class': majority_class,
        'majority_class_name': majority_name,
        'class_priors': dict(zip([int(c) for c in classes_arr],
                                 [float(p) for p in priors])),
    }


def _generate_shap_generalization_contrast(final_records, final_summary,
                                           X_train_raw, y_train,
                                           X_val_raw, y_val,
                                           all_feature_cols, feature_method,
                                           output_dir, skip=False):
    """Post-hoc SHAP: explain selected generalization contrasts; never used for selection."""
    if skip or X_val_raw is None or y_val is None or final_summary is None or final_summary.empty:
        return None
    tables_dir = output_dir / 'Tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    picks = {}
    if 'Test_F1_Macro_mean' in final_summary.columns:
        picks['internal_test_best'] = final_summary.sort_values(
            'Test_F1_Macro_mean', ascending=False).iloc[0]['Model']
    if 'Val_F1_Macro_mean' in final_summary.columns:
        picks['external_val_best'] = final_summary.sort_values(
            'Val_F1_Macro_mean', ascending=False).iloc[0]['Model']
    if 'Generalization_Gap_mean' in final_summary.columns:
        picks['largest_test_to_val_drop'] = final_summary.sort_values(
            'Generalization_Gap_mean', ascending=True).iloc[0]['Model']
    records_df = pd.DataFrame(final_records)
    shap_rows = []
    rng = np.random.default_rng(42)
    idx = np.arange(len(X_val_raw))
    if len(idx) > 300:
        idx = np.sort(rng.choice(idx, size=300, replace=False))
    for role, model_name in picks.items():
        try:
            algo, optimizer = str(model_name).split('-', 1)
            row = records_df[(records_df['Model'] == model_name) &
                             (records_df['Run_Seed'] == SEEDS[0])]
            if row.empty:
                row = records_df[records_df['Model'] == model_name].head(1)
            params = json.loads(row.iloc[0]['Best_Params_JSON']) if not row.empty else dict(_DEFAULT_PARAMS[algo])
            model, preproc, _ = _fit_model_with_fold_preprocessing(
                algo, _clean_model_params(params), X_train_raw, y_train,
                X_val_raw[idx], all_feature_cols, feature_method, SEEDS[0])
            X_val_sel = _transform_with_fold_preprocessing(X_val_raw[idx], preproc)
            _shap_values, shap_mean = _compute_shap_mean_abs(model, X_val_sel)
            for feature, value in zip(preproc['feature_cols'], np.asarray(shap_mean).ravel()):
                shap_rows.append({
                    'Role': role,
                    'Model': model_name,
                    'Feature': feature,
                    'MeanAbsSHAP_ExternalVal': float(value),
                    'Selection_Note': 'post_hoc_explanation_only_not_model_selection',
                })
        except Exception as exc:
            shap_rows.append({
                'Role': role,
                'Model': model_name,
                'Feature': '__ERROR__',
                'MeanAbsSHAP_ExternalVal': np.nan,
                'Selection_Note': str(exc),
            })
    shap_df = pd.DataFrame(shap_rows)
    _write_df(shap_df, tables_dir / 'SHAP_Generalization_Contrast')
    return shap_df


def run_budget_matched_generalization_protocol(
        X_train_raw, y_train, X_test_raw, y_test,
        all_feature_cols, feature_method,
        val_path, le, water_source_mapping,
        output_dir, target_col='y', ion_cols=None,
        n_trials_by_algo=None,
        inner_splits=5, inner_repeats=1,
        final_eval_runs=30,
        pop_size=6, max_iter=3,
        max_grid_candidates=24,
        skip_shap_contrast=False,
        skip_full_shap_archive=True,
        shap_svm_explain_size=160,
        shap_background_size=48,
        seeds=SEEDS):
    """
    Budget-matched repeated-search protocol for all 28 combinations.

    Research intent:
      1. Test whether internal-test performance predicts external validation.
      2. Compare algorithm choice vs optimizer choice for generalization.
      3. Use SHAP only as post-hoc explanation of generalization contrasts.
      4. Never use external validation for model selection.

    Design:
      - Every algorithm-optimizer combination is kept in the study.
      - Hyperparameter search is performed on training data only.
      - Internal test and external validation remain locked scoring sets.
      - Search budgets are capped to keep optimizer comparison computationally
        feasible while preserving equal-treatment comparisons.
    """
    if ion_cols is None:
        ion_cols = ION_COLS
    if n_trials_by_algo is None:
        n_trials_by_algo = {algo: 24 for algo in _ALGORITHMS}

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ['Tables', 'Logs', 'Figures']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / 'Tables'

    print("\n" + "=" * 80)
    print("BUDGET-MATCHED REPEATED GENERALIZATION PROTOCOL")
    print("=" * 80)
    print("Research goals:")
    print("  1. Test whether internal-test performance predicts external validation.")
    print("  2. Compare algorithm choice vs optimizer choice for generalization.")
    print("  3. Use SHAP only as post-hoc explanation of generalization contrasts.")
    print("  External validation is never used for model selection.")
    print("Protocol notes:")
    print("  - All 28 algorithm x optimizer combinations are retained.")
    print("  - Hyperparameter search uses training data only.")
    print("  - Internal test and external validation are locked scoring sets.")
    print("  - Search budgets are capped for tractable, budget-matched comparison.")

    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    all_classes = sorted(np.unique(y_train).tolist())

    X_val_raw, y_val, val_meta = _load_external_validation_raw_for_generalization(
        val_path, all_feature_cols, target_col=target_col, ion_cols=ion_cols,
        le=le, water_source_mapping=water_source_mapping)
    with open(output_dir / 'Logs' / 'validation_policy_budget_matched.json',
              'w', encoding='utf-8') as f:
        json.dump(_json_safe({
            'external_validation_used_for_selection': False,
            'external_validation_role': 'locked descriptive scoring only',
            'validation_loaded_for_scoring_only': True,
            'validation_metadata': val_meta,
        }), f, indent=2, ensure_ascii=False)

    final_records = []
    final_prediction_records = []
    final_model_registry = {}
    search_history_records = []
    search_budget_records = []
    final_seeds = list(seeds[:int(final_eval_runs)])
    final_combo_total = len(_ALGORITHMS) * len(_OPTIMIZERS)
    final_combo_i = 0

    for algo in _ALGORITHMS:
        for optimizer in _OPTIMIZERS:
            final_combo_i += 1
            model_name = f'{algo}-{optimizer}'
            print(f"\n[Final Eval {final_combo_i}/{final_combo_total}] {model_name}",
                  flush=True)
            for run_i, seed in enumerate(final_seeds, 1):
                progress_label = (
                    f"Budget Eval Combo {final_combo_i}/{final_combo_total} | "
                    f"Run {run_i}/{len(final_seeds)} | {model_name}")
                _print_progress(f"{progress_label} | search started")
                try:
                    params, train_cv_f1, search_time = _nested_search_once(
                        algo, optimizer, X_train_raw, y_train,
                        all_feature_cols, feature_method, seed,
                        n_trials_by_algo, pop_size=pop_size, max_iter=max_iter,
                        inner_splits=inner_splits, inner_repeats=inner_repeats,
                        max_grid_candidates=max_grid_candidates,
                        progress_label=progress_label)
                    search_history_records.extend(_search_history_rows(
                        'budget_matched_repeated', algo, optimizer, model_name, params,
                        inner_splits, inner_repeats, run_seed=seed))
                    search_budget_records.append(_search_budget_row(
                        'budget_matched_repeated', algo, optimizer, model_name, params,
                        search_time, inner_splits, inner_repeats, run_seed=seed))

                    model, preproc, X_test_sel = _fit_model_with_fold_preprocessing(
                        algo, _clean_model_params(params), X_train_raw, y_train,
                        X_test_raw, all_feature_cols, feature_method, seed,
                        model_n_jobs=DETERMINISTIC_N_JOBS)
                    y_test_pred = _predict_label_vector(model, X_test_sel)
                    test_proba = _predict_proba_matrix(model, X_test_sel, all_classes)
                    test_metrics = _metric_pack_for_predictions(y_test, y_test_pred)
                    final_prediction_records.extend(_prediction_rows(
                        'budget_matched_repeated_locked_eval', 'internal_test',
                        model_name, algo, optimizer, y_test, y_test_pred,
                        test_proba, np.arange(len(y_test)), all_classes,
                        water_source_mapping, run_seed=seed))

                    rec = {
                        'Phase': 'budget_matched_repeated_locked_eval',
                        'Model': model_name,
                        'Algorithm': algo,
                        'Optimizer': optimizer,
                        'Run_Seed': int(seed),
                        'Train_Internal_CV_F1': float(train_cv_f1),
                        'Search_Time_s': float(search_time),
                        'Test_Accuracy': test_metrics['Accuracy'],
                        'Test_F1_Macro': test_metrics['F1_Macro'],
                        'Test_F1_Weighted': test_metrics['F1_Weighted'],
                        'Test_Kappa': test_metrics['Kappa'],
                        'Test_MCC': test_metrics['MCC'],
                        'Best_Params_JSON': json.dumps(_json_safe(params), ensure_ascii=False),
                        'External_Validation_Used_For_Selection': False,
                    }

                    if X_val_raw is not None and y_val is not None:
                        X_val_sel = _transform_with_fold_preprocessing(X_val_raw, preproc)
                        y_val_pred = _predict_label_vector(model, X_val_sel)
                        val_proba = _predict_proba_matrix(model, X_val_sel, all_classes)
                        val_metrics = _metric_pack_for_predictions(y_val, y_val_pred)
                        final_prediction_records.extend(_prediction_rows(
                            'budget_matched_repeated_locked_eval', 'external_val',
                            model_name, algo, optimizer, y_val, y_val_pred,
                            val_proba, np.arange(len(y_val)), all_classes,
                            water_source_mapping, run_seed=seed))
                        rec.update({
                            'Val_Accuracy': val_metrics['Accuracy'],
                            'Val_F1_Macro': val_metrics['F1_Macro'],
                            'Val_F1_Weighted': val_metrics['F1_Weighted'],
                            'Val_Kappa': val_metrics['Kappa'],
                            'Val_MCC': val_metrics['MCC'],
                            'Generalization_Gap': val_metrics['F1_Macro'] - test_metrics['F1_Macro'],
                        })

                    final_records.append(rec)
                    stored = final_model_registry.get(model_name)
                    if stored is None or train_cv_f1 > stored.get('Train_Internal_CV_F1', -np.inf):
                        final_model_registry[model_name] = {
                            'model': model,
                            'preproc': preproc,
                            'Algorithm': algo,
                            'Optimizer': optimizer,
                            'Run_Seed': int(seed),
                            'Train_Internal_CV_F1': float(train_cv_f1),
                            'Best_Params': _json_safe(params),
                        }

                    msg = (f"  {run_i:02d}/{len(final_seeds)} seed={seed} "
                           f"train_cv={train_cv_f1:.4f} "
                           f"test={test_metrics['F1_Macro']:.4f}")
                    if 'Val_F1_Macro' in rec:
                        msg += (f" val={rec['Val_F1_Macro']:.4f} "
                                f"gap={rec['Generalization_Gap']:+.4f}")
                    print(msg, flush=True)
                    _print_progress(
                        f"{progress_label} | model complete | "
                        f"train_cv={train_cv_f1:.4f} | "
                        f"test={test_metrics['F1_Macro']:.4f} | "
                        f"time={search_time:.1f}s")
                except Exception as exc:
                    print(f"  {run_i:02d}/{len(final_seeds)} seed={seed} FAILED: {exc}",
                          flush=True)

            _checkpoint_nested_progress(
                output_dir, f'budget_{_safe_name_stem(model_name)}_complete',
                [], [], [], final_records, final_prediction_records,
                search_history_records, search_budget_records)

    final_raw = pd.DataFrame(final_records)
    _write_df(final_raw, tables_dir / 'FinalEval_Test_External_Raw')
    final_pred_df = pd.DataFrame(final_prediction_records)
    _write_df(final_pred_df, tables_dir / 'FinalEval_Test_External_Predictions')
    final_diagnostics = _make_prediction_diagnostic_tables(
        final_pred_df, all_classes, water_source_mapping, output_dir,
        'FinalEval_Test_External')

    search_history_df = pd.DataFrame(search_history_records)
    _write_df(search_history_df, tables_dir / 'Search_Convergence_Long')
    search_budget_df = pd.DataFrame(search_budget_records)
    _write_df(search_budget_df, tables_dir / 'Search_Budget_Audit')
    if not search_budget_df.empty:
        budget_summary = search_budget_df.groupby(
            ['Search_Phase', 'Model', 'Algorithm', 'Optimizer'],
            as_index=False).agg(
                N_Searches=('Fitness_Evals_Total', 'size'),
                Fitness_Evals_Total_mean=('Fitness_Evals_Total', 'mean'),
                Fitness_Evals_Total_median=('Fitness_Evals_Total', 'median'),
                Nominal_Model_Fits_Total_mean=('Nominal_Model_Fits_Total', 'mean'),
                Search_Time_s_mean=('Search_Time_s', 'mean'),
                Search_Time_s_sum=('Search_Time_s', 'sum'),
            )
        _write_df(budget_summary, tables_dir / 'Search_Budget_Audit_Summary')
    else:
        budget_summary = pd.DataFrame()

    search_space_audit = _search_space_audit_table(
        n_trials_by_algo, pop_size, max_iter, max_grid_candidates,
        inner_splits, inner_repeats)
    _write_df(search_space_audit, tables_dir / 'Search_Space_Audit')

    final_summary = _summarize_generalization_records(
        final_records, 'budget_matched_repeated_locked_eval')
    _write_df(final_summary, tables_dir / 'FinalEval_Test_External_Summary')

    comparison, corr_df, effect_df = _make_budget_generalization_analysis_tables(
        output_dir, final_summary, final_records)
    _make_supplementary_ranking_and_baseline_tables(
        output_dir, final_summary, y_train, all_classes, water_source_mapping)
    _write_dataset_archives(
        output_dir, X_train_raw, y_train, X_test_raw, y_test,
        X_val_raw, y_val, all_feature_cols, water_source_mapping)
    shap_archive = _generate_full_shap_archive(
        final_model_registry, X_test_raw, y_test, X_val_raw, y_val,
        all_classes, water_source_mapping, output_dir,
        skip=skip_full_shap_archive,
        svm_explain_size=shap_svm_explain_size,
        background_size=shap_background_size)
    shap_df = _generate_shap_generalization_contrast(
        final_records, final_summary, X_train_raw, y_train,
        X_val_raw, y_val, all_feature_cols, feature_method,
        output_dir, skip=skip_shap_contrast)
    regen_manifest = _write_nested_regen_bundle(output_dir, {
        'outer_records': [],
        'outer_predictions': [],
        'outer_fold_indices': [],
        'outer_summary': pd.DataFrame(),
        'rank_stability': pd.DataFrame(),
        'final_records': final_records,
        'final_predictions': final_prediction_records,
        'final_summary': final_summary,
        'final_diagnostics': final_diagnostics,
        'ranking_comparison': comparison,
        'rank_correlations': corr_df,
        'factor_effects': effect_df,
        'search_history': search_history_records,
        'search_budget': search_budget_records,
        'search_budget_summary': budget_summary,
        'search_space_audit': search_space_audit,
        'shap_archive': shap_archive,
        'shap_contrast': shap_df,
        'validation_metadata': val_meta,
        'class_labels': {int(c): _class_name(c, water_source_mapping) for c in all_classes},
        'final_model_registry': final_model_registry,
        'protocol': {
            'mode': 'budget_matched_repeated',
            'inner_splits': int(inner_splits),
            'inner_repeats': int(inner_repeats),
            'final_eval_runs': int(final_eval_runs),
            'max_grid_candidates': int(max_grid_candidates),
            'population_size': int(pop_size),
            'population_max_iter': int(max_iter),
            'n_trials_by_algo': _json_safe(n_trials_by_algo),
            'feature_method': feature_method,
        },
    })
    sci_bundle = _export_sci_figure_table_bundle(
        output_dir, water_source_mapping=water_source_mapping)

    with open(output_dir / 'Logs' / 'budget_matched_generalization_summary.json',
              'w', encoding='utf-8') as f:
        json.dump(_json_safe({
            'protocol': 'budget_matched_repeated_internal_search_with_locked_test_and_external_validation',
            'goal': 'generalization_mechanism_without_using_external_validation_for_selection',
            'inner_splits': int(inner_splits),
            'inner_repeats': int(inner_repeats),
            'final_eval_runs': int(final_eval_runs),
            'max_grid_candidates': int(max_grid_candidates),
            'external_validation_used_for_selection': False,
            'key_outputs': {
                'final_eval_summary': str(tables_dir / 'FinalEval_Test_External_Summary.csv'),
                'ranking_comparison': str(tables_dir / 'Generalization_Ranking_Comparison.csv'),
                'algorithm_optimizer_effects': str(tables_dir / 'Algorithm_vs_Optimizer_EtaSquared.csv'),
                'shap_contrast': str(tables_dir / 'SHAP_Generalization_Contrast.csv'),
                'full_shap_archive': str(tables_dir / 'SHAP_FinalArchive'),
                'search_budget_audit': str(tables_dir / 'Search_Budget_Audit.csv'),
                'search_space_audit': str(tables_dir / 'Search_Space_Audit.csv'),
                'final_predictions': str(tables_dir / 'FinalEval_Test_External_Predictions.csv'),
                'figure_data_manifest': str(output_dir / 'RegenData' / 'figure_data_manifest.json'),
                'sci_figure_table_index': str(tables_dir / 'SCI_FigureTable_Data_Index.csv'),
            },
        }), f, indent=2, ensure_ascii=False)

    return {
        'outer_records': [],
        'outer_summary': pd.DataFrame(),
        'rank_stability': pd.DataFrame(),
        'final_records': final_records,
        'final_summary': final_summary,
        'ranking_comparison': comparison,
        'rank_correlations': corr_df,
        'factor_effects': effect_df,
        'shap_contrast': shap_df,
        'shap_archive': shap_archive,
        'final_predictions': final_pred_df,
        'search_history': search_history_df,
        'search_budget': search_budget_df,
        'regen_manifest': regen_manifest,
        'sci_bundle': sci_bundle,
        'validation_metadata': val_meta,
    }


def run_nested_generalization_protocol(X_train_raw, y_train, X_test_raw, y_test,
                                       all_feature_cols, feature_method,
                                       val_path, le, water_source_mapping,
                                       output_dir, target_col='y', ion_cols=None,
                                       n_trials_by_algo=None,
                                       outer_splits=5, outer_repeats=10,
                                       inner_splits=5, inner_repeats=3,
                                       final_eval_runs=30,
                                       pop_size=20, max_iter=50,
                                       max_grid_candidates=0,
                                       skip_shap_contrast=False,
                                       skip_full_shap_archive=False,
                                       shap_svm_explain_size=240,
                                       shap_background_size=64,
                                       seeds=SEEDS):
    """
    SCI-style protocol for generalization research rather than champion selection.

    External validation is not used in nested CV, hyperparameter search, or any
    model-ranking decision. It is loaded only after the nested-CV and full-training
    internal-test analyses are complete, then used to quantify domain transfer.
    """
    if ion_cols is None:
        ion_cols = ION_COLS
    if n_trials_by_algo is None:
        n_trials_by_algo = {algo: 100 for algo in _ALGORITHMS}
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ['Tables', 'Logs', 'Figures']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / 'Tables'

    print("\n" + "=" * 80)
    print("NESTED GENERALIZATION PROTOCOL")
    print("=" * 80)
    print("Research goals:")
    print("  1. Test whether internal-test performance predicts external validation.")
    print("  2. Compare algorithm choice vs optimizer choice for generalization.")
    print("  3. Use SHAP only as post-hoc explanation of generalization contrasts.")
    print("  External validation is never used for model selection.")

    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    all_classes = sorted(np.unique(y_train).tolist())
    outer_cv = RepeatedStratifiedKFold(
        n_splits=int(outer_splits), n_repeats=int(outer_repeats),
        random_state=42)
    outer_total = int(outer_splits) * int(outer_repeats)
    combo_total = len(_ALGORITHMS) * len(_OPTIMIZERS)
    outer_records = []
    outer_prediction_records = []
    outer_fold_index_records = []
    search_history_records = []
    search_budget_records = []

    for outer_i, (tr_idx, va_idx) in enumerate(outer_cv.split(X_train_raw, y_train), 1):
        outer_seed = seeds[(outer_i - 1) % len(seeds)]
        print(f"\n[Outer {outer_i:03d}/{outer_total:03d}] "
              f"train={len(tr_idx)} validation={len(va_idx)}",
              flush=True)
        for idx in tr_idx:
            outer_fold_index_records.append({
                'Outer_Fold': int(outer_i), 'Outer_Seed': int(outer_seed),
                'Split': 'train', 'Sample_Index': int(idx)
            })
        for idx in va_idx:
            outer_fold_index_records.append({
                'Outer_Fold': int(outer_i), 'Outer_Seed': int(outer_seed),
                'Split': 'validation', 'Sample_Index': int(idx)
            })
        combo_i = 0
        for algo in _ALGORITHMS:
            for optimizer in _OPTIMIZERS:
                combo_i += 1
                model_name = f'{algo}-{optimizer}'
                progress_label = (
                    f"Nested Outer {outer_i}/{outer_total} | "
                    f"Combo {combo_i}/{combo_total} | {model_name}")
                _print_progress(f"{progress_label} | search started")
                try:
                    params, inner_f1, search_time = _nested_search_once(
                        algo, optimizer, X_train_raw[tr_idx], y_train[tr_idx],
                        all_feature_cols, feature_method, outer_seed,
                        n_trials_by_algo, pop_size=pop_size, max_iter=max_iter,
                        inner_splits=inner_splits, inner_repeats=inner_repeats,
                        max_grid_candidates=max_grid_candidates,
                        progress_label=progress_label)
                    search_history_records.extend(_search_history_rows(
                        'nested_outer_cv', algo, optimizer, model_name, params,
                        inner_splits, inner_repeats, outer_fold=outer_i,
                        outer_seed=outer_seed))
                    search_budget_records.append(_search_budget_row(
                        'nested_outer_cv', algo, optimizer, model_name, params,
                        search_time, inner_splits, inner_repeats,
                        outer_fold=outer_i, outer_seed=outer_seed))
                    model, _preproc, X_outer_va = _fit_model_with_fold_preprocessing(
                        algo, _clean_model_params(params),
                        X_train_raw[tr_idx], y_train[tr_idx],
                        X_train_raw[va_idx], all_feature_cols,
                        feature_method, outer_seed)
                    y_pred = _predict_label_vector(model, X_outer_va)
                    y_proba = _predict_proba_matrix(model, X_outer_va, all_classes)
                    outer_prediction_records.extend(_prediction_rows(
                        'nested_outer_cv', 'outer_validation', model_name,
                        algo, optimizer, y_train[va_idx], y_pred, y_proba,
                        va_idx, all_classes, water_source_mapping,
                        outer_fold=outer_i, outer_seed=outer_seed))
                    metrics = _metric_pack_for_predictions(y_train[va_idx], y_pred)
                    p, r, f, _ = precision_recall_fscore_support(
                        y_train[va_idx], y_pred, labels=all_classes,
                        zero_division=0)
                    rec = {
                        'Phase': 'nested_outer_cv',
                        'Outer_Fold': int(outer_i),
                        'Outer_Seed': int(outer_seed),
                        'Model': model_name,
                        'Algorithm': algo,
                        'Optimizer': optimizer,
                        'Inner_CV_F1': float(inner_f1),
                        'Search_Time_s': float(search_time),
                        'Outer_Accuracy': metrics['Accuracy'],
                        'Outer_F1_Macro': metrics['F1_Macro'],
                        'Outer_F1_Weighted': metrics['F1_Weighted'],
                        'Outer_Kappa': metrics['Kappa'],
                        'Outer_MCC': metrics['MCC'],
                        'Best_Params_JSON': json.dumps(_json_safe(params), ensure_ascii=False),
                        'External_Validation_Used': False,
                    }
                    for class_i, class_id in enumerate(all_classes):
                        rec[f'Outer_Precision_Class{class_id}'] = float(p[class_i])
                        rec[f'Outer_Recall_Class{class_id}'] = float(r[class_i])
                        rec[f'Outer_F1_Class{class_id}'] = float(f[class_i])
                    outer_records.append(rec)
                    print(f"  {model_name:<24} inner={inner_f1:.4f} "
                          f"outer={metrics['F1_Macro']:.4f}",
                          flush=True)
                    _print_progress(
                        f"{progress_label} | model complete | "
                        f"inner={inner_f1:.4f} | outer={metrics['F1_Macro']:.4f} | "
                        f"search_time={search_time:.1f}s")
                except Exception as exc:
                    print(f"  {model_name:<24} FAILED: {exc}", flush=True)
        _checkpoint_nested_progress(
            output_dir, f'outer_{outer_i:03d}_complete',
            outer_records, outer_prediction_records, outer_fold_index_records,
            [], [], search_history_records, search_budget_records)

    outer_raw = pd.DataFrame(outer_records)
    _write_df(outer_raw, tables_dir / 'NestedCV_Outer_Raw')
    outer_pred_df = pd.DataFrame(outer_prediction_records)
    _write_df(outer_pred_df, tables_dir / 'NestedCV_Outer_Predictions')
    _write_df(pd.DataFrame(outer_fold_index_records),
              tables_dir / 'NestedCV_Outer_Fold_Indices')
    outer_summary = _summarize_generalization_records(outer_records, 'nested_outer_cv')
    _write_df(outer_summary, tables_dir / 'NestedCV_Outer_Summary')
    rank_stability = _outer_rank_stability(outer_records)
    _write_df(rank_stability, tables_dir / 'NestedCV_Rank_Stability')

    print("\n[Final full-training evaluation] Nested CV is complete; external validation is now loaded for descriptive scoring.")
    X_val_raw, y_val, val_meta = _load_external_validation_raw_for_generalization(
        val_path, all_feature_cols, target_col=target_col, ion_cols=ion_cols,
        le=le, water_source_mapping=water_source_mapping)
    with open(output_dir / 'Logs' / 'validation_policy_nested_generalization.json',
              'w', encoding='utf-8') as f:
        json.dump(_json_safe({
            'external_validation_used_for_selection': False,
            'external_validation_role': 'post-selection descriptive domain-transfer analysis',
            'validation_loaded_after_nested_cv': True,
            'validation_metadata': val_meta,
        }), f, indent=2, ensure_ascii=False)

    final_records = []
    final_prediction_records = []
    final_model_registry = {}
    final_seeds = list(seeds[:int(final_eval_runs)])
    final_combo_total = len(_ALGORITHMS) * len(_OPTIMIZERS)
    final_combo_i = 0
    for algo in _ALGORITHMS:
        for optimizer in _OPTIMIZERS:
            final_combo_i += 1
            model_name = f'{algo}-{optimizer}'
            print(f"\n[Final Eval {final_combo_i}/{final_combo_total}] {model_name}",
                  flush=True)
            for run_i, seed in enumerate(final_seeds, 1):
                progress_label = (
                    f"Final Eval Combo {final_combo_i}/{final_combo_total} | "
                    f"Run {run_i}/{len(final_seeds)} | {model_name}")
                _print_progress(f"{progress_label} | search started")
                try:
                    params, train_cv_f1, search_time = _nested_search_once(
                        algo, optimizer, X_train_raw, y_train,
                        all_feature_cols, feature_method, seed,
                        n_trials_by_algo, pop_size=pop_size, max_iter=max_iter,
                        inner_splits=inner_splits, inner_repeats=inner_repeats,
                        max_grid_candidates=max_grid_candidates,
                        progress_label=progress_label)
                    search_history_records.extend(_search_history_rows(
                        'final_full_training', algo, optimizer, model_name, params,
                        inner_splits, inner_repeats, run_seed=seed))
                    search_budget_records.append(_search_budget_row(
                        'final_full_training', algo, optimizer, model_name, params,
                        search_time, inner_splits, inner_repeats, run_seed=seed))
                    model, preproc, X_test_sel = _fit_model_with_fold_preprocessing(
                        algo, _clean_model_params(params), X_train_raw, y_train,
                        X_test_raw, all_feature_cols, feature_method, seed)
                    y_test_pred = _predict_label_vector(model, X_test_sel)
                    test_proba = _predict_proba_matrix(model, X_test_sel, all_classes)
                    test_metrics = _metric_pack_for_predictions(y_test, y_test_pred)
                    final_prediction_records.extend(_prediction_rows(
                        'full_train_internal_test_external_val', 'internal_test',
                        model_name, algo, optimizer, y_test, y_test_pred,
                        test_proba, np.arange(len(y_test)), all_classes,
                        water_source_mapping, run_seed=seed))
                    if model_name not in final_model_registry:
                        final_model_registry[model_name] = {
                            'model': model,
                            'preproc': preproc,
                            'Algorithm': algo,
                            'Optimizer': optimizer,
                            'Run_Seed': int(seed),
                            'Best_Params': _json_safe(params),
                        }
                    rec = {
                        'Phase': 'full_train_internal_test_external_val',
                        'Model': model_name,
                        'Algorithm': algo,
                        'Optimizer': optimizer,
                        'Run_Seed': int(seed),
                        'Train_Internal_CV_F1': float(train_cv_f1),
                        'Search_Time_s': float(search_time),
                        'Test_Accuracy': test_metrics['Accuracy'],
                        'Test_F1_Macro': test_metrics['F1_Macro'],
                        'Test_F1_Weighted': test_metrics['F1_Weighted'],
                        'Test_Kappa': test_metrics['Kappa'],
                        'Test_MCC': test_metrics['MCC'],
                        'Best_Params_JSON': json.dumps(_json_safe(params), ensure_ascii=False),
                        'External_Validation_Used_For_Selection': False,
                    }
                    if X_val_raw is not None and y_val is not None:
                        X_val_sel = _transform_with_fold_preprocessing(X_val_raw, preproc)
                        y_val_pred = _predict_label_vector(model, X_val_sel)
                        val_proba = _predict_proba_matrix(model, X_val_sel, all_classes)
                        val_metrics = _metric_pack_for_predictions(y_val, y_val_pred)
                        final_prediction_records.extend(_prediction_rows(
                            'full_train_internal_test_external_val', 'external_val',
                            model_name, algo, optimizer, y_val, y_val_pred,
                            val_proba, np.arange(len(y_val)), all_classes,
                            water_source_mapping, run_seed=seed))
                        rec.update({
                            'Val_Accuracy': val_metrics['Accuracy'],
                            'Val_F1_Macro': val_metrics['F1_Macro'],
                            'Val_F1_Weighted': val_metrics['F1_Weighted'],
                            'Val_Kappa': val_metrics['Kappa'],
                            'Val_MCC': val_metrics['MCC'],
                            'Generalization_Gap': val_metrics['F1_Macro'] - test_metrics['F1_Macro'],
                        })
                    final_records.append(rec)
                    msg = f"  {run_i:02d}/{len(final_seeds)} seed={seed} test={test_metrics['F1_Macro']:.4f}"
                    if 'Val_F1_Macro' in rec:
                        msg += f" val={rec['Val_F1_Macro']:.4f} gap={rec['Generalization_Gap']:+.4f}"
                    print(msg, flush=True)
                    _print_progress(
                        f"{progress_label} | model complete | "
                        f"train_cv={train_cv_f1:.4f} | "
                        f"test={test_metrics['F1_Macro']:.4f} | "
                        f"search_time={search_time:.1f}s")
                except Exception as exc:
                    print(f"  {run_i:02d}/{len(final_seeds)} seed={seed} FAILED: {exc}",
                          flush=True)
            _checkpoint_nested_progress(
                output_dir, f'final_{_safe_name_stem(model_name)}_complete',
                outer_records, outer_prediction_records, outer_fold_index_records,
                final_records, final_prediction_records,
                search_history_records, search_budget_records)

    final_raw = pd.DataFrame(final_records)
    _write_df(final_raw, tables_dir / 'FinalEval_Test_External_Raw')
    final_pred_df = pd.DataFrame(final_prediction_records)
    _write_df(final_pred_df, tables_dir / 'FinalEval_Test_External_Predictions')
    final_diagnostics = _make_prediction_diagnostic_tables(
        final_pred_df, all_classes, water_source_mapping, output_dir,
        'FinalEval_Test_External')
    search_history_df = pd.DataFrame(search_history_records)
    _write_df(search_history_df, tables_dir / 'Search_Convergence_Long')
    search_budget_df = pd.DataFrame(search_budget_records)
    _write_df(search_budget_df, tables_dir / 'Search_Budget_Audit')
    if not search_budget_df.empty:
        budget_summary = search_budget_df.groupby(
            ['Search_Phase', 'Model', 'Algorithm', 'Optimizer'],
            as_index=False).agg(
                N_Searches=('Fitness_Evals_Total', 'size'),
                Fitness_Evals_Total_mean=('Fitness_Evals_Total', 'mean'),
                Fitness_Evals_Total_median=('Fitness_Evals_Total', 'median'),
                Nominal_Model_Fits_Total_mean=('Nominal_Model_Fits_Total', 'mean'),
                Search_Time_s_mean=('Search_Time_s', 'mean'),
                Search_Time_s_sum=('Search_Time_s', 'sum'),
            )
        _write_df(budget_summary, tables_dir / 'Search_Budget_Audit_Summary')
    else:
        budget_summary = pd.DataFrame()
    search_space_audit = _search_space_audit_table(
        n_trials_by_algo, pop_size, max_iter, max_grid_candidates,
        inner_splits, inner_repeats)
    _write_df(search_space_audit, tables_dir / 'Search_Space_Audit')
    final_summary = _summarize_generalization_records(
        final_records, 'full_train_internal_test_external_val')
    _write_df(final_summary, tables_dir / 'FinalEval_Test_External_Summary')

    comparison, corr_df, effect_df = _make_generalization_analysis_tables(
        output_dir, outer_summary, rank_stability, final_summary, final_records)
    _make_supplementary_ranking_and_baseline_tables(
        output_dir, final_summary, y_train, all_classes, water_source_mapping)
    _write_dataset_archives(
        output_dir, X_train_raw, y_train, X_test_raw, y_test,
        X_val_raw, y_val, all_feature_cols, water_source_mapping)
    shap_archive = _generate_full_shap_archive(
        final_model_registry, X_test_raw, y_test, X_val_raw, y_val,
        all_classes, water_source_mapping, output_dir,
        skip=skip_full_shap_archive,
        svm_explain_size=shap_svm_explain_size,
        background_size=shap_background_size)
    shap_df = _generate_shap_generalization_contrast(
        final_records, final_summary, X_train_raw, y_train,
        X_val_raw, y_val, all_feature_cols, feature_method,
        output_dir, skip=skip_shap_contrast)
    regen_manifest = _write_nested_regen_bundle(output_dir, {
        'outer_records': outer_records,
        'outer_predictions': outer_prediction_records,
        'outer_fold_indices': outer_fold_index_records,
        'outer_summary': outer_summary,
        'rank_stability': rank_stability,
        'final_records': final_records,
        'final_predictions': final_prediction_records,
        'final_summary': final_summary,
        'final_diagnostics': final_diagnostics,
        'ranking_comparison': comparison,
        'rank_correlations': corr_df,
        'factor_effects': effect_df,
        'search_history': search_history_records,
        'search_budget': search_budget_records,
        'search_budget_summary': budget_summary,
        'search_space_audit': search_space_audit,
        'shap_archive': shap_archive,
        'shap_contrast': shap_df,
        'validation_metadata': val_meta,
        'class_labels': {int(c): _class_name(c, water_source_mapping) for c in all_classes},
        'final_model_registry': final_model_registry,
        'protocol': {
            'outer_splits': int(outer_splits),
            'outer_repeats': int(outer_repeats),
            'inner_splits': int(inner_splits),
            'inner_repeats': int(inner_repeats),
            'final_eval_runs': int(final_eval_runs),
            'max_grid_candidates': int(max_grid_candidates),
            'population_size': int(pop_size),
            'population_max_iter': int(max_iter),
            'n_trials_by_algo': _json_safe(n_trials_by_algo),
            'feature_method': feature_method,
        },
    })
    sci_bundle = _export_sci_figure_table_bundle(
        output_dir, water_source_mapping=water_source_mapping)

    with open(output_dir / 'Logs' / 'nested_generalization_summary.json',
              'w', encoding='utf-8') as f:
        json.dump(_json_safe({
            'protocol': 'repeated_nested_cv_plus_locked_internal_test_and_external_validation',
            'goal': 'generalization_mechanism_not_champion_selection',
            'outer_splits': int(outer_splits),
            'outer_repeats': int(outer_repeats),
            'inner_splits': int(inner_splits),
            'inner_repeats': int(inner_repeats),
            'final_eval_runs': int(final_eval_runs),
            'max_grid_candidates': int(max_grid_candidates),
            'external_validation_used_for_selection': False,
            'key_outputs': {
                'nested_cv_summary': str(tables_dir / 'NestedCV_Outer_Summary.csv'),
                'ranking_comparison': str(tables_dir / 'Generalization_Ranking_Comparison.csv'),
                'algorithm_optimizer_effects': str(tables_dir / 'Algorithm_vs_Optimizer_EtaSquared.csv'),
                'shap_contrast': str(tables_dir / 'SHAP_Generalization_Contrast.csv'),
                'full_shap_archive': str(tables_dir / 'SHAP_FinalArchive'),
                'search_budget_audit': str(tables_dir / 'Search_Budget_Audit.csv'),
                'search_space_audit': str(tables_dir / 'Search_Space_Audit.csv'),
                'final_predictions': str(tables_dir / 'FinalEval_Test_External_Predictions.csv'),
                'figure_data_manifest': str(output_dir / 'RegenData' / 'figure_data_manifest.json'),
                'sci_figure_table_index': str(tables_dir / 'SCI_FigureTable_Data_Index.csv'),
            },
        }), f, indent=2, ensure_ascii=False)

    return {
        'outer_records': outer_records,
        'outer_summary': outer_summary,
        'rank_stability': rank_stability,
        'final_records': final_records,
        'final_summary': final_summary,
        'ranking_comparison': comparison,
        'rank_correlations': corr_df,
        'factor_effects': effect_df,
        'shap_contrast': shap_df,
        'shap_archive': shap_archive,
        'final_predictions': final_pred_df,
        'search_history': search_history_df,
        'search_budget': search_budget_df,
        'regen_manifest': regen_manifest,
        'sci_bundle': sci_bundle,
        'validation_metadata': val_meta,
    }


# ================================================================================
# SECTION 10: MAIN FUNCTION
# ================================================================================

def main():
    """
    Pipeline entry point for the algorithm x optimizer generalization study.

    Two protocols are dispatched on ``--protocol``:

      * ``nested_generalization`` (default): repeated stratified nested
        cross-validation (5-fold x 10 repeats outer; 5-fold x 3 repeats inner)
        over the 4 algorithms x 7 optimizers = 28 combinations, followed by a
        full-training final-evaluation phase that scores each combination on
        the held-out internal test set and the locked external validation set
        across ``--final_eval_runs`` random seeds. External validation is
        loaded only after the nested CV is complete and is never used for
        model selection or hyperparameter search.

      * ``budget_matched_generalization``: a single inner-CV search per
        combination repeated over the same final-evaluation seeds, matched in
        compute budget to the nested protocol so that the two can be compared
        on equal footing.

    Both protocols handle imputation, feature selection and scaling inside
    each fold, persist tabular artefacts under ``Output/Tables`` and
    diagnostic figures under ``Output/Figures``, and write a JSON validation
    policy log that records the role of the external set as post-selection
    descriptive scoring (Cawley & Talbot, 2010).

    Returns
    -------
    nested_payload : object
        Whatever the chosen protocol returns; consumers should treat it as
        protocol-specific.
    """
    print("=" * 80)
    print("SCI Top-Tier Paper: Mine Water Source Identification ")
    print(" SYMMETRIC ALGORITHM x OPTIMIZER MATRIX: 4 algorithms x 7 optimizers (28 combinations)")
    print(f" External validation is reported descriptively only ({GENERALISATION_PRIMARY_METRIC_LABEL})")
    print("=" * 80)

    cli_args = parse_arguments()
    configure_parallelism(cli_args.n_jobs, cli_args.gridsearch_n_jobs,
                          cli_args.search_n_jobs)
    print(f"[Info] Protocol: {cli_args.protocol}", flush=True)
    print(
        f"[Info] Parallelism: model n_jobs={DETERMINISTIC_N_JOBS}, "
        f"GridSearchCV n_jobs={GRIDSEARCH_N_JOBS}, "
        f"search_model_n_jobs={SEARCH_MODEL_N_JOBS}, "
        f"detected_cpu={_available_cpu_count()}",
        flush=True,
    )
    train_path, test_path, output_dir, val_path_main = resolve_paths(cli_args)

    n_trials_rf = cli_args.n_trials_rf
    n_trials_xgb = cli_args.n_trials_xgb
    n_trials_svm = cli_args.n_trials_svm
    n_trials_lgbm = cli_args.n_trials_lgbm
    feature_method = cli_args.feature_method

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ['Figures', 'Tables', 'Models', 'Logs']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    if getattr(cli_args, 'convergence_only', False):
        print("\n[Convergence-Only] Regenerating convergence figures from saved cache...")
        ok = regenerate_convergence_from_cache(output_dir)
        if not ok:
            print("[Convergence-Only] Failed: convergence cache not found or unreadable.")
        return None

    start_time = time.time()
    _global_seed = 42
    _set_all_seeds(_global_seed)

    # === Step 1: Load Data ===
    print("\n[Step 1] Loading data...")
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    print(f" Training: {train_df.shape[0]} samples x {train_df.shape[1]} cols")
    print(f" Test: {test_df.shape[0]} samples x {test_df.shape[1]} cols")

    target_col = resolve_target_column(train_df, 'y')
    ion_cols = ION_COLS

    for _df_name, _df in [('train_df', train_df), ('test_df', test_df)]:
        _resolved_target_col = resolve_target_column(_df, target_col)
        if _resolved_target_col != target_col:
            raise KeyError(
                f"Target column mismatch across files: train uses '{target_col}', "
                f"but {_df_name} uses '{_resolved_target_col}'. Please unify column names."
            )
    _missing_ions = [c for c in ION_COLS if c not in train_df.columns]
    if _missing_ions:
        raise KeyError(f"Ion columns missing: {_missing_ions}")
    # pH (x8) is a mandatory input feature alongside the seven ions. It is
    # checked here  - before any feature engineering  - to fail fast on
    # malformed inputs rather than producing cryptic downstream errors.
    for _df_name, _df in [('train_df', train_df), ('test_df', test_df)]:
        if PH_COL not in _df.columns:
            raise KeyError(
                f"pH column '{PH_COL}' missing in {_df_name}. "
                f"Required 8 features: x1..x7 (ions, mg/L) + x8 (pH)."
            )

    # === Step 2: Feature Engineering + Label Encoding + Feature Selection ===
    print("\n[Step 2] Feature engineering...")
    train_df_fe = create_geochemical_features(train_df.copy(), target_col, ion_cols)
    test_df_fe = create_geochemical_features(test_df.copy(), target_col, ion_cols)

    y_train_raw = train_df_fe[target_col].values
    y_test_raw = test_df_fe[target_col].values

    print("\n[Step 2a] Label encoding (fixed canonical order)...")
    # FIX: Use fixed canonical label encoding to guarantee consistent class IDs
    # across train/test/val splits regardless of which labels appear in each file.
    le, y_train_enc, y_test_enc, _y_val_dummy, water_source_mapping = encode_labels_fixed(
        y_train_raw, y_test_raw
    )

    print(f" Training samples: {len(y_train_enc)} (no oversampling)")
    for cls, count in sorted(dict(zip(*np.unique(y_train_enc, return_counts=True))).items()):
        print(f" {water_source_mapping.get(cls, f'Class {cls}')}: {count}")

    print(f"\n[Step 2b] Feature selection (method='{feature_method}')...")
    all_feature_cols = [c for c in CORE_FEATURE_COLS if c in train_df_fe.columns]
    X_all_train_raw = train_df_fe[all_feature_cols].values
    X_all_test_raw = test_df_fe[all_feature_cols].values

    # Missing value report
    _missing_count = pd.DataFrame(X_all_train_raw).isnull().sum().values
    _missing_rate = pd.DataFrame(X_all_train_raw).isnull().mean().values * 100
    missing_report = pd.DataFrame({'Feature': all_feature_cols,
                                   'Missing_Count': _missing_count,
                                   'Missing_Rate_%': np.round(_missing_rate, 2)})
    missing_report.to_csv(output_dir / 'Tables' / 'missing_value_report.csv', index=False)

    # === Generalization protocol ===
    # No global imputer or feature selector is fitted here. The chosen protocol
    # fits imputation, feature selection, scaling and model training only inside
    # the training data used for each search/evaluation cycle. External
    # validation remains a locked descriptive scoring set and is never used for
    # model selection.
    n_trials_by_algo = {
        'RF': n_trials_rf,
        'XGBoost': n_trials_xgb,
        'SVM': n_trials_svm,
        'LightGBM': n_trials_lgbm,
    }
    nested_payload = run_nested_generalization_protocol(
        X_train_raw=X_all_train_raw,
        y_train=y_train_enc,
        X_test_raw=X_all_test_raw,
        y_test=y_test_enc,
        all_feature_cols=all_feature_cols,
        feature_method=feature_method,
        val_path=val_path_main,
        le=le,
        water_source_mapping=water_source_mapping,
        output_dir=output_dir,
        target_col=target_col,
        ion_cols=ion_cols,
        n_trials_by_algo=n_trials_by_algo,
        outer_splits=cli_args.nested_outer_splits,
        outer_repeats=cli_args.nested_outer_repeats,
        inner_splits=cli_args.nested_inner_splits,
        inner_repeats=cli_args.nested_inner_repeats,
        final_eval_runs=cli_args.final_eval_runs,
        pop_size=cli_args.population_size,
        max_iter=cli_args.population_max_iter,
        max_grid_candidates=cli_args.max_grid_candidates,
        skip_shap_contrast=cli_args.skip_shap_contrast,
        skip_full_shap_archive=cli_args.skip_full_shap_archive,
        shap_svm_explain_size=cli_args.shap_svm_explain_size,
        shap_background_size=cli_args.shap_background_size,
        seeds=SEEDS,
    ) if cli_args.protocol == 'nested_generalization' else run_budget_matched_generalization_protocol(
        X_train_raw=X_all_train_raw,
        y_train=y_train_enc,
        X_test_raw=X_all_test_raw,
        y_test=y_test_enc,
        all_feature_cols=all_feature_cols,
        feature_method=feature_method,
        val_path=val_path_main,
        le=le,
        water_source_mapping=water_source_mapping,
        output_dir=output_dir,
        target_col=target_col,
        ion_cols=ion_cols,
        n_trials_by_algo=n_trials_by_algo,
        inner_splits=cli_args.nested_inner_splits,
        inner_repeats=cli_args.nested_inner_repeats,
        final_eval_runs=cli_args.final_eval_runs,
        pop_size=cli_args.population_size,
        max_iter=cli_args.population_max_iter,
        max_grid_candidates=cli_args.max_grid_candidates,
        skip_shap_contrast=cli_args.skip_shap_contrast,
        skip_full_shap_archive=cli_args.skip_full_shap_archive,
        shap_svm_explain_size=cli_args.shap_svm_explain_size,
        shap_background_size=cli_args.shap_background_size,
        seeds=SEEDS,
    )
    return nested_payload


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    nested_payload = main()
