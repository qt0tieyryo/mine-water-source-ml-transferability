from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.pipeline import Pipeline


def _save_figure(fig, path_stem, dpi=600):
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path_stem) + '.png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(str(path_stem) + '.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')


def _sanitize_name(name):
    safe = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_'
                   for ch in str(name))
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip('_') or 'model'


def _is_svm_model(model):
    return (isinstance(model, Pipeline) and 'svm' in model.named_steps
            or type(model).__name__ == 'SVC')


def _subset_indices(n_rows, max_rows):
    if n_rows <= max_rows:
        return np.arange(n_rows, dtype=int)
    return np.linspace(0, n_rows - 1, num=max_rows, dtype=int)


def _as_2d_shap_matrix(values, n_samples, n_features):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f'Expected a 2-D SHAP matrix, got shape {arr.shape}')
    if arr.shape[0] == n_samples:
        return arr[:, :n_features]
    if arr.shape[1] == n_samples and arr.shape[0] >= n_features:
        return arr[:n_features, :].T
    raise ValueError(f'Unsupported 2-D SHAP shape: {arr.shape}')


def _collapse_multiclass_stack(class_stack):
    """Collapse class-specific SHAP arrays without signed cancellation.

    SHAP returns multiclass explanations either as a list of class matrices or
    as a 3-D array. For probability models such as RandomForestClassifier, the
    signed attributions across classes can sum to zero for every sample/feature.
    A plain signed average therefore erases the signal. We report magnitude as
    mean(abs(SHAP)) across classes and keep the sign of the dominant class only
    so beeswarm plots still have a left/right direction.
    """
    stack = np.asarray(class_stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError(f'Expected class stack with shape (class, sample, feature), got {stack.shape}')
    if stack.shape[0] == 1:
        return stack[0]

    abs_stack = np.abs(stack)
    mean_abs = np.mean(abs_stack, axis=0)
    dominant_class = np.argmax(abs_stack, axis=0)
    dominant_sign = np.take_along_axis(
        np.sign(stack), dominant_class[np.newaxis, :, :], axis=0
    )[0]
    dominant_sign = np.where(dominant_sign == 0, 1.0, dominant_sign)
    return dominant_sign * mean_abs


def _collapse_shap_values(raw_values, n_samples, n_features):
    if isinstance(raw_values, list):
        mats = [_collapse_shap_values(v, n_samples, n_features)
                for v in raw_values]
        mats = [m for m in mats if m is not None and m.size > 0]
        if not mats:
            raise ValueError('No SHAP arrays available after collapsing.')
        return _collapse_multiclass_stack(np.stack(mats, axis=0))

    if hasattr(raw_values, 'values'):
        raw_values = raw_values.values

    arr = np.asarray(raw_values, dtype=float)
    if arr.ndim == 2:
        return _as_2d_shap_matrix(arr, n_samples, n_features)

    if arr.ndim < 2:
        raise ValueError(f'Unsupported SHAP ndim: {arr.ndim}')

    sample_axis = next((i for i, size in enumerate(arr.shape)
                        if size == n_samples), None)
    feature_axis = next((i for i, size in enumerate(arr.shape)
                         if size == n_features and i != sample_axis), None)
    if sample_axis is None or feature_axis is None:
        raise ValueError(f'Cannot infer sample/feature axes from shape {arr.shape}')

    arr = np.moveaxis(arr, [sample_axis, feature_axis], [0, 1])
    if arr.ndim > 2:
        arr = arr.reshape(n_samples, n_features, -1)
        arr = np.moveaxis(arr, 2, 0)
        return _collapse_multiclass_stack(arr)
    return _as_2d_shap_matrix(arr, n_samples, n_features)


def mean_abs_shap(shap_values):
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        matrix = shap_values
    else:
        arr = np.asarray(shap_values)
        if arr.ndim != 2:
            raise ValueError('mean_abs_shap expects a 2-D SHAP matrix.')
        matrix = arr
    return np.mean(np.abs(matrix), axis=0)


def _array_or_none(values):
    if values is None:
        return None
    try:
        return np.asarray(values, dtype=float)
    except Exception:
        return None


def _classwise_shap_values(raw_values, n_samples, n_features):
    """Return class-specific SHAP as (class, sample, feature) when available."""
    if raw_values is None:
        return None
    if isinstance(raw_values, list):
        mats = []
        for values in raw_values:
            try:
                mats.append(_collapse_shap_values(values, n_samples, n_features))
            except Exception:
                continue
        return np.stack(mats, axis=0) if mats else None

    if hasattr(raw_values, 'values'):
        raw_values = raw_values.values

    arr = _array_or_none(raw_values)
    if arr is None:
        return None
    if arr.ndim == 2:
        return _as_2d_shap_matrix(arr, n_samples, n_features)[np.newaxis, :, :]
    if arr.ndim < 3:
        return None

    sample_axis = next((i for i, size in enumerate(arr.shape)
                        if size == n_samples), None)
    feature_axis = next((i for i, size in enumerate(arr.shape)
                         if size == n_features and i != sample_axis), None)
    if sample_axis is None or feature_axis is None:
        return None

    arr = np.moveaxis(arr, [sample_axis, feature_axis], [0, 1])
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    arr = arr.reshape(n_samples, n_features, -1)
    return np.moveaxis(arr, 2, 0)


def _base_values_from(explainer=None, explanation=None):
    if explanation is not None and hasattr(explanation, 'base_values'):
        base = _array_or_none(explanation.base_values)
        if base is not None:
            return base
    if explainer is not None and hasattr(explainer, 'expected_value'):
        return _array_or_none(explainer.expected_value)
    return None


def _shape_or_blank(values):
    arr = _array_or_none(values)
    return '' if arr is None else 'x'.join(str(int(v)) for v in arr.shape)


def compute_shap_robust(model, X, background_size=64, svm_explain_size=120,
                        return_metadata=False):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f'Expected 2-D X, got shape {X.shape}')

    n_samples, n_features = X.shape

    if _is_svm_model(model):
        bg_idx = _subset_indices(n_samples, min(background_size, n_samples))
        ex_idx = _subset_indices(n_samples, min(svm_explain_size, n_samples))
        X_background = X[bg_idx]
        X_explain = X[ex_idx]
        predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
        explainer = shap.KernelExplainer(predict_fn, X_background)
        raw_values = explainer.shap_values(X_explain, nsamples='auto')
        shap_matrix = _collapse_shap_values(raw_values, len(X_explain), n_features)
        aux = {
            'sample_indices': ex_idx,
            'background_indices': bg_idx,
            'base_values': _base_values_from(explainer=explainer),
            'classwise_values': _classwise_shap_values(raw_values, len(X_explain), n_features),
        }
        result = (shap_matrix, X_explain, 'kernel', aux)
        return result if return_metadata else result[:3]

    try:
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(X)
        shap_matrix = _collapse_shap_values(raw_values, n_samples, n_features)
        aux = {
            'sample_indices': np.arange(n_samples, dtype=int),
            'background_indices': None,
            'base_values': _base_values_from(explainer=explainer),
            'classwise_values': _classwise_shap_values(raw_values, n_samples, n_features),
        }
        result = (shap_matrix, X, 'tree', aux)
        return result if return_metadata else result[:3]
    except Exception as tree_exc:
        bg_idx = _subset_indices(n_samples, min(background_size, n_samples))
        ex_idx = _subset_indices(n_samples, min(240, n_samples))
        X_background = X[bg_idx]
        X_explain = X[ex_idx]
        try:
            predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
            explainer = shap.Explainer(predict_fn, X_background)
            explanation = explainer(X_explain)
            shap_matrix = _collapse_shap_values(explanation, len(X_explain), n_features)
            aux = {
                'sample_indices': ex_idx,
                'background_indices': bg_idx,
                'base_values': _base_values_from(explainer=explainer, explanation=explanation),
                'classwise_values': _classwise_shap_values(explanation, len(X_explain), n_features),
            }
            result = (shap_matrix, X_explain, 'generic', aux)
            return result if return_metadata else result[:3]
        except Exception as generic_exc:
            raise RuntimeError(
                f'Robust SHAP failed (tree={tree_exc}; generic={generic_exc})'
            ) from generic_exc


def _plot_beeswarm(shap_matrix, X_used, feature_cols, output_path, title):
    plt.figure(figsize=(11, max(6, len(feature_cols) * 0.45)))
    shap.summary_plot(shap_matrix, X_used, feature_names=feature_cols,
                      show=False, max_display=min(15, len(feature_cols)))
    plt.title(title, fontsize=11)
    plt.tight_layout()
    _save_figure(plt.gcf(), output_path)
    plt.close('all')


def _plot_mean_abs_bar(mean_abs, feature_cols, output_path, title):
    order = np.argsort(mean_abs)[::-1]
    labels = [feature_cols[i] for i in order][::-1]
    values = mean_abs[order][::-1]
    fig, ax = plt.subplots(figsize=(9, max(6, len(labels) * 0.45)))
    ax.barh(labels, values, color='#378ADD', edgecolor='white', alpha=0.9)
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, axis='x')
    plt.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def _compute_interaction_matrix(model, X_used, feature_cols):
    X_subset = X_used[_subset_indices(len(X_used), min(50, len(X_used)))]
    n_samples = X_subset.shape[0]
    n_features = len(feature_cols)
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_interaction_values(X_subset)
    if isinstance(raw, list):
        mats = [np.mean(np.abs(np.asarray(v)), axis=0) for v in raw]
        inter = np.mean(np.stack(mats, axis=0), axis=0)
    else:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 2:
            inter = np.abs(arr)
        elif arr.ndim == 3:
            inter = np.mean(np.abs(arr), axis=0)
        elif arr.ndim >= 4:
            sample_axis = next((i for i, size in enumerate(arr.shape)
                                if size == n_samples), None)
            feature_axes = [i for i, size in enumerate(arr.shape)
                            if size == n_features and i != sample_axis]
            if sample_axis is None or len(feature_axes) < 2:
                raise ValueError(f'Cannot infer interaction axes from shape {arr.shape}')
            arr = np.moveaxis(arr, [sample_axis, feature_axes[0], feature_axes[1]], [0, 1, 2])
            inter = np.mean(np.abs(arr), axis=tuple([0] + list(range(3, arr.ndim))))
        else:
            raise ValueError(f'Unsupported interaction ndim: {arr.ndim}')
    n_feat = min(inter.shape[0], len(feature_cols))
    return inter[:n_feat, :n_feat], X_subset.shape[0]


def _plot_heatmap(df, output_path, title, cmap='RdBu_r', center=None, fmt='.3f'):
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(df.columns) + 4),
                                    max(5, 0.45 * len(df.index) + 3)))
    sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, center=center,
                linewidths=0.5, linecolor='#dddddd', ax=ax,
                cbar_kws={'shrink': 0.85})
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def generate_shap_all_combinations(model_registry, X_val, y_val, feature_cols,
                                    class_labels=None, output_dir=None,
                                    compute_interactions=True, skip_svm=False,
                                    display_feature_names=None,
                                    verbose=True):
    if output_dir is None:
        raise ValueError('output_dir is required')

    X_val = np.asarray(X_val, dtype=float)
    feature_cols = list(feature_cols)
    display_feature_cols = (list(display_feature_names)
                            if display_feature_names is not None
                            else list(feature_cols))

    figures_dir = Path(output_dir) / 'Figures' / 'SHAP_All'
    tables_dir = Path(output_dir) / 'Tables'
    shap_tables_dir = tables_dir / 'SHAP_All'
    figures_dir.mkdir(parents=True, exist_ok=True)
    shap_tables_dir.mkdir(parents=True, exist_ok=True)

    raw_arrays = {}
    classwise_arrays = {}
    x_used_by_model = {}
    sample_indices_by_model = {}
    background_indices_by_model = {}
    base_values_by_model = {}
    mean_abs_by_model = {}
    metadata_rows = []
    interaction_tables = {}

    for model_name, model in model_registry.items():
        if model is None:
            continue
        if skip_svm and _is_svm_model(model):
            continue

        safe_name = _sanitize_name(model_name)
        try:
            shap_matrix, X_used, backend, shap_meta = compute_shap_robust(
                model, X_val, return_metadata=True)
            mean_abs = mean_abs_shap(shap_matrix)
        except Exception as exc:
            if verbose:
                print(f"  [SHAP-All] {model_name}: skipped ({exc})")
            continue

        raw_arrays[model_name] = shap_matrix
        x_used_by_model[model_name] = X_used
        sample_indices_by_model[model_name] = shap_meta.get('sample_indices')
        background_indices_by_model[model_name] = shap_meta.get('background_indices')
        base_values_by_model[model_name] = shap_meta.get('base_values')
        classwise_values = shap_meta.get('classwise_values')
        if classwise_values is not None:
            classwise_arrays[model_name] = classwise_values
        mean_abs_by_model[model_name] = mean_abs
        metadata_rows.append({
            'Model': model_name,
            'Backend': backend,
            'N_SHAP_Samples': int(shap_matrix.shape[0]),
            'N_Features': int(shap_matrix.shape[1]),
            'Is_SVM': bool(_is_svm_model(model)),
            'N_Background_Samples': (
                0 if shap_meta.get('background_indices') is None
                else int(len(shap_meta.get('background_indices')))
            ),
            'Sample_Indices_Saved': True,
            'Base_Value_Shape': _shape_or_blank(shap_meta.get('base_values')),
            'Classwise_SHAP_Shape': _shape_or_blank(classwise_values),
        })

        raw_df = pd.DataFrame(shap_matrix, columns=feature_cols[:shap_matrix.shape[1]])
        raw_df.to_csv(shap_tables_dir / f'SHAP_Raw_{safe_name}.csv', index=False)
        x_used_df = pd.DataFrame(X_used, columns=feature_cols[:X_used.shape[1]])
        x_used_df.insert(0, 'Original_Row_Index', shap_meta.get('sample_indices'))
        x_used_df.to_csv(shap_tables_dir / f'SHAP_Input_XUsed_{safe_name}.csv', index=False)
        if classwise_values is not None:
            np.savez_compressed(
                shap_tables_dir / f'SHAP_Classwise_{safe_name}.npz',
                values=classwise_values,
                sample_indices=shap_meta.get('sample_indices'),
                feature_names=np.asarray(feature_cols[:classwise_values.shape[2]], dtype=str),
                class_labels=np.asarray(class_labels or [], dtype=str),
            )

        mean_df = pd.DataFrame({
            'Feature': feature_cols[:len(mean_abs)],
            'Feature_Display': display_feature_cols[:len(mean_abs)],
            'Mean_Abs_SHAP': mean_abs,
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        mean_df.to_csv(shap_tables_dir / f'SHAP_MeanAbs_{safe_name}.csv', index=False)

        _plot_beeswarm(
            shap_matrix,
            X_used,
            display_feature_cols[:shap_matrix.shape[1]],
            figures_dir / f'Figure_SHAP_All_Beeswarm_{safe_name}',
            f'SHAP Beeswarm - {model_name}',
        )
        _plot_mean_abs_bar(
            mean_abs,
            display_feature_cols[:len(mean_abs)],
            figures_dir / f'Figure_SHAP_All_Bar_{safe_name}',
            f'Mean |SHAP| - {model_name}',
        )

        if compute_interactions and not _is_svm_model(model):
            try:
                inter_mat, n_int = _compute_interaction_matrix(model, X_used, feature_cols)
                inter_df = pd.DataFrame(inter_mat,
                                        index=display_feature_cols[:inter_mat.shape[0]],
                                        columns=display_feature_cols[:inter_mat.shape[1]])
                interaction_tables[model_name] = inter_df
                inter_df.to_csv(shap_tables_dir / f'SHAP_Interaction_{safe_name}.csv')
                _plot_heatmap(
                    inter_df,
                    figures_dir / f'Figure_SHAP_All_Interaction_{safe_name}',
                    f'SHAP Interaction - {model_name} (n={n_int})',
                    cmap='RdBu_r',
                    center=0.0,
                    fmt='.3f',
                )
            except Exception as exc:
                if verbose:
                    print(f"  [SHAP-All] {model_name}: interaction skipped ({exc})")

        if verbose:
            print(f"  [SHAP-All] {model_name}: saved raw SHAP + figures ({backend}).")

    if not raw_arrays:
        raise RuntimeError('No SHAP results were produced for any model.')

    ordered_models = list(raw_arrays.keys())
    _n_mean_feat = len(next(iter(mean_abs_by_model.values())))
    mean_abs_table = pd.DataFrame({
        'Feature': feature_cols[:_n_mean_feat],
        'Feature_Display': display_feature_cols[:_n_mean_feat],
    })
    for model_name in ordered_models:
        mean_abs_table[model_name] = mean_abs_by_model[model_name]
    mean_abs_table.to_csv(tables_dir / 'SHAP_All_MeanAbs.csv', index=False)

    spearman_matrix = pd.DataFrame(np.eye(len(ordered_models)),
                                   index=ordered_models, columns=ordered_models)
    ks_matrix = pd.DataFrame(np.eye(len(ordered_models)),
                             index=ordered_models, columns=ordered_models)
    ks_pvalue_matrix = pd.DataFrame(np.ones((len(ordered_models), len(ordered_models))),
                                    index=ordered_models, columns=ordered_models)

    for i, left_name in enumerate(ordered_models):
        for j in range(i + 1, len(ordered_models)):
            right_name = ordered_models[j]
            left_mean = mean_abs_by_model[left_name]
            right_mean = mean_abs_by_model[right_name]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rho, _ = stats.spearmanr(left_mean, right_mean)
            if rho != rho:
                rho = 1.0 if np.allclose(left_mean, right_mean) else 0.0
            spearman_matrix.iloc[i, j] = spearman_matrix.iloc[j, i] = float(rho)

            left_dist = np.abs(raw_arrays[left_name]).ravel()
            right_dist = np.abs(raw_arrays[right_name]).ravel()
            ks_stat, ks_p = stats.ks_2samp(left_dist, right_dist)
            ks_matrix.iloc[i, j] = ks_matrix.iloc[j, i] = float(ks_stat)
            ks_pvalue_matrix.iloc[i, j] = ks_pvalue_matrix.iloc[j, i] = float(ks_p)

    spearman_matrix.to_csv(tables_dir / 'SHAP_All_Spearman.csv')
    ks_matrix.to_csv(tables_dir / 'SHAP_All_KS_Statistic.csv')
    ks_pvalue_matrix.to_csv(tables_dir / 'SHAP_All_KS_PValue.csv')

    _plot_heatmap(
        spearman_matrix,
        figures_dir / 'Figure_SHAP_All_Spearman_Consistency',
        'SHAP Rank Consistency Across All Models (Spearman rho)',
        cmap='RdYlGn',
        center=0.0,
        fmt='.2f',
    )
    _plot_heatmap(
        ks_matrix,
        figures_dir / 'Figure_SHAP_All_KS_Consistency',
        'KS-SHAP Consistency Across All Models (absolute SHAP distributions)',
        cmap='YlOrRd',
        center=None,
        fmt='.3f',
    )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(tables_dir / 'SHAP_All_Metadata.csv', index=False)

    cache = {
        'raw_arrays': raw_arrays,
        'classwise_arrays': classwise_arrays,
        'x_used_by_model': x_used_by_model,
        'sample_indices_by_model': sample_indices_by_model,
        'background_indices_by_model': background_indices_by_model,
        'base_values_by_model': base_values_by_model,
        'mean_abs_by_model': mean_abs_by_model,
        'mean_abs_table': mean_abs_table,
        'spearman_matrix': spearman_matrix,
        'ks_matrix': ks_matrix,
        'ks_pvalue_matrix': ks_pvalue_matrix,
        'interaction_matrices': interaction_tables,
        'metadata_table': metadata_df,
        'feature_cols': feature_cols,
        'display_feature_cols': display_feature_cols,
        'class_labels': list(class_labels or []),
    }
    joblib.dump(cache, tables_dir / 'SHAP_Cache.pkl')
    if verbose:
        print(f"  [SHAP-All] Cache saved -> {tables_dir / 'SHAP_Cache.pkl'}")
    return cache
