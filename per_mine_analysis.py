
"""Standalone per-mine external-validation analysis.

The script reuses sample-level predictions written by
``main_analysis_pipeline.py`` and does not rerun the main 28-configuration
experiment. It produces mine-specific performance summaries,
class-conditional Kolmogorov-Smirnov diagnostics, exploratory mine-level
shift-performance analyses, class-weight sensitivity results, and figures
under ``<output_dir>/PerMine_Analysis``.

Example:
    python per_mine_analysis.py \
        --output_dir ../Recreated_Model_Output \
        --data_dir ../Input_Data \
        --mine_col auto"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_fonts():
    for path in (r'C:\Windows\Fonts\msyh.ttc',
                 r'C:\Windows\Fonts\Deng.ttf',
                 r'C:\Windows\Fonts\simsun.ttc'):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family'] = font_manager.FontProperties(
                fname=path).get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False


configure_fonts()


ION_COLS = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
PH_COL = 'x8'
CORE_FEATURE_COLS = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
FEATURE_DISPLAY = {'x1': 'K+', 'x2': 'Na+', 'x3': 'Ca2+', 'x4': 'Mg2+',
                   'x5': 'Cl-', 'x6': 'SO42-', 'x7': 'HCO3-', 'x8': 'pH'}

CANONICAL_ORDER = [
    'Ordovician limestone (O)',
    'Goaf water (G)',
    'Taiyuan limestone (T)',
    'Permian sandstone fissure (P)',
]
PATTERNS = {
    'Ordovician limestone (O)': ('ordovician', '\u5965\u7070', '\u5965\u9676'),
    'Goaf water (G)': ('goaf', '\u8001\u7a7a', '\u91c7\u7a7a'),
    'Taiyuan limestone (T)': ('taiyuan', '\u592a\u7070', '\u592a\u539f'),
    'Permian sandstone fissure (P)': ('permian', 'sandstone', '\u7802\u5ca9', '\u4e8c\u53e0'),
}
EXACT_ALIASES = {
    '2': 0, '2.0': 0, 'o': 0, '3': 1, '3.0': 1, 'g': 1,
    '4': 2, '4.0': 2, 't': 2, '5': 3, '5.0': 3, 'p': 3,
}
MINE_COL_CANDIDATES = ['mine', 'mine_id', 'mine_name', 'Mine', 'Mine_ID',
                       'MineName', '\u77ff\u4e95\u540d\u79f0', '\u77ff\u4e95', '\u77ff\u540d', '\u7164\u77ff', '\u77ff\u533a', '\u77ff']

RF_DEFAULT_PARAMS = {'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 8,
                     'min_samples_leaf': 4, 'min_samples_split': 6,
                     'max_samples': 0.8}


def canonicalize(label):
    s = str(label).strip()
    low = s.lower()
    if low in EXACT_ALIASES:
        return CANONICAL_ORDER[EXACT_ALIASES[low]]
    for name, pats in PATTERNS.items():
        if any(p in low for p in pats):
            return name
    return s


def resolve_target_column(df, preferred='y'):
    if preferred in df.columns:
        return preferred
    for cand in ('y', '\u5145\u6c34\u6c34\u6e90', 'label', 'target', 'class'):
        if cand in df.columns:
            return cand
    return df.columns[-1]


def _confusion(y_true, y_pred, k=4):
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def macro_f1_present(y_true, y_pred, k=4):
    """macro-F1 over classes PRESENT in y_true (present-class macro-F1)."""
    cm = _confusion(y_true, y_pred, k)
    present = np.where(cm.sum(axis=1) > 0)[0]
    f1s = []
    for c in present:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) if f1s else np.nan


def mcc_fast(y_true, y_pred, k=4):
    cm = _confusion(y_true, y_pred, k).astype(float)
    t = cm.sum(axis=1); p = cm.sum(axis=0); c = np.trace(cm); s = cm.sum()
    num = c * s - (t * p).sum()
    den = np.sqrt((s**2 - (p**2).sum()) * (s**2 - (t**2).sum()))
    return float(num / den) if den > 0 else 0.0


def balanced_acc(y_true, y_pred, k=4):
    cm = _confusion(y_true, y_pred, k)
    rows = cm.sum(axis=1)
    present = rows > 0
    recalls = np.diag(cm)[present] / rows[present]
    return float(recalls.mean()) if present.any() else np.nan


def per_class_recall(y_true, y_pred, k=4):
    cm = _confusion(y_true, y_pred, k)
    rows = cm.sum(axis=1)
    out = {}
    for c in range(k):
        out[c] = float(cm[c, c] / rows[c]) if rows[c] > 0 else np.nan
    return out


def nanmean_quiet(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def load_external_with_mines(external_path, mine_col='auto', mine_map_file=None):
    """Replicate the pipeline's external-set loading; return
    (X_raw [n,8], y_enc [n], mines [n], df_used)."""
    external_path = Path(external_path)
    if not external_path.exists():
        sys.exit(f'[error] external validation file not found: {external_path}')
    df = pd.read_excel(external_path)
    feature_input = ION_COLS + [PH_COL]
    expected = feature_input + ['y']
    first_col = str(df.columns[0])
    if (first_col.replace('.', '', 1).replace('-', '', 1).isdigit()
            or ION_COLS[0] not in df.columns):
        if df.shape[1] == len(expected):
            df = pd.read_excel(external_path, header=None, names=expected)
        else:
            sys.exit('[error] cannot parse external columns: x1..x8 headers are absent and '
                     'the workbook does not contain exactly nine columns. Provide a valid file.')
    tcol = resolve_target_column(df, 'y')


    mines = None
    if mine_col != 'none':
        if mine_col not in ('auto', None) and mine_col in df.columns:
            mines = df[mine_col].astype(str).str.strip().values
        elif mine_col in ('auto', None):
            for c in MINE_COL_CANDIDATES:
                if c in df.columns:
                    mines = df[c].astype(str).str.strip().values
                    print(f'[info] detected mine column: "{c}"')
                    break
    if mines is None and mine_map_file:
        mp = Path(mine_map_file)
        mdf = pd.read_excel(mp) if mp.suffix.lower() in ('.xlsx', '.xls') \
            else pd.read_csv(mp)
        col = mdf.columns[-1]
        if len(mdf) != len(df):
            sys.exit(f'[error] mine-map rows ({len(mdf)}) do not match external rows '
                     f'({len(df)}); row-wise alignment is impossible.')
        mines = mdf[col].astype(str).str.strip().values
        print(f'[info] loaded mine identifiers from mapping column: "{col}"')
    if mines is None:
        sys.exit('[error] no mine identifier column was found. Specify it with --mine_col '
                 'or provide a row-aligned mapping file with --mine_map_file.')

    for col in feature_input:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    X_raw = df[CORE_FEATURE_COLS].values.astype(float)
    y_names = np.array([canonicalize(v) for v in df[tcol].values], dtype=object)

    mask = np.isin(y_names, CANONICAL_ORDER)
    n_drop = int((~mask).sum())
    if n_drop:
        print(f'[info] removed {n_drop} samples with unknown labels, matching the main pipeline.')
    X_raw, y_names, mines = X_raw[mask], y_names[mask], mines[mask]
    y_enc = np.array([CANONICAL_ORDER.index(v) for v in y_names], dtype=int)
    return X_raw, y_enc, mines, df.loc[mask].reset_index(drop=True)


def load_malan(data_dir, train_domain='train_only'):
    data_dir = Path(data_dir)
    frames = []
    for name in (['train_set.xlsx'] if train_domain == 'train_only'
                 else ['train_set.xlsx', 'test_set.xlsx']):
        p = data_dir / name
        if not p.exists():
            sys.exit(f'[error] file not found: {p}')
        frames.append(pd.read_excel(p))
    parts = []
    for f in frames:
        tcol = resolve_target_column(f, 'y')
        for col in CORE_FEATURE_COLS:
            f[col] = pd.to_numeric(f[col], errors='coerce').astype(float)
        y = np.array([canonicalize(v) for v in f[tcol].values], dtype=object)
        m = np.isin(y, CANONICAL_ORDER)
        parts.append((f.loc[m, CORE_FEATURE_COLS].values.astype(float),
                      np.array([CANONICAL_ORDER.index(v) for v in y[m]])))
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return X, y


def load_predictions(output_dir):
    p = Path(output_dir) / 'Tables' / 'FinalEval_Test_External_Predictions.csv'
    if not p.exists():
        sys.exit(f'[error] main-pipeline prediction file not found: {p}\n'
                 'Confirm that --output_dir points to the main-pipeline output directory.')
    df = pd.read_csv(p)
    ext = df[df['Dataset'] == 'external_val'].copy()
    if ext.empty:
        sys.exit('[error] the prediction file contains no external_val records.')
    return ext


def sanity_check_alignment(pred_ext, y_enc):
    """Verify sample counts, class counts, and row-level label alignment."""
    n = len(y_enc)
    ref = np.bincount(y_enc, minlength=4)
    g0 = pred_ext.groupby(['Model', 'Run_Seed'])
    sizes = g0.size().unique()
    if not (len(sizes) == 1 and sizes[0] == n):
        sys.exit(f'[error] alignment failed: prediction-group sizes {sorted(sizes.tolist())} '
                 f'!= reconstructed external n={n}. Verify that the external file matches the pipeline run.')
    one = g0.get_group(next(iter(g0.groups))).sort_values('Sample_Index')
    got = np.bincount(one['True_Label'].to_numpy(int), minlength=4)
    if not np.array_equal(got, ref):
        sys.exit(f'[error] alignment failed: prediction labels {got.tolist()} differ from '
                 f'reconstructed labels {ref.tolist()}.')
    if not np.array_equal(one['True_Label'].to_numpy(int)[np.argsort(
            one['Sample_Index'].to_numpy(int))], y_enc):
        sys.exit('[error] alignment failed: row-level true-label sequences differ.')
    print(f'[ok] alignment verified: n={n}, O/G/T/P counts={ref.tolist()}')


def per_mine_performance(pred_ext, y_enc, mines, out_tables, out_figs,
                         rep_models, n_boot=500, rng=None):
    rng = rng or np.random.default_rng(20260718)
    mine_names = pd.unique(mines)
    idx_by_mine = {m: np.where(mines == m)[0] for m in mine_names}


    comp_rows = []
    for m in mine_names:
        yy = y_enc[idx_by_mine[m]]
        cnt = np.bincount(yy, minlength=4)
        comp_rows.append({'Mine': m, 'N': len(yy),
                          'n_O': cnt[0], 'n_G': cnt[1],
                          'n_T': cnt[2], 'n_P': cnt[3],
                          'N_Classes_Present': int((cnt > 0).sum()),
                          'Small_Classes(<5)': ','.join(
                              s for s, c in zip('OGTP', cnt) if 0 < c < 5)})
    comp = pd.DataFrame(comp_rows).sort_values('N', ascending=False)
    comp.to_csv(out_tables / 'PerMine_Composition.csv', index=False,
                encoding='utf-8-sig')


    all_rows, main_rows = [], []
    for (model, seed), g in pred_ext.groupby(['Model', 'Run_Seed']):
        g = g.sort_values('Sample_Index')
        yp = g['Pred_Label'].to_numpy(int)
        for m in mine_names:
            ii = idx_by_mine[m]
            yt, yq = y_enc[ii], yp[ii]
            all_rows.append({
                'Model': model, 'Run_Seed': seed, 'Mine': m, 'N': len(ii),
                'Accuracy': float((yt == yq).mean()),
                'Balanced_Accuracy': balanced_acc(yt, yq),
                'MCC': mcc_fast(yt, yq),
                'MacroF1_Present': macro_f1_present(yt, yq),
            })
    per_seed = pd.DataFrame(all_rows)
    agg = (per_seed.groupby(['Model', 'Mine'])
           .agg(N=('N', 'first'),
                Accuracy_mean=('Accuracy', 'mean'), Accuracy_sd=('Accuracy', 'std'),
                BalAcc_mean=('Balanced_Accuracy', 'mean'),
                BalAcc_sd=('Balanced_Accuracy', 'std'),
                MCC_mean=('MCC', 'mean'), MCC_sd=('MCC', 'std'),
                MacroF1p_mean=('MacroF1_Present', 'mean'),
                MacroF1p_sd=('MacroF1_Present', 'std'))
           .reset_index())
    agg.to_csv(out_tables / 'PerMine_Performance_AllModels.csv', index=False,
               encoding='utf-8-sig')


    preds_by_ms = {k: g.sort_values('Sample_Index')['Pred_Label'].to_numpy(int)
                   for k, g in pred_ext.groupby(['Model', 'Run_Seed'])}
    seeds_by_model = {}
    for (mo, se) in preds_by_ms:
        seeds_by_model.setdefault(mo, []).append(se)
    for model in rep_models:
        if model not in seeds_by_model:
            print(f'[warning] representative model {model} is absent from predictions and was skipped.')
            continue
        seeds = seeds_by_model[model]
        for m in mine_names:
            ii = idx_by_mine[m]
            yt = y_enc[ii]

            stat = []
            cls_idx = {c: ii[yt == c] for c in np.unique(yt)}
            for _ in range(n_boot):
                se = seeds[rng.integers(len(seeds))]
                yp = preds_by_ms[(model, se)]
                bi = np.concatenate([rng.choice(v, size=len(v), replace=True)
                                     for v in cls_idx.values()])
                stat.append(macro_f1_present(y_enc[bi], yp[bi]))
            lo, hi = np.percentile(stat, [2.5, 97.5])

            recs = {c: [] for c in range(4)}
            err_counter = {}
            for se in seeds:
                yp = preds_by_ms[(model, se)][ii]
                r = per_class_recall(yt, yp)
                for c in range(4):
                    recs[c].append(r[c])
                wrong = yt != yp
                for a, b in zip(yt[wrong], yp[wrong]):
                    err_counter[(a, b)] = err_counter.get((a, b), 0) + 1
            top_err = max(err_counter, key=err_counter.get) if err_counter else None
            lab = 'OGTP'
            sub = per_seed[(per_seed.Model == model) & (per_seed.Mine == m)]
            main_rows.append({
                'Model': model, 'Mine': m, 'N': len(ii),
                'Classes_Present': ''.join(lab[c] for c in np.unique(yt)),
                'MacroF1_Present_mean': sub['MacroF1_Present'].mean(),
                'MacroF1_Present_boot95_low': lo,
                'MacroF1_Present_boot95_high': hi,
                'MCC_mean': sub['MCC'].mean(),
                'BalancedAcc_mean': sub['Balanced_Accuracy'].mean(),
                'Recall_O': nanmean_quiet(recs[0]),
                'Recall_G': nanmean_quiet(recs[1]),
                'Recall_T': nanmean_quiet(recs[2]),
                'Recall_P': nanmean_quiet(recs[3]),
                'Main_Confusion': (f'{lab[top_err[0]]}->{lab[top_err[1]]}'
                                   if top_err else ''),
                'Descriptive_Only(N<15)': len(ii) < 15,
            })
    main = pd.DataFrame(main_rows)
    main.to_csv(out_tables / 'PerMine_Performance_Main.csv', index=False,
                encoding='utf-8-sig')


    hm = agg[agg.Model.isin(rep_models)].pivot(index='Mine', columns='Model',
                                               values='MCC_mean')
    hm = hm.loc[comp.set_index('Mine').index.intersection(hm.index)]
    fig, ax = plt.subplots(figsize=(1.9 + 1.4 * len(rep_models),
                                    0.55 * len(hm) + 1.6))
    im = ax.imshow(hm.values, cmap='RdYlGn', vmin=-0.1, vmax=0.9, aspect='auto')
    ax.set_xticks(range(hm.shape[1]))
    ax.set_xticklabels(hm.columns, rotation=30, ha='right', fontsize=8)
    nn = comp.set_index('Mine')['N']
    ax.set_yticks(range(hm.shape[0]))
    ax.set_yticklabels([f'{m} (n={nn[m]})' for m in hm.index], fontsize=8)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
    ax.set_title('Per-mine MCC (mean over 30 stochastic fits)', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label='MCC')
    fig.tight_layout()
    fig.savefig(out_figs / 'Fig_PerMine_Heatmap.png', dpi=600)
    plt.close(fig)
    return comp, per_seed, main


def ks_of(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(sps.ks_2samp(a, b).statistic)


def conditional_ks(X_malan, y_malan, X_ext, y_ext, out_tables, out_figs,
                   n_boot=1000, rng=None):
    rng = rng or np.random.default_rng(20260719)
    rows = []
    for j, feat in enumerate(CORE_FEATURE_COLS):
        row = {'Feature': feat, 'Feature_Name': FEATURE_DISPLAY[feat],
               'KS_pooled': ks_of(X_malan[:, j], X_ext[:, j])}
        cond = []
        for c, sh in enumerate('OGTP'):
            a = X_malan[y_malan == c, j]
            b = X_ext[y_ext == c, j]
            ks = ks_of(a, b)
            row[f'KS_{sh}'] = ks
            row[f'n_Malan_{sh}'] = int(np.isfinite(a).sum())
            row[f'n_Ext_{sh}'] = int(np.isfinite(b).sum())
            cond.append(ks)

            if np.isfinite(ks):
                bs = []
                for _ in range(n_boot):
                    aa = rng.choice(a, len(a), replace=True)
                    bb = rng.choice(b, len(b), replace=True)
                    bs.append(ks_of(aa, bb))
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                row[f'KS_{sh}_boot95'] = f'[{lo:.3f}, {hi:.3f}]'
        row['KS_equal_class_mean'] = float(np.nanmean(cond))
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out_tables / 'ConditionalKS.csv', index=False,
               encoding='utf-8-sig')

    cols = ['KS_pooled', 'KS_O', 'KS_G', 'KS_T', 'KS_P', 'KS_equal_class_mean']
    mat = tab.set_index('Feature_Name')[cols]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.imshow(mat.values, cmap='YlOrRd', vmin=0,
                   vmax=np.nanmax(mat.values))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(['Pooled', 'O', 'G', 'T', 'P', 'Equal-class\nmean'],
                       fontsize=8)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
    ax.set_title('KS statistics: pooled vs class-conditional '
                 '(Malan domain vs external domain)', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85, label='KS')
    fig.tight_layout()
    fig.savefig(out_figs / 'Fig_ConditionalKS_Heatmap.png', dpi=600)
    plt.close(fig)
    return tab


def per_mine_ks_association(X_malan, X_ext, mines, per_seed, primary_model,
                            out_tables, out_figs):
    mine_names = pd.unique(mines)
    rows = []
    for m in mine_names:
        ii = np.where(mines == m)[0]
        row = {'Mine': m, 'N': len(ii)}
        kss = []
        for j, feat in enumerate(CORE_FEATURE_COLS):
            ks = ks_of(X_malan[:, j], X_ext[ii, j])
            row[f'KS_{FEATURE_DISPLAY[feat]}'] = ks
            kss.append(ks)
        row['KS_mean'] = float(np.nanmean(kss))
        row['KS_median'] = float(np.nanmedian(kss))
        rows.append(row)
    ks_tab = pd.DataFrame(rows)
    ks_tab.to_csv(out_tables / 'PerMine_KS.csv', index=False,
                  encoding='utf-8-sig')

    perf = (per_seed[per_seed.Model == primary_model]
            .groupby('Mine')
            .agg(MCC=('MCC', 'mean'),
                 MacroF1p=('MacroF1_Present', 'mean')).reset_index())
    j = ks_tab.merge(perf, on='Mine')
    rho, pval = sps.spearmanr(j['KS_mean'], j['MCC'])
    j['Spearman_rho_KSmean_vs_MCC'] = rho
    j['Spearman_p'] = pval
    j.to_csv(out_tables / 'KS_vs_Performance.csv', index=False,
             encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(j['KS_mean'], j['MCC'], s=np.sqrt(j['N']) * 6, alpha=0.75)
    for _, r in j.iterrows():
        ax.annotate(str(r['Mine']), (r['KS_mean'], r['MCC']), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel(f'Mean KS vs Malan domain (8 features)', fontsize=9)
    ax.set_ylabel(f'MCC ({primary_model}, mean of 30 fits)', fontsize=9)
    ax.set_title(f'Exploratory mine-level association '
                 f'(Spearman rho={rho:.2f}, p={pval:.3f}, n={len(j)} mines)',
                 fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_figs / 'Fig_KS_vs_MCC_Scatter.png', dpi=600)
    plt.close(fig)
    print(f'[result] Spearman association between mean per-mine KS and {primary_model} MCC: '
          f'rho={rho:.3f}, p={pval:.3f}; interpret as exploratory, not causal.')
    return ks_tab, j


def weight_sensitivity(data_dir, X_ext_raw, y_ext, xgb_params_by_seed,
                       out_tables, seeds):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print('[warning] xgboost is unavailable; weight sensitivity will run for RF only.')
        XGBClassifier = None

    data_dir = Path(data_dir)
    tr = pd.read_excel(data_dir / 'train_set.xlsx')
    te = pd.read_excel(data_dir / 'test_set.xlsx')

    def prep(df):
        tcol = resolve_target_column(df, 'y')
        for c in CORE_FEATURE_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
        y = np.array([canonicalize(v) for v in df[tcol].values], dtype=object)
        m = np.isin(y, CANONICAL_ORDER)
        return (df.loc[m, CORE_FEATURE_COLS].values.astype(float),
                np.array([CANONICAL_ORDER.index(v) for v in y[m]]))

    Xtr, ytr = prep(tr)
    Xte, yte = prep(te)

    def bal_w(y):
        cls = np.unique(y)
        w = compute_class_weight('balanced', classes=cls, y=y)
        lut = dict(zip(cls, w))
        return np.array([lut[v] for v in y], dtype=float)

    def fit_xgb_two_stage(params, seed, Xa, ya, weighted):
        """Reproduce the two-stage early-stopping and full-refit protocol."""
        p = {k: v for k, v in params.items() if not str(k).startswith('__')}
        sw = bal_w(ya) if weighted else None
        try:
            Xs, Xe, ys, ye_, idx_s, _ = train_test_split(
                Xa, ya, np.arange(len(ya)), test_size=0.15,
                stratify=ya, random_state=seed)
            m1 = XGBClassifier(random_state=seed, verbosity=0,
                               eval_metric='mlogloss',
                               early_stopping_rounds=30, **_xgbp(p))
            m1.fit(Xs, ys, sample_weight=None if sw is None else sw[idx_s],
                   eval_set=[(Xe, ye_)], verbose=False)
            best_n = int(getattr(m1, 'best_iteration', 0) or 0) + 1
            best_n = max(best_n, 10)
        except Exception:
            best_n = int(p.get('n_estimators', 300))
        p2 = _xgbp(p); p2['n_estimators'] = best_n
        m = XGBClassifier(random_state=seed, verbosity=0,
                          eval_metric='mlogloss', **p2)
        m.fit(Xa, ya, sample_weight=sw)
        return m

    def _xgbp(p):
        out = {
            'n_estimators': int(p.get('n_estimators', 300)),
            'max_depth': int(p.get('max_depth', 4)),
            'learning_rate': float(p.get('learning_rate', 0.1)),
            'subsample': float(p.get('subsample', 0.8)),
            'colsample_bytree': float(p.get('colsample_bytree', 0.8)),
            'reg_alpha': float(p.get('reg_alpha',
                               10 ** float(p.get('reg_alpha_log', -1)))),
            'reg_lambda': float(p.get('reg_lambda',
                                10 ** float(p.get('reg_lambda_log', 0)))),
            'min_child_weight': int(p.get('min_child_weight', 5)),
            'gamma': float(p.get('gamma', 0.1)),
            'base_score': 0.5, 'n_jobs': 1,
        }
        return out

    rows, cls_rows = [], []
    lab = 'OGTP'
    for seed in seeds:
        imp = SimpleImputer(strategy='median').fit(Xtr)
        Xa, Xb, Xc = imp.transform(Xtr), imp.transform(Xte), imp.transform(X_ext_raw)
        jobs = [('RF-Default', 'balanced'), ('RF-Default', 'uniform')]
        if XGBClassifier is not None and seed in xgb_params_by_seed:
            jobs += [('XGB-TrainCV', 'balanced'), ('XGB-TrainCV', 'uniform')]
        for model_name, wmode in jobs:
            if model_name == 'RF-Default':
                m = RandomForestClassifier(
                    class_weight='balanced' if wmode == 'balanced' else None,
                    random_state=seed, n_jobs=1,
                    **{k: RF_DEFAULT_PARAMS[k] for k in RF_DEFAULT_PARAMS})
                m.fit(Xa, ytr)
            else:
                m = fit_xgb_two_stage(xgb_params_by_seed[seed], seed, Xa, ytr,
                                      weighted=(wmode == 'balanced'))
            for ds_name, Xd, yd in (('internal_test', Xb, yte),
                                    ('external', Xc, y_ext)):
                yp = m.predict(Xd)
                rows.append({'Model': model_name, 'Weighting': wmode,
                             'Run_Seed': seed, 'Dataset': ds_name,
                             'Accuracy': float((yd == yp).mean()),
                             'Balanced_Accuracy': balanced_acc(yd, yp),
                             'MCC': mcc_fast(yd, yp),
                             'MacroF1': macro_f1_present(yd, yp)})
                rec = per_class_recall(yd, yp)
                cls_rows.append({'Model': model_name, 'Weighting': wmode,
                                 'Run_Seed': seed, 'Dataset': ds_name,
                                 **{f'Recall_{lab[c]}': rec[c]
                                    for c in range(4)}})
    res = pd.DataFrame(rows)
    metric_cols = ['Accuracy', 'Balanced_Accuracy', 'MCC', 'MacroF1']
    summ = (res.groupby(['Model', 'Weighting', 'Dataset'])[metric_cols]
            .agg(['mean', 'std']).round(4).reset_index())
    summ.columns = [
        '_'.join(str(part) for part in col if part)
        if isinstance(col, tuple) else col
        for col in summ.columns
    ]
    summ.to_csv(out_tables / 'WeightSensitivity.csv', index=False,
                encoding='utf-8-sig')
    pc = pd.DataFrame(cls_rows)
    recall_cols = [c for c in pc.columns if c.startswith('Recall_')]
    pc_s = (pc.groupby(['Model', 'Weighting', 'Dataset'])[recall_cols]
            .agg(['mean', 'std']).round(4).reset_index())
    pc_s.columns = [
        '_'.join(str(part) for part in col if part)
        if isinstance(col, tuple) else col
        for col in pc_s.columns
    ]
    pc_s.to_csv(out_tables / 'WeightSensitivity_PerClass.csv', index=False,
                encoding='utf-8-sig')
    print('[done] weight sensitivity completed: balanced reproduces the pipeline; uniform removes weighting.')
    return res, pc


def choose_representative_models(output_dir):
    """Select the RF baseline, the training-CV-leading XGBoost model, and a retrospective external reference."""
    p = Path(output_dir) / 'Tables' / 'FinalEval_Test_External_Raw.csv'
    xgb_best, xgb_params_by_seed = None, {}
    if p.exists():
        raw = pd.read_csv(p)
        xg = raw[raw['Algorithm'] == 'XGBoost']
        if not xg.empty and 'Train_Internal_CV_F1' in xg.columns:
            mean_cv = xg.groupby('Model')['Train_Internal_CV_F1'].mean()
            xgb_best = mean_cv.idxmax()
            print(f'[info] training-CV-selected XGBoost configuration: {xgb_best} '
                  f'(mean training-CV F1={mean_cv.max():.4f}; no external information used)')
            sub = xg[xg.Model == xgb_best]
            if 'Best_Params_JSON' in sub.columns:
                for _, r in sub.iterrows():
                    try:
                        xgb_params_by_seed[int(r['Run_Seed'])] = json.loads(
                            r['Best_Params_JSON'])
                    except Exception:
                        pass
    reps = ['RF-Default']
    if xgb_best:
        reps.append(xgb_best)
    if 'XGBoost-SSA' not in reps:
        reps.append('XGBoost-SSA')
    return reps, xgb_best, xgb_params_by_seed


README = """Per-mine analysis outputs
=========================

PerMine_Composition.csv
    Mine-level sample counts, O/G/T/P composition, represented classes, and
    flags for classes with fewer than five samples.

PerMine_Performance_AllModels.csv
    Mine-level metrics for all 28 configurations, summarized across repeated
    locked evaluations.

PerMine_Performance_Main.csv
    Representative-model results by mine, including present-class macro-F1,
    bootstrap intervals, MCC, balanced accuracy, class recall, and the dominant
    confusion direction. Present-class macro-F1 averages only classes observed
    in that mine, so mines with different class coverage should not be ranked
    directly. Results for mines with N < 15 are descriptive.

ConditionalKS.csv
    Pooled and class-conditional KS statistics using the Malan training domain
    as the default reference. Small O and P reference samples make their
    class-conditional estimates exploratory.

PerMine_KS.csv
    Feature-level KS statistics and their mean and median for each mine.

KS_vs_Performance.csv
    Exploratory Spearman association between mean mine-level KS and the
    selected model's MCC. This analysis is descriptive and not causal.

WeightSensitivity.csv
WeightSensitivity_PerClass.csv
    Comparison of balanced weighting with an unweighted control, reported
    overall and by source class.

Figures/
    Per-mine performance, class-conditional KS, and shift-performance figures.
"""


def main():
    ap = argparse.ArgumentParser(description='Standalone per-mine analysis')
    ap.add_argument('--output_dir', default='../Recreated_Model_Output')
    ap.add_argument('--data_dir', default='../Input_Data')
    ap.add_argument('--external_file', default=None,
                    help='Search the data directory using the main-pipeline filename candidates.')
    ap.add_argument('--mine_col', default='auto')
    ap.add_argument('--mine_map_file', default=None)
    ap.add_argument('--train_domain', choices=['all', 'train_only'],
                    default='train_only', help='Malan KS reference: 153 training samples by default; use all for 192 samples.')
    ap.add_argument('--n_boot', type=int, default=500)
    ap.add_argument('--skip_weight_sensitivity', action='store_true')
    args = ap.parse_args()

    out_root = Path(args.output_dir) / 'PerMine_Analysis'
    out_tables = out_root / 'Tables'
    out_figs = out_root / 'Figures'
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)


    ext_path = args.external_file
    if ext_path is None:
        for cand in ('external_validation_set.xlsx', 'val_set.xlsx',
                     'validation_set.xlsx', '\u9a8c\u8bc1\u96c6.xlsx'):
            p = Path(args.data_dir) / cand
            if p.exists():
                ext_path = p
                break
    if ext_path is None:
        sys.exit('[error] external validation file not found; specify it with --external_file.')
    print(f'[info] external validation file: {ext_path}')

    X_ext, y_ext, mines, _ = load_external_with_mines(
        ext_path, args.mine_col, args.mine_map_file)
    print(f'[info] mines: {len(pd.unique(mines))}; external samples: {len(y_ext)}')

    pred_ext = load_predictions(args.output_dir)
    sanity_check_alignment(pred_ext, y_ext)

    reps, xgb_best, xgb_params_by_seed = choose_representative_models(
        args.output_dir)
    print(f'[info] representative models: {reps}')

    comp, per_seed, _ = per_mine_performance(
        pred_ext, y_ext, mines, out_tables, out_figs, reps,
        n_boot=args.n_boot)

    X_malan, y_malan = load_malan(args.data_dir, args.train_domain)
    conditional_ks(X_malan, y_malan, X_ext, y_ext, out_tables, out_figs)

    available = set(per_seed['Model'].unique())
    primary = xgb_best if (xgb_best in available) else 'RF-Default'
    per_mine_ks_association(X_malan, X_ext, mines, per_seed, primary,
                            out_tables, out_figs)

    if not args.skip_weight_sensitivity:
        seeds = sorted(pred_ext['Run_Seed'].dropna().astype(int).unique())
        weight_sensitivity(args.data_dir, X_ext, y_ext,
                           xgb_params_by_seed, out_tables, seeds)

    (out_root / 'README_PerMine.txt').write_text(README, encoding='utf-8')
    print(f'\n[done] results written to: {out_root}')


if __name__ == '__main__':
    main()
