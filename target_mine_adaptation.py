
"""Target-mine adaptation experiment for cross-mine mine-water source identification.

The script compares a locked source-domain baseline with supervised target-mine
adaptation using non-overlapping local adaptation and evaluation subsets. It
writes per-run results, summary tables, adaptation-size sweeps, and a figure
under ``<output_dir>/LocalCalibration``.

Example:
    python target_mine_adaptation.py \
        --data_dir ../Input_Data \
        --output_dir ../Recreated_Model_Output \
        --mines auto \
        --model RF-Default \
        --n_repeats 30 \
        --calib_frac 0.5 \
        --seed 20240101"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CORE_FEATURE_COLS = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
CANONICAL_ORDER = [
    'Ordovician limestone (O)',
    'Goaf water (G)',
    'Taiyuan limestone (T)',
    'Permian sandstone fissure (P)',
]
PATTERNS = {
    0: ('ordovician', '\u5965\u7070', '\u5965\u9676'),
    1: ('goaf', '\u8001\u7a7a', '\u91c7\u7a7a'),
    2: ('taiyuan', '\u592a\u7070', '\u592a\u539f'),
    3: ('permian', 'sandstone', '\u7802\u5ca9', '\u4e8c\u53e0'),
}
EXACT_ALIASES = {'2': 0, '2.0': 0, 'o': 0, '3': 1, '3.0': 1, 'g': 1,
                 '4': 2, '4.0': 2, 't': 2, '5': 3, '5.0': 3, 'p': 3}
MINE_COL_CANDIDATES = ['mine', 'mine_id', 'mine_name', 'Mine', 'Mine_ID',
                       'MineName', '\u77ff\u4e95\u540d\u79f0', '\u77ff\u4e95', '\u77ff\u540d', '\u7164\u77ff', '\u77ff\u533a', '\u77ff']
ROMAN = {'\u897f\u66f2': 'Xiqu', '\u5c6f\u5170': 'Tunlan', '\u897f\u94ed': 'Ximing', '\u4e1c\u66f2\u77ff': 'Dongqu',
         '\u9547\u57ce\u5e95': 'Zhenchengdi', '\u4e49\u57ce\u77ff': 'Yicheng', '\u675c\u513f\u576a': "Du'erping",
         '\u5b98\u5730\u77ff': 'Guandi', '\u798f\u660c': 'Fuchang', '\u5357\u5cad\u77ff': 'Nanling',
         '\u4e16\u7eaa\u91d1\u946b': 'Shijijinxin', '\u94c2\u9f99': 'Bolong'}


RF_DEFAULT_PARAMS = dict(n_estimators=200, max_features='sqrt', max_depth=8,
                         min_samples_leaf=4, min_samples_split=6, max_samples=0.8)

XGB_DEFAULT_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                          reg_lambda=1.0, min_child_weight=5, gamma=0.1)


def encode_label(label):
    s = str(label).strip()
    low = s.lower()
    if low in EXACT_ALIASES:
        return EXACT_ALIASES[low]
    for cls, pats in PATTERNS.items():
        if any(p in low for p in pats):
            return cls
    for i, name in enumerate(CANONICAL_ORDER):
        if s == name:
            return i
    return -1


def _resolve_target_col(df):
    for cand in ('y', '\u5145\u6c34\u6c34\u6e90', '\u6c34\u6e90', '\u7c7b\u522b', '\u6807\u7b7e', 'label', 'target', 'class'):
        if cand in df.columns:
            return cand
    return df.columns[-1]


def load_xy(path):
    """Load a labeled workbook and return filtered feature and label arrays."""
    path = Path(path)
    if not path.exists():
        sys.exit(f'[error] file not found: {path}')
    df = pd.read_excel(path)
    expected = CORE_FEATURE_COLS + ['y']
    first_col = str(df.columns[0])
    if (first_col.replace('.', '', 1).replace('-', '', 1).isdigit()
            or CORE_FEATURE_COLS[0] not in df.columns):
        if df.shape[1] == len(expected):
            df = pd.read_excel(path, header=None, names=expected)
        else:
            sys.exit(f'[error] cannot parse columns of {path.name}')
    tcol = _resolve_target_col(df)
    X = df[CORE_FEATURE_COLS].apply(pd.to_numeric, errors='coerce').values.astype(float)
    y = np.array([encode_label(v) for v in df[tcol].values])
    keep = y >= 0
    return X[keep], y[keep]


def load_external_with_mines(path, mine_col='auto'):
    """Load the external dataset and return features, labels, and mine identifiers."""
    path = Path(path)
    if not path.exists():
        sys.exit(f'[error] external file not found: {path}')
    df = pd.read_excel(path)
    expected = CORE_FEATURE_COLS + ['y']
    first_col = str(df.columns[0])
    if (first_col.replace('.', '', 1).replace('-', '', 1).isdigit()
            or CORE_FEATURE_COLS[0] not in df.columns):
        if df.shape[1] >= len(expected):
            names = expected + [f'extra{i}' for i in range(df.shape[1] - len(expected))]
            df = pd.read_excel(path, header=None, names=names)

    mines = None
    if mine_col not in ('auto', None) and mine_col in df.columns:
        mines = df[mine_col].astype(str).str.strip().values
    else:
        for c in MINE_COL_CANDIDATES:
            if c in df.columns:
                mines = df[c].astype(str).str.strip().values
                break
    if mines is None:
        sys.exit('[error] no mine column found; pass --mine_col explicitly '
                 f'(available columns: {list(df.columns)})')
    tcol = _resolve_target_col(df)
    X = df[CORE_FEATURE_COLS].apply(pd.to_numeric, errors='coerce').values.astype(float)
    y = np.array([encode_label(v) for v in df[tcol].values])
    keep = y >= 0
    return X[keep], y[keep], mines[keep]


def _confusion(y_true, y_pred, k=4):
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def macro_f1_present(y_true, y_pred, k=4):
    cm = _confusion(y_true, y_pred, k)
    present = np.where(cm.sum(axis=1) > 0)[0]
    f1s = []
    for c in present:
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) if f1s else np.nan


def mcc_fast(y_true, y_pred, k=4):
    cm = _confusion(y_true, y_pred, k).astype(float)
    t = cm.sum(1); p = cm.sum(0); c = np.trace(cm); s = cm.sum()
    num = c * s - (t * p).sum()
    den = np.sqrt((s**2 - (p**2).sum()) * (s**2 - (t**2).sum()))
    return float(num / den) if den > 0 else 0.0


def balanced_acc(y_true, y_pred, k=4):
    cm = _confusion(y_true, y_pred, k)
    rows = cm.sum(1); present = rows > 0
    return float((np.diag(cm)[present] / rows[present]).mean()) if present.any() else np.nan


def build_model(name, seed):
    if name == 'RF-Default':
        return RandomForestClassifier(
            n_estimators=RF_DEFAULT_PARAMS['n_estimators'],
            max_depth=RF_DEFAULT_PARAMS['max_depth'],
            min_samples_split=RF_DEFAULT_PARAMS['min_samples_split'],
            min_samples_leaf=RF_DEFAULT_PARAMS['min_samples_leaf'],
            max_samples=RF_DEFAULT_PARAMS['max_samples'],
            max_features=RF_DEFAULT_PARAMS['max_features'],
            class_weight=None, random_state=seed, n_jobs=1)
    elif name == 'XGBoost-Default':
        from xgboost import XGBClassifier
        p = XGB_DEFAULT_PARAMS
        return XGBClassifier(
            n_estimators=p['n_estimators'], max_depth=p['max_depth'],
            learning_rate=p['learning_rate'], subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'], reg_alpha=p['reg_alpha'],
            reg_lambda=p['reg_lambda'], min_child_weight=p['min_child_weight'],
            gamma=p['gamma'], base_score=0.5, eval_metric='mlogloss',
            random_state=seed, n_jobs=1, verbosity=0)
    else:
        sys.exit(f'[error] unknown --model {name!r} '
                 '(use RF-Default or XGBoost-Default; see NOTE for XGBoost-SSA)')


def fit_predict(model_name, X_tr_raw, y_tr, X_te_raw, seed):
    """Fit preprocessing and one weighted model, then predict the evaluation set."""
    imp = SimpleImputer(strategy='median')
    X_tr = imp.fit_transform(X_tr_raw)
    X_te = imp.transform(X_te_raw)
    sw = compute_sample_weight('balanced', y_tr)
    model = build_model(model_name, seed)
    model.fit(X_tr, y_tr, sample_weight=sw)
    return model.predict(X_te).astype(int)


def largest_remainder_counts(y_pool, requested_n):
    """Allocate an exact sample count by the largest-remainder method."""
    classes, counts = np.unique(y_pool, return_counts=True)
    requested_n = int(requested_n)
    if requested_n > int(counts.sum()):
        raise ValueError(f'requested calib_n={requested_n} exceeds candidate pool={int(counts.sum())}')
    if requested_n < len(classes):
        raise ValueError(f'requested calib_n={requested_n} is smaller than present classes={len(classes)}')
    quotas = requested_n * counts.astype(float) / counts.sum()
    alloc = np.floor(quotas).astype(int)
    alloc = np.where((counts > 0) & (alloc == 0), 1, alloc)
    fractions = quotas - np.floor(quotas)
    while alloc.sum() > requested_n:
        candidates = np.where(alloc > 1)[0]
        drop = candidates[np.argmin(fractions[candidates])]
        alloc[drop] -= 1
    while alloc.sum() < requested_n:
        candidates = np.where(alloc < counts)[0]
        add = candidates[np.argmax(fractions[candidates])]
        alloc[add] += 1
    assert int(alloc.sum()) == requested_n
    return dict(zip(classes.astype(int), alloc.astype(int)))


def stratified_split(y_m, calib_frac, rng, calib_n=None):
    """Return disjoint adaptation and evaluation indices with optional exact size."""
    calib_idx, test_idx = [], []
    for c in np.unique(y_m):
        idx = np.where(y_m == c)[0]
        rng.shuffle(idx)
        if len(idx) == 1:
            calib_idx.append(idx[0]); continue
        k = max(1, int(round(len(idx) * calib_frac)))
        k = min(k, len(idx) - 1)
        calib_idx.extend(idx[:k]); test_idx.extend(idx[k:])
    calib_idx = np.array(sorted(calib_idx)); test_idx = np.array(sorted(test_idx))
    if calib_n is not None and len(calib_idx) > calib_n:

        sel = []
        yc = y_m[calib_idx]
        allocation = largest_remainder_counts(yc, calib_n)
        for c in np.unique(yc):
            ci = calib_idx[yc == c]
            take = allocation[int(c)]
            sel.extend(rng.choice(ci, size=take, replace=False))
        calib_idx = np.array(sorted(sel))
        assert len(calib_idx) == int(calib_n), 'requested/actual calib_n mismatch!'
    assert len(np.intersect1d(calib_idx, test_idx)) == 0, 'calib/test overlap!'
    return calib_idx, test_idx


def run_experiment(Xtr, ytr, Xm, ym, model_name, n_repeats, calib_frac, base_seed):
    rows = []
    for r in range(n_repeats):
        seed = base_seed + r
        rng = np.random.default_rng(seed)
        c_idx, t_idx = stratified_split(ym, calib_frac, rng)
        if len(t_idx) < 3 or len(np.unique(ym[t_idx])) < 2:
            continue
        yte = ym[t_idx]; Xte = Xm[t_idx]

        yp_b = fit_predict(model_name, Xtr, ytr, Xte, seed)

        Xtr2 = np.vstack([Xtr, Xm[c_idx]]); ytr2 = np.concatenate([ytr, ym[c_idx]])
        yp_r = fit_predict(model_name, Xtr2, ytr2, Xte, seed)
        rows.append(dict(
            repeat=r, adaptation_n=len(c_idx), test_n=len(t_idx),
            base_macroF1p=macro_f1_present(yte, yp_b), adapted_macroF1p=macro_f1_present(yte, yp_r),
            base_balacc=balanced_acc(yte, yp_b), adapted_balacc=balanced_acc(yte, yp_r),
            base_mcc=mcc_fast(yte, yp_b), adapted_mcc=mcc_fast(yte, yp_r)))
    return pd.DataFrame(rows)


def size_sweep(Xtr, ytr, Xm, ym, model_name, sizes, n_repeats, base_seed):
    out = []
    for cn in sizes:
        for r in range(n_repeats):
            seed = base_seed + 1000 + r
            rng = np.random.default_rng(seed)
            c_idx, t_idx = stratified_split(ym, 0.5, rng, calib_n=cn)
            if len(c_idx) < 2 or len(t_idx) < 3 or len(np.unique(ym[t_idx])) < 2:
                continue
            yte = ym[t_idx]
            yp_b = fit_predict(model_name, Xtr, ytr, Xm[t_idx], seed)
            Xtr2 = np.vstack([Xtr, Xm[c_idx]]); ytr2 = np.concatenate([ytr, ym[c_idx]])
            yp = fit_predict(model_name, Xtr2, ytr2, Xm[t_idx], seed)
            cnt = np.bincount(ym[c_idx], minlength=4)
            base = macro_f1_present(yte, yp_b)
            adapted = macro_f1_present(yte, yp)
            out.append(dict(
                repeat=r,
                adaptation_n_requested=int(cn),
                adaptation_n_actual=int(len(c_idx)),
                test_n=int(len(t_idx)),
                adaptation_n_O=int(cnt[0]),
                adaptation_n_G=int(cnt[1]),
                adaptation_n_T=int(cnt[2]),
                adaptation_n_P=int(cnt[3]),
                base_macroF1p=base,
                adapted_macroF1p=adapted,
                delta_macroF1p=adapted - base,
            ))
    return pd.DataFrame(out)


def summarize_size_sweep(detail):
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for (mine, requested), g in detail.groupby(['Mine', 'adaptation_n_requested'], sort=False):
        rows.append(dict(
            Mine=mine,
            adaptation_n_requested=int(requested),
            adaptation_n_actual_min=int(g['adaptation_n_actual'].min()),
            adaptation_n_actual_max=int(g['adaptation_n_actual'].max()),
            adaptation_n_O_mean=float(g['adaptation_n_O'].mean()),
            adaptation_n_G_mean=float(g['adaptation_n_G'].mean()),
            adaptation_n_T_mean=float(g['adaptation_n_T'].mean()),
            adaptation_n_P_mean=float(g['adaptation_n_P'].mean()),
            base_macroF1p_mean=float(g['base_macroF1p'].mean()),
            adapted_macroF1p_mean=float(g['adapted_macroF1p'].mean()),
            adapted_macroF1p_sd=float(g['adapted_macroF1p'].std(ddof=1)) if len(g) > 1 else 0.0,
            delta_macroF1p_mean=float(g['delta_macroF1p'].mean()),
            delta_macroF1p_median=float(g['delta_macroF1p'].median()),
            delta_macroF1p_min=float(g['delta_macroF1p'].min()),
            delta_macroF1p_max=float(g['delta_macroF1p'].max()),
            n_eff=int(len(g)),
        ))
    return pd.DataFrame(rows)


def summarize(df, mine):
    def ms(col):
        return float(df[col].mean()), float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    row = {'Mine': mine, 'n_repeats': len(df),
           'adaptation_n_mean': float(df['adaptation_n'].mean()), 'test_n_mean': float(df['test_n'].mean())}
    for metric in ['macroF1p', 'balacc', 'mcc']:
        bm, bs = ms('base_' + metric); rm, rs = ms('adapted_' + metric)
        row[f'base_{metric}'] = round(bm, 4); row[f'base_{metric}_sd'] = round(bs, 4)
        row[f'adapted_{metric}'] = round(rm, 4); row[f'adapted_{metric}_sd'] = round(rs, 4)
        row[f'delta_{metric}'] = round(rm - bm, 4)
        delta = df['adapted_' + metric] - df['base_' + metric]
        row[f'improved_n_{metric}'] = int((delta > 0).sum())
        row[f'delta_{metric}_median'] = round(float(delta.median()), 4)
        row[f'delta_{metric}_min'] = round(float(delta.min()), 4)
        row[f'delta_{metric}_max'] = round(float(delta.max()), 4)
    return row


def make_figure(summ_df, sweep_dict, out_png):
    n = len(summ_df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]; x = np.arange(n); w = 0.36
    ax.bar(x - w/2, summ_df['base_macroF1p'], w, yerr=summ_df['base_macroF1p_sd'],
           capsize=3, label='Baseline (Malan only)', color='#B0B0B0')
    ax.bar(x + w/2, summ_df['adapted_macroF1p'], w, yerr=summ_df['adapted_macroF1p_sd'],
           capsize=3, label='Target-mine adaptation', color='#2C7FB8')
    ax.set_xticks(x); ax.set_xticklabels(summ_df['Mine'], rotation=0)
    ax.set_ylabel('macro-F1 (present classes)'); ax.set_ylim(0, 1)
    ax.set_title('(a) Held-out target-mine performance'); ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for mine, sw in sweep_dict.items():
        if len(sw):
            ax.errorbar(sw['adaptation_n_requested'], sw['adapted_macroF1p_mean'], yerr=sw['adapted_macroF1p_sd'],
                        marker='o', capsize=3, label=mine)
    ax.set_xlabel('Number of local adaptation samples')
    ax.set_ylabel('macro-F1 (present classes)'); ax.set_ylim(0, 1)
    ax.set_title('(b) Recovery vs local sample size'); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=600, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Target-mine adaptation experiment')
    ap.add_argument('--data_dir', default='../Input_Data')
    ap.add_argument('--train_path', default=None)
    ap.add_argument('--external_path', default=None)
    ap.add_argument('--output_dir', default='../Recreated_Model_Output')
    ap.add_argument('--mine_col', default='auto')
    ap.add_argument('--mines', default='auto',
                    help='Comma-separated mine names or "auto" for the two largest mines.')
    ap.add_argument('--model', default='RF-Default', choices=['RF-Default', 'XGBoost-Default'])
    ap.add_argument('--n_repeats', type=int, default=30)
    ap.add_argument('--calib_frac', type=float, default=0.5)
    ap.add_argument('--sizes', default='10,20,40,80')
    ap.add_argument('--seed', type=int, default=20240101)
    args = ap.parse_args()

    dd = Path(args.data_dir)
    train_path = Path(args.train_path) if args.train_path else dd / 'train_set.xlsx'
    ext_path = Path(args.external_path) if args.external_path else dd / 'external_validation_set.xlsx'


    Xtr, ytr = load_xy(train_path)
    print(f'[data] Malan training domain: {len(ytr)} train samples '
          f'(class counts {np.bincount(ytr, minlength=4).tolist()} = O/G/T/P)')

    Xe, ye, mines = load_external_with_mines(ext_path, args.mine_col)
    print(f'[data] external set: {len(ye)} samples across {len(set(mines))} mines')


    uniq, counts = np.unique(mines, return_counts=True)
    order = uniq[np.argsort(-counts)]
    if args.mines.strip().lower() == 'auto':
        targets = list(order[:2])
    else:
        want = [m.strip() for m in args.mines.split(',')]
        rev = {v: k for k, v in ROMAN.items()}
        targets = []
        for w in want:
            if w in uniq:
                targets.append(w)
            elif w in rev and rev[w] in uniq:
                targets.append(rev[w])
            else:
                print(f'[warn] mine {w!r} not found in external set; skipped')
        if not targets:
            sys.exit('[error] none of the requested mines were found')

    out_dir = Path(args.output_dir) / 'LocalCalibration'
    (out_dir / 'Tables').mkdir(parents=True, exist_ok=True)
    (out_dir / 'Figures').mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(',')]

    all_runs, summ_rows, sweeps = [], [], {}
    for m in targets:
        label = ROMAN.get(m, m)
        sel = mines == m
        Xm, ym = Xe[sel], ye[sel]
        print(f'\n[mine] {label} ({m}): N={len(ym)} '
              f'composition {np.bincount(ym, minlength=4).tolist()} (O/G/T/P)')
        df = run_experiment(Xtr, ytr, Xm, ym, args.model, args.n_repeats,
                            args.calib_frac, args.seed)
        if df.empty:
            print('  [skip] not enough test samples/classes'); continue
        df.insert(0, 'Mine', label); all_runs.append(df)
        row = summarize(df, label); summ_rows.append(row)
        print(f'  macro-F1(present): baseline {row["base_macroF1p"]:.3f} '
              f'-> adapted {row["adapted_macroF1p"]:.3f} '
              f'(delta {row["delta_macroF1p"]:+.3f}, improved '
              f'{row["improved_n_macroF1p"]}/{len(df)})')
        sw = size_sweep(Xtr, ytr, Xm, ym, args.model, sizes, args.n_repeats, args.seed)
        sw.insert(0, 'Mine', label); sweeps[label] = sw

    if not summ_rows:
        sys.exit('[error] no mine produced results')

    summ_df = pd.DataFrame(summ_rows)
    summ_df.to_csv(out_dir / 'Tables' / 'LocalCalibration_Summary.csv', index=False)
    pd.concat(all_runs, ignore_index=True).to_csv(
        out_dir / 'Tables' / 'LocalCalibration_PerRun.csv', index=False)
    if sweeps:
        sweep_detail = pd.concat(sweeps.values(), ignore_index=True)
        sweep_detail.to_csv(
            out_dir / 'Tables' / 'LocalCalibration_SizeSweep.csv', index=False)
        sweep_summary = summarize_size_sweep(sweep_detail)
        sweep_summary.to_csv(
            out_dir / 'Tables' / 'LocalCalibration_SizeSweep_Summary.csv', index=False)
        make_figure(
            summ_df,
            {m: sweep_summary[sweep_summary.Mine == m] for m in sweep_summary.Mine.unique()},
            out_dir / 'Figures' / 'Fig_TargetMineAdaptation.png',
        )

    print('\n================ SUMMARY (model = %s) ================' % args.model)
    print(summ_df[['Mine', 'base_macroF1p', 'adapted_macroF1p', 'delta_macroF1p',
                   'improved_n_macroF1p', 'delta_macroF1p_median', 'delta_macroF1p_min',
                   'delta_macroF1p_max', 'base_mcc', 'adapted_mcc', 'delta_mcc']]
          .to_string(index=False))
    print(f'\n[done] outputs written to: {out_dir}')
    print('Repeated splits describe stability and are not independent mine-level experiments.')


if __name__ == '__main__':
    main()
