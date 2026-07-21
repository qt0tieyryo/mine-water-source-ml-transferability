# -*- coding: utf-8 -*-
"""
Target-mine adaptation experiment for cross-mine mine-water source identification.
目标矿适配实验 —— 验证"给锁定模型喂入少量目标矿标记样本即可恢复性能"。

设计(与主程序 main_analysis_pipeline.py / per_mine_analysis.py 完全对齐):
  * 特征        : x1..x8 (K+, Na+, Ca2+, Mg2+, Cl-, SO42-, HCO3-, pH) 全部8个核心特征
  * 类别        : 0=O(奥灰) 1=G(老空) 2=T(太灰) 3=P(砂岩裂隙), 与 CANONICAL_ORDER 一致
  * 预处理      : SimpleImputer(median), 仅在各自"训练数据"上 fit(与 pipeline 一致; 树模型不缩放)
  * 类不平衡    : 训练时使用一次 balanced sample_weight; RF 不再同时叠加 class_weight
  * 锁定模型    : RF-Default(参数固定, 无搜索, 可完全复现) 或 XGBoost-Default(工厂默认参数)
                 —— XGBoost-SSA 每个种子都要重跑 SSA 搜索, 需导入主程序; 这里默认用无搜索的
                    Default 配置即可清楚地演示"目标矿适配"效应。若要严格用 SSA, 见文末 NOTE。

对每个目标矿, 重复 R 次(默认30, 与主程序30种子协议一致):
  (1) 按类别分层, 把该矿的带标签样本拆成【适配子集】和【与之不相交的测试子集】(固定种子);
  (2) baseline    : 仅用马兰训练集(train=153)训练锁定模型 → 在该矿测试子集上评估;
  (3) adapted     : 用 马兰训练集 + 适配子集 训练同一锁定模型 → 在同一测试子集上评估;
  (4) 记录配对结果(同一测试子集 → 配对比较)。
再做一次【适配样本量扫描】(10/20/40/80...), 说明"需要多少本地样本"。

【红线】适配子集与测试子集绝不重叠(脚本内 assert 强制); 预处理只在训练数据上 fit。

产出 <output_dir>/LocalCalibration/ :
  Tables/LocalCalibration_Summary.csv     每矿 baseline vs adapted 的均值/SD + 配对差描述
  Tables/LocalCalibration_PerRun.csv      每矿每次重复的逐条记录(便于自查/画箱线图)
  Tables/LocalCalibration_SizeSweep.csv   适配样本量扫描逐次记录
  Tables/LocalCalibration_SizeSweep_Summary.csv  适配样本量扫描汇总
  Figures/Fig_TargetMineAdaptation.png    前后对比柱状图 + 样本量恢复曲线

用法(在主程序同一目录下):
  python local_calibration.py \
      --data_dir ../Input_Data \
      --output_dir ../Recreated_Model_Output \
      --mines auto \
      --model RF-Default \
      --n_repeats 30 --calib_frac 0.5 --seed 20240101

依赖: numpy pandas scikit-learn matplotlib (--model XGBoost-Default 时另需 xgboost)
作者: De Gao
"""

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

# ------------------------------------------------------------------
# 与主程序一致的常量(勿改)
# ------------------------------------------------------------------
CORE_FEATURE_COLS = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
CANONICAL_ORDER = [
    'Ordovician limestone (O)',
    'Goaf water (G)',
    'Taiyuan limestone (T)',
    'Permian sandstone fissure (P)',
]
SHORT = ['O', 'G', 'T', 'P']
PATTERNS = {
    0: ('ordovician', '奥灰', '奥陶'),
    1: ('goaf', '老空', '采空'),
    2: ('taiyuan', '太灰', '太原'),
    3: ('permian', 'sandstone', '砂岩', '二叠'),
}
EXACT_ALIASES = {'2': 0, '2.0': 0, 'o': 0, '3': 1, '3.0': 1, 'g': 1,
                 '4': 2, '4.0': 2, 't': 2, '5': 3, '5.0': 3, 'p': 3}
MINE_COL_CANDIDATES = ['mine', 'mine_id', 'mine_name', 'Mine', 'Mine_ID',
                       'MineName', '矿井名称', '矿井', '矿名', '煤矿', '矿区', '矿']
ROMAN = {'西曲': 'Xiqu', '屯兰': 'Tunlan', '西铭': 'Ximing', '东曲矿': 'Dongqu',
         '镇城底': 'Zhenchengdi', '义城矿': 'Yicheng', '杜儿坪': "Du'erping",
         '官地矿': 'Guandi', '福昌': 'Fuchang', '南岭矿': 'Nanling',
         '世纪金鑫': 'Shijijinxin', '铂龙': 'Bolong'}

# RF-Default: 主程序中固定参数(无搜索); 类别权重由 sample_weight 单独提供,
# 避免 RF class_weight 与 sample_weight 重复加权。
RF_DEFAULT_PARAMS = dict(n_estimators=200, max_features='sqrt', max_depth=8,
                         min_samples_leaf=4, min_samples_split=6, max_samples=0.8)
# XGBoost-Default: _unified_build_model 中 XGBoost 的工厂默认参数
XGB_DEFAULT_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                          reg_lambda=1.0, min_child_weight=5, gamma=0.1)


# ------------------------------------------------------------------
# 标签规范化 / 数据加载(复现主程序逻辑)
# ------------------------------------------------------------------
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
    return -1  # 未知标签 -> 剔除


def _resolve_target_col(df):
    for cand in ('y', '充水水源', '水源', '类别', '标签', 'label', 'target', 'class'):
        if cand in df.columns:
            return cand
    return df.columns[-1]


def load_xy(path):
    """加载 train_set/test_set 类文件 -> (X_raw[n,8], y_enc[n]); 剔除未知标签。"""
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
    """加载外部集 -> (X_raw[n,8], y_enc[n], mines[n])。"""
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
    # mine column
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


# ------------------------------------------------------------------
# 指标(与 per_mine_analysis.py 完全一致的 bincount 实现)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 锁定模型 factory + fit(对齐 _unified_build_model / _unified_fit)
# ------------------------------------------------------------------
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
    """在训练数据上 fit imputer+model(单次 balanced sample_weight), 预测测试集标签向量。"""
    imp = SimpleImputer(strategy='median')
    X_tr = imp.fit_transform(X_tr_raw)
    X_te = imp.transform(X_te_raw)
    sw = compute_sample_weight('balanced', y_tr)   # == precompute_sample_weight
    model = build_model(model_name, seed)
    model.fit(X_tr, y_tr, sample_weight=sw)
    return model.predict(X_te).astype(int)


# ------------------------------------------------------------------
# 分层拆分: 每类按 calib_frac 分到适配/测试; 单样本类归入适配侧
# ------------------------------------------------------------------
def largest_remainder_counts(y_pool, requested_n):
    """按最大余数法给各类别分配精确样本数。"""
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
    """返回 (adapt_idx, test_idx), 互不相交。calib_n 指定时按每类比例抽固定总量。"""
    calib_idx, test_idx = [], []
    for c in np.unique(y_m):
        idx = np.where(y_m == c)[0]
        rng.shuffle(idx)
        if len(idx) == 1:
            calib_idx.append(idx[0]); continue
        k = max(1, int(round(len(idx) * calib_frac)))
        k = min(k, len(idx) - 1)  # 保证测试侧至少留1个
        calib_idx.extend(idx[:k]); test_idx.extend(idx[k:])
    calib_idx = np.array(sorted(calib_idx)); test_idx = np.array(sorted(test_idx))
    if calib_n is not None and len(calib_idx) > calib_n:
        # 从适配候选侧再分层抽精确 calib_n 个(最大余数法)
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


# ------------------------------------------------------------------
# 主实验
# ------------------------------------------------------------------
def run_experiment(Xtr, ytr, Xm, ym, model_name, n_repeats, calib_frac, base_seed):
    rows = []
    for r in range(n_repeats):
        seed = base_seed + r
        rng = np.random.default_rng(seed)
        c_idx, t_idx = stratified_split(ym, calib_frac, rng)
        if len(t_idx) < 3 or len(np.unique(ym[t_idx])) < 2:
            continue  # 测试子集太小/单类, 跳过该次
        yte = ym[t_idx]; Xte = Xm[t_idx]
        # baseline: 仅马兰
        yp_b = fit_predict(model_name, Xtr, ytr, Xte, seed)
        # adapted: 马兰 + 目标矿适配子集
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
    # (a) before/after macro-F1(present) per mine
    ax = axes[0]; x = np.arange(n); w = 0.36
    ax.bar(x - w/2, summ_df['base_macroF1p'], w, yerr=summ_df['base_macroF1p_sd'],
           capsize=3, label='Baseline (Malan only)', color='#B0B0B0')
    ax.bar(x + w/2, summ_df['adapted_macroF1p'], w, yerr=summ_df['adapted_macroF1p_sd'],
           capsize=3, label='Target-mine adaptation', color='#2C7FB8')
    ax.set_xticks(x); ax.set_xticklabels(summ_df['Mine'], rotation=0)
    ax.set_ylabel('macro-F1 (present classes)'); ax.set_ylim(0, 1)
    ax.set_title('(a) Held-out target-mine performance'); ax.legend(frameon=False, fontsize=9)
    # (b) adaptation-size sweep
    ax = axes[1]
    for mine, sw in sweep_dict.items():
        if len(sw):
            ax.errorbar(sw['adaptation_n_requested'], sw['adapted_macroF1p_mean'], yerr=sw['adapted_macroF1p_sd'],
                        marker='o', capsize=3, label=mine)
    ax.set_xlabel('Number of local adaptation samples')
    ax.set_ylabel('macro-F1 (present classes)'); ax.set_ylim(0, 1)
    ax.set_title('(b) Recovery vs local sample size'); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Target-mine adaptation experiment')
    ap.add_argument('--data_dir', default='../Input_Data')
    ap.add_argument('--train_path', default=None)
    ap.add_argument('--external_path', default=None)
    ap.add_argument('--output_dir', default='../Recreated_Model_Output')
    ap.add_argument('--mine_col', default='auto')
    ap.add_argument('--mines', default='auto',
                    help='comma-separated mine names (Chinese or romanized), or "auto" for the 2 largest')
    ap.add_argument('--model', default='RF-Default', choices=['RF-Default', 'XGBoost-Default'])
    ap.add_argument('--n_repeats', type=int, default=30)
    ap.add_argument('--calib_frac', type=float, default=0.5)
    ap.add_argument('--sizes', default='10,20,40,80')
    ap.add_argument('--seed', type=int, default=20240101)
    args = ap.parse_args()

    dd = Path(args.data_dir)
    train_path = Path(args.train_path) if args.train_path else dd / 'train_set.xlsx'
    ext_path = Path(args.external_path) if args.external_path else dd / 'external_validation_set.xlsx'

    # 马兰训练域统一为 train_set.xlsx 的 153 个训练样本; internal test 不参与本地适配训练。
    Xtr1, ytr1 = load_xy(train_path)
    Xtr, ytr = Xtr1, ytr1
    print(f'[data] Malan training domain: {len(ytr)} train samples '
          f'(class counts {np.bincount(ytr, minlength=4).tolist()} = O/G/T/P)')

    Xe, ye, mines = load_external_with_mines(ext_path, args.mine_col)
    print(f'[data] external set: {len(ye)} samples across {len(set(mines))} mines')

    # 选目标矿
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
    print('论文写法建议: 报告配对提升方向、均值、中位数和范围; '
          '30次重复划分不作为独立矿井实验进行正式显著性推断。')


# ------------------------------------------------------------------
# NOTE — 若要严格用 XGBoost-SSA(而非 Default)做目标矿适配:
#   XGBoost-SSA 的超参数是每个种子由 SSA 优化器现搜出来的, 无固定值。两种做法:
#   (A) 复用主程序: 从 main_analysis_pipeline import _nested_search_once / _unified_fit,
#       在每个 seed 上对 (马兰) 与 (马兰+适配子集) 各搜一次 SSA 再拟合、评估。
#   (B) 近似: 从 RegenData/final_model_registry.pkl 或 Best_Params_JSON 里取该配置某个
#       代表种子的 SSA 最优参数, 固定后当作 'XGBoost-SSA-locked' 复用到本脚本的 build_model。
#   出于可复现性, 本脚本默认用无搜索的 RF-Default / XGBoost-Default 即可清楚演示适配效应。
# ------------------------------------------------------------------
if __name__ == '__main__':
    main()
