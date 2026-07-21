"""
Final figure generator for the manuscript submission package.

Reads the processed figure source data and writes regenerated figure files.

Outputs:
  Figure_Recreated/vector_pdf      vector PDF for each full figure
  Figure_Recreated/png_600dpi      600 dpi PNG for each full figure
  Figure_Recreated/panels_pdf      optional standalone panels
  Figure_Recreated/panels_png_600dpi
  Figure_Recreated/qa              source-data and render checks

The manuscript/supplement mapping is:
  Fig3  internal-test 28-model forest plot
  Fig4  internal/external performance scatter
  Fig5  internal/external confusion matrices for representative models
  Fig6  train/external feature-shift ECDF panels
  Fig7  relative SHAP importance + SHAP rank migration
  Fig8  joint SHAP-KS risk map
  FigS1 convergence trajectories
  FigS2 paired seed-stability comparison (XGBoost-SSA minus LightGBM-Default)
  FigS3 Q-Q plots for the eta-squared decomposition
  FigS4 K+ high-value tail distribution
  FigS5 SHAP rank migration across model roles
  FigS6 target-mine adaptation and adaptation-sample-size sweep
"""

from __future__ import annotations

import argparse
import json
from statistics import NormalDist
from pathlib import Path

import matplotlib

try:
    import fitz
except ImportError:  # Optional dependency used only for PDF vector QA.
    fitz = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


MM = 1 / 25.4
DOUBLE = 174 * MM
SINGLE = 84 * MM
FS_PANEL = 10.5
FS_AXIS_LABEL = 9.5
FS_TICK = 8.0
FS_LEGEND = 7.8
FS_ANNOT = 7.8
FS_CELL = 7.6
FS_DENSE = 7.0
FS_SMALL_TITLE = 8.7

ALGORITHMS = ["RF", "XGBoost", "LightGBM", "SVM"]
OPTIMIZERS = ["Default", "Optuna", "GridSearch", "PSO", "SSA", "DE", "GWO"]
TUNING_OPTIMIZERS = ["Optuna", "GridSearch", "PSO", "SSA", "DE", "GWO"]

ALGO_COLORS = {
    "RF": "#0072B2",
    "XGBoost": "#D55E00",
    "LightGBM": "#009E73",
    "SVM": "#7E57C2",
}

OPTIMIZER_COLORS = {
    "Default": "#6E6E6E",
    "Optuna": "#0072B2",
    "GridSearch": "#D55E00",
    "PSO": "#009E73",
    "SSA": "#CC79A7",
    "DE": "#E69F00",
    "GWO": "#7B61A8",
}

OPTIMIZER_MARKERS = {
    "Default": "o",
    "Optuna": "s",
    "GridSearch": "D",
    "PSO": "^",
    "SSA": "v",
    "DE": "P",
    "GWO": "X",
}

CLASS_ORDER = [
    "Ordovician limestone (O)",
    "Goaf water (G)",
    "Taiyuan limestone (T)",
    "Permian sandstone fissure (P)",
]
CLASS_SHORT = {
    "Ordovician limestone (O)": "O",
    "Goaf water (G)": "G",
    "Taiyuan limestone (T)": "T",
    "Permian sandstone fissure (P)": "P",
}

FEATURES = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
FEATURE_LABELS = {
    "x1": r"$\mathrm{K}^{+}$",
    "x2": r"$\mathrm{Na}^{+}$",
    "x3": r"$\mathrm{Ca}^{2+}$",
    "x4": r"$\mathrm{Mg}^{2+}$",
    "x5": r"$\mathrm{Cl}^{-}$",
    "x6": r"$\mathrm{SO}_{4}^{2-}$",
    "x7": r"$\mathrm{HCO}_{3}^{-}$",
    "x8": r"$\mathrm{pH}$",
}
FEATURE_FROM_DISPLAY = {
    "$K^+$": "x1",
    "$Na^+$": "x2",
    "$Ca^{2+}$": "x3",
    "$Mg^{2+}$": "x4",
    "$Cl^-$": "x5",
    "$SO_4^{2-}$": "x6",
    "$HCO_3^-$": "x7",
    "pH": "x8",
    "K+": "x1",
    "Na+": "x2",
    "Ca2+": "x3",
    "Mg2+": "x4",
    "Cl-": "x5",
    "SO4^2-": "x6",
    "HCO3-": "x7",
}

FEATURE_COLORS = {
    "x1": "#D55E00",
    "x2": "#E69F00",
    "x3": "#A6761D",
    "x4": "#666666",
    "x5": "#009E73",
    "x6": "#CC79A7",
    "x7": "#7E57C2",
    "x8": "#66A61E",
}

SOURCE_FILE_ALIASES = {
    "Search_Convergence_Long.csv": [
        "search_convergence_long.csv",
    ],
    "FinalEval_Test_External_Summary.csv": [
        "final_evaluation_summary.csv",
    ],
    "Generalization_Ranking_Spearman.csv": [
        "generalization_ranking_spearman.csv",
    ],
    "FinalEval_Test_External_Confusion_Matrices_Long.csv": [
        "final_evaluation_confusion_matrices_long.csv",
    ],
    "Algorithm_vs_Optimizer_EtaSquared.csv": [
        "algorithm_vs_optimizer_eta_squared.csv",
    ],
    "feature_arrays.npz": [
        "feature_arrays_submission.npz",
    ],
    "Dataset_DomainShift_KS.csv": [
        "dataset_domain_shift_ks.csv",
    ],
    "SHAP_Generalization_Contrast.csv": [
        "shap_generalization_contrast.csv",
    ],
    "Table_4-9_DomainShift_CoreFeatures_Expanded.csv": [
        "domain_shift_core_features_expanded.csv",
    ],
    "FinalEval_Test_External_Raw.csv": [
        "final_evaluation_raw_records.csv",
    ],
}

TIMES_FONT_SETTINGS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.sf": "Times New Roman",
    "mathtext.default": "regular",
}


def default_source_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    latest_718 = script_dir.parents[0] / "Figure_Source_Data_output7.18"
    if latest_718.exists():
        return latest_718
    latest_source = script_dir / "output6.28" / "Figure_Source_Data_latest_20260628"
    if latest_source.exists():
        return latest_source
    return script_dir.parents[0] / "Figure_Source_Data"


def default_out_dir() -> Path:
    return Path(__file__).resolve().parent / "output7.20" / "Figure_Recreated"


def default_supp_out_dir() -> Path:
    return Path(__file__).resolve().parent / "output7.20" / "Figure_Recreated_Supplementary"


def set_style() -> None:
    matplotlib.rcParams.update(
        {
            **TIMES_FONT_SETTINGS,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.85,
            "axes.labelsize": FS_AXIS_LABEL,
            "axes.titlesize": FS_PANEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": None,
            "savefig.pad_inches": 0.06,
        }
    )


def set_supp_style() -> None:
    matplotlib.rcParams.update(
        {
            **TIMES_FONT_SETTINGS,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.labelsize": FS_AXIS_LABEL,
            "axes.titlesize": FS_PANEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.bbox": None,
            "savefig.pad_inches": 0.06,
        }
    )


def make_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "pdf": out_dir / "vector_pdf",
        "png": out_dir / "png_600dpi",
        "panels_pdf": out_dir / "panels_pdf",
        "panels_png": out_dir / "panels_png_600dpi",
        "qa": out_dir / "qa",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.pdf", "*.png", "*.json"):
            for old in d.glob(pattern):
                old.unlink()
    return dirs


def box_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#222222")
        ax.spines[side].set_linewidth(0.65)
    ax.tick_params(top=False, right=False, colors="#222222", width=0.65)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#D8DEE6", linewidth=0.40, alpha=0.72)
        ax.set_axisbelow(True)


def save_fig(fig: plt.Figure, stem: str, dirs: dict[str, Path]) -> None:
    fig.savefig(dirs["pdf"] / f"{stem}.pdf")
    fig.savefig(dirs["png"] / f"{stem}.png", dpi=600)


def save_panel(fig: plt.Figure, ax: plt.Axes, stem: str, dirs: dict[str, Path], pad: float = 0.16) -> None:
    fig.canvas.draw()
    extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.dpi_scale_trans.inverted()
    ).padded(pad)
    fig.savefig(dirs["panels_pdf"] / f"{stem}.pdf", bbox_inches=extent)
    fig.savefig(dirs["panels_png"] / f"{stem}.png", dpi=600, bbox_inches=extent)


def inspect_pdf_directory(pdf_dir: Path, include_drawings: bool = True) -> dict:
    if fitz is None:
        return {
            "_status": "skipped",
            "_reason": "PyMuPDF is not installed; PDF vector QA was not run.",
        }

    checks = {}
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        doc = fitz.open(pdf)
        font_names = sorted({f[3] for page in doc for f in page.get_fonts(full=True)})
        record = {
            "pages": doc.page_count,
            "images": sum(len(p.get_images(full=True)) for p in doc),
            "fonts": len(font_names),
            "font_names": font_names,
        }
        if include_drawings:
            record["drawings"] = sum(len(p.get_drawings()) for p in doc)
        checks[pdf.name] = record
        doc.close()
    return checks


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def data_file(source_dir: Path, name: str) -> Path:
    aliases = [name] + SOURCE_FILE_ALIASES.get(name, [])
    candidates = []
    for candidate_name in aliases:
        candidates.extend([
            source_dir / candidate_name,
            source_dir / "Tables" / candidate_name,
            source_dir / "RegenData" / candidate_name,
        ])
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def read_csv(source_dir: Path, name: str) -> pd.DataFrame:
    path = data_file(source_dir, name)
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def best_models_from_summary(source_dir: Path) -> dict[str, str]:
    """Return model names directly from the active source-data summary table."""
    df = read_csv(source_dir, "FinalEval_Test_External_Summary.csv")
    required = {"Model", "Test_F1_Macro_mean", "Val_F1_Macro_mean"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required summary columns: {sorted(missing)}")
    return {
        "internal_test_best": str(df.loc[df["Test_F1_Macro_mean"].idxmax(), "Model"]),
        "external_val_best": str(df.loc[df["Val_F1_Macro_mean"].idxmax(), "Model"]),
    }


def shap_role_models(source_dir: Path) -> dict[str, str]:
    """Return SHAP role-to-model mapping, with summary-table fallback."""
    path = data_file(source_dir, "SHAP_Generalization_Contrast.csv")
    mapping: dict[str, str] = {}
    if path.exists():
        df = read_csv(source_dir, "SHAP_Generalization_Contrast.csv")
        if {"Role", "Model"}.issubset(df.columns):
            mapping.update(df.groupby("Role")["Model"].first().to_dict())
    for k, v in best_models_from_summary(source_dir).items():
        mapping.setdefault(k, v)
    return mapping


def ordered_leader_models(source_dir: Path) -> list[str]:
    roles = shap_role_models(source_dir)
    models = [roles["external_val_best"], roles["internal_test_best"]]
    return list(dict.fromkeys(models))


def clean_model_label(model: str) -> str:
    return model.replace("-", "-\n")


def compact_model_label(model: str) -> str:
    """Compact model labels without changing the stored model identifiers."""
    return (
        model.replace("XGBoost", "XGB")
        .replace("LightGBM", "LGBM")
        .replace("GridSearch", "Grid")
    )


def panel_tag(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.025,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        clip_on=False,
    )


def clean_feature_id(value: str) -> str:
    return FEATURE_FROM_DISPLAY.get(str(value), str(value))


def make_cmap(colors: list[str], name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, colors)


BLUE_CMAP = make_cmap(["#F7FBFF", "#D6EAF7", "#74ADD1", "#2B8CBE", "#08306B"], "blue_clean")
GREEN_CMAP = make_cmap(["#F7FCF5", "#D9F0D3", "#74C476", "#238B45", "#00441B"], "green_clean")
PURPLE_CMAP = make_cmap(["#FCFBFD", "#E6E1EF", "#9E9AC8", "#6A51A3", "#3F007D"], "purple_clean")


def text_color(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.52 else "#1F1F1F"


def draw_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    cmap: LinearSegmentedColormap,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "{:.3f}",
    xrotation: float = 35,
    text_size: float = FS_CELL,
) -> None:
    values = matrix.to_numpy(float)
    finite = values[np.isfinite(values)]
    if vmin is None:
        vmin = float(finite.min())
    if vmax is None:
        vmax = float(finite.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            val = float(matrix.loc[row, col])
            color = cmap(norm(val))
            ax.add_patch(
                patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.55,
                )
            )
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=text_size, color=text_color(color))
    ax.set_xlim(-0.5, len(matrix.columns) - 0.5)
    ax.set_ylim(len(matrix.index) - 0.5, -0.5)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=xrotation, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    box_axes(ax)


def add_colorbar(
    fig: plt.Figure,
    ax,
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    label: str | None = None,
    x_offset: float = 1.04,
    tick_side: str = "right",
):
    axes = np.ravel(ax).tolist() if isinstance(ax, np.ndarray) else [ax]
    anchor = axes[-1]
    cax = anchor.inset_axes([x_offset, 0.0, 0.055, 1.0])
    n_steps = 80
    for i in range(n_steps):
        y0 = i / n_steps
        color = cmap(i / (n_steps - 1))
        cax.add_patch(
            patches.Rectangle(
                (0, y0),
                1,
                1 / n_steps,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
            )
        )
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    ticks = np.linspace(vmin, vmax, 5)
    cax.set_yticks((ticks - vmin) / (vmax - vmin))
    cax.set_yticklabels([f"{tick:.2f}" if vmax <= 1 else f"{tick:g}" for tick in ticks])
    if tick_side == "left":
        cax.yaxis.tick_left()
        cax.yaxis.set_label_position("left")
    else:
        cax.yaxis.tick_right()
        cax.yaxis.set_label_position("right")
    if label:
        cax.set_title(label.replace(" ", "\n"), fontsize=FS_LEGEND, pad=4)
    cax.tick_params(labelsize=FS_TICK, length=2.5, width=0.7)
    for side in ("left", "right", "top", "bottom"):
        cax.spines[side].set_visible(True)
        cax.spines[side].set_color("#222222")
        cax.spines[side].set_linewidth(0.6)
    return cax


def add_horizontal_colorbar(
    ax: plt.Axes,
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    label: str,
):
    cax = ax.inset_axes([0.50, 1.065, 0.45, 0.045])
    n_steps = 80
    for i in range(n_steps):
        x0 = i / n_steps
        color = cmap(i / (n_steps - 1))
        cax.add_patch(
            patches.Rectangle(
                (x0, 0),
                1 / n_steps,
                1,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
            )
        )
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    ticks = np.linspace(vmin, vmax, 4)
    cax.set_xticks((ticks - vmin) / (vmax - vmin))
    cax.set_xticklabels([f"{tick:.2f}" for tick in ticks])
    cax.set_yticks([])
    cax.tick_params(axis="x", labelsize=FS_TICK, length=2.5, width=0.7, pad=1)
    cax.set_title(label, fontsize=FS_LEGEND, pad=2)
    for side in ("left", "right", "top", "bottom"):
        cax.spines[side].set_visible(True)
        cax.spines[side].set_color("#222222")
        cax.spines[side].set_linewidth(0.6)
    return cax


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.sort(np.asarray(values, dtype=float))
    vals = vals[np.isfinite(vals)]
    y = np.arange(1, len(vals) + 1) / len(vals)
    return vals, y


def log_ready(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float).copy()
    positive = vals[np.isfinite(vals) & (vals > 0)]
    if len(positive) == 0:
        return vals
    vals[vals <= 0] = positive.min() * 0.5
    return vals


def ks_segment(train_values: np.ndarray, external_values: np.ndarray) -> tuple[float, float, float]:
    left = np.sort(np.asarray(train_values, dtype=float))
    right = np.sort(np.asarray(external_values, dtype=float))
    grid = np.unique(np.concatenate([left, right]))
    y_left = np.searchsorted(left, grid, side="right") / len(left)
    y_right = np.searchsorted(right, grid, side="right") / len(right)
    idx = int(np.argmax(np.abs(y_left - y_right)))
    return float(grid[idx]), float(y_left[idx]), float(y_right[idx])


def figure_03_convergence(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    df = read_csv(source_dir, "Search_Convergence_Long.csv")
    df = df[df["Optimizer"].isin(TUNING_OPTIMIZERS)].copy()
    swarm_optimizers = {"PSO", "SSA", "DE", "GWO"}
    # The archive records four generations for swarm methods.  Each node is the
    # best-so-far score after six additional objective-function evaluations.
    df["Cumulative_Evaluations"] = np.where(
        df["Optimizer"].isin(swarm_optimizers),
        df["Step"].astype(int) * 6,
        df["Step"].astype(int),
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 4.60), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.14, top=0.82, wspace=0.20, hspace=0.28)
    axes = axes.ravel()
    for panel_idx, (ax, algo) in enumerate(zip(axes, ALGORITHMS)):
        sub_algo = df[df["Algorithm"].eq(algo)]
        for opt in TUNING_OPTIMIZERS:
            sub = sub_algo[sub_algo["Optimizer"].eq(opt)]
            if sub.empty:
                continue
            agg = (
                sub.groupby("Cumulative_Evaluations")["Best_Internal_CV_F1"]
                .agg(mean="mean", sd="std")
                .reset_index()
            )
            color = OPTIMIZER_COLORS[opt]
            ax.plot(
                agg["Cumulative_Evaluations"],
                agg["mean"],
                color=color,
                linewidth=1.25,
                marker="o" if opt in swarm_optimizers else None,
                markersize=2.4 if opt in swarm_optimizers else 0,
                label=opt,
                zorder=3,
            )
            last = agg.iloc[-1]
            ax.scatter(last["Cumulative_Evaluations"], last["mean"], color=color, s=13, zorder=4)
        ax.set_title(f"({chr(97 + panel_idx)}) {algo}", loc="left", fontweight="semibold", fontsize=8.5, pad=3)
        ax.set_xlim(1, 24)
        ax.set_xticks([6, 12, 18, 24])
        box_axes(ax, grid_axis="both")
        save_panel(fig, ax, f"FigS1_convergence_{algo}", dirs)
    for i, ax in enumerate(axes):
        if i % 2 == 1:
            ax.set_ylabel("")
        if i < 2:
            ax.tick_params(axis="x", labelbottom=False)
    fig.text(0.5, 0.045, "Cumulative objective-function evaluations", ha="center", va="center", fontsize=9.0)
    fig.text(0.022, 0.48, r"Best-so-far CV macro-$F_{1}$", ha="center", va="center", rotation=90, fontsize=9.0)
    handles = [plt.Line2D([0], [0], color=OPTIMIZER_COLORS[o], lw=1.5, label=o) for o in TUNING_OPTIMIZERS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=6, frameon=False, fontsize=7.3, columnspacing=1.0)
    save_fig(fig, "FigS1_search_convergence", dirs)
    plt.close(fig)
    qa["FigS1"] = {
        "source": "Search_Convergence_Long.csv",
        "rows": int(len(df)),
        "evaluation_axis": "Cumulative objective-function evaluations",
        "swarm_nodes": [6, 12, 18, 24],
        "uncertainty_marker": "final point shown as mean ± SD across 30 repeated searches",
    }


def figure_04_internal_forest(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    df = read_csv(source_dir, "FinalEval_Test_External_Summary.csv").copy()
    ranked = df.sort_values("Test_F1_Macro_mean", ascending=False).reset_index(drop=True)
    y = np.arange(len(ranked))
    fig, ax = plt.subplots(figsize=(6.0, 5.25))
    fig.subplots_adjust(left=0.225, right=0.975, bottom=0.115, top=0.925)
    top_indices = set(ranked.head(3).index)
    for row in sorted(top_indices):
        ax.axhspan(row - 0.5, row + 0.5, color="#FFF8E7", zorder=0.1)
    for alg in ALGORITHMS:
        sub = ranked[ranked["Algorithm"].eq(alg)]
        ypos = sub.index.to_numpy()
        means = sub["Test_F1_Macro_mean"].to_numpy()
        xerr = np.vstack(
            [
                means - sub["Test_F1_Macro_ci95_low"].to_numpy(),
                sub["Test_F1_Macro_ci95_high"].to_numpy() - means,
            ]
        )
        ax.errorbar(
            means,
            ypos,
            xerr=xerr,
            fmt="o",
            markersize=4.1,
            markerfacecolor=ALGO_COLORS[alg],
            markeredgecolor="white",
            markeredgewidth=0.45,
            ecolor="#87919A",
            elinewidth=0.43,
            capsize=0.85,
            color=ALGO_COLORS[alg],
            label=compact_model_label(alg),
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([compact_model_label(m) for m in ranked["Model"]])
    ax.set_xlabel(r"Internal-test macro-$F_{1}$ (mean with 95% CI)")
    ax.set_ylabel("")
    ci_low = float(ranked["Test_F1_Macro_ci95_low"].min())
    ci_high = float(ranked["Test_F1_Macro_ci95_high"].max())
    span = max(ci_high - ci_low, 0.01)
    ax.set_xlim(ci_low - span * 0.025, ci_high + span * 0.025)
    ax.set_ylim(len(ranked) - 0.4, -1.0)
    ax.xaxis.label.set_size(FS_AXIS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_DENSE, pad=2)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.992),
        ncol=4,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.75,
        fontsize=6.9,
    )
    box_axes(ax, grid_axis="x")
    save_fig(fig, "Fig3_internal_test_forest", dirs)
    plt.close(fig)
    best = ranked.head(3)[["Model", "Test_F1_Macro_mean"]].to_dict("records")
    qa["Fig04"] = {"source": "FinalEval_Test_External_Summary.csv", "top3": best}


def figure_05_external(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    df = read_csv(source_dir, "FinalEval_Test_External_Summary.csv").copy()
    matrix = (
        df.pivot_table(index="Algorithm", columns="Optimizer", values="Val_F1_Macro_mean", aggfunc="mean")
        .reindex(index=ALGORITHMS, columns=OPTIMIZERS)
    )
    matrix_display = matrix.rename(columns={"GridSearch": "Grid"})
    spear = read_csv(source_dir, "Generalization_Ranking_Spearman.csv")
    row = spear[spear["Comparison"].eq("InternalTest_vs_ExternalVal")].iloc[0]
    fig = plt.figure(figsize=(6.15, 4.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.15], wspace=0.36)
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.14, top=0.93)
    ax0 = fig.add_subplot(gs[0])
    draw_heatmap(ax0, matrix_display, cmap=GREEN_CMAP, vmin=0.31, vmax=0.42, fmt="{:.3f}", text_size=7.3)
    cax = add_colorbar(fig, ax0, GREEN_CMAP, 0.31, 0.42, None, x_offset=1.010, tick_side="right")
    ticks = np.array([0.31, 0.34, 0.37, 0.40, 0.42])
    cax.set_yticks((ticks - 0.31) / 0.11)
    cax.set_yticklabels([f"{tick:.2f}" for tick in ticks])
    panel_tag(ax0, "(a)")
    ax0.set_xlabel("")
    ax0.set_ylabel("")
    ax0.set_aspect("auto")
    ax0.tick_params(axis="x", labelsize=FS_TICK)
    ax0.tick_params(axis="y", labelsize=FS_TICK)
    # The numerical heatmap duplicates the manuscript table and is deliberately
    # excluded from the main figure; retain only the external-transfer scatter.
    ax0.remove()
    cax.remove()
    ax1 = fig.add_axes([0.13, 0.13, 0.85, 0.82])
    for _, r in df.iterrows():
        ax1.scatter(
            r["Test_F1_Macro_mean"],
            r["Val_F1_Macro_mean"],
            s=31,
            marker=OPTIMIZER_MARKERS.get(r["Optimizer"], "o"),
            color=ALGO_COLORS[r["Algorithm"]],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.86,
            zorder=3,
        )
    ax1.set_xlabel(r"Internal-test macro-$F_{1}$", fontsize=9.0)
    ax1.set_ylabel("External-validation macro-F1", labelpad=2, fontsize=9.0)
    ax1.set_box_aspect(None)
    ax1.set_aspect("auto")
    ax1.axvline(df["Test_F1_Macro_mean"].mean(), color="#8A8A8A", linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)
    ax1.axhline(df["Val_F1_Macro_mean"].mean(), color="#8A8A8A", linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)
    x_span = df["Test_F1_Macro_mean"].max() - df["Test_F1_Macro_mean"].min()
    y_span = df["Val_F1_Macro_mean"].max() - df["Val_F1_Macro_mean"].min()
    ax1.set_xlim(0.735, 0.865)
    ax1.set_ylim(0.305, 0.430)
    ax1.set_xticks([0.75, 0.80, 0.85])
    ax1.set_yticks(np.arange(0.32, 0.421, 0.02))
    box_axes(ax1, grid_axis="both")
    leaders = best_models_from_summary(source_dir)
    key_models = [
        (leaders["external_val_best"], leaders["external_val_best"], (0.805, 0.425), "right"),
        (leaders["internal_test_best"], leaders["internal_test_best"], (0.841, 0.405), "left"),
    ]
    seen_models: set[str] = set()
    for model, role_text, xytext, ha in key_models:
        if model in seen_models or model not in set(df["Model"]):
            continue
        seen_models.add(model)
        r = df[df["Model"].eq(model)].iloc[0]
        ax1.scatter(
            r["Test_F1_Macro_mean"],
            r["Val_F1_Macro_mean"],
            s=54,
            facecolors="none",
            edgecolors="#3A3A3A",
            linewidths=0.42,
            zorder=4,
        )
        arrowprops = {
            "arrowstyle": "-",
            "color": "#555555",
            "linewidth": 0.60,
            "shrinkA": 4,
            "shrinkB": 5,
            "connectionstyle": "arc3,rad=0.0",
        }
        ax1.annotate(
            role_text,
            xy=(r["Test_F1_Macro_mean"], r["Val_F1_Macro_mean"]),
            xytext=xytext,
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=7.0,
            zorder=5,
            arrowprops={**arrowprops, "shrinkA": 2},
        )
    algo_handles = [
        plt.Line2D([0], [0], marker="o", color=ALGO_COLORS[a], lw=0, markersize=3.8, label=a)
        for a in ALGORITHMS
    ]
    opt_handles = [
        plt.Line2D([0], [0], marker=OPTIMIZER_MARKERS[o], color="#444444", lw=0, markersize=3.7, label=o)
        for o in OPTIMIZERS
    ]
    algo_legend = ax1.legend(
        handles=algo_handles,
        title="Algorithm",
        loc="upper left",
        bbox_to_anchor=(0.012, 0.988),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.78,
        fontsize=5.8,
        title_fontsize=5.8,
        ncol=2,
        columnspacing=0.65,
        handletextpad=0.22,
    )
    ax1.add_artist(algo_legend)
    ax1.legend(
        handles=opt_handles,
        title="Tuning strategy",
        loc="lower right",
        bbox_to_anchor=(0.988, 0.018),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.78,
        fontsize=5.6,
        title_fontsize=5.6,
        ncol=2,
        columnspacing=0.55,
        handletextpad=0.18,
    )
    save_panel(fig, ax1, "Fig4_internal_external_scatter", dirs)
    save_fig(fig, "Fig4_external_validation_and_transferability", dirs)
    plt.close(fig)
    qa["Fig05"] = {
        "source": ["FinalEval_Test_External_Summary.csv", "Generalization_Ranking_Spearman.csv"],
        "spearman_internal_external": float(row["Spearman_Rho"]),
        "p": float(row["P_Value"]),
        "internal_test_best": leaders["internal_test_best"],
        "external_val_best": leaders["external_val_best"],
    }


def class_counts_for(df: pd.DataFrame, model: str, dataset: str) -> dict[str, int]:
    sub = df[(df["Dataset"].eq(dataset)) & (df["Model"].eq(model))]
    if "Run_Seed" in sub.columns and not sub.empty:
        sub = sub[sub["Run_Seed"].eq(sub["Run_Seed"].min())]
    return sub.groupby("True_Name")["Count"].sum().reindex(CLASS_ORDER).fillna(0).astype(int).to_dict()


def confusion_matrix_for(df: pd.DataFrame, model: str, dataset: str) -> pd.DataFrame:
    sub = df[(df["Dataset"].eq(dataset)) & (df["Model"].eq(model))]
    mat = (
        sub.groupby(["True_Name", "Pred_Name"])["Row_Normalized"]
        .mean()
        .unstack("Pred_Name")
        .reindex(index=CLASS_ORDER, columns=CLASS_ORDER)
    )
    counts = class_counts_for(df, model, dataset)
    mat.index = [f"{CLASS_SHORT[x]} (n = {counts.get(x, 0)})" for x in mat.index]
    mat.columns = [CLASS_SHORT[x] for x in mat.columns]
    return mat


def figure_06_confusion(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    df = read_csv(source_dir, "FinalEval_Test_External_Confusion_Matrices_Long.csv")
    panels = [("XGBoost-SSA", "(a) XGBoost-SSA"), ("RF-Default", "(b) RF-Default")]
    dataset = "external_val"
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.00))
    fig.subplots_adjust(left=0.105, right=0.885, bottom=0.16, top=0.91, wspace=0.18)
    for panel_idx, (ax, (model, title)) in enumerate(zip(axes, panels)):
        mat = confusion_matrix_for(df, model, dataset)
        draw_heatmap(ax, mat, cmap=BLUE_CMAP, vmin=0, vmax=1, fmt="{:.2f}", xrotation=0, text_size=FS_CELL)
        panel_tag(ax, f"({chr(97 + panel_idx)})")
        ax.text(0.11, 1.025, model, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.6, fontweight="medium")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal")
        ax.tick_params(axis="both", labelsize=7.6)
        if panel_idx == 1:
            ax.set_yticklabels([])
            # The second matrix shares the true-class labels with panel (a);
            # suppress both labels and tick marks to avoid a redundant visual rail.
            ax.tick_params(axis="y", left=False)
        save_panel(fig, ax, f"Fig06_{model}_external_confusion", dirs)
    cax = add_colorbar(fig, axes, BLUE_CMAP, 0, 1, "", x_offset=1.04)
    cax.set_ylabel("Row-normalized proportion", fontsize=7.2, labelpad=4)
    fig.text(0.50, 0.078, "Predicted class", ha="center", va="center", fontsize=9.0)
    fig.text(0.028, 0.51, "True class", ha="center", va="center", rotation=90, fontsize=9.0)
    save_fig(fig, "Fig5_external_confusion_matrices", dirs)
    plt.close(fig)

    qa["Fig06"] = {
        "source": "FinalEval_Test_External_Confusion_Matrices_Long.csv",
        "models": [p[0] for p in panels],
        "dataset": dataset,
    }


def figure_07_eta(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    df = read_csv(source_dir, "Algorithm_vs_Optimizer_EtaSquared.csv")
    response_order = ["Test_F1_Macro", "Val_F1_Macro", "Generalization_Gap"]
    response_labels = ["Internal test", "External validation", "Generalization gap"]
    factor_order = ["Algorithm", "Optimizer", "Algorithm_x_Optimizer", "Residual"]
    factor_labels = {
        "Algorithm": "Algorithm",
        "Optimizer": "Optimizer",
        "Algorithm_x_Optimizer": "Interaction",
        "Residual": "Residual",
    }
    colors = {
        "Algorithm": "#0072B2",
        "Optimizer": "#E69F00",
        "Algorithm_x_Optimizer": "#CC79A7",
        "Residual": "#B8B8B8",
    }
    rows = []
    for response in response_order:
        sub = df[df["Response"].eq(response)]
        raw_vals = {r["Factor"]: float(r["Eta2"]) for _, r in sub.iterrows()}
        vals = {
            "Algorithm": raw_vals.get("Algorithm", 0.0),
            "Optimizer": raw_vals.get("Optimizer", 0.0),
            "Algorithm_x_Optimizer": raw_vals.get("Algorithm_x_Optimizer", 0.0),
            "Residual": raw_vals.get(
                "Run_Residual",
                max(
                    0.0,
                    1.0
                    - raw_vals.get("Algorithm", 0.0)
                    - raw_vals.get("Optimizer", 0.0)
                    - raw_vals.get("Algorithm_x_Optimizer", 0.0),
                ),
            ),
        }
        vals["Response"] = response
        rows.append(vals)
    plot_df = pd.DataFrame(rows).set_index("Response").reindex(response_order)
    fig, ax = plt.subplots(figsize=(5.9, 3.55), constrained_layout=True)
    x = np.arange(len(response_order))
    bottom = np.zeros(len(response_order))
    bar_width = 0.48
    for factor in factor_order:
        vals = plot_df[factor].to_numpy(float)
        ax.bar(
            x,
            vals,
            width=bar_width,
            bottom=bottom,
            color=colors[factor],
            edgecolor="white",
            linewidth=0.7,
            label=factor_labels[factor],
        )
        for xi, btm, val in zip(x, bottom, vals):
            if val >= 0.045:
                ax.text(xi, btm + val / 2, f"{val:.3f}", ha="center", va="center", fontsize=7.1, color="#1F1F1F")
            elif val > 0:
                ymid = btm + val / 2
                ax.annotate(
                    f"{val:.3f}",
                    xy=(xi + bar_width / 2, ymid),
                    xytext=(xi + bar_width / 2 + 0.13, ymid + 0.018),
                    textcoords="data",
                    ha="left",
                    va="center",
                    fontsize=6.8,
                    color="#1F1F1F",
                    arrowprops={
                        "arrowstyle": "-",
                        "color": colors[factor],
                        "linewidth": 0.65,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                    clip_on=False,
                )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(response_labels)
    ax.set_xlim(-0.55, len(response_order) - 0.25)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"Variance explained ($\eta^{2}$)")
    box_axes(ax, grid_axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    save_fig(fig, "Fig07_eta_squared_decomposition", dirs)
    plt.close(fig)
    qa["Fig07"] = {"source": "Algorithm_vs_Optimizer_EtaSquared.csv", "external_eta2": plot_df.loc["Val_F1_Macro"].to_dict()}


def figure_08_domain_shift(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    arrays = np.load(data_file(source_dir, "feature_arrays.npz"), allow_pickle=True)
    feature_names = list(arrays["feature_names"])
    train = arrays["X_train_raw"]
    external = arrays["X_val_raw"]
    ks = read_csv(source_dir, "Dataset_DomainShift_KS.csv")
    ks = ks[(ks["Reference_Split"].eq("train")) & (ks["Compared_Split"].eq("external_val"))].copy()
    ks_map = ks.set_index("Feature")["KS_Statistic"].to_dict()
    p_map = ks.set_index("Feature")["P_Value"].to_dict()
    feature_order = ks[ks["Feature"].isin(FEATURES)].sort_values("KS_Statistic", ascending=False)["Feature"].tolist()

    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 4.20))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.855, wspace=0.31, hspace=0.34)
    axes = axes.ravel()
    for panel_idx, (ax, feat) in enumerate(zip(axes, feature_order)):
        row = panel_idx // 4
        col = panel_idx % 4
        idx = feature_names.index(feat)
        train_values = train[:, idx]
        external_values = external[:, idx]
        if feat != "x8":
            train_values = log_ready(train_values)
            external_values = log_ready(external_values)
        x_train, y_train = ecdf(train_values)
        x_ext, y_ext = ecdf(external_values)
        ax.plot(x_train, y_train, color="#2C7FB8", linewidth=1.16, label="Training")
        ax.plot(x_ext, y_ext, color="#D95F0E", linewidth=1.16, linestyle=(0, (4, 2)), label="External")
        if feat != "x8":
            ax.set_xscale("log")
            ax.minorticks_off()
            positive = np.concatenate([train_values[train_values > 0], external_values[external_values > 0]])
            if len(positive):
                ax.set_xlim(left=max(float(positive.min()) * 0.8, 1e-3))
        x_ks, y_train_ks, y_ext_ks = ks_segment(train_values, external_values)
        ax.plot([x_ks, x_ks], [y_train_ks, y_ext_ks], color="#555555", linewidth=0.65, zorder=5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.text(
            0.0,
            1.055,
            f"({chr(97 + panel_idx)}) {feature_label(feat)}, KS D = {ks_map.get(feat, np.nan):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.7,
            fontweight="bold",
        )
        ax.set_xlabel("")
        if col != 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("")
        box_axes(ax, grid_axis="both")
        save_panel(fig, ax, f"Fig08_{feat}_domain_shift", dirs)
    handles = [
        plt.Line2D([0], [0], color="#2C7FB8", lw=1.6, label="Training domain"),
        plt.Line2D([0], [0], color="#D95F0E", lw=1.6, linestyle=(0, (4, 2)), label="External validation"),
    ]
    fig.text(
        0.5,
        0.055,
        r"Feature value (mg L$^{-1}$ for ions, log scale; pH, linear scale)",
        ha="center",
        va="center",
        fontsize=FS_AXIS_LABEL,
    )
    fig.text(0.022, 0.50, "ECDF", ha="center", va="center", rotation=90, fontsize=FS_AXIS_LABEL)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.975), fontsize=7.6)
    save_fig(fig, "Fig6_feature_domain_shift_ecdf", dirs)
    plt.close(fig)
    qa["Fig08"] = {
        "source": ["feature_arrays.npz", "Dataset_DomainShift_KS.csv"],
        "feature_order": feature_order,
        "ks_core": {f: float(ks_map[f]) for f in FEATURES if f in ks_map},
        "p_core": {f: float(p_map[f]) for f in FEATURES if f in p_map},
    }


def shap_contrast(source_dir: Path) -> pd.DataFrame:
    df = read_csv(source_dir, "SHAP_Generalization_Contrast.csv")
    df = df[df["Role"].isin(["external_val_best", "internal_test_best"])].copy()
    df["Feature_Label"] = df["Feature"].map(feature_label)
    return df


def shap_pivot(source_dir: Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    df = shap_contrast(source_dir)
    models = ordered_leader_models(source_dir)
    pivot = df.pivot_table(index="Feature", columns="Model", values="MeanAbsSHAP_ExternalVal", aggfunc="mean")
    missing = [m for m in models if m not in pivot.columns]
    if missing:
        raise KeyError(f"SHAP_Generalization_Contrast.csv lacks leader model columns: {missing}")
    pivot = pivot[models]
    external_model = models[0]
    return pivot.sort_values(external_model, ascending=False), models, shap_role_models(source_dir)


def figure_09_shap(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    pivot, models, role_models = shap_pivot(source_dir)
    external_model = models[0]
    ordered = pivot.sort_values(external_model, ascending=False)
    percent = ordered.div(ordered.sum(axis=0), axis=1) * 100
    ranks = ordered.rank(ascending=False, method="first").astype(int)
    display_index = [feature_label(f) for f in ordered.index]

    fig = plt.figure(figsize=(DOUBLE, 3.40))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.10, 1.18], wspace=0.38)
    fig.subplots_adjust(left=0.125, right=0.96, bottom=0.15, top=0.88)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    y = np.arange(len(ordered))
    bar_h = 0.30
    offsets = np.linspace(-bar_h / 1.8, bar_h / 1.8, len(models))
    legend_handles = []
    for offset, model in zip(offsets, models):
        color = ALGO_COLORS.get(model.split("-")[0], "#555555")
        vals = percent[model]
        bars = ax0.barh(y + offset, vals, height=bar_h, color=color, alpha=0.86, edgecolor="white", linewidth=0.55)
        for bar, value in zip(bars, vals):
            ax0.text(value + 0.28, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", ha="left", fontsize=6.5, color="#444444")
        legend_handles.append(patches.Patch(facecolor=color, alpha=0.86, label=model))
    panel_tag(ax0, "(a)")
    ax0.set_yticks(y)
    ax0.set_yticklabels(display_index)
    ax0.invert_yaxis()
    ax0.set_xlabel("Normalized mean(|SHAP|) contribution (%)", fontsize=8.6)
    ax0.set_ylabel("")
    ax0.yaxis.label.set_size(FS_AXIS_LABEL)
    ax0.tick_params(axis="y", labelsize=FS_TICK)
    ax0.tick_params(axis="x", labelsize=FS_TICK)
    ax0.set_xlim(0, 28.0)
    box_axes(ax0, grid_axis="x")

    x = np.arange(len(models))
    rank_table = pd.DataFrame({model: ranks[model] for model in models}).sort_values(models[0])
    for feat, row in rank_table.iterrows():
        color = FEATURE_COLORS.get(feat, "#555555")
        label = feature_label(feat)
        change = abs(int(row[models[0]]) - int(row[models[-1]]))
        line_color = color if change >= 3 else "#77838C"
        label_color = color if change >= 3 else "#59656E"
        ax1.plot(x, [row[m] for m in models], color=line_color, marker="o", markersize=0, linewidth=1.15, zorder=2)
        ax1.scatter(x, [row[m] for m in models], color=line_color, s=23, zorder=3, edgecolor="white", linewidth=0.35)
        ax1.text(x[0] - 0.08, row[models[0]], label, ha="right", va="center", fontsize=7.0, color=label_color)
    panel_tag(ax1, "(b)")
    ax1.set_xlim(-0.46, len(models) - 1 + 0.46)
    ax1.set_ylim(8.5, 0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=7.8)
    ax1.set_yticks(range(1, 9))
    ax1.set_ylabel("Within-model SHAP rank")
    ax1.yaxis.label.set_size(9.0)
    box_axes(ax1, grid_axis="y")
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.35, 0.97),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handlelength=1.0,
        columnspacing=1.4,
    )

    save_fig(fig, "Fig7_shap_importance_and_rank_migration", dirs)
    plt.close(fig)
    qa["Fig09"] = {
        "source": "SHAP_Generalization_Contrast.csv",
        "relative_percent": percent.to_dict(),
        "ranks": ranks.to_dict(),
        "models": models,
        "roles": role_models,
        "feature_order": list(ordered.index),
        "order_rule": f"Features sorted by {external_model}",
    }


def feature_ranks(values: pd.Series) -> pd.Series:
    return values.rank(ascending=False, method="first").astype(int)


def figure_10_rank_migration(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    pivot, models, role_models = shap_pivot(source_dir)
    ranks = pd.DataFrame({model: feature_ranks(pivot[model]) for model in models}).sort_values(models[0])
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(5.7, 4.3), constrained_layout=True)
    for feat, row in ranks.iterrows():
        color = FEATURE_COLORS.get(feat, "#555555")
        label = feature_label(feat)
        ax.plot(x, [row[m] for m in models], color=color, marker="o", markersize=5.0, linewidth=1.8)
        ax.text(x[0] - 0.04, row[models[0]], label, ha="right", va="center", fontsize=8.0, color=color)
        ax.text(x[-1] + 0.04, row[models[-1]], label, ha="left", va="center", fontsize=8.0, color=color)
    ax.set_xlim(-0.22, len(models) - 1 + 0.22)
    ax.set_ylim(8.5, 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_yticks(range(1, 9))
    ax.set_ylabel("Within-model SHAP rank")
    box_axes(ax, grid_axis="y")
    save_fig(fig, "Fig10_shap_rank_migration", dirs)
    plt.close(fig)
    qa["Fig10"] = {"source": "SHAP_Generalization_Contrast.csv", "ranks": ranks.to_dict(), "models": models, "roles": role_models}


def figure_11_risk(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    pivot, models, role_models = shap_pivot(source_dir)
    external_model = models[0]
    shap_col = "ExternalBest_SHAP"
    shap_vals = pivot[external_model].rename(shap_col)
    ks = read_csv(source_dir, "Dataset_DomainShift_KS.csv")
    ks = ks[(ks["Reference_Split"].eq("train")) & (ks["Compared_Split"].eq("external_val")) & (ks["Feature"].isin(FEATURES))]
    expanded_path = data_file(source_dir, "Table_4-9_DomainShift_CoreFeatures_Expanded.csv")
    if not expanded_path.exists():
        qa["Fig11"] = {
            "skipped": True,
            "reason": "Table_4-9_DomainShift_CoreFeatures_Expanded.csv not found",
        }
        return
    expanded = read_csv(source_dir, "Table_4-9_DomainShift_CoreFeatures_Expanded.csv")
    feature_col = "Feature" if "Feature" in expanded.columns else expanded.columns[0]
    expanded["Feature"] = expanded[feature_col].map(clean_feature_id)
    cohen = expanded.set_index("Feature")["Cohen's d"]
    df = pd.concat([shap_vals, ks.set_index("Feature")["KS_Statistic"], cohen], axis=1).dropna()

    shap_sorted = np.sort(df[shap_col].to_numpy())
    shap_low_med = (shap_sorted[1] + shap_sorted[2]) / 2
    shap_med_high = (shap_sorted[3] + shap_sorted[4]) / 2
    ks_sorted = np.sort(df["KS_Statistic"].to_numpy())
    ks_low_med = (ks_sorted[1] + ks_sorted[2]) / 2
    ks_med_high = (ks_sorted[3] + ks_sorted[4]) / 2
    print(
        "[Fig11 breakpoints] "
        f"KS low/medium={ks_low_med:.6g}, KS medium/high={ks_med_high:.6g}; "
        f"SHAP low/medium={shap_low_med:.6g}, SHAP medium/high={shap_med_high:.6g}"
    )

    xmin, xmax = 0.135, 0.335
    ymin, ymax = 0.08, 0.72
    fig, ax = plt.subplots(figsize=(DOUBLE, 4.30))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.14, top=0.95)
    ax.axvspan(xmin, ks_low_med, color="#EAF4EF", alpha=0.42, zorder=0)
    ax.axvspan(ks_low_med, ks_med_high, color="#FFF7E6", alpha=0.42, zorder=0)
    ax.axvspan(ks_med_high, xmax, color="#FBE9E8", alpha=0.42, zorder=0)
    for x in [ks_low_med, ks_med_high]:
        ax.axvline(x, color="#B7B7B7", linestyle=(0, (3, 3)), linewidth=0.72)
    shift_colors = {
        "Low shift": "#2C7FB8",
        "Medium shift": "#D98C20",
        "High shift": "#C84C4C",
    }
    shift_markers = {
        "Low shift": "o",
        "Medium shift": "^",
        "High shift": "s",
    }

    def shift_category(value: float) -> str:
        if value <= ks_low_med:
            return "Low shift"
        if value <= ks_med_high:
            return "Medium shift"
        return "High shift"

    offsets = {
        "x5": (9, 5),
        "x7": (12, 10),
        "x1": (10, 7),
        "x2": (10, -3),
        "x6": (10, -9),
        "x8": (12, -9),
        "x3": (-12, 9),
        "x4": (-13, -1),
    }

    for feat, row in df.iterrows():
        category = shift_category(float(row["KS_Statistic"]))
        color = shift_colors[category]
        ax.scatter(
            row["KS_Statistic"],
            row[shap_col],
            s=68,
            marker=shift_markers[category],
            color=color,
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
            alpha=0.94,
        )
        dx, dy = offsets.get(feat, (7, 0))
        ax.annotate(
            feature_label(feat),
            (row["KS_Statistic"], row[shap_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=FS_ANNOT,
        )
    ax.set_xlabel("Training-external KS statistic")
    ax.set_ylabel(rf"Mean(|SHAP|) of {external_model}")
    ax.text(0.190, 0.705, "Stable", color="#666666", fontsize=7.0, ha="left", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.45, "pad": 0.8})
    ax.text(0.286, 0.705, "High shift", color="#666666", fontsize=7.0, ha="left", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.45, "pad": 0.8})
    ax.text((ks_low_med + ks_med_high) / 2, 0.705, "Review", color="#666666", fontsize=6.8, ha="center", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.45, "pad": 0.8})
    box_axes(ax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=shift_markers[label],
            color="none",
            markerfacecolor=shift_colors[label],
            markeredgecolor="white",
            markersize=6.2,
            label=label,
        )
        for label in ["Low shift", "Medium shift", "High shift"]
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.89),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        fontsize=7.8,
        borderpad=0.2,
        labelspacing=0.35,
        handletextpad=0.45,
    )
    save_fig(fig, "Fig8_joint_shap_ks_risk_map", dirs)
    plt.close(fig)
    qa["Fig11"] = {
        "source": ["SHAP_Generalization_Contrast.csv", "Dataset_DomainShift_KS.csv", "Table_4-9_DomainShift_CoreFeatures_Expanded.csv"],
        "values": df.to_dict(),
        "external_model": external_model,
        "roles": role_models,
        "breakpoints": {
            "ks_low_medium": ks_low_med,
            "ks_medium_high": ks_med_high,
            "shap_low_medium": shap_low_med,
            "shap_medium_high": shap_med_high,
        },
    }


def supp_panel_label(ax: plt.Axes, label: str) -> None:
    ax.set_title(label, loc="left", fontweight="bold", pad=5)


def supp_figure_s1(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    raw = read_csv(source_dir, "FinalEval_Test_External_Raw.csv")
    summary = read_csv(source_dir, "FinalEval_Test_External_Summary.csv")
    external_best = str(summary.loc[summary["Val_F1_Macro_mean"].idxmax(), "Model"])
    lightgbm_rows = summary[summary["Algorithm"].eq("LightGBM")].sort_values("Val_F1_Macro_mean", ascending=False)
    lightgbm_best = str(lightgbm_rows.iloc[0]["Model"]) if len(lightgbm_rows) else external_best
    models = list(dict.fromkeys([lightgbm_best, external_best]))
    sub = raw[raw["Model"].isin(models)].copy()
    wide = sub.pivot_table(index="Run_Seed", columns="Model", values="Val_F1_Macro", aggfunc="first").dropna()
    left = wide[models[0]].to_numpy(float)
    right = wide[models[-1]].to_numpy(float)
    # Sign convention matches the manuscript: external best (XGBoost-SSA) minus LightGBM-Default.
    diff = right - left

    rng = np.random.default_rng(20260608)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10000)])
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), constrained_layout=True)
    ax = axes[0]
    bp = ax.boxplot(
        [left, right],
        patch_artist=True,
        widths=0.38,
        tick_labels=models,
        medianprops={"color": "#D55E00", "linewidth": 1.0},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 2.6, "markerfacecolor": "#777777", "markeredgewidth": 0},
    )
    box_colors = [
        ALGO_COLORS.get(models[0].split("-")[0], "#0072B2"),
        ALGO_COLORS.get(models[-1].split("-")[0], "#D55E00"),
    ]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
    for i, (vals, color) in enumerate(zip([left, right], box_colors), start=1):
        jitter = rng.normal(i, 0.025, size=len(vals))
        ax.scatter(jitter, vals, s=6, color=color, alpha=0.42, linewidth=0, zorder=3)
    supp_panel_label(ax, "(a)")
    ax.set_ylabel(r"External macro-$F_{1}$")
    box_axes(ax, "y")

    ax = axes[1]
    ax.axhline(0, color="#888888", linewidth=0.75)
    display_diff = np.sort(diff)
    display_colors = np.where(display_diff >= 0, ALGO_COLORS["XGBoost"], ALGO_COLORS["LightGBM"])
    ax.scatter(np.arange(1, len(display_diff) + 1), display_diff, s=15, color=display_colors, edgecolor="white", linewidth=0.30, zorder=3)
    supp_panel_label(ax, "(b)")
    ax.set_xlabel("Paired run index (sorted)")
    ax.set_ylabel("Paired difference in external macro-F1")
    box_axes(ax, "y")

    ax = axes[2]
    ax.hist(boot, bins=42, density=False, color="#9ECAE1", edgecolor="white", linewidth=0.35)
    ax.axvline(0, color="#8A8A8A", linewidth=0.85, linestyle=(0, (3, 3)), zorder=1)
    ax.axvline(diff.mean(), color="#3B6EA8", linewidth=1.1)
    ax.axvline(ci_low, color="#D95F02", linewidth=0.9, linestyle="--")
    ax.axvline(ci_high, color="#D95F02", linewidth=0.9, linestyle="--")
    ci_text = f"95% CI [{ci_low:.4f}, {ci_high:.4f}]".replace("-", "−")
    supp_panel_label(ax, "(c)")
    ax.text(
        0.145,
        1.028,
        f"Mean Δ = {diff.mean():.4f}; {ci_text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        clip_on=False,
    )
    ax.set_xlabel("Bootstrap mean difference")
    ax.set_ylabel("Bootstrap frequency")
    box_axes(ax)

    save_fig(fig, "FigS2_bootstrap_paired_seed_stability", dirs)
    plt.close(fig)
    qa["FigS2"] = {"source": "FinalEval_Test_External_Raw.csv", "n_paired_seeds": int(len(diff)), "mean_diff": float(diff.mean())}


def supp_qq_panel(ax: plt.Axes, values: np.ndarray, label: str) -> None:
    values = np.asarray(values, dtype=float)
    values = np.sort(values[np.isfinite(values)])
    n_total = len(values)
    probs = (np.arange(1, n_total + 1) - 0.5) / n_total
    normal = NormalDist()
    osm = np.array([normal.inv_cdf(float(p)) for p in probs])
    central_values = values
    central_osm = osm
    slope, intercept = np.polyfit(central_osm, central_values, 1)
    ax.scatter(central_osm, central_values, s=7, color="#0072B2", edgecolor="white", linewidth=0.22)
    xs = np.linspace(np.min(central_osm), np.max(central_osm), 100)
    ax.plot(xs, slope * xs + intercept, color="#777777", linewidth=0.55)
    ax.set_title(label, loc="left", fontsize=8.6, fontweight="semibold", pad=3)
    box_axes(ax)


def supp_figure_s2(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    raw = read_csv(source_dir, "FinalEval_Test_External_Raw.csv")
    residual_columns = ["Test_F1_Macro", "Val_F1_Macro", "Generalization_Gap"]
    residuals: dict[str, np.ndarray] = {}
    for column in residual_columns:
        current = raw[["Algorithm", "Optimizer", column]].dropna().copy()
        # Residuals from the full two-factor model (Algorithm × Optimizer) are
        # the deviations around each algorithm/optimizer cell mean.
        current["residual"] = current[column] - current.groupby(["Algorithm", "Optimizer"])[column].transform("mean")
        sd = float(current["residual"].std(ddof=1))
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"Cannot standardize residuals for {column}")
        residuals[column] = (current["residual"] / sd).to_numpy(float)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.35))
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.20, top=0.87, wspace=0.27)
    supp_qq_panel(axes[0], residuals["Test_F1_Macro"], "(a) Internal macro-F1")
    supp_qq_panel(axes[1], residuals["Val_F1_Macro"], "(b) External macro-F1")
    supp_qq_panel(axes[2], residuals["Generalization_Gap"], "(c) Generalization gap")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.text(0.5, 0.06, "Theoretical normal quantile", ha="center", va="center", fontsize=8.6)
    fig.text(0.025, 0.51, "Standardized residual quantile", ha="center", va="center", rotation=90, fontsize=8.6)
    save_fig(fig, "FigS3_repeated_macro_f1_qq", dirs)
    plt.close(fig)
    qa["FigS3"] = {
        "source": "FinalEval_Test_External_Raw.csv",
        "residual_model": "Full two-factor Algorithm × Optimizer cell-mean model",
        "display": "All standardized residual quantiles are shown",
        "rows_per_panel": {column: int(len(values)) for column, values in residuals.items()},
    }


def supp_figure_s3(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    arr = np.load(data_file(source_dir, "feature_arrays.npz"), allow_pickle=True)
    feature_names = list(arr["feature_names"])
    k_idx = feature_names.index("x1")
    train = np.asarray(arr["X_train_raw"][:, k_idx], dtype=float)
    external = np.asarray(arr["X_val_raw"][:, k_idx], dtype=float)
    log_train = np.log10(log_ready(train) + 1.0)
    log_external = np.log10(log_ready(external) + 1.0)
    combined = np.r_[log_train, log_external]
    pooled_quantiles = {q: float(np.quantile(log_external, q)) for q in [0.95, 0.99]}

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.05), gridspec_kw={"width_ratios": [1.65, 0.75]})
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.88, wspace=0.22)
    ax = axes[0]
    bins = np.linspace(float(np.nanmin(combined)), float(np.nanmax(combined)), 38)
    ax.hist(log_train, bins=bins, density=True, histtype="step", color="#2C7FB8", linewidth=1.15, label="Training")
    ax.hist(log_external, bins=bins, density=True, histtype="step", color="#D95F02", linewidth=1.15, label="External validation")
    ax.axvline(float(np.median(log_train)), color="#2C7FB8", linewidth=0.85, linestyle="-.", label="Training median")
    ax.axvline(float(np.median(log_external)), color="#D95F02", linewidth=0.85, linestyle="-.", label="External median")
    for q, value in pooled_quantiles.items():
        ax.axvline(value, color="#707070", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
        ax.text(value, 0.94, f"External P{int(q * 100)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.0, color="#555555")
    supp_panel_label(ax, "(a)")
    ax.set_xlabel(r"$\log_{10}(\mathrm{K}^{+} + 1)$")
    ax.set_ylabel("Density")
    box_axes(ax)

    ax = axes[1]
    bp = ax.boxplot(
        [log_train, log_external],
        vert=True,
        patch_artist=True,
        widths=0.45,
        tick_labels=["Training", "External\nvalidation"],
        showfliers=False,
        medianprops={"color": "#333333", "linewidth": 0.9},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, color in zip(bp["boxes"], ["#9ECAE1", "#FDD0A2"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for x, vals, color in [(1, log_train, "#2C7FB8"), (2, log_external, "#D95F02")]:
        high = vals[vals >= np.nanquantile(vals, 0.90)]
        jitter = np.random.default_rng(20260608 + x).normal(x, 0.060, len(high))
        ax.scatter(jitter, high, s=3.6, color=color, alpha=0.30, edgecolor="white", linewidth=0.18, zorder=3)
    supp_panel_label(ax, "(b)")
    ax.set_ylabel(r"$\log_{10}(\mathrm{K}^{+} + 1)$")
    box_axes(ax, "y")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.004, 0.992),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.84,
        ncol=1,
        fontsize=5.9,
        columnspacing=0.58,
        handletextpad=0.32,
        handlelength=1.15,
    )

    save_fig(fig, "FigS4_k_distribution_high_value_tail", dirs)
    plt.close(fig)
    qa["FigS4"] = {
        "source": "feature_arrays.npz",
        "transform": "log10(K+ + 1)",
        "pooled_log_quantile_markers": {f"P{int(q * 100)}": value for q, value in pooled_quantiles.items()},
    }


def supp_figure_s4(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    df = read_csv(source_dir, "SHAP_Generalization_Contrast.csv")
    df = df[df["Feature"].isin(FEATURES)].copy()
    df["Rank"] = df.groupby(["Role", "Model"])["MeanAbsSHAP_ExternalVal"].rank(ascending=False, method="first").astype(int)
    role_order = [r for r in ["internal_test_best", "external_val_best", "largest_test_to_val_drop"] if r in set(df["Role"])]
    role_models = df.groupby("Role")["Model"].first().to_dict()
    rank_pivot = df.pivot_table(index="Feature", columns="Role", values="Rank", aggfunc="first").reindex(FEATURES)

    role_labels = {
        "internal_test_best": "RF-Default",
        "external_val_best": "XGB-SSA",
        "largest_test_to_val_drop": "SVM-Grid",
    }
    rank_pivot = rank_pivot.assign(_mean_rank=rank_pivot[role_order].mean(axis=1)).sort_values("_mean_rank").drop(columns="_mean_rank")
    rank_display = rank_pivot.rename(index=feature_label, columns=role_labels)
    fig, ax = plt.subplots(figsize=(5.15, 3.30))
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.17, top=0.88)
    draw_heatmap(ax, rank_display, cmap=PURPLE_CMAP.reversed(), vmin=1, vmax=8, fmt="{:.0f}", xrotation=0, text_size=8.0)
    ax.set_title("SHAP importance rank (1 = most important)", loc="left", fontsize=8.0, fontweight="normal", pad=4)
    ax.set_xlabel("")
    ax.set_ylabel("Feature", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8.0)
    save_fig(fig, "FigS5_shap_rank_migration_roles", dirs)
    plt.close(fig)
    qa["FigS5"] = {"source": "SHAP_Generalization_Contrast.csv", "roles": role_order}


def supp_figure_s6_target_mine_adaptation(source_dir: Path, dirs: dict[str, Path], qa: dict) -> None:
    summary = read_csv(source_dir, "LocalCalibration_Summary.csv")
    per_run = read_csv(source_dir, "LocalCalibration_PerRun.csv")
    try:
        sweep = read_csv(source_dir, "LocalCalibration_SizeSweep_Summary.csv")
    except FileNotFoundError:
        sweep_detail = read_csv(source_dir, "LocalCalibration_SizeSweep.csv")
        sweep = (
            sweep_detail.groupby(["Mine", "adaptation_n_requested"], as_index=False)
            .agg(
                adapted_macroF1p_mean=("adapted_macroF1p", "mean"),
                adapted_macroF1p_sd=("adapted_macroF1p", "std"),
                n_eff=("adapted_macroF1p", "size"),
            )
        )
        sweep["adapted_macroF1p_sd"] = sweep["adapted_macroF1p_sd"].fillna(0.0)
    required_summary = {
        "Mine",
        "base_macroF1p",
        "base_macroF1p_sd",
        "adapted_macroF1p",
        "adapted_macroF1p_sd",
        "delta_macroF1p",
        "improved_n_macroF1p",
    }
    required_sweep = {"Mine", "adaptation_n_requested", "adapted_macroF1p_mean", "adapted_macroF1p_sd", "n_eff"}
    missing_summary = required_summary.difference(summary.columns)
    missing_sweep = required_sweep.difference(sweep.columns)
    if missing_summary or missing_sweep:
        raise KeyError(
            "Missing LocalCalibration columns: "
            f"summary={sorted(missing_summary)}, sweep={sorted(missing_sweep)}"
        )

    mine_order = summary["Mine"].tolist()
    mine_colors = {"Xiqu": "#0072B2", "Tunlan": "#D55E00"}
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE, 78 * MM),
        gridspec_kw={"width_ratios": [0.96, 1.04], "wspace": 0.32},
        constrained_layout=False,
    )

    ax = axes[0]
    baseline_color = "#B7B7B7"
    recal_color = "#2B7DBC"
    for mine_idx, mine in enumerate(mine_order):
        runs = per_run[per_run["Mine"].eq(mine)]
        x_base, x_adapt = mine_idx - 0.15, mine_idx + 0.15
        for _, run in runs.iterrows():
            ax.plot([x_base, x_adapt], [run["base_macroF1p"], run["adapted_macroF1p"]], color="#C9D0D5", linewidth=0.55, alpha=0.20, zorder=1)
        row = summary[summary["Mine"].eq(mine)].iloc[0]
        ax.errorbar(x_base, row["base_macroF1p"], yerr=row["base_macroF1p_sd"], fmt="o", color=baseline_color, markeredgecolor="white", markersize=6.0, capsize=2.2, elinewidth=0.8, label="Baseline mean ± SD" if mine_idx == 0 else None, zorder=4)
        ax.errorbar(x_adapt, row["adapted_macroF1p"], yerr=row["adapted_macroF1p_sd"], fmt="o", color=recal_color, markeredgecolor="white", markersize=6.0, capsize=2.2, elinewidth=0.8, label="Adapted mean ± SD" if mine_idx == 0 else None, zorder=4)
        ax.text(
            mine_idx,
            0.822,
            f"mean Δ = {row['delta_macroF1p']:+.3f}\n{int(row['improved_n_macroF1p'])}/30 improved",
            ha="center",
            va="top",
            fontsize=6.8,
            linespacing=1.05,
        )
    ax.set_xticks(np.arange(len(mine_order)))
    ax.set_xticklabels(mine_order)
    ax.set_ylabel("Present-class macro-F1")
    ax.set_ylim(0.30, 0.85)
    panel_tag(ax, "(a)")
    ax.text(0.09, 1.025, "Held-out target-mine performance", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2, fontweight="semibold")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", frameon=False, fontsize=6.2, ncol=2, handletextpad=0.35, columnspacing=0.8)
    box_axes(ax, "y")

    ax = axes[1]
    sample_sizes = sorted(sweep["adaptation_n_requested"].unique())
    positions = np.arange(len(sample_sizes))
    for mine in mine_order:
        sw = sweep[sweep["Mine"] == mine].sort_values("adaptation_n_requested")
        if sw.empty:
            continue
        ax.errorbar(
            positions + (-0.030 if mine == mine_order[0] else 0.030),
            sw["adapted_macroF1p_mean"],
            yerr=sw["adapted_macroF1p_sd"],
            marker="o",
            markersize=4.4,
            linewidth=1.35,
            capsize=2.8,
            color=mine_colors.get(mine, "#555555"),
            label=f"{mine} adapted",
        )
        base = float(summary.loc[summary["Mine"] == mine, "base_macroF1p"].iloc[0])
        ax.axhline(base, color=mine_colors.get(mine, "#555555"), linewidth=0.75, linestyle=":", alpha=0.75, label=f"{mine} baseline")
    ax.set_xlabel("Number of local adaptation samples")
    ax.set_ylabel("Present-class macro-F1")
    ax.set_xticks(positions)
    ax.set_xticklabels(sample_sizes)
    ax.set_xlim(-0.35, len(sample_sizes) - 0.65)
    ax.set_ylim(0.40, 0.80)
    panel_tag(ax, "(b)")
    ax.text(0.09, 1.025, "Recovery versus local sample size", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2, fontweight="semibold")
    ax.legend(frameon=False, loc="upper left", fontsize=6.2, ncol=2, columnspacing=0.65, handlelength=1.35)
    box_axes(ax, "y")

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.88)
    save_fig(fig, "FigS6_TargetMineAdaptation", dirs)
    plt.close(fig)
    qa["FigS6"] = {
        "source": [
            "LocalCalibration_Summary.csv",
            "LocalCalibration_SizeSweep.csv",
            "LocalCalibration_SizeSweep_Summary.csv",
        ],
        "mines": mine_order,
        "n_repeats": summary["n_repeats"].astype(int).tolist() if "n_repeats" in summary.columns else None,
        "delta_macroF1p": summary["delta_macroF1p"].round(4).tolist(),
    }


def inspect_supp_outputs(dirs: dict[str, Path], qa: dict) -> None:
    qa["pdf_vector_checks"] = inspect_pdf_directory(dirs["pdf"])
    (dirs["qa"] / "supplementary_figure_checks.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def inspect_outputs(dirs: dict[str, Path], qa: dict) -> None:
    qa["pdf_vector_checks"] = inspect_pdf_directory(dirs["pdf"])
    qa["panel_pdf_vector_checks"] = inspect_pdf_directory(
        dirs["panels_pdf"], include_drawings=False
    )
    (dirs["qa"] / "figure_data_and_vector_checks.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=default_source_dir())
    parser.add_argument("--out_dir", type=Path, default=default_out_dir())
    parser.add_argument("--supp_out_dir", type=Path, default=default_supp_out_dir())
    parser.add_argument("--which", choices=["all", "main", "supp"], default="all")
    args = parser.parse_args()

    if args.which in {"all", "main"}:
        set_style()
        dirs = make_dirs(args.out_dir)
        qa: dict = {"source_dir": str(args.source_dir), "figures": {}}
        builders = [
            ("Fig3", figure_04_internal_forest),
            ("Fig4", figure_05_external),
            ("Fig5", figure_06_confusion),
            ("Fig6", figure_08_domain_shift),
            ("Fig7", figure_09_shap),
            ("Fig8", figure_11_risk),
        ]
        for label, fn in builders:
            print(f"[figure] {label}")
            fn(args.source_dir, dirs, qa["figures"])
        inspect_outputs(dirs, qa)
        print(f"[done main] {args.out_dir}")

    if args.which in {"all", "supp"}:
        set_supp_style()
        supp_dirs = make_dirs(args.supp_out_dir)
        supp_qa: dict = {"source_dir": str(args.source_dir), "figures": {}}
        supp_builders = [
            ("FigS1", figure_03_convergence),
            ("FigS2", supp_figure_s1),
            ("FigS3", supp_figure_s2),
            ("FigS4", supp_figure_s3),
            ("FigS5", supp_figure_s4),
            ("FigS6", supp_figure_s6_target_mine_adaptation),
        ]
        for label, fn in supp_builders:
            print(f"[supp] {label}")
            fn(args.source_dir, supp_dirs, supp_qa["figures"])
        inspect_supp_outputs(supp_dirs, supp_qa)
        print(f"[done supp] {args.supp_out_dir}")


if __name__ == "__main__":
    main()
