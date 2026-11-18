"""Plot Niño3.4 corr/RMSE/MAE for three evaluation periods (exfig11 style).

Rows 1-3 read the 1991-2023, 1991-2007, and 2008-2023 blocks in
corr_rmse_mae_1991_2023.txt, including the updated 3D-Geoformer metrics.
The three lists under each model are corr, RMSE, and MAE, respectively.
JMA and MetFrance are omitted.
Run with: python exfig12.py (saves exfig12.png and exfig12.pdf).
"""

import ast
import os
from pathlib import Path
import re
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import exfig1 as template


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "corr_rmse_mae_1991_2023.txt"
EVAL_PERIODS = ("1991-2023", "1991-2007", "2008-2023")
METRICS = ("corr", "rmse", "mae")
EXCLUDED_MODELS = {"JMA", "MetFrance"}
MODEL_ALIASES = {
    "SINTEXF2": "SINTEX-F2",
    "GEOSS2S": "NASA-GEOSS2S",
    "SPEAR": "GFDL-SPEAR",
}
PANEL_TITLES = {
    "corr": "Correlation skill for Niño3.4",
    "rmse": "RMSE for Niño3.4",
    "mae": "MAE for Niño3.4",
}
Y_AXIS = {
    "corr": ("Correlation", (0, 1.0), [0.1, 0.3, 0.5, 0.7, 0.9]),
    "rmse": ("RMSE (°C)", (0, 1.25), [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]),
    "mae": ("MAE (°C)", (0, 1.0), [0.2, 0.4, 0.6, 0.8, 1.0]),
}
PANEL_LABELS = list("abcdefghi")

skill = template.skill
DPI = template.DPI
FIG_WIDTH_IN = template.FIG_WIDTH_IN
PANEL_HEIGHT_IN = template.PANEL_HEIGHT_IN
# Keep the panel size and spacing used by exfig11.py.
TOP_MARGIN_IN = template.TOP_MARGIN_IN + 0.25
ROW_GAP_IN = template.ROW_GAP_IN + 0.20
FIG_HEIGHT_IN = (
    TOP_MARGIN_IN
    + PANEL_HEIGHT_IN * len(EVAL_PERIODS)
    + ROW_GAP_IN * (len(EVAL_PERIODS) - 1)
    + (template.LEGEND_GAP_IN + 0.1)
    + template.LEGEND_HEIGHT_IN
    + template.BOTTOM_MARGIN_IN
)
FIGSIZE = (FIG_WIDTH_IN, FIG_HEIGHT_IN)


def parse_period_metrics(path):
    """Read the three requested text-file periods, keeping corr/rmse/mae order."""
    blocks = {}
    period = model = None
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"\d{4}-\d{4}", line):
            period, model = line, None
            if period in blocks:
                raise ValueError(f"{path}:{line_number}: duplicate period {period}")
            blocks[period] = {}
        elif period is not None and period not in EVAL_PERIODS:
            continue
        elif line.startswith("["):
            if period is None or model is None:
                raise ValueError(f"{path}:{line_number}: metric list has no period/model")
            values = np.asarray(ast.literal_eval(line), dtype=float)
            if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"{path}:{line_number}: expected a nonempty, finite metric list")
            blocks[period][model].append(values)
        else:
            if period is None:
                raise ValueError(f"{path}:{line_number}: model appears before a period")
            model = MODEL_ALIASES.get(line, line)
            if model in blocks[period]:
                raise ValueError(f"{path}:{line_number}: duplicate model {model} in {period}")
            blocks[period][model] = []

    data = {}
    empty_blocks = []
    for period, models in blocks.items():
        if period not in EVAL_PERIODS:
            continue
        data[period] = {}
        for model, arrays in models.items():
            if model in EXCLUDED_MODELS:
                continue
            if not arrays:
                empty_blocks.append(f"{model} ({period})")
                continue
            if len(arrays) != len(METRICS):
                raise ValueError(f"{period}/{model}: expected corr, rmse, mae; got {len(arrays)} lists")
            if len({len(values) for values in arrays}) != 1:
                raise ValueError(f"{period}/{model}: corr/rmse/mae lead counts differ")
            data[period][model] = {
                metric: (np.arange(1, len(values) + 1), values)
                for metric, values in zip(METRICS, arrays)
            }
    if empty_blocks:
        warnings.warn("No metric lists; skipping " + ", ".join(empty_blocks), stacklevel=2)
    return data


def load_data():
    data = parse_period_metrics(RESULTS_FILE)
    style = skill.load_assignments(
        BASE_DIR / "nc_plot_nino34_rmse.py", {"colors", "linewidths", "plot_order"}
    )
    style["plot_order"] = [name for name in style["plot_order"] if name not in EXCLUDED_MODELS]
    for period in EVAL_PERIODS:
        if not data.get(period):
            raise ValueError(f"Missing evaluation data for {period}")
        for model, payload in data[period].items():
            if model not in style["plot_order"]:
                raise ValueError(f"No plotting style for model {model}")
            for metric in METRICS:
                if metric not in payload:
                    raise ValueError(f"Missing {metric} for {period}/{model}")
                leads, values = payload[metric]
                if len(leads) != len(values):
                    raise ValueError(f"Lead/value lengths differ for {period}/{model}/{metric}")
    return data, style


def panel_geometry():
    total_w = template.AX_RIGHT - template.AX_LEFT
    panel_w = total_w / (3 + template.AX_WSPACE * 2)
    gap_w = template.AX_WSPACE * panel_w
    xs = [template.AX_LEFT + col * (panel_w + gap_w) for col in range(3)]
    row_bottoms = [
        (FIG_HEIGHT_IN - TOP_MARGIN_IN - row * (PANEL_HEIGHT_IN + ROW_GAP_IN)
         - PANEL_HEIGHT_IN) / FIG_HEIGHT_IN
        for row in range(len(EVAL_PERIODS))
    ]
    return xs, row_bottoms, panel_w, PANEL_HEIGHT_IN / FIG_HEIGHT_IN


def plot_metric_panel(ax, data, style, period, metric, panel_label):
    for model in style["plot_order"]:
        if model not in data:
            continue
        leads, values = data[model][metric]
        ax.plot(
            leads, values, color=style["colors"][model],
            linewidth=style["linewidths"][model], solid_capstyle="butt",
            label=model, zorder=4 if model == "TDA-Net" else 3,
        )
    ax.set_xlim(0, 25)
    ax.set_xlabel("Lead time (month)", fontsize=skill.LABEL_SIZE, labelpad=8)
    ax.set_xticks(skill.LEAD_TICKS)
    ax.set_xticks(np.arange(1, 25), minor=True)
    ylabel, ylim, yticks = Y_AXIS[metric]
    ax.set_ylabel(ylabel, fontsize=skill.LABEL_SIZE, labelpad=8)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    if metric == "corr":
        ax.axhline(0.5, color="black", linewidth=2.0, linestyle="--", zorder=2)
    skill.add_panel_label(ax, panel_label)
    skill.add_panel_title(ax, f"{PANEL_TITLES[metric]} ({period})")
    skill.style_skill_axis(ax)


def add_bottom_model_legend(ax, data, style):
    ax.axis("off")
    models = [name for name in style["plot_order"]
              if any(name in data[period] for period in EVAL_PERIODS)]
    labels = skill.row_major_legend_input(models, ncol=6)
    handles = [Line2D([0], [0], color=style["colors"][name],
                      lw=style["linewidths"][name]) for name in labels]
    legend = ax.legend(
        handles, labels, loc="center", bbox_to_anchor=(0.5, 0.5), ncol=6,
        frameon=True, fancybox=False, edgecolor="black",
        fontsize=skill.SKILL_LEGEND_SIZE, handlelength=2.2, handletextpad=0.7,
        columnspacing=1.5, labelspacing=0.75, borderaxespad=0.0, borderpad=0.6,
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_alpha(1.0)


def build_figure(data, style):
    template.apply_style()
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    xs, row_bottoms, panel_w, panel_h = panel_geometry()
    for row, period in enumerate(EVAL_PERIODS):
        for col, metric in enumerate(METRICS):
            ax = fig.add_axes([xs[col], row_bottoms[row], panel_w, panel_h])
            plot_metric_panel(ax, data[period], style, period, metric, PANEL_LABELS[row * 3 + col])
    legend_ax = fig.add_axes([
        template.AX_LEFT, template.BOTTOM_MARGIN_IN / FIG_HEIGHT_IN,
        template.AX_RIGHT - template.AX_LEFT, template.LEGEND_HEIGHT_IN / FIG_HEIGHT_IN,
    ])
    add_bottom_model_legend(legend_ax, data, style)
    return fig


def main():
    data, style = load_data()
    fig = build_figure(data, style)
    for suffix in ("png", "pdf"):
        output_path = BASE_DIR / f"exfig12.{suffix}"
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Saved! {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
