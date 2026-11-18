"""Combine the Nino3/Nino4 skill panels and phase panels into one 9-panel figure."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nc_plot_combined_nino3_nino4_corr_rmse_mae_6panels as skill
import nc_plot_combined_phase_3panels as phase


DPI = skill.DPI
FIG_WIDTH_IN = skill.FIGSIZE[0]

TOP_MARGIN_IN = phase.AX_TOP_MARGIN_IN
PANEL_HEIGHT_IN = phase.AX_HEIGHT_IN
ROW_GAP_IN = 1.1
LEGEND_GAP_IN = 0.80
LEGEND_HEIGHT_IN = 0.72
BOTTOM_MARGIN_IN = 0.35

FIG_HEIGHT_IN = (
    TOP_MARGIN_IN
    + PANEL_HEIGHT_IN * 3
    + ROW_GAP_IN * 2
    + LEGEND_GAP_IN
    + LEGEND_HEIGHT_IN
    + BOTTOM_MARGIN_IN
)
FIGSIZE = (FIG_WIDTH_IN, FIG_HEIGHT_IN)

AX_LEFT = 0.06
AX_RIGHT = 0.985
AX_WSPACE = 0.20
PANEL_LABELS = list("abcdefghi")
EXCLUDED_MODELS = {"JMA", "MetFrance"}


def apply_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.unicode_minus": False,
        "axes.linewidth": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.minor.width": 1.6,
        "ytick.minor.width": 1.6,
    })


def panel_geometry():
    total_w = AX_RIGHT - AX_LEFT
    panel_w = total_w / (3 + AX_WSPACE * 2)
    gap_w = AX_WSPACE * panel_w
    panel_h = PANEL_HEIGHT_IN / FIG_HEIGHT_IN
    xs = [AX_LEFT + col * (panel_w + gap_w) for col in range(3)]

    row_bottoms = []
    for row in range(3):
        row_top_in = FIG_HEIGHT_IN - TOP_MARGIN_IN - row * (PANEL_HEIGHT_IN + ROW_GAP_IN)
        row_bottoms.append((row_top_in - PANEL_HEIGHT_IN) / FIG_HEIGHT_IN)

    return xs, row_bottoms, panel_w, panel_h


def load_phase_data():
    phase_data = phase.parse_phase_results(phase.BASE_RESULTS_FILE)
    phase_data.update(phase.parse_phase_results(phase.RESULTS_FILE))
    phase.validate_phase_data(phase_data)
    return phase_data


def add_bottom_model_legend(ax, colors, linewidths):
    ax.axis("off")
    ncol = 6
    models = [name for name in phase.MODEL_LEGEND_ORDER if name not in EXCLUDED_MODELS]
    labels = phase.row_major_legend_input(models, ncol)
    handles = [
        Line2D([0], [0], color=colors[name], lw=linewidths[name])
        for name in labels
    ]
    legend = ax.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=phase.SKILL_LEGEND_SIZE,
        handlelength=2.2,
        handletextpad=0.7,
        columnspacing=1.5,
        labelspacing=0.75,
        borderaxespad=0.0,
        borderpad=0.6,
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_alpha(1.0)


def main():
    apply_style()

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    xs, row_bottoms, panel_w, panel_h = panel_geometry()

    for i, (source_path, title, metric) in enumerate(skill.PANEL_SPECS):
        row = i // 3
        col = i % 3
        ax = fig.add_axes([xs[col], row_bottoms[row], panel_w, panel_h])
        skill.plot_metric_panel(
            ax,
            source_path,
            title,
            metric,
            PANEL_LABELS[i],
        )

    phase_data = load_phase_data()
    style = phase.load_assignments(
        phase.STYLE_SOURCE,
        {"colors", "linewidths", "plot_order"},
    )
    colors = style["colors"]
    linewidths = style["linewidths"]
    plot_order = [name for name in style["plot_order"] if name not in EXCLUDED_MODELS]

    for j, (metric, title, ylabel, ylim, yticks) in enumerate(phase.PHASE_PANELS):
        ax = fig.add_axes([xs[j], row_bottoms[2], panel_w, panel_h])
        phase.plot_phase_panel(
            ax,
            phase_data,
            metric,
            title,
            ylabel,
            ylim,
            yticks,
            colors,
            linewidths,
            plot_order,
            PANEL_LABELS[6 + j],
        )

    legend_ax = fig.add_axes([
        AX_LEFT,
        BOTTOM_MARGIN_IN / FIG_HEIGHT_IN,
        AX_RIGHT - AX_LEFT,
        LEGEND_HEIGHT_IN / FIG_HEIGHT_IN,
    ])
    add_bottom_model_legend(legend_ax, colors, linewidths)

    fig.savefig("nc_combined_nino3_nino4_corr_rmse_mae_phase_9panels.png",
                dpi=600, bbox_inches="tight")
    fig.savefig("nc_combined_nino3_nino4_corr_rmse_mae_phase_9panels.pdf",
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
