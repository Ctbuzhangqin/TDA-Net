import ast
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation


# =========================
# Global style controls
# =========================
TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 15
PANEL_LABEL_SIZE = 19
SKILL_LEGEND_SIZE = 15

LEAD_TICKS = [0, 4, 8, 12, 16, 20, 24]

FIGSIZE = (17.2, 10.2)
DPI = 300

PANEL_SPECS = [
    ("nc_plot_nino3_corr.py", "Correlation skill for Niño3 prediction", "corr"),
    ("nc_plot_nino3_rmse.py", "RMSE for Niño3 prediction", "rmse"),
    ("results_mae_nino3.txt", "MAE for Niño3 prediction", "mae"),
    ("nc_plot_nino4_corr.py", "Correlation skill for Niño4 prediction", "corr"),
    ("nc_plot_nino4_rmse.py", "RMSE for Niño4 prediction", "rmse"),
    ("results_mae_nino4.txt", "MAE for Niño4 prediction", "mae"),
]
PANEL_LABELS = list("abcdef")

MODEL_LEGEND_ORDER = [
    "TDA-Net",
    "ORCA-DL",
    "3D-Geoformer",
    "SEAS5",
    "SINTEX-F2",
    "CCSM4",
    "CESM1",
    "NASA-GEOSS2S",
    "NCEP-CFSv2",
    "CanSIPS-IC3",
    "CanSIPS-IC4",
    "GFDL-SPEAR",
]

# Keep the three-column geometry close to nc_plot_combined_rmse_3panels.py.
AX_LEFT = 0.06
AX_RIGHT = 0.985
AX_TOP_MARGIN_IN = 0.55
AX_HEIGHT_IN = 3.75
AX_WSPACE = 0.20
ROW_GAP_IN = 0.95
LEGEND_AXIS = [0.06, 0.015, 0.925, 0.065]


def load_assignments(script_path, wanted_names):
    script_path = Path(script_path)
    tree = ast.parse(script_path.read_text(), filename=str(script_path))
    env = {"np": np}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not any(name in wanted_names for name in target_names):
            continue
        mod = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(mod)
        exec(compile(mod, str(script_path), "exec"), env)

    missing = sorted(set(wanted_names) - set(env))
    if missing:
        raise KeyError(f"{script_path} is missing assignments: {missing}")
    return {name: env[name] for name in wanted_names}


def load_mae_data(results_path):
    data = {}
    current_name = None

    for raw_line in Path(results_path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("["):
            if current_name is None:
                continue
            values = ast.literal_eval(line)
            data[current_name] = (np.arange(1, len(values) + 1), values)
            current_name = None
        else:
            current_name = line

    return data


def load_metric_payload(source_path, metric, style_path):
    if metric == "mae":
        style = load_assignments(
            style_path,
            {"colors", "linewidths", "plot_order"},
        )
        style["data"] = load_mae_data(source_path)
        return style

    return load_assignments(
        source_path,
        {"data", "colors", "linewidths", "plot_order"},
    )


def style_skill_axis(ax):
    ax.tick_params(axis="both", which="major", direction="in",
                   length=7, width=1.8, labelsize=TICK_SIZE, pad=8)
    ax.tick_params(axis="x", which="minor", direction="in", length=4, width=1.5)
    ax.grid(True, which="major", linestyle=":", linewidth=1.7, color="#9e9e9e", zorder=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")


def add_panel_label(ax, label):
    offset = ScaledTranslation(-18 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0, 1.0, label,
        transform=ax.transAxes + offset,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def add_panel_title(ax, title):
    title_offset = ScaledTranslation(18 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0, 1.0, title,
        transform=ax.transAxes + title_offset,
        fontsize=TITLE_SIZE,
        ha="left",
        va="bottom",
    )


def plot_metric_panel(ax, source_path, title, metric, panel_label):
    style_path = "nc_plot_nino3_rmse.py" if "nino3" in source_path else "nc_plot_nino4_rmse.py"
    ns = load_metric_payload(source_path, metric, style_path)
    data = ns["data"]
    colors = ns["colors"]
    linewidths = ns["linewidths"]
    plot_order = ns["plot_order"]

    for name in plot_order:
        if name not in data:
            continue
        x, y = data[name]
        ax.plot(
            x, y,
            color=colors[name],
            linewidth=linewidths[name],
            solid_capstyle="butt",
            label=name,
            zorder=4 if name == "TDA-Net" else 3,
        )

    ax.set_xlim(0, 25)
    ax.set_xlabel("Lead time (month)", fontsize=LABEL_SIZE, labelpad=8)
    ax.set_xticks(LEAD_TICKS)
    ax.set_xticks(np.arange(1, 25, 1), minor=True)

    if metric == "corr":
        ax.axhline(y=0.5, color="black", linewidth=2.0, linestyle="--", zorder=2)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Correlation", fontsize=LABEL_SIZE, labelpad=8)
        ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    elif metric == "rmse":
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("RMSE (°C)", fontsize=LABEL_SIZE, labelpad=8)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    elif metric == "mae":
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("MAE (°C)", fontsize=LABEL_SIZE, labelpad=8)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    add_panel_label(ax, panel_label)
    add_panel_title(ax, title)
    style_skill_axis(ax)

    return colors, linewidths


def row_major_legend_input(labels, ncol):
    """Reorder labels so Matplotlib's column-filled legend displays row-major."""
    nrow = int(np.ceil(len(labels) / ncol))
    ordered = []
    for col in range(ncol):
        for row in range(nrow):
            idx = row * ncol + col
            if idx < len(labels):
                ordered.append(labels[idx])
    return ordered


def add_shared_model_legend(ax, colors, linewidths):
    ax.axis("off")
    ncol = 6
    labels = row_major_legend_input(MODEL_LEGEND_ORDER, ncol)
    handles = [
        Line2D([0], [0], color=colors[name], lw=linewidths[name])
        for name in labels
    ]
    legend = ax.legend(
        handles, labels,
        loc="center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=SKILL_LEGEND_SIZE,
        handlelength=2.2,
        handletextpad=0.7,
        columnspacing=1.5,
        labelspacing=0.75,
        borderaxespad=0.0,
        borderpad=0.6,
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_alpha(1.0)


def add_panel_axes(fig):
    fig_w, fig_h = FIGSIZE
    total_w = AX_RIGHT - AX_LEFT
    ax_w = total_w / (3 + AX_WSPACE * 2)
    gap_w = AX_WSPACE * ax_w
    ax_h = AX_HEIGHT_IN / fig_h
    row_gap = ROW_GAP_IN / fig_h
    top = 1.0 - AX_TOP_MARGIN_IN / fig_h

    axes = []
    for row in range(2):
        ax_top = top - row * (ax_h + row_gap)
        ax_bottom = ax_top - ax_h
        for col in range(3):
            axes.append(fig.add_axes([
                AX_LEFT + col * (ax_w + gap_w),
                ax_bottom,
                ax_w,
                ax_h,
            ]))
    return axes


def main():
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

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    axes = add_panel_axes(fig)

    legend_payload = None
    for i, (source_path, title, metric) in enumerate(PANEL_SPECS):
        payload = plot_metric_panel(
            axes[i],
            source_path,
            title,
            metric,
            PANEL_LABELS[i],
        )
        if legend_payload is None:
            legend_payload = payload

    legend_ax = fig.add_axes(LEGEND_AXIS)
    add_shared_model_legend(legend_ax, *legend_payload)

    fig.savefig("nc_combined_nino3_nino4_corr_rmse_mae_6panels.png",
                dpi=600, bbox_inches="tight")
    fig.savefig("nc_combined_nino3_nino4_corr_rmse_mae_6panels.pdf",
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
