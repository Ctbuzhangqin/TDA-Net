import ast
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import nc_plot_combined_5panels as combined


LEAD_TICKS = [0, 4, 8, 12, 16, 20, 24]

TOP_PANELS = [
    ("nc_plot_nino34_corr.py", "Correlation skill for Niño3.4 prediction", "corr"),
    ("nc_plot_nino34_rmse.py", "RMSE for Niño3.4 prediction", "rmse"),
    ("results_mae.txt", "MAE for Niño3.4 prediction", "mae"),
]


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


def load_metric_payload(source_path, metric):
    if metric == "mae":
        style = combined.load_assignments(
            "nc_plot_nino34_rmse.py",
            {"colors", "linewidths", "plot_order"},
        )
        style["data"] = load_mae_data(source_path)
        return style

    return combined.load_assignments(
        source_path,
        {"data", "colors", "linewidths", "plot_order"},
    )


def plot_metric_panel(ax, source_path, title, metric, panel_label):
    ns = load_metric_payload(source_path, metric)
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
    ax.set_xlabel("Lead time (month)", fontsize=combined.LABEL_SIZE, labelpad=8)
    ax.set_xticks(LEAD_TICKS)
    ax.set_xticks(np.arange(0, 25, 1), minor=True)

    if metric == "corr":
        ax.axhline(y=0.5, color="black", linewidth=2.0, linestyle="--", zorder=2)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Correlation", fontsize=combined.LABEL_SIZE, labelpad=8)
        ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    elif metric == "rmse":
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("RMSE (°C)", fontsize=combined.LABEL_SIZE, labelpad=8)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    elif metric == "mae":
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("MAE (°C)", fontsize=combined.LABEL_SIZE, labelpad=8)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    combined.add_panel_label(ax, panel_label)
    combined.add_panel_title(ax, title, combined.TITLE_SIZE)
    combined.style_skill_axis(ax)

    return colors, linewidths


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

    fig = plt.figure(figsize=combined.FIGSIZE, dpi=combined.DPI)
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 0.02, 0.82, 0.82])

    legend_payload = None
    for i, (source_path, title, metric) in enumerate(TOP_PANELS):
        ax = fig.add_subplot(gs[0, i])
        payload = plot_metric_panel(
            ax,
            source_path,
            title,
            metric,
            combined.PANEL_LABELS[i],
        )
        if legend_payload is None:
            legend_payload = payload

    legend_ax = fig.add_subplot(gs[1, :])
    combined.add_shared_model_legend(legend_ax, *legend_payload)

    for idx, lead in enumerate(combined.ONI_LEADS):
        ax = fig.add_subplot(gs[2 + idx, :])
        combined.plot_oni_panel(ax, lead, combined.PANEL_LABELS[3 + idx])

    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        top=0.955,
        bottom=0.06,
        wspace=0.20,
        hspace=0.4,
    )

    fig.savefig("nc_combined_nino34_rmse_mae_5panels.png", dpi=600, bbox_inches="tight")
    fig.savefig("nc_combined_nino34_rmse_mae_5panels.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
