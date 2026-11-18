# Figure 1.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nc_plot_combined_5panels as combined
import nc_plot_combined_nino34_rmse_mae_5panels as base
import nc_plot_combined_phase_3panels as phase


FIGSIZE = (17.2, 14.5)
# here
PHASE_LEADS = [6, 12]
PANEL_LABELS = list("abcdefghi")

BAR_TITLE_SIZE = combined.TITLE_SIZE
BAR_LABEL_SIZE = combined.LABEL_SIZE
BAR_X_TICK_SIZE = 14
BAR_Y_TICK_SIZE = combined.ONI_TICK_SIZE
BAR_LEGEND_SIZE = combined.ONI_LEGEND_SIZE
BAR_WIDTH = 0.72
LOWER_ONI_WIDTH_FRACTION = 0.52
LOWER_PANEL_GAP = 0.04
IN_PHASE_COLOR = "#e6a157"
NOT_IN_PHASE_COLOR = "#9ecae1"
OUT_OF_PHASE_COLOR = "#d62728"
BAR_EXCLUDED_MODELS = {"JMA", "MetFrance"}

MODEL_LABELS = {
    "TDA-Net": "TDA",
    "ORCA-DL": "ORCA",
    "3D-Geoformer": "3D-Geo",
    "SEAS5": "SEAS5",
    "SINTEX-F2": "SINTEX",
    "CCSM4": "CCSM4",
    "CESM1": "CESM1",
    "NASA-GEOSS2S": "NASA",
    "NCEP-CFSv2": "NCEP",
    "CanSIPS-IC3": "CanIC3",
    "CanSIPS-IC4": "CanIC4",
    "GFDL-SPEAR": "GFDL",
}


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


def load_phase_data():
    phase_data = phase.parse_phase_results(phase.BASE_RESULTS_FILE)
    phase_data.update(phase.parse_phase_results(phase.RESULTS_FILE))
    phase.validate_phase_data(phase_data)
    return phase_data


def value_at_lead(payload, metric, lead):
    for idx, payload_lead in enumerate(payload["lead"]):
        if payload_lead == lead:
            return payload[metric][idx]
    return np.nan


def phase_values_for_lead(phase_data, model_order, lead):
    values = {
        "in_phase": [],
        "not_in_phase": [],
        "out_of_phase": [],
    }
    for model in model_order:
        payload = phase_data.get(model)
        for metric in values:
            values[metric].append(
                value_at_lead(payload, metric, lead) if payload is not None else np.nan
            )
    return {metric: np.asarray(metric_values, dtype=float)
            for metric, metric_values in values.items()}


def style_bar_axis(ax):
    ax.tick_params(axis="x", which="major", direction="in",
                   length=7, width=1.8, labelsize=BAR_X_TICK_SIZE, pad=2)
    ax.tick_params(axis="y", which="major", direction="in",
                   length=7, width=1.8, labelsize=BAR_Y_TICK_SIZE, pad=8)
    ax.grid(True, which="major", axis="y",
            linestyle=":", linewidth=1.7, color="#9e9e9e", zorder=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")


def setup_bar_xaxis(ax, model_order):
    x = np.arange(len(model_order))
    ax.set_xlim(-0.65, len(model_order) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS.get(model, model) for model in model_order],
        rotation=90,
        ha="center",
        va="top",
        fontsize=BAR_X_TICK_SIZE,
    )
    return x


def add_bar_title(ax, title):
    combined.add_panel_title(ax, title, BAR_TITLE_SIZE)


def plot_in_phase_bar(ax, phase_data, model_order, lead, panel_label):
    x = setup_bar_xaxis(ax, model_order)
    values = phase_values_for_lead(phase_data, model_order, lead)["in_phase"]
    valid = np.isfinite(values)

    ax.bar(
        x[valid],
        values[valid],
        width=BAR_WIDTH,
        color=IN_PHASE_COLOR,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    ax.set_ylim(0, 80)
    yticks = [0, 20, 40, 60, 80]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick}%" for tick in yticks])

    combined.add_panel_label(ax, panel_label)
    add_bar_title(ax, f"In-phase (Lead month = {lead})")
    style_bar_axis(ax)


def plot_not_in_phase_bar(ax, phase_data, model_order, lead, panel_label,
                          show_legend=False):
    x = setup_bar_xaxis(ax, model_order)
    values = phase_values_for_lead(phase_data, model_order, lead)
    not_in_phase = values["not_in_phase"]
    out_of_phase = values["out_of_phase"]
    valid = np.isfinite(not_in_phase) & np.isfinite(out_of_phase)

    ax.bar(
        x[valid],
        not_in_phase[valid],
        width=BAR_WIDTH,
        color=NOT_IN_PHASE_COLOR,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
        label="Not-in-phase",
    )
    ax.bar(
        x[valid],
        out_of_phase[valid],
        width=BAR_WIDTH,
        color=OUT_OF_PHASE_COLOR,
        edgecolor="black",
        linewidth=0.45,
        zorder=4,
        label="Out-of-phase",
    )
    ax.set_ylim(0, 80)
    yticks = [0, 20, 40, 60, 80]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick}%" for tick in yticks])

    combined.add_panel_label(ax, panel_label)
    add_bar_title(ax, f"Not-in-phase (Lead month = {lead})")
    style_bar_axis(ax)

    if show_legend:
        handles = [
            Line2D([0], [0], color=NOT_IN_PHASE_COLOR, lw=8, label="Not-in-phase"),
            Line2D([0], [0], color=OUT_OF_PHASE_COLOR, lw=8, label="Out-of-phase"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False,
                  fontsize=BAR_LEGEND_SIZE, handlelength=1.5, borderaxespad=0.2)


def lower_row_axes(fig, gs, row):
    bbox = gs[row, :].get_position(fig)
    gap = LOWER_PANEL_GAP
    oni_w = bbox.width * LOWER_ONI_WIDTH_FRACTION
    bar_w = (bbox.width - oni_w - 2 * gap) / 2
    y0 = bbox.y0
    height = bbox.height
    x0 = bbox.x0

    return (
        fig.add_axes([x0, y0, oni_w, height]),
        fig.add_axes([x0 + oni_w + gap, y0, bar_w, height]),
        fig.add_axes([x0 + oni_w + gap + bar_w + gap, y0, bar_w, height]),
    )


def main():
    apply_style()

    style = combined.load_assignments(
        "nc_plot_nino34_rmse.py",
        {"colors", "linewidths", "plot_order"},
    )
    colors = style["colors"]
    linewidths = style["linewidths"]
    model_order = style["plot_order"]
    bar_model_order = [
        model for model in model_order
        if model not in BAR_EXCLUDED_MODELS
    ]
    phase_data = load_phase_data()

    fig = plt.figure(figsize=FIGSIZE, dpi=combined.DPI)
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[1.0, 0.02, 0.82, 0.82],
        width_ratios=[1.0, 1.0, 1.0],
        left=0.06,
        right=0.985,
        top=0.955,
        bottom=0.095,
        wspace=0.20,
        hspace=0.48,
    )

    for i, (source_path, title, metric) in enumerate(base.TOP_PANELS):
        ax = fig.add_subplot(gs[0, i])
        base.plot_metric_panel(ax, source_path, title, metric, PANEL_LABELS[i])
        for line in list(ax.lines):
            if line.get_label() in BAR_EXCLUDED_MODELS:
                line.remove()

    legend_ax = fig.add_subplot(gs[1, :])
    # Match the original 14-model legend width and spread the six columns evenly.
    original_legend = combined.add_shared_model_legend(legend_ax, colors, linewidths)
    legend_width = original_legend.get_window_extent().transformed(
        legend_ax.transAxes.inverted()
    ).width
    combined.add_shared_model_legend(
        legend_ax, colors, linewidths,
        ncol=6,
        excluded_models=BAR_EXCLUDED_MODELS,
        bbox_to_anchor=(0.5 - legend_width / 2, -0.1, legend_width, 0),
        mode="expand",
    )

    for row_idx, lead in enumerate(PHASE_LEADS):
        row = 2 + row_idx
        oni_label = PANEL_LABELS[3] if lead == 6 else PANEL_LABELS[6]
        in_label = PANEL_LABELS[4] if lead == 6 else PANEL_LABELS[7]
        not_label = PANEL_LABELS[5] if lead == 6 else PANEL_LABELS[8]

        oni_ax, in_ax, not_ax = lower_row_axes(fig, gs, row)
        combined.plot_oni_panel(oni_ax, lead, oni_label)

        plot_in_phase_bar(in_ax, phase_data, bar_model_order, lead, in_label)
        plot_not_in_phase_bar(
            not_ax,
            phase_data,
            bar_model_order,
            lead,
            not_label,
            show_legend=True,
        )


    fig.savefig("nc_combined_nino34_rmse_mae_phase_9panels.png",
                dpi=600, bbox_inches="tight")
    fig.savefig("nc_combined_nino34_rmse_mae_phase_9panels.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print("Saved! nc_combined_nino34_rmse_mae_phase_9panels.png")


if __name__ == "__main__":
    main()
