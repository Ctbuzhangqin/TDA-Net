import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation

from nc_plot_seasonal_common import MONTH_CYCLE, RESULT_COLUMN_ORDER


TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 15
PANEL_LABEL_SIZE = 19

FIGSIZE = (15.2, 16.85)  # Five rows for nine model panels.
DPI = 300
MAX_LEAD = 24
INIT_MONTHS = MONTH_CYCLE.copy()
TARGET_START_MONTH = "Feb"

INPUT_FILE = "results_seasonal_supp.txt"
CORR_OUTPUT_PREFIX = "nc_seasonal_supp_corr_9panels"
RMSE_OUTPUT_PREFIX = "nc_seasonal_supp_rmse_9panels"

ORIGINAL_FIGSIZE = (15.2, 10.2)
ORIGINAL_LEFT = 0.065
ORIGINAL_RIGHT = 0.985
ORIGINAL_TOP = 0.955
ORIGINAL_BOTTOM = 0.12
ORIGINAL_WSPACE = 0.14
ORIGINAL_HSPACE = 0.42

MODEL_PANELS = [
    ("SEAS5", "SEAS5"),
    ("SINTEX-F2", "SINTEX-F2"),
    ("COLA-RSMAS-CCSM4", "CCSM4"),
    ("COLA-RSMAS-CESM1", "CESM1"),
    ("NASA-GEOSS2S", "NASA-GEOSS2S"),
    ("NCEP-CFSv2", "NCEP-CFSv2"),
    ("CanSIPS-IC3", "CanSIPS-IC3"),
    ("CanSIPS-IC4", "CanSIPS-IC4"),
    ("GFDL-SPEAR", "GFDL-SPEAR"),
]

SEASON_TO_MONTH = {
    "DJF": "Jan",
    "JFM": "Feb",
    "FMA": "Mar",
    "MAM": "Apr",
    "AMJ": "May",
    "MJJ": "Jun",
    "JJA": "Jul",
    "JAS": "Aug",
    "ASO": "Sep",
    "SON": "Oct",
    "OND": "Nov",
    "NDJ": "Dec",
}


def add_panel_label(ax, label):
    offset = ScaledTranslation(-18 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + offset,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def add_panel_title(ax, title):
    title_offset = ScaledTranslation(4 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        title,
        transform=ax.transAxes + title_offset,
        fontsize=TITLE_SIZE,
        ha="left",
        va="bottom",
    )


def style_seasonal_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        length=3,
        width=1.2,
        labelsize=TICK_SIZE,
        pad=3,
    )


def build_target_months():
    n_targets = 12 + MAX_LEAD - 1
    start_idx = MONTH_CYCLE.index(TARGET_START_MONTH)
    target_months = []
    target_years = []
    for i in range(n_targets):
        m_idx = (start_idx + i) % 12
        year = (start_idx + i) // 12
        target_months.append(MONTH_CYCLE[m_idx])
        target_years.append(year)
    return target_months, target_years


def apply_seasonal_axis_labels(ax, n_targets):
    target_months, target_years = build_target_months()
    xtick_pos = np.arange(0, n_targets, 3)
    xtick_labels = [rf"{target_months[i]}$_{target_years[i]}$" for i in xtick_pos]

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=TICK_SIZE)
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(INIT_MONTHS, fontsize=TICK_SIZE)

    ax.set_xlabel("Target month", fontsize=LABEL_SIZE, labelpad=2)
    ax.set_ylabel("Initialization month", fontsize=LABEL_SIZE, labelpad=3)
    ax.set_xlim(-0.5, n_targets - 0.5)
    ax.set_ylim(-0.5, 11.5)
    style_seasonal_axis(ax)


def add_year_separators(ax, n_targets):
    _, target_years = build_target_months()
    for xpos in np.flatnonzero(np.diff(target_years[:n_targets])) + 0.5:
        if xpos <= n_targets - 0.5:
            ax.axvline(
                xpos,
                color="#1e9bd7",
                linestyle="--",
                linewidth=1.5,
            )


def read_metric_sections(path):
    section_names = {
        "Seasonal Corr": "corr",
        "Seasonal RMSE": "rmse",
    }
    data = {"corr": {}, "rmse": {}}
    current_section = None
    current_model = None
    current_lines = []

    def flush_model():
        nonlocal current_model, current_lines
        if current_section is not None and current_model is not None:
            data[current_section][current_model] = "\n".join(current_lines)
        current_model = None
        current_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped in section_names:
                flush_model()
                current_section = section_names[stripped]
                continue
            if current_section is None or set(stripped) == {"-"}:
                continue
            if stripped.startswith("Lead"):
                continue
            if re.match(r"^\d+\s+", stripped):
                if current_model is None:
                    raise ValueError(f"Data row appears before a model name: {stripped}")
                current_lines.append(stripped)
                continue

            flush_model()
            current_model = stripped

    flush_model()
    return data


def parse_skill_by_lead(raw_results):
    values_by_month = {month: {} for month in MONTH_CYCLE}
    parsed_leads = []

    for line in raw_results.strip().splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        parsed_leads.append(int(parts[0]))

    lead_offset = 1 if parsed_leads and min(parsed_leads) == 0 else 0

    for line in raw_results.strip().splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        lead = int(parts[0]) + lead_offset
        for season, value in zip(RESULT_COLUMN_ORDER, parts[1:]):
            values_by_month[SEASON_TO_MONTH[season]][lead] = float(value)

    skill = np.full((len(INIT_MONTHS), MAX_LEAD), np.nan)
    for row_i, month in enumerate(INIT_MONTHS):
        for lead, value in values_by_month[month].items():
            if 1 <= lead <= MAX_LEAD:
                skill[row_i, lead - 1] = value
    return skill


def build_plot_matrix(skill_by_lead):
    n_targets = 12 + MAX_LEAD - 1
    z = np.full((12, n_targets), np.nan)
    for init_i in range(12):
        for lead in range(1, MAX_LEAD + 1):
            target_i = init_i + lead - 1
            if 0 <= target_i < n_targets:
                z[init_i, target_i] = skill_by_lead[init_i, lead - 1]
    return z


def build_hatch_mask(z):
    hatch_mask = np.zeros_like(z, dtype=bool)
    hatch_mask[~np.isnan(z) & (z < 0.5)] = True
    return hatch_mask


def build_corr_colormap():
    corr_bounds = np.linspace(0.3, 0.95, 14)
    cmap = LinearSegmentedColormap.from_list(
        "enso_red",
        ["#fff1ec", "#ff9b7f", "#fb5a49", "#b30017"],
        N=len(corr_bounds) - 1,
    )
    cmap.set_bad("#cfcfcf")
    cmap.set_under("#fff1ec")
    cmap.set_over("#b30017")
    return cmap, BoundaryNorm(corr_bounds, cmap.N)


def build_rmse_colormap():
    colors = [
        "#08306b",
        "#08519c",
        "#2171b5",
        "#4292c6",
        "#6baed6",
        "#9ecae1",
        "#c7e9c0",
        "#ffffbf",
        "#fee08b",
        "#fdae61",
        "#f46d43",
        "#d73027",
        "#a50026",
        "#6b001f",
    ]
    cmap = ListedColormap(colors, name="rmse_low_blue_high_red")
    cmap.set_bad("#cfcfcf")
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    bounds = np.linspace(0.1, 1.5, len(colors) + 1)
    return cmap, BoundaryNorm(bounds, cmap.N)


def plot_corr_panel(ax, tables, source_name, display_name, panel_label, cmap, norm):
    if source_name not in tables:
        raise ValueError(f"Missing corr table: {source_name}")

    z = build_plot_matrix(parse_skill_by_lead(tables[source_name]))
    hatch_mask = build_hatch_mask(z)
    n_targets = z.shape[1]

    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(-0.5, n_targets - 0.5, -0.5, 11.5),
    )

    for r in range(12):
        for c in range(n_targets):
            if hatch_mask[r, c]:
                rect = Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor=(1, 1, 1, 0),
                    edgecolor="0.35",
                    hatch="///",
                    linewidth=0.0,
                )
                ax.add_patch(rect)

    add_year_separators(ax, n_targets)
    apply_seasonal_axis_labels(ax, n_targets)
    add_panel_label(ax, panel_label)
    add_panel_title(ax, f"Niño3.4 correlation skill ({display_name})")
    return im


def plot_rmse_panel(ax, tables, source_name, display_name, panel_label, cmap, norm):
    if source_name not in tables:
        raise ValueError(f"Missing RMSE table: {source_name}")

    z = build_plot_matrix(parse_skill_by_lead(tables[source_name]))
    n_targets = z.shape[1]

    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(-0.5, n_targets - 0.5, -0.5, 11.5),
    )

    add_year_separators(ax, n_targets)
    apply_seasonal_axis_labels(ax, n_targets)
    add_panel_label(ax, panel_label)
    add_panel_title(ax, f"Niño3.4 RMSE ({display_name})")
    return im


def add_column_colorbar(fig, ax, im, label, ticks=None, unit=None):
    pos = ax.get_position()
    fig_w, fig_h = FIGSIZE
    source_w, source_h = ORIGINAL_FIGSIZE
    cax = fig.add_axes(
        [
            pos.x0 + 0.05 * source_w / fig_w,
            pos.y0 - 0.09 * source_h / fig_h,
            pos.width - 0.1 * source_w / fig_w,
            0.018 * source_h / fig_h,
        ]
    )
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    if ticks is not None:
        cbar.set_ticks(ticks)
    cbar.set_label(label, fontsize=LABEL_SIZE, labelpad=3)
    cbar.ax.tick_params(
        axis="x",
        which="major",
        direction="in",
        length=4,
        width=1.2,
        labelsize=TICK_SIZE,
        pad=3,
    )
    cbar.outline.set_linewidth(1.2)
    if unit is not None:
        cbar.ax.text(
            1.06,
            0.5,
            unit,
            transform=cbar.ax.transAxes,
            fontsize=LABEL_SIZE,
            va="center",
            ha="left",
        )


def build_panel_axes(fig, n_panels):
    fig_w, fig_h = FIGSIZE
    source_w, source_h = ORIGINAL_FIGSIZE

    panel_w = (
        (ORIGINAL_RIGHT - ORIGINAL_LEFT)
        * source_w
        / (2 + ORIGINAL_WSPACE)
    )
    panel_h = (
        (ORIGINAL_TOP - ORIGINAL_BOTTOM)
        * source_h
        / (3 + 2 * ORIGINAL_HSPACE)
    )
    gap_x = ORIGINAL_WSPACE * panel_w
    gap_y = ORIGINAL_HSPACE * panel_h
    left_x = ORIGINAL_LEFT * source_w
    right_x = left_x + panel_w + gap_x
    center_x = (fig_w - panel_w) / 2
    top_y = fig_h - (1 - ORIGINAL_TOP) * source_h

    axes = []
    for i in range(n_panels):
        if i == n_panels - 1 and n_panels % 2 == 1:
            row = i // 2
            x = center_x
        else:
            row = i // 2
            x = left_x if i % 2 == 0 else right_x

        y = top_y - (row + 1) * panel_h - row * gap_y
        axes.append(
            fig.add_axes(
                [x / fig_w, y / fig_h, panel_w / fig_w, panel_h / fig_h]
            )
        )
    return axes


def plot_metric_figure(
    tables,
    plot_panel,
    cmap,
    norm,
    output_prefix,
    colorbar_label,
    colorbar_ticks=None,
    colorbar_unit=None,
):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    axes = build_panel_axes(fig, len(MODEL_PANELS))
    panel_labels = list("abcdefghijklmnopqrstuvwxyz")

    im = None
    for i, (source_name, display_name) in enumerate(MODEL_PANELS):
        im = plot_panel(
            axes[i],
            tables,
            source_name,
            display_name,
            panel_labels[i],
            cmap,
            norm,
        )

    add_column_colorbar(
        fig,
        axes[-1],
        im,
        colorbar_label,
        ticks=colorbar_ticks,
        unit=colorbar_unit,
    )

    fig.savefig(f"{output_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "mathtext.default": "regular",
            "axes.unicode_minus": False,
            "axes.linewidth": 1.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.minor.width": 1.6,
            "ytick.minor.width": 1.6,
            "hatch.linewidth": 1.0,
        }
    )

    sections = read_metric_sections(INPUT_FILE)
    corr_cmap, corr_norm = build_corr_colormap()
    rmse_cmap, rmse_norm = build_rmse_colormap()

    plot_metric_figure(
        sections["corr"],
        plot_corr_panel,
        corr_cmap,
        corr_norm,
        CORR_OUTPUT_PREFIX,
        "Correlation",
    )
    plot_metric_figure(
        sections["rmse"],
        plot_rmse_panel,
        rmse_cmap,
        rmse_norm,
        RMSE_OUTPUT_PREFIX,
        "RMSE",
        colorbar_ticks=np.arange(0.2, 1.51, 0.2),
        colorbar_unit=r"°C",
    )


if __name__ == "__main__":
    main()
