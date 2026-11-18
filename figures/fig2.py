import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation

from nc_plot_seasonal_ablation_diff import read_model_tables
from nc_plot_seasonal_common import (
    MONTH_CYCLE,
    MODEL_TABLES,
    build_hatch_mask,
    build_plot_matrix,
    build_target_months,
    parse_skill_by_lead,
)


TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 15
PANEL_LABEL_SIZE = 19

FIGSIZE = (15.2, 10.2)
DPI = 300
INIT_MONTHS = MONTH_CYCLE.copy()
TARGET_START_MONTH = "Feb"

LEFT_PANELS = [
    ("3D-Geoformer", "a"),
    ("ORCA-DL", "b"),
    ("TDA-Net", "c"),
]

RIGHT_PANELS = [
    ("3D-Geoformer", "d"),
    ("ORCA-DL", "e"),
    ("TDA-Net", "f"),
]

RMSE_FILE = "results_rmse_seasonal.txt"
OUTPUT_PREFIX = "nc_combined_seasonal_6panels"


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


def apply_seasonal_axis_labels(ax, n_targets):
    target_months, target_years = build_target_months(start_month=TARGET_START_MONTH)
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


def add_year_separators(ax):
    _, target_years = build_target_months(start_month=TARGET_START_MONTH)
    for xpos in np.flatnonzero(np.diff(target_years)) + 0.5:
        ax.axvline(
            xpos,
            color="#1e9bd7",
            linestyle="--",
            linewidth=1.5,
        )


def plot_model_panel(ax, model_name, panel_label):
    skill_by_lead = parse_skill_by_lead(MODEL_TABLES[model_name], init_months=INIT_MONTHS)
    z = build_plot_matrix(skill_by_lead)
    hatch_mask = build_hatch_mask(z)
    n_targets = z.shape[1]

    corr_bounds = np.linspace(0.3, 0.95, 14)
    cmap = LinearSegmentedColormap.from_list(
        "enso_red",
        ["#fff1ec", "#ff9b7f", "#fb5a49", "#b30017"],
        N=len(corr_bounds) - 1,
    )
    cmap.set_bad("#cfcfcf")
    cmap.set_under("#fff1ec")
    cmap.set_over("#b30017")
    norm = BoundaryNorm(corr_bounds, cmap.N)

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

    add_year_separators(ax)
    apply_seasonal_axis_labels(ax, n_targets)
    add_panel_label(ax, panel_label)
    add_panel_title(ax, f"Niño3.4 correlation skill ({model_name})")
    return im


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
    bounds = np.linspace(0.1, 1.5, len(colors)+1)
    return cmap, BoundaryNorm(bounds, cmap.N), bounds


def plot_rmse_panel(ax, tables, model_name, panel_label, cmap, norm):
    if model_name not in tables:
        raise ValueError(f"Missing model table: {model_name}")

    rmse_by_lead = parse_skill_by_lead(tables[model_name], init_months=INIT_MONTHS)
    z = build_plot_matrix(rmse_by_lead)
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

    add_year_separators(ax)
    apply_seasonal_axis_labels(ax, n_targets)
    add_panel_label(ax, panel_label)
    add_panel_title(ax, f"Niño3.4 RMSE ({model_name})")
    return im


def add_column_colorbar(fig, ax, im, label, ticks=None, unit=None):
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0+0.05, pos.y0 - 0.090, pos.width - 0.1, 0.018])
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

    rmse_tables = read_model_tables(RMSE_FILE)
    rmse_cmap, rmse_norm, _ = build_rmse_colormap()
    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, dpi=DPI)

    corr_im = None
    for row, (model_name, panel_label) in enumerate(LEFT_PANELS):
        corr_im = plot_model_panel(axes[row, 0], model_name, panel_label)

    rmse_im = None
    for row, (model_name, panel_label) in enumerate(RIGHT_PANELS):
        rmse_im = plot_rmse_panel(
            axes[row, 1], rmse_tables, model_name, panel_label, rmse_cmap, rmse_norm
        )

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.955,
        bottom=0.12,
        wspace=0.14,
        hspace=0.48,
    )

    add_column_colorbar(fig, axes[2, 0], corr_im, "Correlation")
    add_column_colorbar(
        fig,
        axes[2, 1],
        rmse_im,
        "RMSE",
        ticks=np.arange(0.2, 1.51, 0.2),
        unit=r"°C",
    )

    fig.savefig(f"{OUTPUT_PREFIX}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_PREFIX}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
