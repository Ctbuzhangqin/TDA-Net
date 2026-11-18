import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.transforms import ScaledTranslation

from nc_plot_seasonal_ablation_diff import read_model_tables
from nc_plot_seasonal_common import (
    MONTH_CYCLE,
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
    ("baseline+SLAM", "a"),
    ("baseline+SLAM+LADR", "b"),
    ("baseline+SLAM+LADR+Overlap", "c"),
]

RIGHT_PANELS = [
    ("baseline+SLAM", "d"),
    ("baseline+SLAM+LADR", "e"),
    ("baseline+SLAM+LADR+Overlap", "f"),
]

CORR_ABLATION_FILE = "results_seasonal_ablation.txt"
ABLATION_FILE = "results_rmse_seasonal_ablation.txt"
BASE_MODEL = "baseline"
OUTPUT_PREFIX = "nc_combined_seasonal_rmse_6panels"


def add_panel_label(ax, label):
    offset = ScaledTranslation(-18 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes+offset,
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
        transform=ax.transAxes+title_offset,
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


def build_rmse_diff_colormap():
    colors = [
        "#053061",
        "#2166ac",
        "#4393c3",
        "#67a9cf",
        "#92c5de",
        "#d1e5f0",
        "#f7f7f7",
        "#f7f7f7",
        "#fddbc7",
        "#f4a582",
        "#d6604d",
        "#b2182b",
        "#67001f",
    ]
    cmap = ListedColormap(colors, name="rmse_diff")
    cmap.set_bad("#cfcfcf")
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    bounds = np.linspace(-0.6, 0.6, len(colors)+1)
    return cmap, BoundaryNorm(bounds, cmap.N), bounds


def plot_correlation_diff_panel(ax, tables, variant_model, panel_label):
    missing = [name for name in [BASE_MODEL, variant_model] if name not in tables]
    if missing:
        raise ValueError(f"Missing model table(s): {', '.join(missing)}")

    baseline = build_plot_matrix(
        parse_skill_by_lead(tables[BASE_MODEL], init_months=INIT_MONTHS)
    )
    variant = build_plot_matrix(
        parse_skill_by_lead(tables[variant_model], init_months=INIT_MONTHS)
    )
    diff = variant - baseline
    n_targets = diff.shape[1]

    colors = [
        "#2400c8",
        "#2526e6",
        "#2d55f0",
        "#3f82ef",
        "#56a9ee",
        "#72c5e8",
        "#9eddeb",
        "#c7eef1",
        "#e7f6f6",
        "#ffffff",
        "#ffffff",
        "#f6d49a",
        "#f5b66e",
        "#f48b5a",
        "#f85f4c",
        "#fb3d3f",
        "#f51f31",
        "#e80e28",
        "#cf082d",
        "#b3002d",
    ]
    cmap = ListedColormap(colors, name="corr_diff")
    cmap.set_bad("#cfcfcf")
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    diff_bounds = np.linspace(-0.25, 0.25, 21)
    norm = BoundaryNorm(diff_bounds, cmap.N)

    im = ax.imshow(
        diff,
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
    add_panel_title(
        ax,
        f"Niño3.4 correlation skill difference ({ablation_short_name(variant_model)})",
    )
    return im


def ablation_short_name(model_name):
    prefix = "baseline+"
    if model_name.startswith(prefix):
        return model_name[len(prefix):]
    return model_name


def plot_ablation_diff_panel(ax, tables, variant_model, panel_label, cmap, norm):
    missing = [name for name in [BASE_MODEL, variant_model] if name not in tables]
    if missing:
        raise ValueError(f"Missing model table(s): {', '.join(missing)}")

    baseline = build_plot_matrix(
        parse_skill_by_lead(tables[BASE_MODEL], init_months=INIT_MONTHS)
    )
    variant = build_plot_matrix(
        parse_skill_by_lead(tables[variant_model], init_months=INIT_MONTHS)
    )
    diff = variant - baseline
    n_targets = diff.shape[1]

    im = ax.imshow(
        diff,
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
    add_panel_title(
        ax,
        f"Niño3.4 RMSE difference ({ablation_short_name(variant_model)})",
    )
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
        }
    )

    corr_ablation_tables = read_model_tables(CORR_ABLATION_FILE)
    ablation_tables = read_model_tables(ABLATION_FILE)
    diff_cmap, diff_norm, _ = build_rmse_diff_colormap()

    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, dpi=DPI)

    corr_diff_im = None
    for row, (variant_model, panel_label) in enumerate(LEFT_PANELS):
        corr_diff_im = plot_correlation_diff_panel(
            axes[row, 0], corr_ablation_tables, variant_model, panel_label
        )

    diff_im = None
    for row, (variant_model, panel_label) in enumerate(RIGHT_PANELS):
        diff_im = plot_ablation_diff_panel(
            axes[row, 1], ablation_tables, variant_model, panel_label, diff_cmap, diff_norm
        )

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.955,
        bottom=0.12,
        wspace=0.14,
        hspace=0.48,
    )

    add_column_colorbar(
        fig,
        axes[2, 0],
        corr_diff_im,
        "Correlation difference",
        ticks=np.arange(-0.25, 0.2501, 0.1),
    )
    add_column_colorbar(
        fig,
        axes[2, 1],
        diff_im,
        "RMSE difference",
        ticks=np.arange(-0.6, 0.6001, 0.2),
        unit=r"°C",
    )

    fig.savefig(f"{OUTPUT_PREFIX}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_PREFIX}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
