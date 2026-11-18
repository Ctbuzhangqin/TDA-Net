
from _paths import DATA_ROOT, PREDICTION_ROOT, STAT_ROOT, SST_MEAN_FILE_PATH, SST_STD_FILE_PATH
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.transforms import ScaledTranslation

import nc_plot_d20 as d20
import nc_plot_thetao_tcc as thetao_tcc


TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 15
PANEL_LABEL_SIZE = 19
DPI = 300
FIGSIZE = (17.2, 14.0)
PANEL_LABELS = list("abcdefg")
D20_PRED_EXP = "tda_ensemble"
D20_DISPLAY_NAME = "TDA-Net"


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


def add_panel_title(ax, title, fontsize=TITLE_SIZE):
    title_offset = ScaledTranslation(18 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        title,
        transform=ax.transAxes + title_offset,
        fontsize=fontsize,
        ha="left",
        va="bottom",
    )


def style_spines(ax, linewidth):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
        spine.set_color("black")


def plot_d20_panel(ax, data, title, panel_label):
    time_values = d20.month_axis(d20.PLOT_START_YEAR, d20.PLOT_END_YEAR)
    norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
    ax.imshow(
        np.flip(data, axis=1).T,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        interpolation="bilinear",
        extent=[time_values.min(), time_values.max(), d20.PLOT_LONS.min(), d20.PLOT_LONS.max()],
    )
    ax.set_xlim(d20.PLOT_START_YEAR, d20.PLOT_END_YEAR + 11 / 12)
    ax.set_ylim(d20.PLOT_LONS.min(), d20.PLOT_LONS.max())

    xticks = np.arange(d20.PLOT_START_YEAR, d20.PLOT_END_YEAR + 1, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=TICK_SIZE)

    yticks = np.arange(120, 271, 30)
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [d20.lon_label(y) if i % 2 == 0 else "" for i, y in enumerate(yticks)],
        fontsize=TICK_SIZE,
    )
    ax.tick_params(axis="both", which="major", direction="out", length=9, width=1.6, pad=7)
    style_spines(ax, 1.6)
    add_panel_label(ax, panel_label)
    add_panel_title(ax, title)


def plot_tcc_panel(ax, tcc_masked, lead, panel_label):
    vmin, vmax = -0.2, 1.0
    levels = np.linspace(vmin, vmax, 25)
    cmap = thetao_tcc.build_tcc_colormap()
    ax.set_facecolor("#cfcfcf")
    ax.contourf(
        thetao_tcc.PLOT_LONS,
        thetao_tcc.DEPTHS,
        tcc_masked,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend="both",
    )
    try:
        cs = ax.contour(
            thetao_tcc.PLOT_LONS,
            thetao_tcc.DEPTHS,
            tcc_masked,
            levels=[0.5],
            colors="black",
            linewidths=1.2,
        )
        ax.clabel(cs, fmt={0.5: "0.5"}, fontsize=16, inline=True, inline_spacing=2)
    except ValueError:
        pass

    ax.set_ylim(1000, 10)
    ax.set_xlim(0, 360)
    xticks = [0, 60, 120, 180, 240, 300, 360]
    ax.set_xticks(xticks)
    ax.set_xticklabels([thetao_tcc.lon_label(x) for x in xticks], fontsize=TICK_SIZE)
    ax.set_yticks(thetao_tcc.DEPTHS)
    ax.set_yticklabels(
        [str(d) if d in thetao_tcc.LABELED_DEPTH_TICKS else "" for d in thetao_tcc.DEPTHS],
        fontsize=TICK_SIZE,
    )
    ax.set_ylabel("Depth (m)", fontsize=LABEL_SIZE, labelpad=3)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=1.2,
        labelsize=TICK_SIZE,
        top=False,
        right=False,
    )
    style_spines(ax, 1.2)
    add_panel_label(ax, panel_label)
    add_panel_title(ax, f"TCC (lead month = {lead})")


def main():
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.unicode_minus": False,
        "axes.linewidth": 1.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
    })

    d20.PRED_DIR = d20.Path(str(PREDICTION_ROOT / f"{D20_PRED_EXP}/preds"))
    d20.INIT_TIMES_FILE = d20.Path(str(PREDICTION_ROOT / f"{D20_PRED_EXP}/init_times.json"))

    print("Preparing D20 panels...")
    d20_panels = [("Equatorial mean d20 anomalies from GODAS", d20.get_true_d20_anomaly())]
    for lead in d20.LEADS:
        d20_panels.append(
            (
                f"Equatorial mean d20 anomalies from {D20_DISPLAY_NAME} (lead month = {lead})",
                d20.get_pred_d20_anomaly(lead),
            )
        )

    print("Preparing thetao TCC panels...")
    true_clim = thetao_tcc.get_true_climatology()
    tcc_panels = [(lead, thetao_tcc.calc_tcc_for_lead(true_clim, lead)) for lead in thetao_tcc.LEADS]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(
        5,
        4,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 1.59],
        width_ratios=[1.0, 1.0, 1.0, 0.03],
    )

    d20_axes = []
    for row_idx, (title, data) in enumerate(d20_panels):
        ax = fig.add_subplot(gs[row_idx, :3])
        plot_d20_panel(ax, data, title, PANEL_LABELS[row_idx])
        d20_axes.append(ax)

    for col_idx, (lead, tcc_masked) in enumerate(tcc_panels):
        ax = fig.add_subplot(gs[4, col_idx])
        plot_tcc_panel(ax, tcc_masked, lead, PANEL_LABELS[4 + col_idx])

    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.96,
        bottom=0.065,
        wspace=0.28,
        hspace=0.48,
    )

    d20_slot = gs[:4, 3].get_position(fig)
    tcc_slot = gs[4, 3].get_position(fig)

    d20_cbar_y0 = d20_axes[-1].get_position().y0
    d20_cbar_y1 = d20_axes[0].get_position().y1
    d20_cax = fig.add_axes([d20_slot.x0 - 0.03, d20_cbar_y0, d20_slot.width, d20_cbar_y1 - d20_cbar_y0])
    d20_sm = ScalarMappable(norm=TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20), cmap="RdBu_r")
    d20_sm.set_array([])
    d20_cbar = fig.colorbar(d20_sm, cax=d20_cax, orientation="vertical", extend="both")
    d20_cbar.ax.set_title("m", fontsize=LABEL_SIZE, pad=8)
    d20_cbar.ax.tick_params(labelsize=TICK_SIZE, length=5, width=1.2)
    d20_cbar.outline.set_linewidth(1.2)

    tcc_cax = fig.add_axes([tcc_slot.x0 - 0.03, tcc_slot.y0, tcc_slot.width, tcc_slot.height])
    tcc_sm = ScalarMappable(norm=Normalize(vmin=-0.2, vmax=1.0), cmap=thetao_tcc.build_tcc_colormap())
    tcc_sm.set_array([])
    tcc_ticks = np.arange(-0.2, 1.01, 0.2)
    tcc_cbar = fig.colorbar(tcc_sm, cax=tcc_cax, orientation="vertical", extend="both", ticks=tcc_ticks)
    # tcc_cbar.set_label("TCC", fontsize=LABEL_SIZE, labelpad=8)
    tcc_cbar.ax.tick_params(labelsize=TICK_SIZE, length=5, width=1.2)
    tcc_cbar.outline.set_linewidth(1.2)

    fig.savefig("nc_combined_d20_thetao_tcc.png", dpi=600, bbox_inches="tight")
    fig.savefig("nc_combined_d20_thetao_tcc.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved nc_combined_d20_thetao_tcc.png")
    print("Saved nc_combined_d20_thetao_tcc.pdf")


if __name__ == "__main__":
    main()
