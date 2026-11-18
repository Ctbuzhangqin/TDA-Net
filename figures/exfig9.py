import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation

import plot_thetao_so as ts
import plot_uo_vo_zos as uvz


OUTPUT_PNG = "nc_combined_uo_vo_zos_thetao_so.png"
OUTPUT_PDF = "nc_combined_uo_vo_zos_thetao_so.pdf"
EXP_NAME = "tda_ensemble"
DISPLAY_NAME = "TDA-Net"

FIGSIZE = (10, 15.0)
DPI = 300
TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 15
PANEL_LABEL_SIZE = 19


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
    title_offset = ScaledTranslation(18 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        title,
        transform=ax.transAxes + title_offset,
        fontsize=TITLE_SIZE,
        ha="left",
        va="bottom",
    )


def compute_uvz_panels():
    ocean_mask = uvz.load_land_sea_mask(uvz.VAR_MAPPING["zos"][1], uvz.VAR_MAPPING["zos"][2])
    lat_weights = uvz.compute_latitude_weights(ocean_mask)

    true_zos = uvz.get_true_djf_mean("zos", *uvz.VAR_MAPPING["zos"])
    true_uo = uvz.get_true_djf_mean("uo", *uvz.VAR_MAPPING["uo"])
    true_vo = uvz.get_true_djf_mean("vo", *uvz.VAR_MAPPING["vo"])

    pred_zos = uvz.get_pred_djf_mean(EXP_NAME, "zos", uvz.VAR_MAPPING["zos"][1], uvz.VAR_MAPPING["zos"][2])
    pred_uo = uvz.get_pred_djf_mean(EXP_NAME, "uo", uvz.VAR_MAPPING["uo"][1], uvz.VAR_MAPPING["uo"][2])
    pred_vo = uvz.get_pred_djf_mean(EXP_NAME, "vo", uvz.VAR_MAPPING["vo"][1], uvz.VAR_MAPPING["vo"][2])

    true_curr = np.concatenate([true_uo.flatten(), true_vo.flatten()])
    pred_curr = np.concatenate([pred_uo.flatten(), pred_vo.flatten()])
    curr_weights = np.concatenate([lat_weights.flatten(), lat_weights.flatten()])
    corr_zos = uvz.weighted_pearson_corr(pred_zos.flatten(), true_zos.flatten(), lat_weights.flatten())
    corr_curr = uvz.weighted_pearson_corr(pred_curr, true_curr, curr_weights)

    def apply_mask(field):
        return np.where(ocean_mask, field, np.nan)

    return [
        {
            "title": "GODAS",
            "zos": apply_mask(true_zos),
            "uo": apply_mask(true_uo),
            "vo": apply_mask(true_vo),
        },
        {
            "title": (
                f"{DISPLAY_NAME} (lead month = {uvz.LEAD}) | "
                rf"$\mathrm{{Correlation}}_{{\mathrm{{SSH}}}} = ${corr_zos:.2f}, "
                rf"$\mathrm{{Correlation}}_{{\mathrm{{UV}}}} = ${corr_curr:.2f}"
            ),
            "zos": apply_mask(pred_zos),
            "uo": apply_mask(pred_uo),
            "vo": apply_mask(pred_vo),
        },
    ]


def compute_ts_panels():
    true_t = ts.get_true_djf_mean_all_layers("thetao", *ts.VAR_MAPPING["thetao"])
    true_s = ts.get_true_djf_mean_all_layers("so", *ts.VAR_MAPPING["so"])
    true_t_points, true_s_points, true_d_points = ts.extract_valid_ts_points(true_t, true_s)

    pred_t = ts.get_pred_djf_mean_all_layers(EXP_NAME, "thetao", ts.VAR_MAPPING["thetao"][1])
    pred_s = ts.get_pred_djf_mean_all_layers(EXP_NAME, "so", ts.VAR_MAPPING["so"][1])
    pred_t_points, pred_s_points, pred_d_points = ts.extract_valid_ts_points(pred_t, pred_s)

    corr_ts = ts.calc_ts_distribution_corr(
        true_s_points, true_t_points, pred_s_points, pred_t_points
    )
    return [
        {
            "title": "GODAS",
            "t": true_t_points,
            "s": true_s_points,
            "d": true_d_points,
        },
        {
            "title": f"{DISPLAY_NAME} (lead month = {ts.LEAD})",
            "corr": corr_ts,
            "t": pred_t_points,
            "s": pred_s_points,
            "d": pred_d_points,
        },
    ]


def plot_uvz_panel(ax, cfg, vmin, vmax):
    lat = np.linspace(-63.5, 63.5, 128)
    lon = np.linspace(0.5, 359.5, 360)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    skip = (slice(None, None, 5), slice(None, None, 8))

    ax.set_facecolor("lightgray")
    cf = ax.contourf(
        lon_grid,
        lat_grid,
        cfg["zos"],
        levels=30,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        extend="both",
    )
    q = ax.quiver(
        lon_grid[skip],
        lat_grid[skip],
        cfg["uo"][skip],
        cfg["vo"][skip],
        scale=6,
        width=0.003,
        headwidth=4,
        headlength=5,
        color="black",
    )

    ax.set_xlim([0, 360])
    ax.set_ylim([-63.5, 63.5])

    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_yticklabels(["60°S", "30°S", "0°", "30°N", "60°N"])
    ax.tick_params(axis="both", labelsize=TICK_SIZE, direction="in", length=6, width=1.4)
    # ax.set_xlabel("Longitude", fontsize=LABEL_SIZE, labelpad=6)
    # ax.set_ylabel("Latitude", fontsize=LABEL_SIZE, labelpad=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    return cf, q


def plot_ts_panel(ax, cfg):
    subsample = 10
    sc = ax.scatter(
        cfg["s"][::subsample],
        cfg["t"][::subsample],
        c=cfg["d"][::subsample],
        cmap="Spectral",
        s=1,
        alpha=0.5,
        vmin=0,
        vmax=1000,
    )
    ax.set_xlim([30.5, 39.5])
    ax.set_ylim([-2, 32])
    ax.set_xlabel(rf"Salinity (g $\mathrm{{kg}}^{{-1}}$)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Temperature (°C)", fontsize=LABEL_SIZE)
    ax.grid(True, linestyle=":", alpha=0.6)
    if "corr" in cfg:
        ax.text(
            0.97,
            0.03,
            f"Correlation = {cfg['corr']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=TICK_SIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )
    ax.tick_params(axis="both", labelsize=TICK_SIZE, direction="in", length=6, width=1.4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    return sc


def main():
    plt.rcParams.update(
        {
            "font.family": "Arial",
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

    print("Loading GODAS and TDA-Net SSH/current fields...")
    uvz_panels = compute_uvz_panels()
    all_zos = np.concatenate([panel["zos"].ravel() for panel in uvz_panels])
    vmin, vmax = np.nanpercentile(all_zos, [1, 99])

    print("Loading GODAS and TDA-Net T-S fields...")
    ts_panels = compute_ts_panels()

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 1.05],
        hspace=0.20,
        wspace=0.18,
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])

    cf, _ = plot_uvz_panel(ax_a, uvz_panels[0], vmin, vmax)
    add_panel_label(ax_a, "a")
    add_panel_title(ax_a, uvz_panels[0]["title"])
    cf, q = plot_uvz_panel(ax_b, uvz_panels[1], vmin, vmax)
    add_panel_label(ax_b, "b")
    add_panel_title(ax_b, uvz_panels[1]["title"])
    sc = plot_ts_panel(ax_c, ts_panels[0])
    add_panel_label(ax_c, "c")
    add_panel_title(ax_c, ts_panels[0]["title"])
    sc = plot_ts_panel(ax_d, ts_panels[1])
    add_panel_label(ax_d, "d")
    add_panel_title(ax_d, ts_panels[1]["title"])

    fig.subplots_adjust(left=0.07, right=0.92, top=0.965, bottom=0.08)

    a_pos = ax_a.get_position()
    b_pos = ax_b.get_position()
    ab_bottom = b_pos.y0
    ab_top = a_pos.y1
    ab_height = ab_top - ab_bottom
    cbar_ab_height = 0.58 * ab_height
    cbar_ab_bottom = ab_bottom + 0.5 * (ab_height - cbar_ab_height)
    cbar_ab_ax = fig.add_axes([
        b_pos.x1 + 0.024,
        cbar_ab_bottom,
        0.012,
        cbar_ab_height,
    ])
    cbar_ab = fig.colorbar(cf, cax=cbar_ab_ax, orientation="vertical")
    cbar_ab.ax.tick_params(labelsize=TICK_SIZE, direction="in", length=4, width=1.2, pad=3)
    cbar_ab.outline.set_linewidth(1.2)
    cbar_ab.ax.text(
        0.5,
        1.08,
        "m",
        transform=cbar_ab.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=LABEL_SIZE,
    )

    ax_b.quiverkey(
        q,
        X=b_pos.x1 + 0.050,
        Y=cbar_ab_bottom - 0.015,
        U=0.5,
        label=rf"0.5 m $\mathrm{{s}}^{{-1}}$",
        labelpos="S",
        coordinates="figure",
        fontproperties={"size": TICK_SIZE},
    )

    d_pos = ax_d.get_position()
    cbar_cd_ax = fig.add_axes([d_pos.x1 + 0.024, d_pos.y0 + 0.08 * d_pos.height, 0.012, 0.84 * d_pos.height])
    cbar_cd = fig.colorbar(sc, cax=cbar_cd_ax, orientation="vertical")
    cbar_cd.ax.tick_params(labelsize=TICK_SIZE, direction="in", length=4, width=1.2, pad=3)
    cbar_cd.ax.invert_yaxis()
    cbar_cd.outline.set_linewidth(1.2)
    cbar_cd.ax.text(
        0.5,
        1.04,
        "m",
        transform=cbar_cd.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=LABEL_SIZE,
    )

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
