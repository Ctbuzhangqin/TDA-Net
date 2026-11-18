import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.transforms import ScaledTranslation


# =========================
# Global style controls
# =========================
TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 15
PANEL_LABEL_SIZE = 19
SKILL_LEGEND_SIZE = 15
ONI_TITLE_SIZE = TITLE_SIZE
ONI_LABEL_SIZE = LABEL_SIZE
ONI_TICK_SIZE = TICK_SIZE
ONI_LEGEND_SIZE = 15

FIGSIZE = (17.2, 14.5)
DPI = 300
START_YEAR = 2011
END_YEAR = 2023
MARK_EVERY = 1

SKILL_SCRIPTS = [
    ("nc_plot_nino34_corr.py", "Correlation skill for Niño3.4 prediction", "corr"),
    ("nc_plot_nino3_corr.py", "Correlation skill for Niño3 prediction", "corr"),
    ("nc_plot_nino4_corr.py", "Correlation skill for Niño4 prediction", "corr"),
]
ONI_LEADS = [6, 12]
PANEL_LABELS = list("abcde")
ONI_WARM_COLOR = "#d62728"
ONI_COLD_COLOR = "#1f77b4"

# A panel's original two-column legend order: left column followed by right column.
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
    "JMA",
    "MetFrance",
]


class RightAlignedMarkerHandler(HandlerLine2D):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        marker = Line2D(
            [xdescent + width * 0.92],
            [ydescent + height / 2],
            linestyle="None",
        )
        self.update_prop(marker, orig_handle, legend)
        marker.set_transform(trans)
        return [marker]


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


def add_panel_title(ax, title, fontsize):
    title_offset = ScaledTranslation(12 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0, 1.0, title,
        transform=ax.transAxes + title_offset,
        fontsize=fontsize,
        ha="left",
        va="bottom",
    )


def plot_skill_panel(ax, script_path, title, metric, panel_label):
    ns = load_assignments(script_path, {"data", "colors", "linewidths", "plot_order"})
    data = ns["data"]
    colors = ns["colors"]
    linewidths = ns["linewidths"]
    plot_order = ns["plot_order"]

    for name in plot_order:
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
    ax.set_xticks([2, 6, 10, 14, 18, 22])
    ax.set_xticks(np.arange(1, 25, 1), minor=True)

    ax.axhline(y=0.5, color="black", linewidth=2.0, linestyle="--", zorder=2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Correlation", fontsize=LABEL_SIZE, labelpad=8)
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])

    add_panel_label(ax, panel_label)
    add_panel_title(ax, title, TITLE_SIZE)
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


def add_shared_model_legend(ax, colors, linewidths, ncol=7, excluded_models=(),
                            bbox_to_anchor=(0.5, -0.1), mode=None):
    ax.axis("off")
    labels = row_major_legend_input(
        [name for name in MODEL_LEGEND_ORDER if name not in excluded_models],
        ncol,
    )
    handles = [
        Line2D([0], [0], color=colors[name], lw=linewidths[name])
        for name in labels
    ]
    legend = ax.legend(
        handles, labels,
        loc="center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        mode=mode,
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
    return legend


def oni_phase(oni, threshold=0.5):
    oni = np.asarray(oni)
    return np.where(oni >= threshold, 1, np.where(oni <= -threshold, -1, 0))


def corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return np.nan
    return np.corrcoef(a[mask], b[mask])[0, 1]


def load_oni_data(lead):
    obs = np.load("oni_true_2011-2023.npy")
    pred = np.load(f"oni_pred_lead_{lead}.npy")
    n = min(len(obs), len(pred))
    dates = pd.date_range(f"{START_YEAR}-01-01", periods=n, freq="MS")
    x = dates.year + (dates.month - 1) / 12.0
    return x, obs[:n], pred[:n]


def style_oni_axis(ax):
    ax.grid(False)
    ax.tick_params(axis="both", which="major", length=7, labelsize=ONI_TICK_SIZE, pad=6)
    ax.tick_params(axis="x", which="minor", length=4)
    ax.tick_params(axis="y", which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_color("black")


def add_oni_threshold_shading(ax, x, oni, alpha=0.08):
    ax.fill_between(
        x, 0.5, oni,
        where=oni >= 0.5,
        color=ONI_WARM_COLOR,
        alpha=alpha,
        interpolate=True,
        zorder=1,
    )
    ax.fill_between(
        x, -0.5, oni,
        where=oni <= -0.5,
        color=ONI_COLD_COLOR,
        alpha=alpha,
        interpolate=True,
        zorder=1,
    )


def plot_oni_panel(ax, lead, panel_label):
    x, obs_oni, pred_oni = load_oni_data(lead)
    mark_mask = np.arange(len(x)) % MARK_EVERY == 0

    obs_phase = oni_phase(obs_oni)
    pred_phase = oni_phase(pred_oni)
    valid = np.isfinite(obs_oni) & np.isfinite(pred_oni)
    agree = valid & (obs_phase == pred_phase)
    agreement_rate = agree[valid].mean()
    out_of_phase = valid & (obs_phase * pred_phase == -1)
    out_of_phase_rate = out_of_phase[valid].mean()
    r = corr(obs_oni, pred_oni)

    add_oni_threshold_shading(ax, x, obs_oni, alpha=0.09)
    add_oni_threshold_shading(ax, x, pred_oni, alpha=0.06)

    line_obs, = ax.plot(
        x, obs_oni,
        color="black", marker="o", markersize=4.8,
        linewidth=2.2, markevery=MARK_EVERY,
        zorder=4,
        label="GODAS",
    )
    line_pred, = ax.plot(
        x, pred_oni,
        color="red", marker="s", markersize=4.8,
        linewidth=2.2, markevery=MARK_EVERY,
        zorder=5,
        label="TDA-Net",
    )

    ax.axhline(0.5, color=ONI_WARM_COLOR, linestyle=(0, (4, 3)), linewidth=1.5, zorder=2)
    ax.axhline(-0.5, color=ONI_COLD_COLOR, linestyle=(0, (4, 3)), linewidth=1.5, zorder=2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.8, zorder=2)

    strip_y = -2.28
    ax.scatter(
        x[agree & mark_mask], np.full(np.sum(agree & mark_mask), strip_y),
        color="#e6a157", marker="s", s=24, zorder=4,
    )
    ax.scatter(
        x[valid & ~agree & mark_mask],
        np.full(np.sum(valid & ~agree & mark_mask), strip_y),
        color="#9ecae1", marker="x", s=34, linewidths=1.7, zorder=5,
    )

    ax.set_xlim(x.min() - 0.45, x.max() + 0.45)
    ax.set_ylim(-2.65, 3.25)
    ax.set_xlabel("Time (year)", fontsize=ONI_LABEL_SIZE)
    ax.set_ylabel("ONI (°C)", fontsize=ONI_LABEL_SIZE)
    ax.set_xticks(np.arange(START_YEAR, END_YEAR + 1, 2))
    ax.set_xticks(np.arange(START_YEAR, END_YEAR + 1, 1), minor=True)
    ax.set_yticks(np.arange(-2.0, 3.1, 1.0))
    style_oni_axis(ax)

    add_panel_label(ax, panel_label)
    # add_panel_title(
    #     ax,
    #     f"ONI prediction (Lead month = {lead}, Correlation = {r:.2f}, "
    #     f"In-phase = {agreement_rate:.1%}, Out-of-phase = {out_of_phase_rate:.1%})",
    #     ONI_TITLE_SIZE,
    # )
    add_panel_title(
        ax,
        f"ONI prediction (Lead month = {lead}, Correlation = {r:.2f})",
        ONI_TITLE_SIZE,
    )

    phase_agreement_handle = Line2D(
        [0], [0], color="#e6a157", marker="s", linestyle="None",
        markersize=7, label="In-phase",
    )
    phase_disagreement_handle = Line2D(
        [0], [0], color="#9ecae1", marker="x", linestyle="None",
        markersize=8, markeredgewidth=1.8, label="Not-in-phase",
    )

    ax.legend(
        [line_obs, line_pred, phase_agreement_handle, phase_disagreement_handle],
        ["GODAS", "TDA-Net", "In-phase", "Not-in-phase"],
        loc="upper center",
        bbox_to_anchor=(0.7, 0.985),
        ncol=2,
        frameon=False,
        fontsize=ONI_LEGEND_SIZE,
        handlelength=2.0,
        columnspacing=0.2,
        handletextpad=0.7,
        handler_map={
            phase_agreement_handle: RightAlignedMarkerHandler(),
            phase_disagreement_handle: RightAlignedMarkerHandler(),
        },
    )


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
    # Increase/decrease the second value to change the legend row height.
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 0.02, 0.82, 0.82])

    legend_payload = None
    for i, (script_path, title, metric) in enumerate(SKILL_SCRIPTS):
        ax = fig.add_subplot(gs[0, i])
        payload = plot_skill_panel(ax, script_path, title, metric, PANEL_LABELS[i])
        if legend_payload is None:
            legend_payload = payload

    legend_ax = fig.add_subplot(gs[1, :])
    add_shared_model_legend(legend_ax, *legend_payload)

    for idx, lead in enumerate(ONI_LEADS):
        ax = fig.add_subplot(gs[2 + idx, :])
        plot_oni_panel(ax, lead, PANEL_LABELS[3 + idx])

    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        top=0.955,
        bottom=0.06,
        wspace=0.20,
        # Smaller hspace reduces the vertical gap below the shared legend and between panels.
        hspace=0.4,
    )

    fig.savefig("nc_combined_5panels.png", dpi=600, bbox_inches="tight")
    fig.savefig("nc_combined_5panels.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
