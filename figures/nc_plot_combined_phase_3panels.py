import ast
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

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

FIGSIZE = (17.2, 5.6)
DPI = 300

RESULTS_FILE = "results_phase.txt"
BASE_RESULTS_FILE = "results_phase_old.txt"
STYLE_SOURCE = "nc_plot_nino34_rmse.py"
PANEL_LABELS = list("abc")

PHASE_PANELS = [
    ("in_phase", "In-phase rate for Niño3.4 prediction", "In-phase (%)", (0, 100), [0, 20, 40, 60, 80, 100]),
    ("not_in_phase", "Not-in-phase rate for Niño3.4 prediction", "Not-in-phase (%)", (0, 80), [0, 20, 40, 60, 80]),
    ("out_of_phase", "Out-of-phase rate for Niño3.4 prediction", "Out-of-phase (%)", (0, 20), [0, 4, 8, 12, 16, 20]),
]

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

MODEL_ALIASES = {
    "NUIST": "NUIST-CFS1.2",
    "NUIST-CFS1.2": "NUIST-CFS1.2",
    "SINTEX-F": "SINTEX-F2",
    "SINTEX-F2": "SINTEX-F2",
    "A01": "GFDL-FLOR-A06",
    "A06": "GFDL-FLOR-A06",
    "B01": "GFDL-FLOR-B01",
    "aer04": "GFDL-aer04",
}
MODEL_ALIASES.update({name: name for name in MODEL_LEGEND_ORDER})

MODEL_HEADER_NAMES = set(MODEL_ALIASES)
MODEL_HEADER_NAMES.update({
    "CNN",
    "CCSM3",
    "CanCM3",
    "CanCM4",
    "GFDL-aer04",
    "GFDL-FLOR-A06",
    "GFDL-FLOR-B01",
})

# Match the physical size and spacing of the ABC axes in nc_plot_combined_rmse_3panels.py.
AX_LEFT = 0.06
AX_RIGHT = 0.985
AX_TOP_MARGIN_IN = 0.6525
AX_HEIGHT_IN = 3.7529
AX_WSPACE = 0.20
LEGEND_AXIS = [0.06, 0.01, 0.925, 0.06]


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


def add_panel_title(ax, title):
    title_offset = ScaledTranslation(18 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0, 1.0, title,
        transform=ax.transAxes + title_offset,
        fontsize=TITLE_SIZE,
        ha="left",
        va="bottom",
    )


def empty_model_payload():
    return {
        "lead": [],
        "in_phase": [],
        "not_in_phase": [],
        "out_of_phase": [],
    }


def normalize_model_name(raw_name):
    return MODEL_ALIASES.get(raw_name.strip())


def append_row(data, model, lead, in_phase, not_in_phase, out_of_phase):
    payload = data.setdefault(model, empty_model_payload())
    payload["lead"].append(int(lead))
    payload["in_phase"].append(float(in_phase))
    payload["not_in_phase"].append(float(not_in_phase))
    payload["out_of_phase"].append(float(out_of_phase))


def set_array_metric(data, model, metric, values):
    payload = data.setdefault(model, empty_model_payload())
    payload[metric] = [float(v) for v in values]
    if not payload["lead"]:
        payload["lead"] = list(range(1, len(values) + 1))


def parse_phase_results(path):
    path = Path(path)
    data = {}
    current_model = None
    phase_columns = None
    column_aliases = {
        "in_phase": {"in%", "inphase%"},
        "not_in_phase": {"not%", "notin%", "notinphase%"},
        "out_of_phase": {"out%", "outphase%", "outofphase%"},
    }

    cnn_pattern = re.compile(
        r"lead\s+(\d+):.*?"
        r"In-phase\s+([-+]?\d+(?:\.\d+)?)%.*?"
        r"Not-in-phase\s+([-+]?\d+(?:\.\d+)?)%.*?"
        r"Out-of-phase\s+([-+]?\d+(?:\.\d+)?)%",
        re.IGNORECASE,
    )

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        model = normalize_model_name(stripped)
        if model:
            current_model = model
            phase_columns = None
            data.setdefault(current_model, empty_model_payload())
            continue

        if stripped in MODEL_HEADER_NAMES:
            current_model = None
            phase_columns = None
            continue

        if current_model is None:
            continue

        if re.match(r"^In-phase\s*%:", stripped):
            values = ast.literal_eval(stripped.split(":", 1)[1].strip())
            set_array_metric(data, current_model, "in_phase", values)
            continue

        if re.match(r"^Not in-phase\s*%:", stripped):
            values = ast.literal_eval(stripped.split(":", 1)[1].strip())
            set_array_metric(data, current_model, "not_in_phase", values)
            continue

        if re.match(r"^Out-of-phase\s*%:", stripped):
            values = ast.literal_eval(stripped.split(":", 1)[1].strip())
            set_array_metric(data, current_model, "out_of_phase", values)
            continue

        match = cnn_pattern.search(stripped)
        if match:
            append_row(data, current_model, *match.groups())
            continue

        parts = stripped.split()
        if parts[0].lower() == "lead":
            # Locate percentages by header, since MAE and count columns vary.
            headers = [part.lower().replace("-", "") for part in parts]
            phase_columns = []
            for aliases in column_aliases.values():
                matches = [i for i, header in enumerate(headers) if header in aliases]
                if len(matches) != 1:
                    raise ValueError(
                        f"{path}:{line_number}: ambiguous or missing phase column for {current_model}"
                    )
                phase_columns.append(matches[0])
            continue

        if not re.fullmatch(r"[-+]?\d+", parts[0]):
            continue

        if phase_columns is None or max(phase_columns) >= len(parts):
            raise ValueError(
                f"{path}:{line_number}: missing phase header or values for {current_model}"
            )
        append_row(data, current_model, parts[0], *(parts[i] for i in phase_columns))

    normalize_leads(data)
    return data


def normalize_leads(data):
    for payload in data.values():
        if payload["lead"] and min(payload["lead"]) == 0:
            payload["lead"] = [lead + 1 for lead in payload["lead"]]


def validate_phase_data(data):
    missing = [name for name in MODEL_LEGEND_ORDER if name not in data]
    if missing:
        raise KeyError(f"{RESULTS_FILE} is missing models: {missing}")

    for model, payload in data.items():
        lead_count = len(payload["lead"])
        for metric in ("in_phase", "not_in_phase", "out_of_phase"):
            if len(payload[metric]) != lead_count:
                raise ValueError(
                    f"{model} has {lead_count} leads but {len(payload[metric])} values for {metric}"
                )
            values = np.asarray(payload[metric], dtype=float)
            if np.any(~np.isfinite(values) | (values < 0) | (values > 100)):
                raise ValueError(f"{model} {metric} must contain finite percentages in [0, 100]")

        in_phase = np.asarray(payload["in_phase"], dtype=float)
        not_in_phase = np.asarray(payload["not_in_phase"], dtype=float)
        out_of_phase = np.asarray(payload["out_of_phase"], dtype=float)
        # Allow for percentages rounded to two decimal places in the source.
        if not np.allclose(in_phase + not_in_phase, 100, atol=0.02, rtol=0):
            raise ValueError(f"{model} In-phase and Not-in-phase percentages must sum to 100")
        if np.any(out_of_phase > not_in_phase + 0.02):
            raise ValueError(f"{model} Out-of-phase percentage cannot exceed Not-in-phase")


def plot_phase_panel(ax, phase_data, metric, title, ylabel, ylim, yticks,
                     colors, linewidths, plot_order, panel_label):
    for name in plot_order:
        payload = phase_data[name]
        ax.plot(
            payload["lead"],
            payload[metric],
            color=colors[name],
            linewidth=linewidths[name],
            solid_capstyle="butt",
            label=name,
            zorder=4 if name == "TDA-Net" else 3,
        )

    ax.set_xlim(0, 25)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Lead time (month)", fontsize=LABEL_SIZE, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, labelpad=8)
    ax.set_xticks(LEAD_TICKS)
    ax.set_yticks(yticks)
    ax.set_xticks(np.arange(1, 25, 1), minor=True)

    add_panel_label(ax, panel_label)
    add_panel_title(ax, title)
    style_skill_axis(ax)


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
    ncol = 7
    labels = row_major_legend_input(MODEL_LEGEND_ORDER, ncol)
    handles = [
        Line2D([0], [0], color=colors[name], lw=linewidths[name])
        for name in labels
    ]
    legend = ax.legend(
        handles, labels,
        loc="center",
        bbox_to_anchor=(0.5, -0.1),
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


def add_abc_axes(fig):
    fig_w, fig_h = FIGSIZE
    total_w = AX_RIGHT - AX_LEFT
    ax_w = total_w / (3 + AX_WSPACE * 2)
    gap_w = AX_WSPACE * ax_w
    ax_h = AX_HEIGHT_IN / fig_h
    ax_top = 1.0 - AX_TOP_MARGIN_IN / fig_h
    ax_bottom = ax_top - ax_h

    return [
        fig.add_axes([AX_LEFT + i * (ax_w + gap_w), ax_bottom, ax_w, ax_h])
        for i in range(3)
    ]


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

    phase_data = parse_phase_results(BASE_RESULTS_FILE)
    phase_data.update(parse_phase_results(RESULTS_FILE))
    validate_phase_data(phase_data)
    style = load_assignments(STYLE_SOURCE, {"colors", "linewidths", "plot_order"})
    colors = style["colors"]
    linewidths = style["linewidths"]
    plot_order = style["plot_order"]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    axes = add_abc_axes(fig)

    for i, (metric, title, ylabel, ylim, yticks) in enumerate(PHASE_PANELS):
        plot_phase_panel(
            axes[i],
            phase_data,
            metric,
            title,
            ylabel,
            ylim,
            yticks,
            colors,
            linewidths,
            plot_order,
            PANEL_LABELS[i],
        )

    legend_ax = fig.add_axes(LEGEND_AXIS)
    add_shared_model_legend(legend_ax, colors, linewidths)

    fig.savefig("nc_combined_phase_3panels.png", dpi=600, bbox_inches="tight")
    fig.savefig("nc_combined_phase_3panels.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
