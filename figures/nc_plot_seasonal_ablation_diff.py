import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from nc_plot_seasonal_common import (
    INIT_MONTHS,
    build_plot_matrix,
    build_target_months,
    parse_skill_by_lead,
)


plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.unicode_minus"] = False

INPUT_FILE = "results_seasonal_ablation.txt"
BASE_MODEL = "baseline"
SLAM_MODEL = "baseline + SLAM"
OUTPUT_PREFIX = "nc_seasonal_ab_slam_diff"


def read_model_tables(path):
    tables = {}
    current_model = None
    current_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or set(stripped) == {"-"}:
                continue
            if stripped.startswith("Lead"):
                if current_model is not None:
                    current_lines.append(stripped)
                continue
            if stripped.split()[0].isdigit():
                if current_model is None:
                    raise ValueError(f"Data row appears before a model name: {line}")
                current_lines.append(stripped)
                continue

            if current_model is not None:
                tables[current_model] = "\n".join(current_lines)
            current_model = stripped
            current_lines = []

    if current_model is not None:
        tables[current_model] = "\n".join(current_lines)

    return tables


def plot_ablation_diff():
    tables = read_model_tables(INPUT_FILE)
    missing = [name for name in [BASE_MODEL, SLAM_MODEL] if name not in tables]
    if missing:
        raise ValueError(f"Missing model table(s): {', '.join(missing)}")

    baseline = build_plot_matrix(parse_skill_by_lead(tables[BASE_MODEL]))
    slam = build_plot_matrix(parse_skill_by_lead(tables[SLAM_MODEL]))
    diff = slam - baseline

    target_months, target_years = build_target_months()
    n_targets = diff.shape[1]

    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=300)

    colors = [
        "#2400c8", "#2526e6", "#2d55f0", "#3f82ef", "#56a9ee",
        "#72c5e8", "#9eddeb", "#c7eef1", "#ffffff", "#ffffff",
        "#f6d49a", "#f5b66e", "#f48b5a", "#f85f4c", "#fb3d3f",
        "#f51f31", "#e80e28", "#cf082d", "#b3002d"
    ]
    cmap = LinearSegmentedColormap.from_list("corr_diff", colors)
    cmap.set_bad("#cfcfcf")
    norm = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25)

    im = ax.imshow(
        diff,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(-0.5, n_targets - 0.5, -0.5, 11.5)
    )

    for xpos in [11, 23]:
        ax.axvline(
            xpos,
            color="#1e9bd7",
            linestyle="--",
            linewidth=1.5
        )

    xtick_pos = np.arange(0, n_targets, 3)
    xtick_labels = [rf"{target_months[i]}$^{target_years[i]}$"
                    for i in xtick_pos]

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=12)
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(INIT_MONTHS, fontsize=12)

    ax.set_xlabel("Target month", fontsize=14, labelpad=2)
    ax.set_ylabel("Initial time", fontsize=14, labelpad=2)
    ax.set_title(
        f"Seasonality of Nino3.4 correlation skill diff ({SLAM_MODEL[9:]})",
        fontsize=14,
        pad=8
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        length=3,
        width=1.2,
        pad=3
    )

    ax.set_xlim(-0.5, n_targets - 0.5)
    ax.set_ylim(-0.5, 11.5)

    plt.tight_layout(pad=0.6)
    fig.savefig(f"{OUTPUT_PREFIX}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUTPUT_PREFIX}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_ablation_diff()
