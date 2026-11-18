import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# 1. 替换成你自己的数据
# =========================
# key: 模型名称
# value: (lead_time数组, correlation数组)
# 如果某个模型只预测到12个月，就只写到12；不用补到24
data = {
    "TDA-Net": (
        np.arange(1, 25),
        [0.9824637905085998, 0.9580733005516165, 0.9173785009045409, 0.8654276722448893, 0.8105639774757203, 0.7569641146531382, 0.7130959041980987, 0.6680165331311647, 0.6424330342021027, 0.6315197100942334, 0.6344233680449861, 0.6160363193334624, 0.6023725525484759, 0.5873328059567613, 0.5807953154067087, 0.5714094816846366, 0.5568738562783124, 0.537914182783814, 0.5247437549111547, 0.5227156335761887, 0.5405348036609566, 0.564099077383398, 0.5856549923124102, 0.5972708689284951]
    ),
    "ORCA-DL": (
        np.arange(1, 25),
        [0.9775486742756992, 0.9613292787377252, 0.9350527380400575, 0.8933360576178837, 0.8369047110296585, 0.7733723458287626, 0.7398798642822434, 0.7032063660039761, 0.6621662862989551, 0.6212849238200574, 0.5792913230650883, 0.5135354285475044, 0.45903200541537753, 0.4103278943384238, 0.35530886320565136, 0.30808915598155984, 0.2779386056297268, 0.24859188158554804, 0.2604231202174281, 0.278599071884758, 0.2972492708976897, 0.3174716249441875, 0.34128359948302217, 0.32901924004913563]
    ),
    "3D-Geoformer": (
        np.arange(1, 21),
        [0.97226824162435, 0.9697585122716419, 0.9571072626721411, 0.9353080120797134, 0.907563565806898, 0.877685725524402, 0.8440600562977705, 0.8091209044810594, 0.7712567362994694, 0.7308213588030837, 0.689743497682797, 0.6404854303777258, 0.587443196156114, 0.5488584115802118, 0.5125343776162977, 0.46686688975379464, 0.4239952629652983, 0.3954416636191314, 0.38043793487217153, 0.37377776392246675]
    ),
    "SEAS5": (
        np.arange(1, 7),
        [0.9619785923433881, 0.9495500511682631, 0.9344896894955008, 0.9172711771611842, 0.8974778303330198, 0.8661946608829302]
    ),
    "SINTEX-F2": (
        np.arange(1, 12),
        [0.9456229655033193, 0.9255628668041099, 0.8937104082250468, 0.85112700956792, 0.8001725806139873, 0.7555421506244165, 0.7188899060416337, 0.679732564467098, 0.6113278493162692, 0.53248941637839, 0.4823244594766853]
    ),
    "CCSM4": (
        np.arange(1, 13),
        [0.9715, 0.9598, 0.9376, 0.9031, 0.868, 0.8362, 0.8064, 0.7761, 0.7399, 0.7076, 0.6856, 0.6629]
    ),
    "CESM1": (
        np.arange(1, 13),
        [0.9702, 0.9554, 0.9368, 0.8973, 0.8662, 0.8487, 0.8397, 0.8246, 0.7908, 0.7493, 0.7089, 0.6784]
    ),
    "NASA-GEOSS2S": (
        np.arange(1, 10),
        [0.9514, 0.9334, 0.9039, 0.879, 0.8696, 0.8587, 0.8368, 0.8017, 0.7416]
    ),
    "NCEP-CFSv2": (
        np.arange(1, 11),
        [0.9714, 0.9567, 0.9381, 0.9114, 0.882, 0.8506, 0.8144, 0.7849, 0.754, 0.7253]
    ),
    "CanSIPS-IC3": (
        np.arange(1, 13),
        [0.9698, 0.971, 0.9638, 0.9454, 0.9174, 0.8829, 0.8343, 0.7841, 0.7339, 0.6839, 0.6384, 0.5955]
    ),
    "CanSIPS-IC4": (
        np.arange(1, 13),
        [0.9642, 0.9646, 0.9609, 0.9474, 0.9231, 0.8901, 0.8409, 0.7815, 0.7108, 0.6379, 0.5837, 0.523]
    ),
    "GFDL-SPEAR": (
        np.arange(1, 13),
        [0.9633, 0.9562, 0.9409, 0.9183, 0.8961, 0.8627, 0.8165, 0.7647, 0.706, 0.6398, 0.5727, 0.5353]
    ),
}


# =========================
# 2. 颜色和线宽设置
# =========================
colors = {
    "TDA-Net": "#ff0000",
    "ORCA-DL": "#1f77b4",
    "3D-Geoformer": "#7a4fa3",
    "SEAS5": "#2ca02c",
    "SINTEX-F2": "#58a5f0",
    "CCSM4": "#5d63d9",
    "CESM1": "#8c564b",
    "NASA-GEOSS2S": "#ff7f0e",
    "NCEP-CFSv2": "#17becf",
    "CanSIPS-IC3": "#9467bd",
    "CanSIPS-IC4": "#e377c2",
    "GFDL-SPEAR": "#bcbd22",
}
linewidths = {
    "TDA-Net": 4.0,
    "ORCA-DL": 3.0,
    "3D-Geoformer": 3.0,
    "SEAS5": 3.0,
    "SINTEX-F2": 3.0,
    "CCSM4": 3.0,
    "CESM1": 3.0,
    "NASA-GEOSS2S": 3.0,
    "NCEP-CFSv2": 3.0,
    "CanSIPS-IC3": 3.0,
    "CanSIPS-IC4": 3.0,
    "GFDL-SPEAR": 3.0,
}


# =========================
# 3. 开始绘图
# =========================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(5.5, 5.1), dpi=300)

plot_order = [
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
]

for name in plot_order:
    x, y = data[name]
    ax.plot(
        x, y,
        color=colors[name],
        linewidth=linewidths[name],
        solid_capstyle="butt",
        label=name,
        zorder=4 if name == "TDA-Net" else 3
    )

# 0.5 黑色虚线
ax.axhline(
    y=0.5,
    color="black",
    linewidth=2.0,
    linestyle="--",
    zorder=2
)

# =========================
# 4. 坐标轴、网格、刻度
# =========================
ax.set_xlim(0, 25)
ax.set_ylim(0, 1.0)

ax.set_xlabel("Lead time (months)", fontsize=15, labelpad=8)
ax.set_ylabel("Correlation skill", fontsize=15, labelpad=8)

ax.set_xticks([2, 6, 10, 14, 18, 22])
ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])

# 小刻度，每个月一个
ax.set_xticks(np.arange(1, 25, 1), minor=True)

ax.tick_params(axis="both", which="major", direction="in",
               length=7, width=1.8, labelsize=13, pad=8)
ax.tick_params(axis="x", which="minor", direction="in",
               length=4, width=1.5)

# 灰色点状网格
ax.grid(
    True,
    which="major",
    linestyle=":",
    linewidth=1.7,
    color="#9e9e9e",
    zorder=1
)

# 边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.8)
    spine.set_color("black")

# =========================
# 5. 标题和左上角 B
# =========================
ax.set_title("Correlation skills for Niño3 prediction",
             fontsize=16, pad=10)

ax.text(
    -0.05, 1.02, "B",
    transform=ax.transAxes,
    fontsize=19,
    fontweight="bold",
    va="bottom",
    ha="left"
)

# =========================
# 6. 图例：分左右两列手动排版，更接近原图
# =========================
left_labels = [
    "TDA-Net",
    "ORCA-DL",
    "3D-Geoformer",
    "SEAS5",
    "SINTEX-F2",
    "CCSM4",
]

right_labels = [
    "CESM1",
    "NASA-GEOSS2S",
    "NCEP-CFSv2",
    "CanSIPS-IC3",
    "CanSIPS-IC4",
    "GFDL-SPEAR",
]

left_handles = [
    Line2D([0], [0], color=colors[name], lw=linewidths[name])
    for name in left_labels
]

right_handles = [
    Line2D([0], [0], color=colors[name], lw=linewidths[name])
    for name in right_labels
]

leg1 = ax.legend(
    left_handles,
    left_labels,
    loc="lower left",
    bbox_to_anchor=(0.02, 0.01),
    frameon=False,
    fontsize=11.5,
    handlelength=2.0,
    handletextpad=0.7,
    borderaxespad=0.0,
    labelspacing=0.35
)

ax.add_artist(leg1)

ax.legend(
    right_handles,
    right_labels,
    loc="lower left",
    bbox_to_anchor=(0.47, 0.01),
    frameon=False,
    fontsize=11.5,
    handlelength=2.0,
    handletextpad=0.7,
    borderaxespad=0.0,
    labelspacing=0.35
)

plt.tight_layout()

# 保存图片
plt.savefig("nc_nino3_correlation_skill.png", dpi=600, bbox_inches="tight")
plt.savefig("nc_nino3_correlation_skill.pdf", bbox_inches="tight")

# plt.show()