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
        [0.9893930616276861, 0.9725958659612367, 0.9538681065336028, 0.9322375803794619, 0.9027699643411127, 0.8659556683453763, 0.8331268180885449, 0.7997773179133181, 0.7669657313825684, 0.7362701529922405, 0.7164718300032321, 0.6907469086886942, 0.6668767464224693, 0.6468804885702373, 0.642659157754419, 0.6261249508437082, 0.6102463867412912, 0.5891520099544315, 0.5697609099556845, 0.5492606504514258, 0.5362133003949894, 0.5341497008335028, 0.5374533259351257, 0.5412590166526426]
    ),
    "ORCA-DL": (
        np.arange(1, 25),
        [0.9917730482185143, 0.97862581115761, 0.962508504038507, 0.9389079662040122, 0.9146792926559965, 0.8779736136827515, 0.8468955441733922, 0.8143440868883376, 0.7843252348584181, 0.7498163919658826, 0.7120068952783243, 0.6532467235684758, 0.6012906721701065, 0.5605113139857203, 0.5221364234520662, 0.4899069791134664, 0.46216158444253114, 0.4433846138254144, 0.4291449583643892, 0.4140119134741871, 0.3954442049106828, 0.3859407411920676, 0.37856260199561886, 0.3740383945570582]
    ),
    "3D-Geoformer": (
        np.arange(1, 21),
        [0.9347673853986196, 0.9356541493006559, 0.9210513938423374, 0.8972477166319178, 0.8665122543131698, 0.8299219875124823, 0.7889919534764174, 0.7460763536028565, 0.6999832204363001, 0.6567549863170185, 0.6199719488458607, 0.5745658325799449, 0.5315335318236541, 0.5096246480269575, 0.4955606593912512, 0.46885879477279985, 0.4367299124446797, 0.41475522130728393, 0.40324242936931004, 0.3991664144415332]    ),
    "SEAS5": (
        np.arange(1, 7),
        [0.8964909965746033, 0.9119939141435653, 0.9027838832211725, 0.8804815606388552, 0.8486209880493895, 0.8019716508151584]
    ),
    "SINTEX-F2": (
        np.arange(1, 12),
        [0.9211989164265781, 0.9072350325223515, 0.8869681566175973, 0.8543237897579778, 0.8134942807771809, 0.7757861020915234, 0.7253001498186273, 0.6634746636459584, 0.6107680798757478, 0.5631890763671589, 0.5087123032593691]
    ),
    "CCSM4": (
        np.arange(1, 13),
        [0.9153, 0.9178, 0.8857, 0.8557, 0.8397, 0.8149, 0.7669, 0.71, 0.6595, 0.6083, 0.5589, 0.5254]
    ),
    "CESM1": (
        np.arange(1, 13),
        [0.9143, 0.9218, 0.8982, 0.8701, 0.8474, 0.8221, 0.7965, 0.7639, 0.7298, 0.6868, 0.6384, 0.6009]
    ),
    "NASA-GEOSS2S": (
        np.arange(1, 10),
        [0.9484, 0.9377, 0.9204, 0.8981, 0.872, 0.8375, 0.7979, 0.7524, 0.6841]
    ),
    "NCEP-CFSv2": (
        np.arange(1, 11),
        [0.9235, 0.9256, 0.9117, 0.8854, 0.8393, 0.7766, 0.6998, 0.6177, 0.5687, 0.5265]
    ),
    "CanSIPS-IC3": (
        np.arange(1, 13),
        [0.9333, 0.9547, 0.9496, 0.926, 0.8889, 0.8375, 0.7801, 0.7233, 0.6602, 0.5999, 0.5525, 0.5206]
    ),
    "CanSIPS-IC4": (
        np.arange(1, 13),
        [0.9339, 0.9499, 0.9445, 0.9274, 0.8951, 0.8481, 0.7871, 0.7199, 0.6572, 0.5926, 0.539, 0.4908]
    ),
    "GFDL-SPEAR": (
        np.arange(1, 13),
        [0.9625, 0.9662, 0.942, 0.9094, 0.8783, 0.8409, 0.7977, 0.7441, 0.6845, 0.6209, 0.5558, 0.4978]
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
ax.set_title("Correlation skills for Niño4 prediction",
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
plt.savefig("nc_nino4_correlation_skill.png", dpi=600, bbox_inches="tight")
plt.savefig("nc_nino4_correlation_skill.pdf", bbox_inches="tight")

# plt.show()