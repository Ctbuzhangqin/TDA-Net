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
        [0.98444937177278, 0.9660164839193282, 0.9436394383281468, 0.9175331887062458, 0.8854810901934272, 0.8471959682242515, 0.8185712959096876, 0.7873814989201321, 0.7577942575447737, 0.7291156800954143, 0.7142154863191049, 0.6901795058356404, 0.6676420074451118, 0.6497171093023875, 0.646727047924635, 0.6344256754948051, 0.6167028999935132, 0.5920555759704279, 0.5705304893479183, 0.5562082900600493, 0.5597172180860218, 0.5758002278215791, 0.5998501855700232, 0.6186951465418224]
    ),
    "ORCA-DL": (
        np.arange(1, 25),
        [0.9830363184056496, 0.9735956593007781, 0.9561253781917556, 0.9324840693536839, 0.9014222229891936, 0.8615883411285645, 0.8411820476733943, 0.8125237011203729, 0.777749445181064, 0.7305039706019096, 0.6822664078947974, 0.6075732885475749, 0.551007433661465, 0.5034122090141979, 0.45697754579217514, 0.4203302721364883, 0.39407506268085807, 0.3688112568097208, 0.36637710471435014, 0.3612580255970906, 0.35720458121060134, 0.361191707332628, 0.3780034862625346, 0.3763891402394807]
    ),
    "3D-Geoformer": (
        np.arange(1, 21),
        [0.993636317708267, 0.9812654787871282, 0.9610594047954546, 0.9350893208404644, 0.9038985422422332, 0.8704299900494035, 0.8330753578960685, 0.7931892977253062, 0.7495596857021126, 0.7053548502529732, 0.6651543598557804, 0.6179551777203268, 0.5725538270986907, 0.5455305997725578, 0.5229457488428326, 0.48670029123930275, 0.44728937315864264, 0.41946386578244016, 0.4043369666375728, 0.39785532813203717]
    ),
    "SEAS5": (
        np.arange(1, 7),
        [0.967518579104147, 0.950378130233932, 0.9276608718893724, 0.9031610281948121, 0.8755883232734941, 0.8367646879269004]
    ),
    "SINTEX-F2": (
        np.arange(1, 12),
        [0.9602759877937115, 0.935462280039919, 0.9061270975609621, 0.8710496771404467, 0.8261937373859491, 0.7852304709489698, 0.7462669464528787, 0.6953271452134366, 0.6375803863060092, 0.5696434572298806, 0.5048189006732343]
    ),
    "CCSM4": (
        np.arange(1, 13),
        [0.9905, 0.9683, 0.9339, 0.8982, 0.8672, 0.8327, 0.7969, 0.7632, 0.724, 0.6875, 0.6635, 0.6469]
    ),
    "CESM1": (
        np.arange(1, 13),
        [0.9901, 0.9691, 0.9499, 0.9184, 0.8913, 0.8676, 0.8516, 0.8253, 0.7886, 0.7474, 0.7061, 0.6762]
    ),
    "NASA-GEOSS2S": (
        np.arange(1, 10),
        [0.9834, 0.9489, 0.9204, 0.9035, 0.8933, 0.8734, 0.8446, 0.8029, 0.7375]
    ),
    "NCEP-CFSv2": (
        np.arange(1, 11),
        [0.9869, 0.9691, 0.9492, 0.924, 0.8947, 0.8587, 0.8158, 0.7763, 0.7419, 0.7178]
    ),
    "CanSIPS-IC3": (
        np.arange(1, 13),
        [0.9943, 0.9844, 0.9684, 0.9422, 0.9082, 0.8679, 0.8181, 0.7659, 0.7104, 0.6554, 0.6117, 0.5765]
    ),
    "CanSIPS-IC4": (
        np.arange(1, 13),
        [0.995, 0.9855, 0.9693, 0.947, 0.9161, 0.8761, 0.8233, 0.7584, 0.6934, 0.6271, 0.5686, 0.5162]
    ),
    "GFDL-SPEAR": (
        np.arange(1, 13),
        [0.9921, 0.9762, 0.955, 0.9271, 0.8993, 0.8615, 0.8187, 0.7666, 0.7062, 0.643, 0.5794, 0.5256]
    ),
    "JMA": (
        np.arange(1, 6),
        [0.952637, 0.927105, 0.894449, 0.859399, 0.814462]
    ),
    "MetFrance": (
        np.arange(1, 6),
        [0.952637, 0.927105, 0.894449, 0.859399, 0.814462]
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
    "JMA": "#7f7f7f",
    "MetFrance": "#d62728",
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
    "JMA": 3.0,
    "MetFrance": 3.0,
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
    "JMA",
    "MetFrance",
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
ax.set_title("Correlation skills for Niño3.4 prediction",
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
    "CESM1",
]

right_labels = [
    "NASA-GEOSS2S",
    "NCEP-CFSv2",
    "CanSIPS-IC3",
    "CanSIPS-IC4",
    "GFDL-SPEAR",
    "JMA",
    "MetFrance",
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
plt.savefig("nc_nino34_correlation_skill.png", dpi=600, bbox_inches="tight")
plt.savefig("nc_nino34_correlation_skill.pdf", bbox_inches="tight")

# plt.show()