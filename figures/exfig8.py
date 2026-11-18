"""
绘制多个模型的 TCC 对比图（2 模型 x 3 lead month = 6 子图）
- 模型: ORCA-DL, TDA-Net
- lead month: 6, 12, 18
- 数据文件: tcc_map_tos_{model_name}_lead{lead}.npz
- 输出: 6 个带 a-f 编号的 TCC 对比热力图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation


# 配置
MODELS = ['ORCA-DL', 'TDA-Net']
LEADS = [6, 12, 18]
PANEL_LABELS = list("abcdef")

# 与 nc_plot_combined_5panels.py 保持一致的字体控制
TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 15
PANEL_LABEL_SIZE = 19
DPI = 300


def add_panel_label(ax, label):
    """按 nc_plot_combined_5panels.py 的相对坐标框位置添加编号。"""
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
    """按 nc_plot_combined_5panels.py 的标题位置和字号添加标题。"""
    title_offset = ScaledTranslation(18 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0.0, 1.0, title,
        transform=ax.transAxes + title_offset,
        fontsize=TITLE_SIZE,
        ha="left",
        va="bottom",
    )


def add_map_gridlines(ax, row, col):
    """添加地图网格线，并把纬度标注改为 30 度间隔。"""
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--',
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = col == 0
    gl.bottom_labels = row == len(MODELS) - 1
    gl.ylocator = mticker.FixedLocator(np.arange(-60, 61, 30))
    gl.xlabel_style = {'size': TICK_SIZE}
    gl.ylabel_style = {'size': TICK_SIZE}
    return gl


def load_tcc_data(model_name, lead):
    """加载 TCC 数据"""
    filepath = f'tcc_map_tos_{model_name}_lead{lead}.npz'
    try:
        data = np.load(filepath)
        tcc_map = data['tcc_map']
        ocean_mask = data['ocean_mask']
        return tcc_map, ocean_mask
    except FileNotFoundError:
        print(f"警告: 未找到文件 {filepath}")
        return None, None


def plot_tcc_comparison():
    """绘制 TCC 对比图（保留原 3x3 布局中的后两行）。"""
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

    fig = plt.figure(figsize=(18, 11.2), dpi=DPI)
    gs = GridSpec(
        3, 3,
        figure=fig,
        hspace=0.08,
        wspace=0.08,
        left=0.045,
        right=0.94,
        top=0.95,
        bottom=0.13,
    )

    # 创建经纬度网格
    lat = np.linspace(-63.5, 63.5, 128)
    lon = np.linspace(0.5, 359.5, 360)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # 设置 colormap 和 levels
    levels = np.linspace(-1, 1, 21)
    cmap = plt.cm.RdBu_r
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # 存储所有子图的 contourf 对象（用于共享 colorbar）
    cf = None

    # 遍历模型和 lead month，创建子图
    for i, model in enumerate(MODELS):
        for j, lead in enumerate(LEADS):
            ax = fig.add_subplot(
                gs[i + 1, j],
                projection=ccrs.PlateCarree(central_longitude=180),
            )
            ax.set_box_aspect(0.5)
            panel_idx = i * len(LEADS) + j

            # 加载数据
            tcc_map, ocean_mask = load_tcc_data(model, lead)

            if tcc_map is not None and ocean_mask is not None:
                # 将陆地区域设为 NaN
                tcc_map_masked = np.where(ocean_mask, tcc_map, np.nan)

                # 设置地图范围
                ax.set_extent([0, 360, -63.5, 63.5], crs=ccrs.PlateCarree())
                ax.set_aspect("auto")
                # 添加海岸线和陆地
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)

                # 绘制 TCC 热力图
                cf = ax.contourf(
                    lon_grid, lat_grid, tcc_map_masked,
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    extend='both',
                )

                # 添加 TCC=0.5 等高线
                ax.contour(
                    lon_grid, lat_grid, tcc_map_masked,
                    levels=[0.5],
                    colors='black',
                    linewidths=1.5,
                    transform=ccrs.PlateCarree(),
                )

            else:
                # 数据缺失，显示空白子图
                ax.text(
                    0.5, 0.5, 'Data Not Available',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=14,
                    color='red',
                )

            add_panel_label(ax, PANEL_LABELS[panel_idx])
            add_panel_title(ax, f'{model} (lead month = {lead})')
            add_map_gridlines(ax, i, j)

    # 添加共享的 colorbar（放在底部，坐标保持与原 9 子图脚本一致）
    if cf is not None:
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
        cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
        # cbar.set_label('Temporal Correlation Coefficient (TCC)', fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

    png_path = './nc_tcc_comparison_models_leads_labeled_no_nuist.png'
    pdf_path = './nc_tcc_comparison_models_leads_labeled_no_nuist.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\n图形已保存至 {png_path}")
    print(f"PDF 已保存至 {pdf_path}")
    plt.close(fig)


def compute_statistics():
    """计算并输出各模型各 lead month 的 TCC 统计信息"""
    print("\n" + "=" * 80)
    print("TCC 统计信息")
    print("=" * 80)
    print(f"{'Model':<15} {'lead month':<12} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'>0.5 %':<10}")
    print("-" * 80)

    for model in MODELS:
        for lead in LEADS:
            tcc_map, ocean_mask = load_tcc_data(model, lead)

            if tcc_map is not None:
                # 只统计海洋区域的有效值
                valid_tcc = tcc_map[~np.isnan(tcc_map)]

                if len(valid_tcc) > 0:
                    mean_tcc = np.mean(valid_tcc)
                    median_tcc = np.median(valid_tcc)
                    min_tcc = np.min(valid_tcc)
                    max_tcc = np.max(valid_tcc)
                    above_05_pct = np.sum(valid_tcc > 0.5) / len(valid_tcc) * 100

                    print(f"{model:<15} {lead:<12} {mean_tcc:<10.4f} {median_tcc:<10.4f} "
                          f"{min_tcc:<10.4f} {max_tcc:<10.4f} {above_05_pct:<10.2f}")
                else:
                    print(f"{model:<15} {lead:<12} {'N/A':<10} {'N/A':<10} "
                          f"{'N/A':<10} {'N/A':<10} {'N/A':<10}")
            else:
                print(f"{model:<15} {lead:<12} {'N/A':<10} {'N/A':<10} "
                      f"{'N/A':<10} {'N/A':<10} {'N/A':<10}")

    print("-" * 80)


def main():
    print("=" * 80)
    print("TCC 多模型对比图绘制")
    print("=" * 80)

    # 1. 绘制对比图
    print("\n[1] 绘制 TCC 对比图...")
    plot_tcc_comparison()

    # 2. 计算统计信息
    print("\n[2] 计算统计信息...")
    compute_statistics()

    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
