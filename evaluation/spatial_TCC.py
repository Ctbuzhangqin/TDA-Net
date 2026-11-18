"""
计算多变量异常值的时间相关系数 (TCC) 并绘制热力图
- 支持多层变量（uo, vo, thetao, so）和单层变量（zos, tos）
- 对于多层变量，支持指定层索引（默认取第0层），而非多层均值
- 预测文件: ./output/test/{exp_name}/preds/{var}_step{lead}.pt
- 初始化时间: ./output/test/{exp_name}/init_times.json
- 真实值文件: ./dataset/GODAS/{var}/{YYYY}_{M}.npy
- 统计量: ./stat/mean/{stat_name}.npy, ./stat/std/{stat_name}.npy
  - uo -> ucur, vo -> vcur, thetao -> pottmp, so -> salt, zos -> sshg, tos -> sst
- 气候态基准期: 1991-2020
- 评估时间: 2011-2023
- 输出: TCC 热力图，标注 TCC=0.5 等高线
"""

import numpy as np
import torch
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# 配置
exp_name = 'tda_ensemble'
PRED_DIR = f'./output/test/{exp_name}/preds'
INIT_TIMES_FILE = f'./output/test/{exp_name}/init_times.json'
TRUE_DIR = './dataset/GODAS'
STAT_DIR = './stat'
CLIM_START_YEAR = 1991
CLIM_END_YEAR = 2020
EVAL_START = 2011
EVAL_END = 2023

# 绘图配置
LEAD_TO_PLOT = 18  # 指定要绘制的 lead

# 变量映射: 预测变量名 -> (真实值目录名, 统计量文件名, 是否多层, 层索引)
# 层索引: 仅对多层变量有效，None 表示使用默认值 0
VAR_MAPPING = {
    # 'uo':      ('uo',      'ucur',  True,  0),   # 16 层, 取第0层
    # 'vo':      ('vo',      'vcur',  True,  0),   # 16 层, 取第0层
    # 'thetao':  ('thetao',  'pottmp', True,  0),   # 16 层, 取第0层
    # 'so':      ('so',      'salt',  True,  0),   # 16 层, 取第0层
    # 'zos':     ('zos',     'sshg',  False, None), # 单层
    'tos':     ('tos',     'sst',   False, None), # 单层 (tos 作为参考保留)
}


def load_land_sea_mask(stat_name, is_multilayer):
    """从 mean 文件加载 land/sea mask，ocean=True, land=False"""
    mean = np.load(f'{STAT_DIR}/mean/{stat_name}.npy')
    # mean shape: (12, [depth,] lat, lon) 或 (12, lat, lon)
    if is_multilayer:  # 多层变量
        ocean_mask = ~np.isnan(mean[0, 0])  # (lat, lon)
    else:  # 单层变量
        ocean_mask = ~np.isnan(mean[0])  # (lat, lon)
    return ocean_mask


def load_true_data(var_name, start_year=1980, end_year=2023):
    """加载真实值数据"""
    data_list = []
    time_list = []
    true_dir = f'{TRUE_DIR}/{var_name}'

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            filepath = f'{true_dir}/{year}_{month}.npy'
            try:
                data = np.load(filepath)
                data_list.append(data)
                time_list.append((year, month))
            except FileNotFoundError:
                break
        else:
            continue
        break

    return np.array(data_list), time_list


def compute_climatology(true_data, time_list, start_year, end_year):
    """计算月气候态 (12个月)，支持 (time, lat, lon) 格式的二维数据"""
    climatology = np.zeros((12, true_data.shape[1], true_data.shape[2]))
    counts = np.zeros(12)

    for i, (year, month) in enumerate(time_list):
        if start_year <= year <= end_year:
            climatology[month - 1] += true_data[i]
            counts[month - 1] += 1

    for m in range(12):
        if counts[m] > 0:
            climatology[m] /= counts[m]

    return climatology


def load_pred_data_pt(var_name, lead):
    """加载 .pt 格式的预测数据"""
    filepath = f'{PRED_DIR}/{var_name}_step{lead}.pt'
    pred_data = torch.load(filepath, weights_only=False)
    if isinstance(pred_data, torch.Tensor):
        pred_data = pred_data.numpy()

    init_times = json.load(open(INIT_TIMES_FILE))

    pred_times = []
    for init_time in init_times:
        year, month = map(int, init_time.split('_'))
        init_date = datetime(year, month, 1)
        pred_date = init_date + relativedelta(months=lead)
        pred_times.append(f"{pred_date.year}-{pred_date.month:02d}")

    return pred_data, pred_times


def compute_pred_climatology(pred_data, pred_times, start_year, end_year):
    """计算预测数据的月气候态 (12个月)，支持 (time, lat, lon) 格式"""
    climatology = np.zeros((12, pred_data.shape[1], pred_data.shape[2]))
    counts = np.zeros(12)

    for i, time_str in enumerate(pred_times):
        year = int(time_str.split('-')[0])
        month = int(time_str.split('-')[1])
        if start_year <= year <= end_year:
            climatology[month - 1] += pred_data[i]
            counts[month - 1] += 1

    for m in range(12):
        if counts[m] > 0:
            climatology[m] /= counts[m]

    return climatology


def denormalize_pred_data(pred_data, pred_times, mean, std):
    """按预测目标月份对预测数据做反标准化"""
    pred_data_denorm = []
    for i, time_str in enumerate(pred_times):
        month = int(time_str.split('-')[1])
        pred_data_denorm.append(pred_data[i] * std[month - 1] + mean[month - 1])
    return np.array(pred_data_denorm)


def compute_tcc_map(pred_anomalies, true_anomalies, ocean_mask):
    """
    计算每个格点的时间相关系数

    参数:
        pred_anomalies: (N, H, W) 预测异常值
        true_anomalies: (N, H, W) 真实异常值
        ocean_mask: (H, W) 海洋掩码

    返回:
        tcc_map: (H, W) 时间相关系数地图
    """
    H, W = ocean_mask.shape
    tcc_map = np.full((H, W), np.nan)

    # 对每个海洋格点计算时间相关系数
    for i in range(H):
        for j in range(W):
            if ocean_mask[i, j]:
                # 提取该格点的时间序列
                pred_series = pred_anomalies[:, i, j]
                true_series = true_anomalies[:, i, j]

                # 移除 NaN
                valid_mask = ~(np.isnan(pred_series) | np.isnan(true_series))
                pred_valid = pred_series[valid_mask]
                true_valid = true_series[valid_mask]

                # 计算相关系数
                if len(pred_valid) > 2:
                    corr, _ = pearsonr(pred_valid, true_valid)
                    tcc_map[i, j] = corr

    return tcc_map


def plot_tcc_map(tcc_map, ocean_mask, lead, exp_name, var_name, layer_info=''):
    """
    绘制 TCC 热力图

    参数:
        tcc_map: (H, W) TCC 地图
        ocean_mask: (H, W) 海洋掩码
        lead: lead time
        exp_name: 实验名称
        var_name: 变量名称
        layer_info: 层信息字符串 (如 '_layer0')
    """
    # 创建经纬度网格
    lat = np.linspace(-63.5, 63.5, 128)
    lon = np.linspace(0.5, 359.5, 360)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # 将陆地区域设为 NaN
    tcc_map_masked = np.where(ocean_mask, tcc_map, np.nan)

    # 创建图形
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))

    # 设置地图范围
    ax.set_extent([0, 360, -63.5, 63.5], crs=ccrs.PlateCarree())

    # 添加海岸线和陆地
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)

    # 绘制 TCC 热力图
    # 使用 RdBu_r colormap，范围 [-1, 1]
    levels = np.linspace(-1, 1, 21)
    cmap = plt.cm.RdBu_r
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    cf = ax.contourf(lon_grid, lat_grid, tcc_map_masked,
                     levels=levels, cmap=cmap, norm=norm,
                     transform=ccrs.PlateCarree(), extend='both')

    # 添加 TCC=0.5 等高线
    cs = ax.contour(lon_grid, lat_grid, tcc_map_masked,
                    levels=[0.5], colors='black', linewidths=1.5,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    # 添加网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # 保存图形
    save_path = f'./tcc_map_{var_name}{layer_info}_{exp_name}_lead{lead}.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图形已保存至 {save_path}")
    plt.close()


def process_variable(var_name, true_var_name, stat_name, is_multilayer, layer_idx):
    """
    处理单个变量的 TCC 计算

    参数:
        var_name: 预测变量名 (如 'uo', 'vo', 'tos')
        true_var_name: 真实值目录名 (如 'uo', 'tos')
        stat_name: 统计量文件名 (如 'ucur', 'sst')
        is_multilayer: 是否多层变量
        layer_idx: 多层变量的层索引 (默认0)，单层变量忽略此参数
    """
    # 确定层索引
    if is_multilayer and layer_idx is None:
        layer_idx = 0

    layer_info = f'_layer{layer_idx}' if is_multilayer else ''

    print(f"\n{'='*70}")
    print(f"处理变量: {var_name} (真实值: {true_var_name}, 统计量: {stat_name})")
    if is_multilayer:
        print(f"多层变量, 使用第 {layer_idx} 层")
    else:
        print(f"单层变量")
    print(f"Lead: {LEAD_TO_PLOT}")
    print(f"{'='*70}")

    # 1. 加载 land/sea mask
    ocean_mask = load_land_sea_mask(stat_name, is_multilayer)
    print(f"\n[1] Land/sea mask: ocean={np.sum(ocean_mask)}, land={np.sum(~ocean_mask)}")

    # 2. 加载统计量 (mean/std)
    print("\n[2] 加载统计量和真实值数据...")
    mean = np.load(f'{STAT_DIR}/mean/{stat_name}.npy')
    std = np.load(f'{STAT_DIR}/std/{stat_name}.npy')
    print(f"    Mean 形状: {mean.shape}, Std 形状: {std.shape}")

    # 3. 加载真实值数据
    true_data, time_list = load_true_data(true_var_name)
    print(f"    原始真实值数据形状: {true_data.shape}")
    print(f"    时间范围: {time_list[0]} ~ {time_list[-1]}")

    # 反标准化真实值数据
    if is_multilayer:
        # true_data: (time, depth, lat, lon), mean/std: (12, depth, lat, lon)
        true_data_denorm = []
        for i, (year, month) in enumerate(time_list):
            denorm = true_data[i] * std[month - 1] + mean[month - 1]
            true_data_denorm.append(denorm)
        true_data = np.array(true_data_denorm)
        # 提取指定层，变为 (time, lat, lon)
        true_data = true_data[:, layer_idx, :, :]
        # mean/std 也提取对应层，用于后续预测值的反标准化
        mean_layer = mean[:, layer_idx, :, :]  # (12, lat, lon)
        std_layer = std[:, layer_idx, :, :]    # (12, lat, lon)
    else:
        # true_data: (time, lat, lon), mean/std: (12, lat, lon)
        true_data_denorm = []
        for i, (year, month) in enumerate(time_list):
            denorm = true_data[i] * std[month - 1] + mean[month - 1]
            true_data_denorm.append(denorm)
        true_data = np.array(true_data_denorm)
        mean_layer = mean  # (12, lat, lon)
        std_layer = std    # (12, lat, lon)

    print(f"    反标准化后真实值数据形状: {true_data.shape}")

    # 4. 计算真实值气候态 (已是二维 (time, lat, lon))
    print(f"\n[3] 计算真实值气候态 ({CLIM_START_YEAR}-{CLIM_END_YEAR})...")
    true_climatology = compute_climatology(true_data, time_list, CLIM_START_YEAR, CLIM_END_YEAR)

    # 5. 构建真实值时间索引
    time_to_idx = {}
    for i, (year, month) in enumerate(time_list):
        time_to_idx[(year, month)] = i

    # 6. 加载预测数据
    print(f"\n[4] 加载 Lead {LEAD_TO_PLOT} 预测数据...")
    pred_data, pred_times = load_pred_data_pt(var_name, LEAD_TO_PLOT)
    print(f"    原始预测数据形状: {pred_data.shape}")

    if is_multilayer:
        # pred_data: (time, depth, lat, lon)
        # 先反标准化完整数据，再提取指定层
        pred_data = denormalize_pred_data(pred_data, pred_times, mean, std)
        pred_data = pred_data[:, layer_idx, :, :]  # (time, lat, lon)
    else:
        # pred_data: (time, lat, lon)
        pred_data = denormalize_pred_data(pred_data, pred_times, mean_layer, std_layer)

    print(f"    处理后预测数据形状: {pred_data.shape}")

    # 计算预测气候态
    pred_climatology = compute_pred_climatology(pred_data, pred_times, CLIM_START_YEAR, CLIM_END_YEAR)

    # 7. 收集评估时间段内的异常值
    print(f"\n[5] 收集评估时间段 ({EVAL_START}-{EVAL_END}) 的异常值...")
    pred_anomalies = []
    true_anomalies = []

    for i, time_str in enumerate(pred_times):
        year = int(time_str.split('-')[0])
        month = int(time_str.split('-')[1])

        # 只评估 EVAL_START ~ EVAL_END 范围
        if year < EVAL_START or year > EVAL_END:
            continue

        # 获取对应真实值
        if (year, month) not in time_to_idx:
            continue

        true_idx = time_to_idx[(year, month)]

        # 计算异常值
        pred_anomaly = pred_data[i] - pred_climatology[month - 1]
        true_anomaly = true_data[true_idx] - true_climatology[month - 1]

        pred_anomalies.append(pred_anomaly)
        true_anomalies.append(true_anomaly)

    pred_anomalies = np.array(pred_anomalies)  # (N, H, W)
    true_anomalies = np.array(true_anomalies)
    print(f"    收集到 {len(pred_anomalies)} 个时间步")

    # 8. 计算 TCC 地图
    print("\n[6] 计算每个格点的时间相关系数...")
    tcc_map = compute_tcc_map(pred_anomalies, true_anomalies, ocean_mask)

    # 统计信息
    valid_tcc = tcc_map[~np.isnan(tcc_map)]
    print(f"    TCC 统计:")
    print(f"      均值: {np.mean(valid_tcc):.4f}")
    print(f"      中位数: {np.median(valid_tcc):.4f}")
    print(f"      最小值: {np.min(valid_tcc):.4f}")
    print(f"      最大值: {np.max(valid_tcc):.4f}")
    print(f"      TCC > 0.5 的格点比例: {np.sum(valid_tcc > 0.5) / len(valid_tcc) * 100:.2f}%")

    # 9. 绘制 TCC 地图
    print("\n[7] 绘制 TCC 热力图...")
    plot_tcc_map(tcc_map, ocean_mask, LEAD_TO_PLOT, exp_name, var_name, layer_info)

    # 10. 保存 TCC 数据
    save_path = f'./tcc_map_{var_name}{layer_info}_{exp_name}_lead{LEAD_TO_PLOT}.npz'
    np.savez(save_path,
             tcc_map=tcc_map,
             ocean_mask=ocean_mask,
             var_name=var_name,
             layer_idx=layer_idx if is_multilayer else -1,
             lead=LEAD_TO_PLOT,
             eval_start=EVAL_START,
             eval_end=EVAL_END)
    print(f"TCC 数据已保存至 {save_path}")

    return tcc_map


def main():
    print("=" * 70)
    print("多变量时间相关系数 (TCC) 计算")
    print(f"Lead: {LEAD_TO_PLOT}, 评估时间: {EVAL_START}-{EVAL_END}")
    print("=" * 70)

    all_results = {}

    for var_name, (true_var_name, stat_name, is_multilayer, layer_idx) in VAR_MAPPING.items():
        try:
            tcc_map = process_variable(var_name, true_var_name, stat_name, is_multilayer, layer_idx)
            all_results[var_name] = tcc_map
        except Exception as e:
            print(f"\n错误: 处理变量 {var_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 总结
    print("\n" + "=" * 70)
    print("所有变量 TCC 总结")
    print("=" * 70)
    for var_name, tcc_map in all_results.items():
        valid_tcc = tcc_map[~np.isnan(tcc_map)]
        is_multilayer = VAR_MAPPING[var_name][2]
        layer_idx = VAR_MAPPING[var_name][3]
        layer_str = f" (层 {layer_idx})" if is_multilayer else ""
        print(f"  {var_name}{layer_str}: 均值={np.mean(valid_tcc):.4f}, "
              f"中位数={np.median(valid_tcc):.4f}, "
              f"TCC>0.5 比例={np.sum(valid_tcc > 0.5) / len(valid_tcc) * 100:.1f}%")

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
