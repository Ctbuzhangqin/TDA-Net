"""
按起报月份计算 Nino3.4 指数的预测相关系数 (24个lead × 12个起报月份)
- 预测文件: ./output/test/tda_ensemble/preds/tos_step1.pt ~ tos_step24.pt
- 初始化时间: ./output/test/tda_ensemble/init_times.json
- 真实值文件: ./dataset/GODAS/tos/YYYY_M.npy
- Nino3.4 区域: [59:69, 190:241]
- 气候态: 1991-2020年
- 评估时间: 2011-2023年
- 按起报月份分组计算相关系数
"""

import numpy as np
import torch
import json
from scipy.stats import pearsonr
from dateutil.relativedelta import relativedelta
from datetime import datetime


# File locations are overridden by scripts/evaluate_nino.py.
SST_MEAN_FILE = 'resources/sst_mean.npy'
SST_STD_FILE = 'resources/sst_std.npy'
# 配置
exp_name = 'tda_ensemble'
PRED_DIR = f'./output/test/{exp_name}/preds'
INIT_TIMES_FILE = f'./output/test/{exp_name}/init_times.json'
TRUE_DIR = './dataset/GODAS/tos'
NINO34_LAT_SLICE = slice(59, 69)
NINO34_LON_SLICE = slice(190, 241)
CLIM_START_YEAR = 1991
CLIM_END_YEAR = 2020
EVAL_START = 2011
EVAL_END = 2023
NUM_LEADS = 24


def runmean(data, window=3):
    """计算滑动平均"""
    n = len(data)
    result = np.zeros(n)

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        result[i] = np.nanmean(data[start:end])

    return result


def load_true_data(start_year=1980, end_year=EVAL_END):
    """加载真实值数据"""
    data_list = []
    time_list = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            filepath = f'{TRUE_DIR}/{year}_{month}.npy'
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
    """计算月气候态 (12个月)"""
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


def get_latitude_weights(lat_slice):
    """计算纬度权重 (基于 cos(latitude))"""
    start_idx = lat_slice.start
    stop_idx = lat_slice.stop
    n_lats = stop_idx - start_idx
    lats = np.linspace(-5, 5, n_lats)
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.sum()
    return weights.reshape(-1, 1)


def compute_nino34_index(data, climatology, time_list):
    """计算 Nino3.4 指数 (SST异常的区域平均，基于纬度加权)"""
    nino34_indices = []
    lat_weights = get_latitude_weights(NINO34_LAT_SLICE)

    for i, (_, month) in enumerate(time_list):
        anomaly = data[i] - climatology[month - 1]
        nino34_region = anomaly[NINO34_LAT_SLICE, NINO34_LON_SLICE]
        valid_mask = ~np.isnan(nino34_region)
        weighted_sum = np.nansum(nino34_region * lat_weights)
        weight_sum = np.sum(valid_mask.astype(float) * lat_weights)
        nino34_index = weighted_sum / weight_sum if weight_sum > 0 else np.nan
        nino34_indices.append(nino34_index)

    return runmean(np.array(nino34_indices))


def load_pred_data_pt(lead):
    """加载 .pt 格式的预测数据"""
    filepath = f'{PRED_DIR}/tos_step{lead}.pt'
    pred_data = torch.load(filepath, weights_only=False)
    init_times = json.load(open(INIT_TIMES_FILE))

    pred_times = []
    init_months = []
    for init_time in init_times:
        year, month = map(int, init_time.split('_'))
        init_date = datetime(year, month, 1)
        pred_date = init_date + relativedelta(months=lead)
        pred_times.append(f"{pred_date.year}-{pred_date.month:02d}")
        init_months.append(month)  # 记录起报月份

    return pred_data, pred_times, init_months


def compute_pred_climatology(pred_data, pred_times, start_year, end_year):
    """计算预测数据的月气候态 (12个月)"""
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


def compute_pred_nino34_index(pred_data, climatology, pred_times):
    """计算预测的 Nino3.4 指数 (基于纬度加权)"""
    nino34_indices = []
    lat_weights = get_latitude_weights(NINO34_LAT_SLICE)

    for i, time_str in enumerate(pred_times):
        month = int(time_str.split('-')[1])
        anomaly = pred_data[i] - climatology[month - 1]
        nino34_region = anomaly[NINO34_LAT_SLICE, NINO34_LON_SLICE]
        valid_mask = ~np.isnan(nino34_region)
        weighted_sum = np.nansum(nino34_region * lat_weights)
        weight_sum = np.sum(valid_mask.astype(float) * lat_weights)
        nino34_index = weighted_sum / weight_sum if weight_sum > 0 else np.nan
        nino34_indices.append(nino34_index)

    return runmean(np.array(nino34_indices))


def get_true_nino34_for_lead(true_nino34, time_list, pred_times):
    """根据预测时间获取对应的真实 Nino3.4 指数"""
    time_to_idx = {}
    for i, (year, month) in enumerate(time_list):
        time_to_idx[(year, month)] = i

    true_indices = []
    valid_mask = []

    for time_str in pred_times:
        year = int(time_str.split('-')[0])
        month = int(time_str.split('-')[1])

        if (year, month) in time_to_idx:
            true_indices.append(true_nino34[time_to_idx[(year, month)]])
            valid_mask.append(True)
        else:
            true_indices.append(np.nan)
            valid_mask.append(False)

    return np.array(true_indices), np.array(valid_mask)


def get_season_name(month):
    """
    根据月份获取三月季节名称
    由于 runmean(window=3)，每个月代表前后三个月的平均
    例如：1月 -> DJF (12-1-2月)
    """
    season_names = [
        'DJF',  # 1月: Dec-Jan-Feb
        'JFM',  # 2月: Jan-Feb-Mar
        'FMA',  # 3月: Feb-Mar-Apr
        'MAM',  # 4月: Mar-Apr-May
        'AMJ',  # 5月: Apr-May-Jun
        'MJJ',  # 6月: May-Jun-Jul
        'JJA',  # 7月: Jun-Jul-Aug
        'JAS',  # 8月: Jul-Aug-Sep
        'ASO',  # 9月: Aug-Sep-Oct
        'SON',  # 10月: Sep-Oct-Nov
        'OND',  # 11月: Oct-Nov-Dec
        'NDJ',  # 12月: Nov-Dec-Jan
    ]
    return season_names[month - 1]


def main():
    print("=" * 80)
    print("Nino3.4 预测相关系数计算 - 按起报月份分组")
    print("=" * 80)

    # 1. 加载真实值数据
    print("\n[1] 加载真实值数据...")
    mean = np.load(SST_MEAN_FILE)
    std = np.load(SST_STD_FILE)
    true_data, time_list = load_true_data()
    true_data = true_data.reshape(-1, mean.shape[0], mean.shape[1], mean.shape[2])
    true_data = true_data * std + mean
    true_data = true_data.reshape(-1, mean.shape[1], mean.shape[2])
    print(f"    真实值数据形状: {true_data.shape}")
    print(f"    时间范围: {time_list[0]} ~ {time_list[-1]}")

    # 2. 计算气候态
    print(f"\n[2] 计算气候态 ({CLIM_START_YEAR}-{CLIM_END_YEAR})...")
    climatology = compute_climatology(true_data, time_list, CLIM_START_YEAR, CLIM_END_YEAR)

    # 3. 计算真实值的 Nino3.4 指数
    print("\n[3] 计算真实值 Nino3.4 指数...")
    true_nino34 = compute_nino34_index(true_data, climatology, time_list)

    # 4. 按起报月份计算相关系数和 RMSE
    print(f"\n[4] 按起报月份计算各 lead 的相关系数和 RMSE ({EVAL_START}-{EVAL_END})...")

    # 存储结果: (24 leads, 12 months)
    correlations_seasonal = np.zeros((NUM_LEADS, 12))
    rmse_seasonal = np.zeros((NUM_LEADS, 12))
    sample_counts = np.zeros((NUM_LEADS, 12), dtype=int)

    for lead in range(1, NUM_LEADS + 1):
        # 加载预测数据
        pred_data, pred_times, init_months = load_pred_data_pt(lead)
        pred_data = denormalize_pred_data(pred_data, pred_times, mean, std)
        # 计算预测气候态
        pred_climatology = compute_pred_climatology(pred_data, pred_times, CLIM_START_YEAR, CLIM_END_YEAR)

        # 计算预测的 Nino3.4 指数
        pred_nino34 = compute_pred_nino34_index(pred_data, pred_climatology, pred_times)

        # 获取对应的真实 Nino3.4 指数
        true_nino34_aligned, valid_mask = get_true_nino34_for_lead(true_nino34, time_list, pred_times)

        # 按起报月份分组
        for init_month in range(1, 13):
            # 筛选该起报月份的样本
            month_mask = np.array(init_months) == init_month
            combined_mask = month_mask & valid_mask

            pred_month = pred_nino34[combined_mask]
            true_month = true_nino34_aligned[combined_mask]
            pred_times_month = [pred_times[i] for i in range(len(pred_times)) if combined_mask[i]]

            # 筛选评估时间范围
            eval_mask = []
            for time_str in pred_times_month:
                year = int(time_str.split('-')[0])
                eval_mask.append(EVAL_START <= year <= EVAL_END)
            eval_mask = np.array(eval_mask)

            pred_eval = pred_month[eval_mask]
            true_eval = true_month[eval_mask]

            # 移除 NaN
            mask = ~(np.isnan(pred_eval) | np.isnan(true_eval))
            pred_clean = pred_eval[mask]
            true_clean = true_eval[mask]

            # 计算相关系数和 RMSE
            if len(pred_clean) > 2:
                corr, _ = pearsonr(pred_clean, true_clean)
                rmse = np.sqrt(np.mean((pred_clean - true_clean) ** 2))
            else:
                corr = np.nan
                rmse = np.nan

            correlations_seasonal[lead - 1, init_month - 1] = corr
            rmse_seasonal[lead - 1, init_month - 1] = rmse
            sample_counts[lead - 1, init_month - 1] = len(pred_clean)

    # 5. 输出结果
    print("\n" + "=" * 110)
    print("相关系数 (按起报季节)")
    print("=" * 110)
    print(f"{'Lead':<6}", end='')
    for m in range(1, 13):
        season = get_season_name(m)
        print(f"{season:<9}", end='')
    print()
    print("-" * 110)

    for lead in range(1, NUM_LEADS + 1):
        print(f"{lead:<6}", end='')
        for m in range(12):
            corr = correlations_seasonal[lead - 1, m]
            if np.isnan(corr):
                print(f"{'--':<9}", end='')
            else:
                print(f"{corr:<9.3f}", end='')
        print()

    print("\n" + "=" * 110)
    print("RMSE (按起报季节)")
    print("=" * 110)
    print("{:<6}".format("Lead"), end="")
    for m in range(1, 13):
        season = get_season_name(m)
        print(f"{season:<9}", end="")
    print()
    print("-" * 110)

    for lead in range(1, NUM_LEADS + 1):
        print(f"{lead:<6}", end="")
        for m in range(12):
            rmse = rmse_seasonal[lead - 1, m]
            if np.isnan(rmse):
                print("{:<9}".format("--"), end="")
            else:
                print(f"{rmse:<9.3f}", end="")
        print()

    # 6. 保存结果
    results = {
        'leads': np.arange(1, NUM_LEADS + 1),
        'months': np.arange(1, 13),
        'season_names': [get_season_name(m) for m in range(1, 13)],
        'correlations_seasonal': correlations_seasonal,
        'rmse_seasonal': rmse_seasonal,
        'sample_counts': sample_counts,
    }
    save_path = f'./nino34_correlation_seasonal_{exp_name}.npz'
    np.savez(save_path, **results)
    print(f"\n结果已保存至 {save_path}")

    # 7. 输出样本数统计
    print("\n" + "=" * 110)
    print("样本数 (按起报季节)")
    print("=" * 110)
    print(f"{'Lead':<6}", end='')
    for m in range(1, 13):
        season = get_season_name(m)
        print(f"{season:<9}", end='')
    print()
    print("-" * 110)

    for lead in range(1, NUM_LEADS + 1):
        print(f"{lead:<6}", end='')
        for m in range(12):
            count = sample_counts[lead - 1, m]
            print(f"{count:<9}", end='')
        print()


if __name__ == '__main__':
    main()
