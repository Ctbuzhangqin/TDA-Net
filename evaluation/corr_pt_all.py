"""
计算 Nino3.4 指数的预测相关系数（适配 predict_all.py 的输出）

- 预测文件: ./output/test/<exp_name>/preds/tos_step1.pt ~ tos_step24.pt
- 初始化时间: ./output/test/<exp_name>/init_times.json
- 真实值文件: ./dataset/GODAS/tos/YYYY_M.npy
- Nino3.4 区域: [59:69, 190:241]
- 气候态: 1991-2020年

与旧版 corr_pt.py 的区别：
- 不再假设不同 lead 需要用不同的数组切片索引对齐评估区间
- 直接按预测目标时间 pred_times 过滤 EVAL_START ~ EVAL_END
- 按实际年份读取 TRUE_DIR 下的真实月份，默认截至 EVAL_END
"""

import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr


# File locations are overridden by scripts/evaluate_nino.py.
SST_MEAN_FILE = 'resources/sst_mean.npy'
SST_STD_FILE = 'resources/sst_std.npy'
# 配置
exp_name = 'tda_ensemble'

PRED_DIR = f'./output/test/{exp_name}/preds'
INIT_TIMES_FILE = f'./output/test/{exp_name}/init_times.json'
TRUE_DIR = './dataset/GODAS/tos'
NINO34_LAT_SLICE = slice(59, 69)
NINO34_LON_SLICE = slice(190, 241)  # nino3.4：slice(190, 241) nino4: slice(160, 211), nino3: slice(210, 271)
# 气候态时间，闭区间
CLIM_START_YEAR = 1991
CLIM_END_YEAR = 2020
# 在GODAS数据集上评估的时间段
EVAL_START = 2011
EVAL_END = 2023
# 预报时效
NUM_LEADS = 24


def ensure_numpy_array(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def parse_year_month(time_str):
    year, month = map(int, time_str.split('_'))
    return year, month


def parse_pred_time(time_str):
    year, month = map(int, time_str.split('-'))
    return year, month


def add_months(time_str, months):
    year, month = parse_year_month(time_str)
    month_index = year * 12 + (month - 1) + months
    new_year, new_month_idx = divmod(month_index, 12)
    return f"{new_year}-{new_month_idx + 1:02d}"


def runmean(data, window=3):
    n = len(data)
    result = np.zeros(n)

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        result[i] = np.nanmean(data[start:end])

    return result


def load_true_data(start_year=1980, end_year=EVAL_END):
    data_list = []
    time_list = []

    file_list = []
    for filename in os.listdir(TRUE_DIR):
        if not filename.endswith('.npy'):
            continue
        stem = filename[:-4]
        year, month = parse_year_month(stem)
        if year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        file_list.append((year, month, filename))

    file_list.sort(key=lambda x: x[0] * 12 + x[1])

    for year, month, filename in file_list:
        data = np.load(os.path.join(TRUE_DIR, filename))
        data_list.append(data)
        time_list.append((year, month))

    return np.array(data_list), time_list


def compute_climatology(true_data, time_list, start_year=1985, end_year=2018):
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
    n_lats = lat_slice.stop - lat_slice.start
    lats = np.linspace(-5, 5, n_lats)
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.sum()
    return weights.reshape(-1, 1)


def compute_nino34_index(data, climatology, time_list):
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
    filepath = f'{PRED_DIR}/tos_step{lead}.pt'
    pred_data = ensure_numpy_array(torch.load(filepath, weights_only=False))

    with open(INIT_TIMES_FILE, 'r') as f:
        init_times = json.load(f)

    pred_times = [add_months(init_time, lead) for init_time in init_times]
    return pred_data, pred_times


def compute_pred_climatology(pred_data, pred_times, start_year=1985, end_year=2018):
    pred_data = ensure_numpy_array(pred_data)
    climatology = np.zeros((12, pred_data.shape[1], pred_data.shape[2]))
    counts = np.zeros(12)

    for i, time_str in enumerate(pred_times):
        year, month = parse_pred_time(time_str)
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
    pred_data = ensure_numpy_array(pred_data)
    nino34_indices = []
    lat_weights = get_latitude_weights(NINO34_LAT_SLICE)

    for i, time_str in enumerate(pred_times):
        _, month = parse_pred_time(time_str)
        anomaly = pred_data[i] - climatology[month - 1]
        nino34_region = anomaly[NINO34_LAT_SLICE, NINO34_LON_SLICE]
        valid_mask = ~np.isnan(nino34_region)
        weighted_sum = np.nansum(nino34_region * lat_weights)
        weight_sum = np.sum(valid_mask.astype(float) * lat_weights)
        nino34_index = weighted_sum / weight_sum if weight_sum > 0 else np.nan
        nino34_indices.append(nino34_index)

    return runmean(np.array(nino34_indices))


def get_true_nino34_for_lead(true_nino34, time_list, pred_times):
    time_to_idx = {(year, month): i for i, (year, month) in enumerate(time_list)}

    true_indices = []
    valid_mask = []

    for time_str in pred_times:
        year, month = parse_pred_time(time_str)
        if (year, month) in time_to_idx:
            true_indices.append(true_nino34[time_to_idx[(year, month)]])
            valid_mask.append(True)
        else:
            true_indices.append(np.nan)
            valid_mask.append(False)

    return np.array(true_indices), np.array(valid_mask)


def get_eval_time_mask(pred_times, start_year, end_year=None):
    mask = []
    for time_str in pred_times:
        year, _ = parse_pred_time(time_str)
        in_window = year >= start_year and (end_year is None or year <= end_year)
        mask.append(in_window)
    return np.array(mask)


def oni_phase(oni):
    oni = np.asarray(oni)
    return np.where(oni > 0.5, 1, np.where(oni < -0.5, -1, 0))


def compute_phase_stats(true_oni, pred_oni):
    valid = np.isfinite(true_oni) & np.isfinite(pred_oni)
    n = int(np.sum(valid))
    if n == 0:
        return {
            'in_phase_count': 0,
            'not_in_phase_count': 0,
            'out_of_phase_count': 0,
            'in_phase_rate': np.nan,
            'not_in_phase_rate': np.nan,
            'out_of_phase_rate': np.nan,
        }

    true_phase = oni_phase(true_oni[valid])
    pred_phase = oni_phase(pred_oni[valid])
    in_phase_count = int(np.sum(true_phase == pred_phase))
    out_of_phase_count = int(np.sum(true_phase * pred_phase == -1))
    not_in_phase_count = n - in_phase_count

    return {
        'in_phase_count': in_phase_count,
        'not_in_phase_count': not_in_phase_count,
        'out_of_phase_count': out_of_phase_count,
        'in_phase_rate': in_phase_count / n,
        'not_in_phase_rate': not_in_phase_count / n,
        'out_of_phase_rate': out_of_phase_count / n,
    }


def main():
    print("=" * 60)
    print("Nino3.4 预测相关系数计算 (.pt 格式, for predict_all.py)")
    print("=" * 60)

    print("\n[1] 加载真实值数据...")
    mean = np.load(SST_MEAN_FILE)
    std = np.load(SST_STD_FILE)
    true_data, time_list = load_true_data()
    true_data = true_data.reshape(-1, mean.shape[0], mean.shape[1], mean.shape[2])
    true_data = true_data * std + mean
    true_data = true_data.reshape(-1, mean.shape[1], mean.shape[2])
    latest_true_year = time_list[-1][0]
    eval_end_year = latest_true_year if EVAL_END is None else min(EVAL_END, latest_true_year)
    print(f"    真实值数据形状: {true_data.shape}")
    print(f"    时间范围: {time_list[0]} ~ {time_list[-1]}")
    print(f"    评估年份: {EVAL_START} ~ {eval_end_year}")

    print(f"\n[2] 计算气候态 ({CLIM_START_YEAR}-{CLIM_END_YEAR})...")
    climatology = compute_climatology(true_data, time_list, CLIM_START_YEAR, CLIM_END_YEAR)
    print(f"    气候态形状: {climatology.shape}")

    print("\n[3] 计算真实值 Nino3.4 指数...")
    true_nino34 = compute_nino34_index(true_data, climatology, time_list)
    print(f"    真实 Nino3.4 指数长度: {len(true_nino34)}")

    print("\n[4] 计算各 lead 的相关系数...")
    print(f'''{exp_name}
Nino3.4 Index Correlation and RMSE''')
    print("-" * 112)
    print(
        f"{'Lead':<8} {'Correlation':<15} {'RMSE':<15} {'N':<8} "
        f"{'InPhase%':<12} {'NotInPhase%':<14} {'OutPhase%':<12} {'EvalRange':<10}"
    )
    print("-" * 112)

    correlations = []
    rmses = []
    maes = []
    counts = []
    in_phase_counts = []
    not_in_phase_counts = []
    out_of_phase_counts = []
    in_phase_rates = []
    not_in_phase_rates = []
    out_of_phase_rates = []

    for lead in range(1, NUM_LEADS + 1):
        pred_data, pred_times = load_pred_data_pt(lead)
        pred_data = denormalize_pred_data(pred_data, pred_times, mean, std)
        pred_climatology = compute_pred_climatology(pred_data, pred_times, CLIM_START_YEAR, CLIM_END_YEAR)
        pred_nino34 = compute_pred_nino34_index(pred_data, pred_climatology, pred_times)
        true_nino34_aligned, valid_mask = get_true_nino34_for_lead(true_nino34, time_list, pred_times)

        eval_time_mask = get_eval_time_mask(pred_times, EVAL_START, eval_end_year)
        nan_mask = ~(np.isnan(pred_nino34) | np.isnan(true_nino34_aligned))
        mask = valid_mask & eval_time_mask & nan_mask

        pred_eval = pred_nino34[mask]
        true_eval = true_nino34_aligned[mask]
        pred_times_eval = np.array(pred_times)[mask]

        # if lead == 18 or lead == 3 or lead == 6 or lead == 12:
        #     np.save(f'oni_pred_lead_{lead}.npy', pred_eval)
        # if lead == 15:
        #     np.save(f'oni_pred_lead_{lead}.npy', pred_eval)
        # if lead == 1:
        #     np.save('oni_true_2011-2023.npy', true_eval)
        # if lead == 1:
        #     save_mask = np.array([
        #         1980 <= year <= 2023 for year, month in time_list
        #     ])
        #     np.save('oni_1980_2023.npy', true_nino34[save_mask])

        if len(pred_eval) > 2:
            corr, _ = pearsonr(pred_eval, true_eval)
            rmse = np.sqrt(np.mean((pred_eval - true_eval) ** 2))
            mae = np.mean(np.abs(pred_eval - true_eval))
        else:
            corr = np.nan
            rmse = np.nan
            mae = np.nan

        phase_stats = compute_phase_stats(true_eval, pred_eval)

        correlations.append(corr)
        rmses.append(rmse)
        maes.append(mae)
        counts.append(len(pred_eval))
        in_phase_counts.append(phase_stats['in_phase_count'])
        not_in_phase_counts.append(phase_stats['not_in_phase_count'])
        out_of_phase_counts.append(phase_stats['out_of_phase_count'])
        in_phase_rates.append(phase_stats['in_phase_rate'])
        not_in_phase_rates.append(phase_stats['not_in_phase_rate'])
        out_of_phase_rates.append(phase_stats['out_of_phase_rate'])

        if len(pred_times_eval) > 0:
            eval_range = f"{pred_times_eval[0]}~{pred_times_eval[-1]}"
        else:
            eval_range = "N/A"

        print(
            f"{lead:<8} {corr:<15.4f} {rmse:<15.4f} {len(pred_eval):<8} "
            f"{phase_stats['in_phase_rate'] * 100:<12.2f} "
            f"{phase_stats['not_in_phase_rate'] * 100:<14.2f} "
            f"{phase_stats['out_of_phase_rate'] * 100:<12.2f} {eval_range}"
        )
    print(correlations)
    print(rmses)
    print(maes)
    print("-" * 112)

    results = {
        'leads': np.arange(1, NUM_LEADS + 1),
        'correlations': np.array(correlations),
        'rmses': np.array(rmses),
        'maes': np.array(maes),
        'counts': np.array(counts),
        'in_phase_counts': np.array(in_phase_counts),
        'not_in_phase_counts': np.array(not_in_phase_counts),
        'out_of_phase_counts': np.array(out_of_phase_counts),
        'in_phase_rates': np.array(in_phase_rates),
        'not_in_phase_rates': np.array(not_in_phase_rates),
        'out_of_phase_rates': np.array(out_of_phase_rates),
    }
    np.savez('./nino34_correlation_pt_all.npz', **results)
    print("\n结果已保存至 ./nino34_correlation_pt_all.npz")

    # print("\n" + "=" * 60)
    # print("摘要")
    # print("=" * 60)
    # for i in range(NUM_LEADS):
    #     print(f"Lead {i+1:02d}: 相关系数 = {correlations[i]:.4f}")
    #     print(f"         RMSE = {rmses[i]:.4f}")
    #     print(f"         N = {counts[i]}")


if __name__ == '__main__':
    main()
