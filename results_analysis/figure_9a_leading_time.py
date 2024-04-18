import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from config.file_path_config import base_path, pos_dataset_path


def get_max_time_diff_row(group):
    return group[group['time_diff'] == group['time_diff'].max()]

def start_plot():
    pos_df = pd.read_pickle(pos_dataset_path)
    pos_df['time'] = pd.to_datetime(pos_df['time'], unit='s')
    pos_df['sample_time'] = pd.to_datetime(pos_df['sample_time'], unit='s')

    pos_df['time'] = pos_df['time'] - pd.Timedelta(days=10)
    pos_df['sample_time'] = pos_df['sample_time'] - pd.Timedelta(days=10)

    pred_csv = pd.read_csv(base_path + 'model_comparision/vm_threshold_0.42_seed_80_First_pred.csv', header=0,
                           usecols=['nc_ip', 'instance_id', 'sample_time', 'y_p', 'y_t'])
    pred_csv['sample_time'] = pd.to_datetime(pred_csv['sample_time'], unit='s')

    pred_csv = pred_csv[(pred_csv['y_p'] == 1) & (pred_csv['y_t'] == 1)]

    result = pd.merge(pos_df, pred_csv, on=['nc_ip', 'instance_id', 'sample_time'], how='inner')

    result['time_diff'] = result['time'] - result['sample_time']

    max_diff_rows = pd.concat([get_max_time_diff_row(group) for _, group in result.groupby('instance_id')])

    max_diff_rows['duration_days'] = np.ceil(max_diff_rows['time_diff'].dt.total_seconds() / (24 * 3600))

    max_diff_rows = max_diff_rows.reset_index(drop=True)

    duration_days_counts = max_diff_rows['duration_days'].value_counts()

    sorted_duration_days = duration_days_counts.sort_index()
    cumulative = np.cumsum(sorted_duration_days) / sum(sorted_duration_days)

    plt.figure(figsize=(6, 6))
    plt.plot(cumulative, linestyle='-', linewidth=2.5)
    ax = plt.gca()
    x_major_step = (ax.get_xticks()[1] - ax.get_xticks()[0]) / 5
    y_major_step = (ax.get_yticks()[1] - ax.get_yticks()[0]) / 5
    ax.xaxis.set_minor_locator(MultipleLocator(x_major_step))
    ax.yaxis.set_minor_locator(MultipleLocator(y_major_step))

    target_cumulative = 0.34110
    target_index = [index for index, value in enumerate(cumulative) if value >= target_cumulative][0]
    target_days = sorted_duration_days.index[target_index]
    target_value = cumulative[target_days]
    ax.scatter(target_days, target_value, color='red', s=20, edgecolor='black', linewidths=0, zorder=3)
    ax.annotate('(34%)',
                xy=(target_days, target_value),
                xytext=(target_days + 50, target_value),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=1),
                verticalalignment='center',
                horizontalalignment='right')

    target_cumulative = 0.63566
    target_index = [index for index, value in enumerate(cumulative) if value >= target_cumulative][0]
    target_days = sorted_duration_days.index[target_index]
    target_value = cumulative[target_days]
    ax.scatter(target_days, target_value, color='red', s=20, edgecolor='black', linewidths=0, zorder=3)
    ax.annotate('(63%)',
                xy=(target_days, target_value),
                xytext=(target_days + 50, target_value),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=1),
                verticalalignment='center',
                horizontalalignment='right')

    target_cumulative = 0.71318

    target_index = [index for index, value in enumerate(cumulative) if value >= target_cumulative][0]
    target_days = sorted_duration_days.index[target_index]
    target_value = cumulative[target_days]
    ax.scatter(target_days, target_value, color='red', s=20, edgecolor='black', linewidths=0, zorder=3)
    ax.annotate('(73%)',
                xy=(target_days, target_value),
                xytext=(target_days + 50, target_value),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=1),
                verticalalignment='center',
                horizontalalignment='right')


    target_cumulative = 0.76744
    target_index = [index for index, value in enumerate(cumulative) if value >= target_cumulative][0]
    target_days = sorted_duration_days.index[target_index]
    target_value = cumulative[target_days]
    ax.scatter(target_days, target_value, color='red', s=20, edgecolor='black', linewidths=0, zorder=3)
    ax.annotate('(76%)',
                xy=(target_days, target_value),
                xytext=(target_days + 50, target_value),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=1),
                verticalalignment='center',
                horizontalalignment='right')


    target_cumulative = 0.875
    target_index = [index for index, value in enumerate(cumulative) if value >= target_cumulative][0]
    target_days = sorted_duration_days.index[target_index]
    target_value = cumulative[target_days]
    ax.scatter(target_days, target_value, color='red', s=20, edgecolor='black', linewidths=0, zorder=3)
    ax.annotate('(87%)',
                xy=(target_days, target_value),
                xytext=(target_days + 50, target_value),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=1),
                verticalalignment='center',
                horizontalalignment='right')

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.savefig(base_path+'figure/figure_9a.pdf', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    start_plot()