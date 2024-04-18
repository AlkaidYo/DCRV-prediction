import sys

sys.path.append('../')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator
import argparse
from config.file_path_config import base_path, ref_10_node_migration_cost, ref_10_node_repair_cost, \
    ref_10_node_crash_cost

cost_vm_migration = 1
cost_nc_migration = ref_10_node_migration_cost[0] + ref_10_node_repair_cost[0]  # 5,10,15
cost_nc_downtime = ref_10_node_crash_cost[1] + ref_10_node_repair_cost[0]  # 50,250,750


def operate_arrays(truth_down_num, TP_value, FP_value, FN_value, operation):
    return [operation(truth_down_num, x, y, z) for x, y, z in zip(TP_value, FP_value, FN_value)]


def func_1(truth_down_num, TP_value, FP_value, FN_value):
    return (cost_nc_downtime * truth_down_num / 0.8 - (
            cost_nc_migration * TP_value + cost_nc_migration * FP_value + cost_nc_downtime * (
            FN_value + truth_down_num * 0.25))) / (
            cost_nc_downtime * truth_down_num / 0.8)


def func_2(truth_down_num, TP_value, FP_value, FN_value):
    return (cost_nc_downtime * truth_down_num / 0.8 - (
            TP_value + FP_value + cost_nc_downtime * (FN_value + truth_down_num * 0.25))) / (
            cost_nc_downtime * truth_down_num / 0.8)


def start_plot(seed_value=80):
    confusion_metrix_df_vm = pd.read_csv(
        base_path + 'results/XGB/vm_test_result_tree_30_seed_' + str(seed_value) + '_First.csv', header=0)
    confusion_metrix_df_nc = pd.read_csv(
        base_path + 'results/XGB/nc_test_result_tree_30_seed_' + str(seed_value) + '_First.csv', header=0)
    truth_down_num = confusion_metrix_df_nc['tp'].iloc[0] + confusion_metrix_df_nc['fn'].iloc[0]
    start = 0.4
    stop = 1.0
    step = int((stop - start) / 0.02)
    x_values = np.linspace(start, stop, step)

    VM_TP_value = confusion_metrix_df_vm['tp'].tolist()
    VM_FP_value = confusion_metrix_df_vm['fp'].tolist()
    VM_TN_value = confusion_metrix_df_vm['tn'].tolist()
    VM_FN_value = confusion_metrix_df_vm['fn'].tolist()

    NC_TP_value = confusion_metrix_df_nc['tp'].tolist()
    NC_FP_value = confusion_metrix_df_nc['fp'].tolist()
    NC_TN_value = confusion_metrix_df_nc['tn'].tolist()
    NC_FN_value = confusion_metrix_df_nc['fn'].tolist()

    y_func2 = operate_arrays(truth_down_num, VM_TP_value, VM_FP_value, VM_FN_value, func_2)
    y_func1 = operate_arrays(truth_down_num, NC_TP_value, NC_FP_value, NC_FN_value, func_1)

    plt.figure(figsize=(6, 6))
    plt.plot(x_values, y_func2, marker='o', label='VM', linewidth=2.5)
    plt.plot(x_values, y_func1, marker='x', label='Node', linewidth=2.5)
    plt.xlabel('Threshold')

    plt.legend()
    ax = plt.gca()

    x_major_step = (ax.get_xticks()[1] - ax.get_xticks()[0]) / 5
    y_major_step = (ax.get_yticks()[1] - ax.get_yticks()[0]) / 5

    ax.xaxis.set_minor_locator(MultipleLocator(x_major_step))
    ax.yaxis.set_minor_locator(MultipleLocator(y_major_step))

    max_y_func1 = max(y_func1)
    max_x_func1 = x_values[y_func1.index(max_y_func1)]
    ax.annotate('Max: {:.3f}'.format(max_y_func1),
                xy=(max_x_func1, max_y_func1),
                xytext=(max_x_func1, max_y_func1 - 0.13),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=1),
                verticalalignment='top',
                horizontalalignment='center')

    max_y_func2 = max(y_func2)
    max_x_func2 = x_values[y_func2.index(max_y_func2)]
    ax.annotate('Max: {:.3f}'.format(max_y_func2),
                xy=(max_x_func2, max_y_func2),
                xytext=(max_x_func2 + 0.03, max_y_func2 - 0.13),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=1),
                verticalalignment='top',
                horizontalalignment='center')

    max_y_func2 = max(y_func2)
    max_x_F1 = 0.74 / 0.02 - 20
    max_y_F1 = y_func2[int(max_x_F1)]
    ax.annotate('Best F1: {:.3f}'.format(max_y_F1),
                xy=(0.74, max_y_F1),
                xytext=(0.74 + 0.13, max_y_F1 + 0.23),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=1),
                verticalalignment='top',
                horizontalalignment='center')

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.savefig(base_path + 'figure/figure_8a.pdf', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='figure_8a')
    parser.add_argument('--seed', default='80',
                        help='the value of path/to/results/XGB/vm_test_result_tree_30_seed_????_First.csv at the position of the question mark(Default value is 80)')
    args = parser.parse_args()
    seed_value = args.seed
    start_plot(seed_value)
