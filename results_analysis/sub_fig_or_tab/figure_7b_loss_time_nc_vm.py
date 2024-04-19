import sys

sys.path.append('../')
import pandas as pd
import argparse

from config.file_path_config import base_path
from config.file_path_config import base_path, ref_10_node_migration_cost, ref_10_node_repair_cost, \
    ref_10_node_crash_cost

loss_time_vm_migration = 1
loss_time_nc_migration = 2  # 2,4,12

cost_vm_migration = 1
cost_nc_migration = ref_10_node_migration_cost[0] + ref_10_node_repair_cost[0]  # 5,10,15
cost_nc_downtime = ref_10_node_crash_cost[1] + ref_10_node_repair_cost[0]  # 50,250,750

truth_down_num = None

def operate_arrays(truth_down_num, TP_value, FP_value, FN_value, operation):
    return [operation(truth_down_num, x, y, z) for x, y, z in zip(TP_value, FP_value, FN_value)]


def func_NC_250(truth_down_num, TP_value, FP_value, FN_value):
    return (300 * truth_down_num / 0.8 - (
            cost_nc_migration * TP_value + cost_nc_migration * FP_value + 300 * (
            FN_value + truth_down_num * 0.25))) / (
            300 * truth_down_num / 0.8)


def func_VM_250(truth_down_num, TP_value, FP_value, FN_value):
    return (300 * truth_down_num / 0.8 - (
            TP_value + FP_value + 300 * (FN_value + truth_down_num * 0.25))) / (
            300 * truth_down_num / 0.8)


# time loss
def func_1_5(truth_down_num, TP_value, FP_value, FN_value):
    return loss_time_nc_migration * TP_value + loss_time_nc_migration * FP_value + 5 * 60 * (FN_value)


def func_1_50(truth_down_num, TP_value, FP_value, FN_value):
    return loss_time_nc_migration * TP_value + loss_time_nc_migration * FP_value + 50 * 60 * (FN_value)


def func_1_500(truth_down_num, TP_value, FP_value, FN_value):
    return loss_time_nc_migration * TP_value + loss_time_nc_migration * FP_value + 500 * 60 * (FN_value)


# time loss
def func_2_5(truth_down_num, TP_value, FP_value, FN_value):
    return loss_time_vm_migration * TP_value + loss_time_vm_migration * FP_value + 5 * 60 * (FN_value)


def func_2_50(truth_down_num, TP_value, FP_value, FN_value):
    return loss_time_vm_migration * TP_value + loss_time_vm_migration * FP_value + 50 * 60 * (FN_value)


def func_2_500(truth_down_num, TP_value, FP_value, FN_value):
    return loss_time_vm_migration * TP_value + loss_time_vm_migration * FP_value + 500 * 60 * (FN_value)


def start_plot(seed_value):
    confusion_metrix_df_vm = pd.read_csv(
        base_path + 'results/XGB/' + 'vm_test_result_' + 'tree_30' + '_seed_' + str(
            seed_value) + '_First.csv', header=0)
    confusion_metrix_df_nc = pd.read_csv(
        base_path + 'results/XGB/' + 'nc_test_result_' + 'tree_30' + '_seed_' + str(
            seed_value) + '_First.csv', header=0)

    VM_TP_value = confusion_metrix_df_vm['tp'].tolist()
    VM_FP_value = confusion_metrix_df_vm['fp'].tolist()
    VM_TN_value = confusion_metrix_df_vm['tn'].tolist()
    VM_FN_value = confusion_metrix_df_vm['fn'].tolist()

    NC_TP_value = confusion_metrix_df_nc['tp'].tolist()
    NC_FP_value = confusion_metrix_df_nc['fp'].tolist()
    NC_TN_value = confusion_metrix_df_nc['tn'].tolist()
    NC_FN_value = confusion_metrix_df_nc['fn'].tolist()

    return VM_TP_value, VM_FP_value, VM_TN_value, VM_FN_value, NC_TP_value, NC_FP_value, NC_TN_value, NC_FN_value


def bar1(VM_TP_value, VM_FP_value, VM_TN_value, VM_FN_value, NC_TP_value, NC_FP_value, NC_TN_value, NC_FN_value):
    loss_time_VM = operate_arrays(truth_down_num, VM_TP_value, VM_FP_value, VM_FN_value, func_2_5)
    loss_time_NC = operate_arrays(truth_down_num, NC_TP_value, NC_FP_value, NC_FN_value, func_1_5)

    cost_reduction_vm = operate_arrays(truth_down_num, VM_TP_value, VM_FP_value, VM_FN_value, func_VM_250)
    cost_reduction_nc = operate_arrays(truth_down_num, NC_TP_value, NC_FP_value, NC_FN_value, func_NC_250)

    max_y_func1 = max(cost_reduction_nc)
    max_y_func1 = loss_time_NC[cost_reduction_nc.index(max_y_func1)]

    max_y_func2 = max(cost_reduction_vm)
    max_y_func2 = loss_time_VM[cost_reduction_vm.index(max_y_func2)]

    return max_y_func1, max_y_func2


def bar2(VM_TP_value, VM_FP_value, VM_TN_value, VM_FN_value, NC_TP_value, NC_FP_value, NC_TN_value, NC_FN_value):

    loss_time_VM = operate_arrays(truth_down_num, VM_TP_value, VM_FP_value, VM_FN_value, func_2_50)
    loss_time_NC = operate_arrays(truth_down_num, NC_TP_value, NC_FP_value, NC_FN_value, func_1_50)


    cost_reduction_vm = operate_arrays(truth_down_num, VM_TP_value, VM_FP_value, VM_FN_value, func_VM_250)
    cost_reduction_nc = operate_arrays(truth_down_num, NC_TP_value, NC_FP_value, NC_FN_value, func_NC_250)

    max_y_func1 = max(cost_reduction_nc)
    max_y_func1 = loss_time_NC[cost_reduction_nc.index(max_y_func1)]

    max_y_func2 = max(cost_reduction_vm)
    max_y_func2 = loss_time_VM[cost_reduction_vm.index(max_y_func2)]

    return max_y_func1, max_y_func2


def bar3(VM_TP_value, VM_FP_value, VM_TN_value, VM_FN_value, NC_TP_value, NC_FP_value, NC_TN_value, NC_FN_value):

    loss_time_VM = operate_arrays(truth_down_num, VM_TP_value, VM_FP_value, VM_FN_value, func_2_500)
    loss_time_NC = operate_arrays(truth_down_num, NC_TP_value, NC_FP_value, NC_FN_value, func_1_500)


    cost_reduction_vm = operate_arrays(truth_down_num, VM_TP_value, VM_FP_value, VM_FN_value, func_VM_250)
    cost_reduction_nc = operate_arrays(truth_down_num, NC_TP_value, NC_FP_value, NC_FN_value, func_NC_250)

    max_y_func1 = max(cost_reduction_nc)
    max_y_func1 = loss_time_NC[cost_reduction_nc.index(max_y_func1)]

    max_y_func2 = max(cost_reduction_vm)
    max_y_func2 = loss_time_VM[cost_reduction_vm.index(max_y_func2)]

    return max_y_func1, max_y_func2


def save_txt(nc_loss_time, vm_loss_time):
    with open(base_path + 'temp_dir/figure_7b_nc_vm.csv', 'w') as file:
        file.write(','.join(map(str, vm_loss_time)) + '\n')
        file.write(','.join(map(str, nc_loss_time)) + '\n')


def start(seed_value=80):
    global truth_down_num
    df_seed_80 = pd.read_csv(base_path + 'results/XGB/nc_test_result_tree_30_seed_' + str(seed_value) + '_First.csv')
    truth_down_num = df_seed_80['tp'].iloc[0] + df_seed_80['fn'].iloc[0]

    VM_TP_value, VM_FP_value, VM_TN_value, VM_FN_value, NC_TP_value, NC_FP_value, NC_TN_value, NC_FN_value = start_plot(
        seed_value)
    nc_5, vm_5 = bar1(VM_TP_value, VM_FP_value, VM_TN_value, VM_FN_value, NC_TP_value, NC_FP_value, NC_TN_value,
                      NC_FN_value)
    nc_50, vm_50 = bar2(VM_TP_value, VM_FP_value, VM_TN_value, VM_FN_value, NC_TP_value, NC_FP_value, NC_TN_value,
                        NC_FN_value)
    nc_500, vm_500 = bar3(VM_TP_value, VM_FP_value, VM_TN_value, VM_FN_value, NC_TP_value, NC_FP_value, NC_TN_value,
                          NC_FN_value)
    nc_loss_time = [nc_5, nc_50, nc_500]
    vm_loss_time = [vm_5, vm_50, vm_500]
    save_txt(nc_loss_time, vm_loss_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Figure 7b: loss compute time.')
    parser.add_argument('--seed', default='80',
                        help='the value of path/to/results/XGB/vm_test_result_tree_30_seed_????_First.csv at the position of the question mark(Default value is 80)')
    args = parser.parse_args()
    seed_value = args.seed
    df_seed_80 = pd.read_csv(base_path + 'results/XGB/nc_test_result_tree_30_seed_' + str(seed_value) + '_First.csv')
    truth_down_num = df_seed_80['tp'].iloc[0] + df_seed_80['fn'].iloc[0]
    start(seed_value)
