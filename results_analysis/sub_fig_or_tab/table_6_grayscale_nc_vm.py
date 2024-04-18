import sys

sys.path.append('../')
import pandas as pd
import numpy as np
from config.file_path_config import base_path
from config.file_path_config import base_path, ref_10_node_migration_cost, ref_10_node_repair_cost, \
    ref_10_node_crash_cost

cost_vm_migration = 1
cost_nc_migration = ref_10_node_migration_cost[0] + ref_10_node_repair_cost[0]# 5,10,15
cost_nc_downtime = ref_10_node_crash_cost[1] + ref_10_node_repair_cost[0] # 50,250,750


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


def start():
    confusion_metrix_df_vm = pd.read_csv(base_path+'results/XGB/vm_test_result_tree_30_seed_0_First.csv', header=0)
    confusion_metrix_df_nc = pd.read_csv(base_path+'results/XGB/nc_test_result_tree_30_seed_0_First.csv', header=0)
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

    print('Grayscale test node migration: ', y_func1[0])
    print('Grayscale test VM migration: ', y_func2[0])


if __name__ == '__main__':
    start()