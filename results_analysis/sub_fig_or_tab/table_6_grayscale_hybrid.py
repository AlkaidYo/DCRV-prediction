import sys

sys.path.append('../')

import pandas as pd
import numpy as np

from config.file_path_config import base_path
from config.file_path_config import base_path, ref_10_node_migration_cost, ref_10_node_repair_cost, \
    ref_10_node_crash_cost

cost_vm_migration = 1
cost_nc_migration = ref_10_node_migration_cost[0] + ref_10_node_repair_cost[0]  # 5,10,15
cost_nc_downtime = ref_10_node_crash_cost[1] + ref_10_node_repair_cost[0]  # 50,250,750
max_cost = 0
truth_down_num = None


def operate_arrays(truth_down_num, TP_value, FP_value, FN_value, operation):
    return [operation(truth_down_num, x, y, z) for x, y, z in zip(TP_value, FP_value, FN_value)]


def func_2(first_tp, first_fp, second_tp, second_fp, second_fn):
    return ((cost_nc_downtime * (truth_down_num) / 0.8 - (
            first_tp + first_fp + cost_nc_migration * second_tp + cost_nc_migration * second_fp + cost_nc_downtime * (
            second_fn + (truth_down_num) * 0.25)))
            / (cost_nc_downtime * (truth_down_num) / 0.8))


def start_plot():

    threshold_list = np.round(np.arange(0.4, 1.0, 0.02), 2)
    threshold_list_i = [0.4]
    hyb_cost_list = []

    for i in threshold_list_i:

        threshold_i_df = pd.read_csv(base_path + 'grayscale_testing/vm_threshold_0.4_seed_0_First_pred.csv',
                                     header=0,
                                     usecols=['nc_ip', 'instance_id', 'y_p', 'y_t'])
        filtered_df = threshold_i_df[(threshold_i_df['y_p'] == 1)].copy()

        idx = filtered_df.groupby('nc_ip')['instance_id'].idxmin()
        earliest_instance_ids = filtered_df.loc[idx, 'instance_id'].tolist()
        fitst_migration_df = filtered_df[filtered_df['instance_id'].isin(earliest_instance_ids)]

        first_tp = fitst_migration_df[(fitst_migration_df['y_t'] == 1)]['instance_id'].nunique()
        first_fp = fitst_migration_df[(fitst_migration_df['y_t'] == 0)]['instance_id'].nunique()

        for j in threshold_list:
            second_threshold_i_df = pd.read_csv(
                base_path + 'grayscale_testing/nc_threshold_0.72_seed_0_Second_pred_0.4_seed_0_First_pred.csv',
                header=0,
                usecols=['nc_ip', 'instance_id', 'y_p', 'y_t'])


            second_tp = second_threshold_i_df[
                (second_threshold_i_df['y_p'] == 1) & (second_threshold_i_df['y_t'] == 1)].copy()
            second_tp_list = second_tp['nc_ip'].unique().tolist()
            second_tp = len(second_tp_list)

            second_fp = second_threshold_i_df[
                (second_threshold_i_df['y_p'] == 1) & (second_threshold_i_df['y_t'] == 0)].copy()
            second_fp = second_fp['nc_ip'].nunique()

            second_fn = second_threshold_i_df[
                (second_threshold_i_df['y_p'] == 0) & (second_threshold_i_df['y_t'] == 1)].copy()
            second_fn_list = second_fn['nc_ip'].unique().tolist()
            second_fn_list = [item for item in second_fn_list if item not in second_tp_list]
            second_fn = len(second_fn_list)


            y_func2 = func_2(first_tp, first_fp, second_tp, second_fp, second_fn)
            hyb_cost_list.append(y_func2)
    return hyb_cost_list


def save_txt(hyb_cost_list):
    with open(base_path + 'temp_dir/table_6_hybrid.csv', 'w') as file:
        file.write(','.join(map(str, hyb_cost_list)) + '\n')


def start():
    global truth_down_num
    df_seed_80 = pd.read_csv(base_path + 'results/XGB/nc_test_result_tree_30_seed_0_First.csv')
    truth_down_num = df_seed_80['tp'].iloc[0] + df_seed_80['fn'].iloc[0]
    hyb_cost_list = start_plot()
    save_txt(hyb_cost_list)

