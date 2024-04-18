import sys

sys.path.append('../')
import pandas as pd
from config.file_path_config import base_path
import argparse

loss_time_vm_migration = 1
loss_time_nc_migration = 2  # 2,4,12
truth_down_num = None


def func_1_5(first_tp, first_fp, second_tp, second_fp, second_fn):
    return loss_time_vm_migration * (first_tp + first_fp) + loss_time_nc_migration * (
            second_tp + second_fp) + 5 * 60 * (second_fn)


def func_1_50(first_tp, first_fp, second_tp, second_fp, second_fn):
    return loss_time_vm_migration * (first_tp + first_fp) + loss_time_nc_migration * (
            second_tp + second_fp) + 50 * 60 * (second_fn)


def func_1_500(first_tp, first_fp, second_tp, second_fp, second_fn):
    return loss_time_vm_migration * (first_tp + first_fp) + loss_time_nc_migration * (
            second_tp + second_fp) + 500 * 60 * (second_fn)


def start_plot(seed_value):
    for i in [0.4]:
        threshold_i_df = pd.read_csv(base_path + 'model_comparision/vm_threshold_' + str(
            i) + '_seed_' + str(seed_value) + '_First_pred.csv',
                                     header=0,
                                     usecols=['nc_ip', 'instance_id', 'sample_time', 'y_p', 'y_t'])

        filtered_df = threshold_i_df[(threshold_i_df['y_p'] == 1)].copy()

        filtered_df['sample_time'] = pd.to_datetime(filtered_df['sample_time'])
        idx = filtered_df.groupby('nc_ip')['sample_time'].idxmin()
        earliest_instance_ids = filtered_df.loc[idx, 'instance_id'].tolist()
        earliest_instance_ids = list(set(earliest_instance_ids))
        fitst_migration_df = filtered_df[filtered_df['instance_id'].isin(earliest_instance_ids)]

        first_tp = fitst_migration_df[(fitst_migration_df['y_t'] == 1)]['instance_id'].nunique()
        first_fp = fitst_migration_df[(fitst_migration_df['y_t'] == 0)]['instance_id'].nunique()

        for j in [0.84]:

            second_threshold_i_df = pd.read_csv(
                base_path + 'model_comparision/vm_threshold_' + str(j) + '_seed_80_Second_pred_' + str(
                    i) + '_seed_80_First_pred.csv',
                header=0,
                usecols=['nc_ip', 'instance_id', 'sample_time', 'y_p', 'y_t'])


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


            hyb_5 = func_1_5(first_tp, first_fp, second_tp, second_fp, second_fn)
            hyb_50 = func_1_50(first_tp, first_fp, second_tp, second_fp, second_fn)
            hyb_500 = func_1_500(first_tp, first_fp, second_tp, second_fp, second_fn)
            return [hyb_5, hyb_50, hyb_500]


def save_txt(hyb_loss_time):
    with open(base_path + 'temp_dir/figure_7b_hybrid.csv', 'w') as file:
        file.write(','.join(map(str, hyb_loss_time)) + '\n')


def start(seed_value=80):
    global truth_down_num
    df_seed_80 = pd.read_csv(base_path + 'results/XGB/nc_test_result_tree_30_seed_' + str(seed_value) + '_First.csv')
    truth_down_num = df_seed_80['tp'].iloc[0] + df_seed_80['fn'].iloc[0]

    mix_loss_time = start_plot(seed_value)
    save_txt(mix_loss_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Figure 7b: loss compute time.')
    parser.add_argument('--seed', default='80',
                        help='the value of path/to/results/XGB/vm_test_result_tree_30_seed_????_First.csv at the position of the question mark(Default value is 80)')
    args = parser.parse_args()
    seed_value = args.seed
    df_seed_80 = pd.read_csv(base_path + 'results/XGB/nc_test_result_tree_30_seed_' + str(seed_value) + '_First.csv')
    truth_down_num = df_seed_80['tp'].iloc[0] + df_seed_80['fn'].iloc[0]
    start(seed_value)
