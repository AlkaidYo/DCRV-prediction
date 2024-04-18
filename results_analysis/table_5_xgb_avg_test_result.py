
import sys
sys.path.append('../')
from config.file_path_config import base_path
import pandas as pd
import glob
tree_high_list = [30]
def avg_xgb_metrics():
    filenames_1 = glob.glob(base_path + 'results/XGB/vm_test_result_tree_30_seed_?_First.csv')
    filenames_2 = glob.glob(base_path + 'results/XGB/vm_test_result_tree_30_seed_??_First.csv')
    filenames_3 = glob.glob(base_path + 'results/XGB/vm_test_result_tree_30_seed_???_First.csv')
    filenames_4 = glob.glob(base_path + 'results/XGB/vm_test_result_tree_30_seed_????_First.csv')

    filenames = filenames_4 + filenames_3 + filenames_2 + filenames_1

    dataframes = [pd.read_csv(filename, header=0) for filename in filenames]

    combined_df = pd.concat(dataframes, axis=0)

    combined_df.reset_index(drop=True, inplace=True)

    averages_df = combined_df.groupby(combined_df.index % 30).mean()

    print(' Test Precision:' + str(averages_df.iloc[17, 1]))
    print(' Test Recall:' + str(averages_df.iloc[17, 2]))
    print(' Test F1 Score:' + str(averages_df.iloc[17, 3]))
    averages_df.to_csv(base_path + 'temp_dir/table_5_avg_xgb.csv', index=False)


if __name__ == '__main__':
    avg_xgb_metrics()
