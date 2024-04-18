import sys
sys.path.append('../')

base_path = '/path/to/project/'

train_dataset_path='/path/to/file'
gray_dataset_path='/path/to/file'
pos_dataset_path='/path/to/file'

xgb_cached_model_path = base_path + 'cached_model/xgb.dat'
rf_cached_model_path = base_path + 'cached_model/rf.dat'
lr_cached_model_path = base_path + 'cached_model/lr.pth'

ref_10_node_crash_cost = [100, 250, 500]  #
ref_10_node_repair_cost = [50]  # Crash:Repair = 2,5,10
ref_10_node_migration_cost = [10]  # Crash:Migration = 10,25,50