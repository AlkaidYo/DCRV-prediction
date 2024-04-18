import sys
import os

sys.path.append('../')

import numpy as np
import gc
import xgboost as xgb_model
import random
from config.file_path_config import xgb_cached_model_path, base_path, gray_dataset_path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import logging
import warnings
import joblib
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(funcName)s %(lineno)d %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

logger.info('init params......')
params = {
    'learning_rate': 0.05,
    'max_depth': 15,
    'min_child_weight': 5,
    'gamma': 0.5,
    'subsample': 1,
    'colsample_bytree': 0.7,
    'reg_alpha': 10,
    'reg_lambda': 5,
    'scale_pos_weight': 8,
    'n_estimators': 50,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}
seed_value = [80]
tree_hight = [30]
all_dataset_pkl = pd.read_pickle(gray_dataset_path)


def xgb_matrix(train_df):
    X_train = train_df.iloc[:, 2:-1]
    y_train = train_df.iloc[:, -1]
    return X_train, y_train


def agg_instance_id(cla_str, test_df, y_pred):
    test_df['y_pred'] = y_pred
    test_df['y_pred'] = test_df.groupby(cla_str)['y_pred'].transform(lambda x: 1 if x.any() else x)
    test_df = test_df.drop_duplicates()
    return test_df['label'], test_df['y_pred']



def test_metric(cla_str, X_test_instance_id, y_pred):
    y_test, y_pred = agg_instance_id(cla_str, X_test_instance_id, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = test_confusion_matrix(cla_str, y_test, y_pred)

    logger.info(cla_str + ' Test Accuracy:' + str(accuracy))
    logger.info(cla_str + ' Test Precision:' + str(precision))
    logger.info(cla_str + ' Test Recall:' + str(recall))
    logger.info(cla_str + ' Test F1 Score:' + str(f1))
    return accuracy, precision, recall, f1, tn, fp, fn, tp


def test_confusion_matrix(cla_str, y_true, y_pred):
    if sum(y_true) == 0 and sum(y_pred) == 0:
        return len(y_pred), 0, 0, 0
    else:
        matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = matrix.ravel()
        return tn, fp, fn, tp


def main_test(dtest, test_set, seed_value, tree_high, apdx, pre_rec_list):
    logger.info('load cached model')
    model = joblib.load(xgb_cached_model_path)
    logger.info('predicting......')
    predictions = model.predict(dtest)

    logger.info('start agg instance id...')
    X_test_instance_id_nc_ip = test_set['nc_ip']
    X_test_instance_id_vm_id = test_set['instance_id']

    X_test_instance_id_nc_ip = pd.concat([X_test_instance_id_nc_ip, y_test], axis=1)
    X_test_instance_id_vm_id = pd.concat([X_test_instance_id_vm_id, y_test], axis=1)

    nc_metric_file = base_path + 'results/XGB/' + 'nc_test_result_' + 'tree_' + str(tree_high) + '_seed_' + str(
        seed_value) + apdx + '.csv'
    with open(nc_metric_file, mode='w') as file:
        file.write('accuracy, precision, recall,f1,tn,fp,fn,tp' + '\n')
        for i in pre_rec_list:
            i = 0.72
            logger.info('---------------------------------------------------------------')
            logger.info('nc ip threshold is: ' + str(i))
            y_pred = [1 if p > i else 0 for p in predictions]
            threshold_i_pred = test_set.iloc[:, :2].assign(y_p=y_pred).assign(y_t=test_set['label'])
            threshold_i_pred.to_csv(
                'nc_threshold_' + str(round(i, 2)) + '_seed_' + str(seed_value) + apdx + '_pred.csv', index=False)
            accuracy, precision, recall, f1, tn, fp, fn, tp = test_metric('nc_ip', X_test_instance_id_nc_ip, y_pred)
            file.write(str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' +
                       str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(tp) + '\n')

    vm_metric_file = base_path + 'results/XGB/' + 'vm_test_result_' + 'tree_' + str(tree_high) + '_seed_' + str(
        seed_value) + apdx + '.csv'
    with open(vm_metric_file, mode='w') as file:
        file.write('accuracy,precision,recall,f1,tn,fp,fn,tp' + '\n')
        for i in pre_rec_list:
            logger.info('---------------------------------------------------------------')
            logger.info('vm id threshold is: ' + str(i))
            y_pred = [1 if p > i else 0 for p in predictions]
            threshold_i_pred = test_set.iloc[:, :2].assign(y_p=y_pred).assign(y_t=test_set['label'])
            threshold_i_pred.to_csv(
                'vm_threshold_' + str(round(i, 2)) + '_seed_' + str(seed_value) + apdx + '_pred.csv', index=False)
            accuracy, precision, recall, f1, tn, fp, fn, tp = test_metric('instance_id', X_test_instance_id_vm_id, y_pred)
            file.write(str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' +
                       str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(tp) + '\n')


if __name__ == '__main__':
    first_test_set = all_dataset_pkl
    logger.info('---------------------------------------------------------------')
    logger.info('---------------------------------------------------------------')
    logger.info('First: VM Migration')
    logger.info('xgb dataset init')
    X_test, y_test = xgb_matrix(first_test_set)

    dtest = xgb_model.DMatrix(X_test, label=y_test)
    for j in tree_hight:
        params['max_depth'] = j
        logger.info('---------------------------------------------------------------')
        logger.info('---------------------------------------------------------------')
        logger.info('tree_hight: ' + str(j))
        main_test(dtest, first_test_set, 0, j, '_First', [0.4])

    logger.info('---------------------------------------------------------------')
    logger.info('---------------------------------------------------------------')
    logger.info('Second: VM Migration')
    f = base_path + 'grayscale_testing/vm_threshold_0.4_seed_0_First_pred.csv'
    threshold_i_df = pd.read_csv(f, header=0, usecols=['nc_ip', 'instance_id', 'y_p'])
    filtered_df = threshold_i_df[(threshold_i_df['y_p'] == 1)].copy()

    idx = filtered_df.groupby('nc_ip')['instance_id'].idxmin()
    earliest_instance_ids = filtered_df.loc[idx, 'instance_id'].tolist()
    second_test_set = first_test_set[~first_test_set['instance_id'].isin(earliest_instance_ids)]

    X_test, y_test = xgb_matrix(second_test_set)

    dtest = xgb_model.DMatrix(X_test, label=y_test)
    filename = os.path.basename(f)
    for j in tree_hight:
        params['max_depth'] = j
        main_test(dtest, second_test_set, 0, j, '_Second_pred_' + filename[13:-9], [0.8])

    del threshold_i_df
    del filtered_df
    gc.collect()
