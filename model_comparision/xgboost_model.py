import sys
import os

sys.path.append('../')

import numpy as np
import gc
import xgboost as xgb_model
import random
from config.file_path_config import train_dataset_path, xgb_cached_model_path, base_path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from utils.train_vaild_test_set import sampling_train_test_set
import pandas as pd
import logging
import warnings
import joblib
import glob

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
seed_value = random.sample(range(1, 5001), 100)
while 80 in seed_value:
    index = seed_value.index(80)
    seed_value[index] = 5001

tree_hight = [30]
pre_rec_value_start = 0.40
pre_rec_value_stop = 1.0
all_dataset_pkl = pd.read_pickle(train_dataset_path)


def xgb_matrix(train_df):
    X_train = train_df.iloc[:, 3:-1]
    y_train = train_df.iloc[:, -1]
    return X_train, y_train


def agg_instance_id(cla_str, test_df, y_pred):
    test_df['y_pred'] = y_pred
    test_df['y_pred'] = test_df.groupby(cla_str)['y_pred'].transform(lambda x: 1 if x.any() else x)
    test_df = test_df.drop_duplicates()
    return test_df['label'], test_df['y_pred']


def main_train(dtrain, params_c):
    logger.info('xgb training')
    model = xgb_model.train(params_c, dtrain)
    joblib.dump(model, xgb_cached_model_path)


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


def main_test(dtest, test_set, seed_value, tree_high, apdx):
    logger.info('load cached model')
    model = joblib.load(xgb_cached_model_path)
    logger.info('predicting......')
    predictions = model.predict(dtest)

    logger.info('start agg instance id...')
    X_test_instance_id_nc_ip = test_set['nc_ip']
    X_test_instance_id_vm_id = test_set['instance_id']

    X_test_instance_id_nc_ip = pd.concat([X_test_instance_id_nc_ip, y_test], axis=1)
    X_test_instance_id_vm_id = pd.concat([X_test_instance_id_vm_id, y_test], axis=1)

    pre_rec_list = np.arange(pre_rec_value_start, pre_rec_value_stop, 0.02)
    nc_metric_file = base_path + 'results/XGB/nc_test_result_tree_' + str(tree_high) + '_seed_' + str(
        seed_value) + apdx + '.csv'
    with open(nc_metric_file, mode='w') as file:
        file.write('accuracy, precision, recall,f1,tn,fp,fn,tp' + '\n')
        for i in pre_rec_list:
            logger.info('---------------------------------------------------------------')
            logger.info('nc ip threshold is: ' + str(i))
            y_pred = [1 if p > i else 0 for p in predictions]
            accuracy, precision, recall, f1, tn, fp, fn, tp = test_metric('nc_ip', X_test_instance_id_nc_ip, y_pred)
            file.write(str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' +
                       str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(tp) + '\n')

    vm_metric_file = base_path + 'results/XGB/vm_test_result_tree_' + str(tree_high) + '_seed_' + str(
        seed_value) + apdx + '.csv'
    with open(vm_metric_file, mode='w') as file:
        file.write('accuracy,precision,recall,f1,tn,fp,fn,tp' + '\n')
        for i in pre_rec_list:
            logger.info('---------------------------------------------------------------')
            logger.info('vm id threshold is: ' + str(i))
            y_pred = [1 if p > i else 0 for p in predictions]
            accuracy, precision, recall, f1, tn, fp, fn, tp = test_metric('instance_id', X_test_instance_id_vm_id,
                                                                          y_pred)
            file.write(str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' +
                       str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(tp) + '\n')


if __name__ == '__main__':
    for i in seed_value:
        train_set, first_test_set = sampling_train_test_set(all_dataset_pkl, 0.7, i)
        logger.info('---------------------------------------------------------------')
        logger.info('---------------------------------------------------------------')
        logger.info('First: VM Migration')
        logger.info('xgb dataset init')
        X_train, y_train = xgb_matrix(train_set)
        X_test, y_test = xgb_matrix(first_test_set)

        dtrain = xgb_model.DMatrix(X_train, label=y_train)
        dtest = xgb_model.DMatrix(X_test, label=y_test)
        for j in tree_hight:
            params['max_depth'] = j
            logger.info('---------------------------------------------------------------')
            logger.info('---------------------------------------------------------------')
            logger.info('seed value: ' + str(i))
            logger.info('tree_hight: ' + str(j))
            main_train(dtrain, params)
            main_test(dtest, first_test_set, i, j, '_First')
