import sys
sys.path.append('../')
from utils.train_vaild_test_set import sampling_train_test_set
import numpy as np
import pandas as pd
import random
from config.file_path_config import train_dataset_path, rf_cached_model_path, base_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(funcName)s %(lineno)d %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

params = {
    'max_depth': 30,
    'n_estimators': 100,
    'n_jobs': -1,
    'criterion': 'gini',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
}
seed_value = random.sample(range(1, 5001), 100)
max_depth_list = [30]
all_dataset_pkl = pd.read_pickle(train_dataset_path)
pre_rec_value_start = 0.20
pre_rec_value_stop = 0.80


def xgb_matrix(train_df):
    X_train = train_df.iloc[:, 3:-1]
    y_train = train_df.iloc[:, -1]
    return X_train, y_train


def main_train(X_train, y_train, params):
    logger.info('rf training')
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    joblib.dump(clf, rf_cached_model_path)


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

    logger.info(cla_str + ' Test Accuracy:' + str(accuracy))
    logger.info(cla_str + ' Test Precision:' + str(precision))
    logger.info(cla_str + ' Test Recall:' + str(recall))
    logger.info(cla_str + ' Test F1 Score:' + str(f1))

    return accuracy, precision, recall, f1


def main_test(X_test, y_test, test_set, seed_value, depth_value):
    logger.info('load cached model')
    model = joblib.load(rf_cached_model_path)

    logger.info('predicting......')
    predictions = model.predict_proba(X_test)[:, 1]
    logger.info('start agg instance id...')
    X_test_instance_id_nc_ip = test_set['nc_ip']
    X_test_instance_id_vm_id = test_set['instance_id']

    X_test_instance_id_nc_ip = pd.concat([X_test_instance_id_nc_ip, y_test], axis=1)
    X_test_instance_id_vm_id = pd.concat([X_test_instance_id_vm_id, y_test], axis=1)

    pre_rec_list = np.arange(pre_rec_value_start, pre_rec_value_stop, 0.02)
    nc_metric_file = base_path + 'results/RF/' + 'nc_test_result_' + 'tree_' + str(depth_value) + '_seed_' + str(
        seed_value) + '_First.csv'
    with open(nc_metric_file, mode='w') as file:
        file.write('accuracy, precision, recall, f1' + '\n')
        for i in pre_rec_list:
            logger.info('---------------------------------------------------------------')
            logger.info('nc ip threshold is: ' + str(i))
            y_pred = [1 if p > i else 0 for p in predictions]
            accuracy, precision, recall, f1 = test_metric('nc_ip', X_test_instance_id_nc_ip, y_pred)
            file.write(str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + '\n')

    vm_metric_file = base_path + 'results/RF/' + 'vm_test_result_' + 'tree_' + str(depth_value) + '_seed_' + str(
        seed_value) + '_First.csv'
    with open(vm_metric_file, mode='w') as file:
        file.write('accuracy, precision, recall, f1' + '\n')
        for i in pre_rec_list:
            logger.info('---------------------------------------------------------------')
            logger.info('vm id threshold is: ' + str(i))
            y_pred = [1 if p > i else 0 for p in predictions]
            accuracy, precision, recall, f1 = test_metric('instance_id', X_test_instance_id_vm_id, y_pred)
            file.write(str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + '\n')


if __name__ == '__main__':
    for i in seed_value:
        train_set, test_set = sampling_train_test_set(all_dataset_pkl, 0.7, i)
        logger.info('dataset init')
        X_train, y_train = xgb_matrix(train_set)
        X_test, y_test = xgb_matrix(test_set)

        for j in max_depth_list:
            params['max_depth'] = j
            logger.info('---------------------------------------------------------------')
            logger.info('---------------------------------------------------------------')
            logger.info('seed value: ' + str(i))
            logger.info('max_depth: ' + str(j))
            main_train(X_train, y_train, params)
            main_test(X_test, y_test, test_set, i, j)
