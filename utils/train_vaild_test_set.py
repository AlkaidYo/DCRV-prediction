import sys
import os
sys.path.append('../')
sys.path.append(os.getcwd())

import warnings
import random
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(name)s %(funcName)s %(lineno)d %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def sampling_train_test_set(dataset_all, sample_rate, seed_value):
    nc_ip_column = dataset_all['nc_ip'].tolist()
    logger.info('searching nc ip finished')
    unique_lst = list(set(nc_ip_column))
    random.seed(seed_value)

    logger.info('sampling nc_ip')
    sampled_lst = random.sample(unique_lst, int(len(unique_lst) * sample_rate))
    df_train = dataset_all[dataset_all['nc_ip'].isin(sampled_lst)]
    df_test = dataset_all[~dataset_all['nc_ip'].isin(sampled_lst)]
    return df_train, df_test

