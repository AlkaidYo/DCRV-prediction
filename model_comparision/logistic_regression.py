import sys
import os

sys.path.append('../')

from config.file_path_config import base_path, lr_cached_model_path, train_dataset_path
from utils.torch_dataloader import LogDataset
from utils.train_vaild_test_set import sampling_train_test_set
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(funcName)s %(lineno)d %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

pre_rec_value_start = 0.15
pre_rec_value_stop = 0.72

imp_feats = ['c' + str(i) for i in range(1, 80 + 1)]


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


class LinearClassifier(nn.Module):
    def __init__(self, input_size):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out


class Trainer:
    def __init__(self, input_size, num_epochs=2, learning_rate=0.05):
        self.input_size = input_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model = LinearClassifier(input_size)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        logger.info('init trainer success')

    def train(self, train_df):
        train_df = train_df[['instance_id', 'nc_ip', 'sample_time'] + imp_feats + ['label']]
        train_dataset = LogDataset(train_df)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=1, pin_memory=True)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info('torch.cuda.is_available is: ' + str(torch.cuda.is_available()))
        self.model.to(device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        for epoch in range(self.num_epochs):
            logger.info('Epoch [{}/{}]'.format(epoch + 1, self.num_epochs))
            for inputs, labels in train_loader:
                inputs = inputs.float()
                labels = labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 2 == 0:
                logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, loss.item()))

    def predict(self, test_files):
        logger.info('start predict......')
        test_files = test_files[['instance_id', 'nc_ip', 'sample_time'] + imp_feats + ['label']]
        test_dataset = LogDataset(test_files)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        predictions = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.float()
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                predictions.extend(outputs.flatten().tolist())
        logger.info('finish predict......')
        return torch.tensor(predictions)


def main(feats_num, train_df, test_df):
    if not os.path.exists(lr_cached_model_path):
        model = Trainer(feats_num, num_epochs=50)
        model.train(train_df)
        torch.save(model, lr_cached_model_path)
    model = torch.load(lr_cached_model_path)
    predictions = model.predict(test_df)
    logger.info('start agg instance id...')
    X_test_instance_id_nc_ip = test_set['nc_ip']
    X_test_instance_id_vm_id = test_set['instance_id']

    X_test_instance_id_nc_ip = pd.concat([X_test_instance_id_nc_ip, y_test], axis=1)
    X_test_instance_id_vm_id = pd.concat([X_test_instance_id_vm_id, y_test], axis=1)

    pre_rec_list = np.arange(pre_rec_value_start, pre_rec_value_stop, 0.02)
    nc_metric_file = base_path + 'results/LR/' + 'nc_test_result.csv'
    with open(nc_metric_file, mode='w') as file:
        file.write('accuracy, precision, recall, f1' + '\n')
        for i in pre_rec_list:
            logger.info('---------------------------------------------------------------')
            logger.info('nc ip threshold is: ' + str(i))
            y_pred = [1 if p > i else 0 for p in predictions]
            accuracy, precision, recall, f1 = test_metric('nc_ip', X_test_instance_id_nc_ip, y_pred)
            file.write(str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + '\n')

    vm_metric_file = base_path + 'results/LR/' + 'vm_test_result.csv'
    with open(vm_metric_file, mode='w') as file:
        file.write('accuracy, precision, recall, f1' + '\n')
        for i in pre_rec_list:
            logger.info('---------------------------------------------------------------')
            logger.info('vm id threshold is: ' + str(i))
            y_pred = [1 if p > i else 0 for p in predictions]
            print(predictions)
            accuracy, precision, recall, f1 = test_metric('instance_id', X_test_instance_id_vm_id, y_pred)
            file.write(str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + '\n')


if __name__ == '__main__':
    all_dataset_pkl = pd.read_pickle(train_dataset_path)
    train_set, test_set = sampling_train_test_set(all_dataset_pkl, 0.7, 70)
    feats_num = len(imp_feats)
    y_test = test_set.iloc[:, -1]
    main(feats_num, train_set, test_set)
