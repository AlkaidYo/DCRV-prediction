import sys
import torch

sys.path.append('../')
from torch.utils.data import Dataset
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(funcName)s %(lineno)d %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class LogDataset(Dataset):
    def __init__(self, file_df):
        self.file_df = file_df

    def __len__(self):
        return self.file_df.shape[0]

    def __getitem__(self, idx):
        tensor_feats = torch.tensor(self.file_df.iloc[idx, 3:-1].astype(float).values)
        tensor_label = torch.tensor(self.file_df.iloc[idx, -1].astype(float))
        return tensor_feats, tensor_label
