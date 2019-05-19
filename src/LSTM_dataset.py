import os, sys
import pandas as pd
import torch.utils.data as thd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PonziAndNonPonziDataset(Dataset):
    def __init__(self, dir, file):
        '''
        :param file: ponzi and non ponzi data file name
        :param dir: path storing the file
        :param transform:
        '''
        self.df = pd.read_csv(dir+file)
        self.train_Y = self.df['label'].values.tolist()
        self.train_X = self.df.drop('label', axis=1).drop('address', axis=1).values

    def __len__(self):
        return self.train_X.shape[0]

    def __getitem__(self, item_index):
        '''
        :param item_index: the index of the wanted item
        :return: contract address, opcode sequence, ponzi label: 0==ponzi, 1==non_ponzi
        '''
        return torch.Tensor(self.train_X[item_index].astype(int)), self.train_Y[item_index]


train_dataset = PonziAndNonPonziDataset('../dataset/op_int_equal_dataframe/', 'op_int_equal_new.csv')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=50,
                                           shuffle=True,
                                           num_workers=2)