import os
import pandas as pd
import numpy as np
import json
import os
import pickle
import matplotlib
from matplotlib import pyplot as plt
import datetime
import torch
from torch.utils.data import Dataset
from model import INPUT_NORMALIZATION
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, roc_curve, auc, \
    precision_recall_curve, average_precision_score
import time


N_LAYERS = 2
FC_FEATURES = 16
STATIC_INPUT_SIZE = 2
DYNAMIC_INPUT_SIZE = 20
INPUT_SIZE = DYNAMIC_INPUT_SIZE + STATIC_INPUT_SIZE
RNN_TYPE = 'GRU'
NORMALIZATION_CFG = False
SEQ_LENGTH = 100

db_path_train = 'Data/db_2019.pickle'
db_path_validation = 'Data/db_2020.pickle'
db_path_test = 'Data/db_2020.pickle'

rg_path_train = 'Data/rain_gauge_19.xlsx'
rg_path_validation = 'Data/rain_gauge_20.xlsx'
rg_path_test = 'Data/rain_gauge_20.xlsx'

# valid_sequences_train = 'Data/train.csv'
# valid_sequences_validation = 'Data/validation.csv'
# valid_sequences_test = 'Data/test.csv'

set_basic = [1, 2, 3, 4, 8, 10, 11, 29]

# Sets used for the number of links experiment - Used for training.
set_A = [1, 2, 3, 4]
set_B = [1, 2, 3, 4, 8, 10, 11, 29]
set_C = [1, 2, 3, 4, 8, 10, 11, 21, 29, 31, 32, 35, 37, 40, 42, 48, 49, 52]

# Sets used for the number of links experiment - Used for validation/testing without hop 3.
set_AA = [1, 2, 4]
set_BB = [1, 2, 4, 8, 10, 11, 29]
set_CC = [1, 2, 4, 8, 10, 11, 21, 29, 31, 32, 35, 37, 40, 42, 48, 49, 52]

# hops_to_exclude_train = [5, 16, 19, 30, 33, 34, 36, 38, 41, 43, 44, 45, 46, 47, 50, 51, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66]  # 2019-2020
hops_to_keep_train = [1, 2, 4, 8, 10, 29]
hops_to_exclude_train = [i for i in range(1,  67) if i not in hops_to_keep_train]
# hops_to_exclude_test = [7, 16, 17, 18, 19, 20, 24, 33, 34, 36, 38, 41, 43, 45, 50, 51, 53, 55, 56, 57, 58, 63, 64, 65, 66]  # 2020 - 2021
hops_to_keep_test = [1, 2, 4, 8, 10, 29]
hops_to_exclude_test = [i for i in range(1,  67) if i not in hops_to_keep_test]


class RandomNoise():
    """
    Class used to add random uniform noise in the range of [-0.5, 0.5] dB for each link.
    The noise is added to each sample.
    """
    def __call__(self, data):
        random_noise = np.random.rand(data.shape[0], data.shape[1]) - 0.5
        data_noised = data + random_noise
        return data_noised


def rolling_wet_dry(rsl_df, window='30min'):
    """
        Function that applies the rolling standard deviation method for the wet/dry classification task.
        :param rsl_df: Pandas dataframe of the RSL samples.
        :param window: string representing the window size of calculating the standard deviation.
        :return: Two values:
                - Pandas dataframe of the rolling standard deviation values.
                - Float representing the threshold used for determining wet or dry.
    """
    # window = '60min'
    s_t = rsl_df.rolling(window, center=False).std()
    sigma = 1.2 * s_t.quantile(q=0.8)
    # std_df = rsl_df.rolling(window).std()
    # wd_df = (std_df > sigma).astype(int)
    return s_t, sigma


if __name__ == '__main__':
    from Smbit import Rehovot, RSL, load_rain_gauge

    time_start = time.time()
    # Select threshold and dataset type
    threshold = 0.8
    dataset = 'train'
    ds_train = '2019-11-01' # train
    de_train = '2019-12-01' # train

    hops_to_exclude = hops_to_exclude_train if dataset == 'train' else hops_to_exclude_test

    rehovot_dataset = Rehovot(dataset=dataset, seq_len=SEQ_LENGTH, hops_to_exclude=hops_to_exclude)

    # Select specific links or all links that weren't excluded
    # links_name = ['1_down', '2_up', '2_down', '3_up', '3_down', '4_up', '4_down']
    links_name = ['4_down']
    # links_name = rehovot_dataset.dynamic_data.columns.values

    link_index = [np.argwhere(rehovot_dataset.dynamic_data.columns == link_name)[0, 0] for link_name in links_name]
    # hops_id = np.array([int(x.replace('_down','').replace('_up','')) for x in rehovot_dataset.dynamic_data.columns])

    num_links = len(link_index)

    rsl_data = rehovot_dataset.dynamic_data[links_name]
    meta_data = rehovot_dataset.static_data[link_index]
    rg_data = rehovot_dataset.rg

    # Find sigma

    std_df, sigma = rolling_wet_dry(rsl_data)
    wd_df = (std_df > sigma).astype(int)
    # rsl_data['sum'] = rsl_data.rolling('60min').sum()

    rsl_data_avg = rsl_data.resample('10T', closed='right', label='right').mean()
    std_df_avg = std_df.resample('10T', closed='right', label='right').mean()
    wd_df_avg = (std_df_avg > sigma).astype(int)

    plt.figure()
    plt.plot(wd_df_avg[links_name])
    plt.plot(rsl_data_avg[links_name])
    plt.plot(std_df_avg[links_name])

    # wd_df[links_name].plot()
    # rg_data.plot(color='black')
    plt.show()
    print('Done')
