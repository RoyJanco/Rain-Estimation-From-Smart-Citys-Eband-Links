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
from Smbit import Rehovot, RSL, load_rain_gauge
from model import INPUT_NORMALIZATION, TwoStepNetworkGeneric
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import time
from utils import rolling_wet_dry, hops_to_exclude_train, hops_to_exclude_test
from short_links_model import calc_rain
from test_regression_classification import infer_reference, infer_entire_dataset, parse_model_name
N_LAYERS = 2
FC_FEATURES = 16
STATIC_INPUT_SIZE = 2
DYNAMIC_INPUT_SIZE = 20
INPUT_SIZE = DYNAMIC_INPUT_SIZE + STATIC_INPUT_SIZE
RNN_TYPE = 'GRU'
NORMALIZATION_CFG = False
SEQ_LENGTH = 48

global device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU available')
else:
    device = torch.device('cpu')
    print('GPU not available, training on CPU.')
print(device)


def load_model(model_name):
    """
    Function that loads the model.
    :param model_name: string representing the model name.
    :return: the corresponding model.
    """
    model_config = parse_model_name(model_name)
    model_path = os.path.join('Models/TwoStep/TimePeriods', model_name)
    model = TwoStepNetworkGeneric(n_layers=model_config['nl'], rnn_type=model_config['rt'], normalization_cfg=INPUT_NORMALIZATION,
                                  enable_tn=False,
                                  tn_alpha=0.9,
                                  tn_affine=False,
                                  rnn_input_size=INPUT_SIZE,
                                  rnn_n_features=model_config['h'],
                                  metadata_input_size=STATIC_INPUT_SIZE,
                                  metadata_n_features=FC_FEATURES,
                                  threshold=0.7,
                                  model_type=model_config['mt'],
                                  fully_size=model_config['fz'])


    state_dict = torch.load(model_path, map_location=torch.device(device))
    # print(state_dict.keys())
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def plot_results(df):
    """
    Function that plots the RMSE graph of the different models.
    :param df: Pandas dataframe containing the RMSE of the different models for the training duration experiment.
    """
    ax = df.plot.bar(rot=0, edgecolor='black')
    ax.set_ylabel('RMSE [mm/h]')
    if SAVE:
        plt.savefig(f"Out/TimePeriods/time_periods_{dataset}.eps")


if __name__ == '__main__':
    time_start = time.time()
    dataset = 'validation'
    SAVE = False
    model_names_rnn = [
        'RegressionTwoStep_rt-GRU_mt-skip_fz-large_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200_train1.pt',
        'RegressionTwoStep_rt-GRU_mt-skip_fz-large_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200_train2.pt',
        'RegressionTwoStep_rt-GRU_mt-skip_fz-large_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200_train3.pt',
        'RegressionTwoStep_rt-GRU_mt-skip_fz-large_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200_train4.pt',
    ]
    model_names_short_links = [
        'ShortLinksModels/model_params_1down_train1.pkl',
        'ShortLinksModels/model_params_1down_train2.pkl',
        'ShortLinksModels/model_params_1down_train3.pkl',
        'ShortLinksModels/model_params_1down_train4.pkl'
    ]
    model_short_names = ['Train 4', 'Reference']

    output_file_name = f'time_periods-{dataset}.csv'
    output_path = os.path.join('Out/TimePeriods', output_file_name)

    hops_to_exclude = hops_to_exclude_train if dataset == 'train' else hops_to_exclude_test


    rehovot_dataset = Rehovot(dataset=dataset, seq_len=SEQ_LENGTH, hops_to_exclude=hops_to_exclude)

    # Select specific links or all links that weren't excluded
    # links_name = ['1_down', '2_up', '2_down', '3_up', '3_down', '4_up', '4_down', '8_up', '8_down', '10_up', '10_down']
    # links_name = ['2_down']
    links_name = rehovot_dataset.dynamic_data.columns.values

    link_index = [np.argwhere(rehovot_dataset.dynamic_data.columns == link_name)[0, 0] for link_name in links_name]
    # hops_id = np.array([int(x.replace('_down','').replace('_up','')) for x in rehovot_dataset.dynamic_data.columns])

    num_links = len(link_index)

    rsl_data = rehovot_dataset.dynamic_data[links_name]
    meta_data = rehovot_dataset.static_data[link_index]
    attenuation_data = rehovot_dataset.attenuation[links_name]
    rg_data = rehovot_dataset.rg


    resuls_df = pd.DataFrame(columns=['Label', 'RNN', 'Model'])
    models_scores_dict = {}

    predictions_list = []
    gt_list = []
    # Loop through models
    for i in range(len(model_names_short_links)):
        # model_path = os.path.join('Models', model_name)
        # Load pre trained model
        # print(f'Evaluating model {i}/{len(model_names)}')
        # print(model_name)

        # Evaluate short link model i
        model_params_short_links = (pd.read_pickle(model_names_short_links[i]))[links_name]
        metrics_model, _, _ = infer_reference(attenuation_data, model_params_short_links, meta_data, rg_data, rehovot_dataset.valid_sequences)

        # Evaluate rnn model i
        model = load_model(model_name=model_names_rnn[i])
        metrics_rnn, _, _, _, _ = infer_entire_dataset(model, rsl_data, meta_data,  rg_data, rehovot_dataset.valid_sequences)

        row = {'Label': f'TR_{i+1}', 'RNN': metrics_rnn['RMSE'], 'Model': metrics_model['RMSE']}

        resuls_df = resuls_df.append(row, ignore_index=True)
        # predictions_list.append(y_pred)
        # gt_list.append(y_true)
        # models_scores_dict[model_short_names[i-1]] = (true_labels, model_wd_probs)
    resuls_df.set_index('Label', inplace=True)
    plot_results(resuls_df)
    time_end = time.time()
    print(f'Finished testing {len(model_names_rnn)} models. Took {(time_end-time_start):.2f} seconds.')
    # if SAVE:
    #     resuls_df.to_csv(output_path, float_format='%.3f', index=False)
    plt.show()
    print('Done')