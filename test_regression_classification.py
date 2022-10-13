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
from sklearn.metrics import mean_squared_error, confusion_matrix, matthews_corrcoef, roc_curve, auc, \
    precision_recall_curve, average_precision_score
import time
from utils import rolling_wet_dry, hops_to_exclude_train, hops_to_exclude_test
from short_links_model import calc_rain

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


def parse_model_name(model_name):
    """
    Function that parses the model name.
    :param model_name: string representing the model name.
    :return: dictionary representing the model configuration.
    """
    model_name_split = model_name.replace('.pt','').split('_')
    model_config = {item.split('-')[0]: item.split('-')[1] for item in model_name_split if '-' in item}
    model_config['rt'] = str(model_config['rt']) if 'rt' in model_config.keys() else 'GRU'
    model_config['mt'] = str(model_config['mt']) if 'mt' in model_config.keys() else 'skip' # model type - 'rnn' or 'skip'
    model_config['fz'] = str(model_config['fz']) if 'fz' in model_config.keys() else 'large' # fully connected size 'small' or 'large'
    model_config['sl'] = int(model_config['sl'])    # sequence length
    model_config['h'] = int(model_config['h'])      # hidden size
    model_config['nl'] = int(model_config['nl']) if 'nl' in model_config.keys() else 2   # number of layers
    model_config['d'] = float(model_config['d'])    # dropout probability
    model_config['wd'] = float(model_config['wd'])  # weight decay
    model_config['e'] = int(model_config['e'])      # epochs
    return model_config


def load_model(threshold, model_name):
    """
    Function that loads the model.
    :param threshold: float which is the threshold for determining wet/dry.
    :param model_name: string representing the model name.
    :return: the corresponding model.
    """
    model_config = parse_model_name(model_name)
    model_path = os.path.join('Models/TwoStep/GeneralModels', model_name)
    model = TwoStepNetworkGeneric(n_layers=model_config['nl'], rnn_type=model_config['rt'], normalization_cfg=INPUT_NORMALIZATION,
                                  enable_tn=False,
                                  tn_alpha=0.9,
                                  tn_affine=False,
                                  rnn_input_size=INPUT_SIZE,
                                  rnn_n_features=model_config['h'],
                                  metadata_input_size=STATIC_INPUT_SIZE,
                                  metadata_n_features=FC_FEATURES,
                                  threshold=threshold,
                                  model_type=model_config['mt'],
                                  fully_size=model_config['fz'])

    state_dict = torch.load(model_path, map_location=torch.device(device))
    # print(state_dict.keys())
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def organize_data(rsl_data, static_data, rg_data, ds_rg, de):
    """
    Function that organizes the input data of the network.
    :param rsl_data: Pandas series representing the RSL measurements of the specified link.
    :param static_data: ndarray of shape [2,] representing the metadata of the link (length, frequency).
    :param rg_data: Pandas dataframe containing the rain gauge measurements.
    :param ds_rg: string representing the start date of the subsequence.
    :param de: string representing the end date of the subsequence.
    :return: Five values:
            - ndarray of shape [N_s, 22] representing the input of the rnn.
            - Pandas dataframe representing the rain gauge measurements.
            - Pandas series representing the RSL measurements downsampled to the same sampling rate of the rain gauge.
    """
    T = 20  # ratio between rsl and rg sampling rate, i.e 20 samples of rsl for each rg sample.
    ds_rg = pd.to_datetime(ds_rg)
    de = pd.to_datetime(de)
    # ds_rg = pd.to_datetime(rg_data.index[0])
    # de = pd.to_datetime(min(rg_data.index[-1], rsl_data.last_valid_index()))
    ds_rsl = ds_rg - pd.Timedelta(9.5, "min")

    rg_data = rg_data[ds_rg:de]  # For regression
    seq_len = len(rg_data)

    # Order rsl samples in matrix form of (T x seq_len)
    dynamic_data = rsl_data[ds_rsl:de]
    last_index = (len(dynamic_data)//T)*T
    dynamic_data = dynamic_data[:last_index]
    # rsl_time_series_averaged = rsl_data[ds_rg:de].resample('10T').mean()
    rsl_time_series_averaged = rsl_data[ds_rg:de].resample('10T', closed='right', label='right').mean()  # This is the right way to do it

    # Remove samples from the end such that it will divide by 20

    dynamic_data_reshaped = dynamic_data.values.T.reshape((-1, len(rg_data), T))
    static_data_reshaped = np.repeat(np.expand_dims(static_data, axis=1), len(rg_data), axis=1)
    # Add static data to the end of dynamic data
    dynamic_data_reshaped = np.concatenate((dynamic_data_reshaped, static_data_reshaped), axis=2)

    return dynamic_data_reshaped, rg_data, rsl_time_series_averaged


def infer(model, rsl, rg_data, rsl_avg):
    """
    Function that applies the rnn based model on the RSL measurements.
    :param model: the model.
    :param rsl: numpy array of shape [N_l, N_s, 22] containing the RSL measurements of all links.
    :param rg_data: Pandas dataframe of shape [N_s] containing the rain gauge measurements.
    :param rsl_avg: Pandas dataframe of shape [N_s, N_l] containing the RSL downsample to match the sampling rate of the rain gauge.
    :return: Two values:
            - numpy array of shape [1, N_s, N_l] containing the rainfall estimation of all links.
            - A tensor of shape [N_l, N_s, 1] containing the wet/dry scores (before the sigmoid) of the rnn of all links.
    """
    with torch.no_grad():
        # Initialize hidden state
        h = model.init_state(batch_size=rsl.shape[0])
        model.eval()
        dynamic_data = (torch.Tensor(rsl)).to(device)
        # output, h = model(dynamic_data, 0, h)  # For One step
        output, wd_scores, h = model(dynamic_data, 0, h) # For Two step
        output = output.detach().cpu().numpy().astype(float).T
    return output, wd_scores


# def evaluate_model(y_true, y_pred):
#     """
#     Function that produces bias and RMSE plots for different rain intensities measured by the rain gauge.
#     :param y_true: numpy array
#     :param y_pred: numpy array
#     """
#     # Calculate RMSE
#     rain_levels = [5, 10, 15, 30, 100]  # lowest level is 1
#     mean_NBIAS = []
#     mean_NRMSE = []
#     error = y_pred - y_true
#
#     for i in range(len(rain_levels)):
#         if i == 0:
#             ind_level = (y_true > 1) & (y_true <= rain_levels[i])
#         else:
#             ind_level = (y_true > rain_levels[i-1]) & (y_true <= rain_levels[i])
#         mean_rg_level = y_true[ind_level].mean()
#         bias = error[ind_level].mean()
#         mean_NBIAS.append(bias / mean_rg_level)
#         mse = (error[ind_level] ** 2).mean()
#         mean_NRMSE.append(np.sqrt(mse) / mean_rg_level)
#
#     print(f'MEAN NBIAS:\n {mean_NBIAS}.\n MEAN NRMSE:\n {mean_NRMSE}')
#
#     rain_level_labels = ['1<R<5', '5<R<10', '10<R<15', '15<R<30', '30<R']
#     plt.figure(1)
#     plt.bar(rain_level_labels, mean_NBIAS)
#     plt.title('MEAN NBIAS')
#     # if SAVE:
#     #     plt.savefig(f"Out/regression_results_bias_{dataset}.eps")
#     plt.xlabel('Rain rate range [mm/h]')
#     plt.figure(2)
#     plt.bar(rain_level_labels, mean_NRMSE)
#     plt.title('MEAN NRMSE')
#     plt.xlabel('Rain rate range [mm/h]')
#     # if SAVE:
#     #     plt.savefig(f"Out/regression_results_rmse_{dataset}.eps")
#     # plt.show()


def compare_models(y_pred, y_true):
    """
    Function that compares the models in terms of BIAS and RMSE for different levels of rain intensity.
    :param y_true: list of numpy array containing the rain gauge measurements. Each array in the list corresponds to different model.
    :param y_pred: list of numpy array containing the estimated rain of all links. Each array in the list corresponds to different model.
    """
    # Calculate RMSE
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    rain_levels = [5, 10, 15, 30]  # lowest level is 1
    BIAS_array = []
    RMSE_array = []

    error = y_pred - y_true

    # Iterate through models
    for m in range(y_pred.shape[0]):
        mean_NBIAS = []
        mean_NRMSE = []
        for i in range(len(rain_levels)):
            if i == 0:
                ind_level = (y_true[m] > 1) & (y_true[m] <= rain_levels[i])
            else:
                ind_level = (y_true[m] > rain_levels[i-1]) & (y_true[m] <= rain_levels[i])
            mean_rg_level = y_true[m, ind_level].mean()
            bias = error[m, ind_level].mean()
            mean_NBIAS.append(bias / mean_rg_level)
            mse = (error[m, ind_level] ** 2).mean()
            mean_NRMSE.append(np.sqrt(mse) / mean_rg_level)
        BIAS_array.append(np.array(mean_NBIAS))
        RMSE_array.append(np.array(mean_NRMSE))
    # print(f'MEAN NBIAS:\n {mean_NBIAS}.\n MEAN NRMSE:\n {mean_NRMSE}')
    BIAS_array = np.array(BIAS_array)
    RMSE_array = np.array(RMSE_array)
    rain_level_labels = ['1<R<5', '5<R<10', '10<R<15', '15<R']
    x = 2*np.arange(len(rain_level_labels))
    width = 0.25
    plt.figure(1)
    for m in range(y_pred.shape[0]):
        plt.bar(x + m*width, BIAS_array[m], width=width, label=model_short_names[m], edgecolor='black')
    plt.title('MEAN NBIAS')
    plt.xlabel('Rain rate range [mm/h]')
    plt.ylabel('NBIAS')
    plt.xticks(x + width, rain_level_labels)
    plt.legend()
    if SAVE:
        plt.savefig(f'Out/GeneralTest/regression_results_bias_{dataset}.eps')

    plt.figure(2)
    for m in range(y_pred.shape[0]):
        plt.bar(x + m*width, RMSE_array[m], width=width, label=model_short_names[m], edgecolor='black')
    plt.title('MEAN NRMSE')
    plt.xlabel('Rain rate range [mm/h]')
    plt.ylabel('NRMSE')
    plt.xticks(x + width, rain_level_labels)
    plt.legend()
    if SAVE:
        plt.savefig(f'Out/GeneralTest/regression_results_rmse_{dataset}.eps')

    # plt.show()


def infer_reference_per_sequence(attenuation_data, model_params, lengths, ds_rg, de):
    """
    Function that applies the short links model on the attenuation values of all links for the specified dates.
    :param attenuation_data: Pandas dataframe of shape [N_s, N_l] containing the attenuation of all links.
    :param model_params: Pandas dataframe containing the short links model parameters of all links. ('W', 'b').
    :param lengths: numpy array of shape [N_l,] containing the path length of each link.
    :param ds_rg: Timestamp of the start date for the inference.
    :param de: Timestamp of the end date for the inference.
    :return: numpy array containing the estimated rain of all links.
    """
    T = 20  # ratio between rsl and rg sampling rate, i.e 20 samples of rsl for each rg sample.
    ds_rg = pd.to_datetime(ds_rg)
    de = pd.to_datetime(de)
    ds_rsl = ds_rg - pd.Timedelta(9.5, "min")
    attenuation_data = attenuation_data[ds_rsl:de]
    rain_estimated = calc_rain(attenuation_data, model_params, lengths)
    rain_estimated = rain_estimated.resample('10T', closed='right', label='right').mean().values
    return rain_estimated


def infer_reference_per_sequence_reg_class(attenuation_data, std, model_params, lengths, ds_rg, de):
    """
    Function that applies the short links model on the attenuation values of all links for the dates specified by ds_rg and de.
    :param attenuation_data: Pandas dataframe of shape [N_s, N_l] containing the attenuation of all links.
    :param std: Pandas dataframe of shape [N_s, N_l]  containing the rolling standard deviation of all links.
    :param model_params: Pandas dataframe containing the short links model parameters of all links. ('W', 'b').
    :param lengths: numpy array of shape [N_l,] containing the path length of each link.
    :param ds_rg: Timestamp of the start date for the inference.
    :param de: Timestamp of the end date for the inference.
    :return: Two values:
            - numpy array of shape [N_s/20, N_l] containing the estimated rain of all links using the short links model.
            - numpy array of shape [N_s/20, N_l] containing the rolling standard deviation of all links using the short links model.
    where: N_s is the number of samples of the RSL and N_l is the number of links.
    """
    T = 20  # ratio between rsl and rg sampling rate, i.e 20 samples of rsl for each rg sample.
    ds_rg = pd.to_datetime(ds_rg)
    de = pd.to_datetime(de)
    ds_rsl = ds_rg - pd.Timedelta(9.5, "min")
    attenuation_data = attenuation_data[ds_rsl:de]
    std_data = std[ds_rsl:de]
    rain_estimated = calc_rain(attenuation_data, model_params, lengths)
    rain_estimated = rain_estimated.resample('10T', closed='right', label='right').mean().values
    std_data = std_data.resample('10T', closed='right', label='right').mean().values

    return rain_estimated, std_data


def infer_reference(attenuation_data, model_params, meta_data, rg_data, sequences_to_evaluate):
    """
    Function that applies the short links model on the attenuation values of all links.
    :param attenuation_data: Pandas dataframe of shape [N_s, N_l] containing the attenuation of all links.
    :param model_params: Pandas dataframe containing the short links model parameters of all links. ('W', 'b').
    :param meta_data: numpy array of shape [N_l, 2] containing the metadata of all links (length and frequency).
    :param rg_data: Pandas dataframe of shape [N_s/20] containing the rain gauge measurements.
    :param sequences_to_evaluate: Pandas dataframe containing the start and end date of rain events to evaluate.
    :return: Three values:
            - Dictionary containing the rain estimation and classification metrics results.
            - numpy array containing the estimated rain of all links.
            - numpy array containing the ground truth rain measured by the rain gauge.
    """
    first_iteration = True
    for i in range(len(sequences_to_evaluate)):
        ds_test = sequences_to_evaluate['ds'][i]
        de_test = sequences_to_evaluate['de'][i]
        rg_data_sub = rg_data[ds_test:de_test]
        # rsl_avg = rsl_data[ds_test:de_test].resample('10T', closed='right', label='right').mean()
        rain_estimated = infer_reference_per_sequence(attenuation_data, model_params, meta_data[:, 0], ds_test, de_test)
        if first_iteration:
            rain_estimated_array = rain_estimated
            rain_array = rg_data_sub.values.squeeze()
            first_iteration = False
        else:
            rain_estimated_array = np.concatenate(([rain_estimated_array, rain_estimated]), axis=0)
            rain_array = np.concatenate((rain_array, rg_data_sub.values.squeeze()))

    y_true = rain_array.repeat(attenuation_data.shape[1])
    y_pred = rain_estimated_array.flatten()
    metrics_dict = calc_metrics_regression(y_true, y_pred)
    # evaluate_model(y_true, y_pred)

    return metrics_dict, y_pred, y_true


def infer_reference_reg_class(attenuation_data, std, model_params, meta_data, rg_data, sequences_to_evaluate):
    """
    Function that applies the short links model on the attenuation values of all links.
    :param attenuation_data: Pandas dataframe of shape [N_s, N_l] containing the attenuation of all links.
    :param std: Pandas dataframe of shape [N_s, N_l]  containing the rolling standard deviation of all links.
    :param model_params: Pandas dataframe containing the short links model parameters of all links. ('W', 'b').
    :param meta_data: numpy array of shape [N_l, 2] containing the metadata of all links (length and frequency).
    :param rg_data: Pandas dataframe of shape [N_s/20] containing the rain gauge measurements.
    :param sequences_to_evaluate: Pandas dataframe containing the start and end date of rain events to evaluate.
    :return: Five values:
            - Dictionary containing the rain estimation and classification metrics results.
            - numpy array containing the estimated rain of all links.
            - numpy array containing the ground truth rain measured by the rain gauge.
            - numpy array containing the rolling standard deviation of the attenuation of all links.
            - numpy array containing the ground truth classification labels (1 - Wet, 0 - Dry).
    """
    first_iteration = True
    for i in range(len(sequences_to_evaluate)):
        ds_test = sequences_to_evaluate['ds'][i]
        de_test = sequences_to_evaluate['de'][i]
        rg_data_sub = rg_data[ds_test:de_test]
        # rsl_avg = rsl_data[ds_test:de_test].resample('10T', closed='right', label='right').mean()
        rain_estimated, wd_std = infer_reference_per_sequence_reg_class(attenuation_data, std, model_params, meta_data[:, 0], ds_test, de_test)
        if first_iteration:
            rain_estimated_array = rain_estimated
            wd_std_array = wd_std
            rain_array = rg_data_sub.values.squeeze()
            first_iteration = False
        else:
            rain_estimated_array = np.concatenate(([rain_estimated_array, rain_estimated]), axis=0)
            wd_std_array = np.concatenate(([wd_std_array, wd_std]), axis=0)
            rain_array = np.concatenate((rain_array, rg_data_sub.values.squeeze()))

    y_true_classification = (rain_array > 0).astype(int).repeat(rsl_data.shape[1])
    y_pred_classification = (wd_std_array > sigma_rsd.values).astype(int).flatten()
    y_true_reg = rain_array.repeat(attenuation_data.shape[1])
    # y_pred_reg = rain_estimated_array.flatten()
    y_pred_reg = (rain_estimated_array * (wd_std_array > sigma_rsd.values)).flatten()  # Multiply by w/d indicator
    metrics_dict_reg = calc_metrics_regression(y_true_reg, y_pred_reg)
    metrics_dict_classification = calc_metrics_classification(y_true_classification, y_pred_classification)
    metrics_dict_reg.update(metrics_dict_classification)
    # evaluate_model(y_true, y_pred)

    return metrics_dict_reg, y_pred_reg, y_true_reg, wd_std_array.flatten(), y_true_classification


def calc_metrics_regression(y_true, y_pred):
    """
    Function that calculates the bias and rmse for wet samples only.
    :param y_true: numpy array containing the rain gauge measurements.
    :param y_pred: numpy array containing the estimated rain of all links.
    :return: dictionary containing the BIAS and RMSE
    """
    ind = y_true > 0
    BIAS = np.mean(y_pred[ind] - y_true[ind])
    RMSE = mean_squared_error(y_true[ind], y_pred[ind], squared=False)
    metrics_dict = {'BIAS': BIAS, 'RMSE': RMSE}
    print(f'BIAS = {BIAS:.2f}.\n RMSE = {RMSE:.2f}.')
    return metrics_dict


def calc_metrics_classification(y_true, y_pred):
    """
    Function that calculates the classification metrics
    :param y_true: numpy array containing the rain gauge measurements.
    :param y_pred: numpy array containing the estimated rain of all links.
    :return: dictionary containing classification metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    TPR = tp / (tp + fn)  # Sensitivity, Recall
    FPR = fp / (fp + tn)
    FNR = fn / (tp + fn)
    TNR = tn / (fp + tn)  # Specificity, Selectivity
    ACC = (tp + tn) / (tp + fp + fn + tn)
    BAL_ACC = (TPR + TNR)/2

    Recall = tp / (tp + fn)
    Precision = tp / (tp + fp)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    metrics_dict = {'ACC': ACC, 'Balanced ACC': BAL_ACC, 'Recall': Recall, 'Precision': Precision, 'F1': F1}
    print(f'TPR = {TPR:.2f}.\n FPR = {FPR:.2f}.\n FNR = {FNR:.2f}.\n TNR = {TNR:.2f}.')
    print(f'Precision = {Precision:.2f}. Recall = {Recall:.2f}')
    print(f'Accuracy = {ACC:.2f}.')
    print(f'Balanced accuracy = {BAL_ACC:.2f}.')
    print(f'F1 = {F1:.2f}.')

    MCC = matthews_corrcoef(y_true, y_pred)
    print(f'MCC = {MCC:.2f}')

    # plot_roc(y_true, wet_dry_array.flatten())
    return metrics_dict


def infer_entire_dataset(model, rsl_data, meta_data, rg_data, sequences_to_evaluate, threshold=0.7):
    """
    Function that applies the rnn based model on the RSL measurements of all links.
    :param model: the model.
    :param rsl_data: Pandas dataframe of shape [N_s, N_l]  containing the RSL measurements of all links.
    :param meta_data: numpy array of shape [N_l, 2] containing the metadata of all links (length and frequency).
    :param rg_data: Pandas dataframe of shape [N_s/20] containing the rain gauge measurements.
    :param sequences_to_evaluate: Pandas dataframe containing the start and end date of rain events to evaluate.
    :param threshold: float used of determining wet/dry.
    :return: Five values:
            - Dictionary containing the rain estimation and classification metrics results.
            - numpy array containing the estimated rain of all links.
            - numpy array containing the ground truth rain measured by the rain gauge.
            - numpy array containing the wet/dry probabilities of the rnn model.
            - numpy array containing the ground truth classification labels (1 - Wet, 0 - Dry).
    """
    df_wd = pd.DataFrame()
    first_iteration = True
    for i in range(len(sequences_to_evaluate)):
        ds_test = sequences_to_evaluate['ds'][i]
        de_test = sequences_to_evaluate['de'][i]
        dynamic_data, rg_data2, rsl_avg = organize_data(rsl_data, meta_data,  rg_data, ds_test, de_test)
        rain_estimates, wd_scores = infer(model, dynamic_data, rg_data2, rsl_avg)
        rain_estimates = rain_estimates.squeeze()
        wd_scores = wd_scores.squeeze()
        wd_probs = torch.sigmoid(wd_scores)
        wd_probs = wd_probs.detach().cpu().numpy().astype(float).T

        if first_iteration:
            rain_estimates_array = rain_estimates
            wd_probs_array = wd_probs
            rain_array = rg_data2.values.squeeze()
            first_iteration = False
        else:
            rain_estimates_array = np.concatenate(([rain_estimates_array, rain_estimates]), axis=0)
            wd_probs_array = np.concatenate(([wd_probs_array, wd_probs]), axis=0)
            rain_array = np.concatenate((rain_array, rg_data2.values.squeeze()))


        # data = pd.DataFrame(rsl_avg)
        # data['wd'] = wet_dry_probabilities
        # data['rg'] = rg_data
        # df_wd = df_wd.append(data)
    # df_wd.plot()
    # plt.show()

    # plt.plot(wet_dry_array)
    # plt.plot(rain_array)
    # plt.show()

    # Calculate confusion matrix
    y_true_reg = rain_array.repeat(rsl_data.shape[1])
    y_pred_reg = (rain_estimates_array).flatten()

    y_true_classification = (rain_array > 0).astype(int).repeat(rsl_data.shape[1])
    y_pred_classification = (wd_probs_array > threshold).astype(int).flatten()
    metrics_dict_reg = calc_metrics_regression(y_true_reg, y_pred_reg)
    metrics_dict_classification = calc_metrics_classification(y_true_classification, y_pred_classification)
    metrics_dict_reg.update(metrics_dict_classification)
    # evaluate_model(y_true, y_pred)

    return metrics_dict_reg, y_pred_reg, y_true_reg, wd_probs_array.flatten(), y_true_classification


def plot_roc(df_models):
    """
    Function that plots the roc curve of the various models.
    :param df_models: dictionary where the keys are the model short names. The values are tuple of two elements where the first
    one is a numpy array containing the ground truth wet/dry labels from the rain gauge. The second one is the wet/dry probabilities
    of the specific model.
    """
    plt.figure()
    lw = 2
    for model_name in df_models:
        fpr, tpr, thresholds = roc_curve(df_models[model_name][0], df_models[model_name][1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,
            lw=lw,
            label=f'{model_name} AUC={roc_auc:.2f}'
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    if SAVE:
        plt.savefig(f'Out/GeneralTest/classification_models_roc_curve_{dataset}.eps')


def plot_pr(df_models):
    """
    Function that plots the precision-recall curve of the various models.
    :param df_models: dictionary where the keys are the model short names. The values are tuple of two elements where the first
    one is a numpy array containing the ground truth wet/dry labels from the rain gauge. The second one is the wet/dry probabilities
    of the specific model.
    """
    plt.figure()
    lw = 2
    for model_name in df_models:
        precision, recall, thresholds = precision_recall_curve(df_models[model_name][0], df_models[model_name][1])
        # Find optimal threshold in terms of F1
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_max = np.nanargmax(f1)
        thr_best = thresholds[f1_max]
        print(f'best threshold = {thr_best}, precision={precision[f1_max]}, recall={recall[f1_max]}, f1={f1[f1_max]}.')
        average_precision = average_precision_score(df_models[model_name][0], df_models[model_name][1])
        plt.plot(recall, precision,
            lw=lw,
            label=f'{model_name} AP={average_precision:.2f}'
        )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall curve")
    plt.legend(loc="lower right")


if __name__ == '__main__':
    time_start = time.time()
    dataset = 'validation'
    # threshold = 0.7
    # thresholds = [0.72, 0.76, 0.72, 0.75, '']  # For TimePeriods
    thresholds = [0.75, 0.73, 0.75, 0.77, '']  # For GeneralTest

    SAVE = False
    # model_names = [
    #     'RegressionTwoStep_rt-GRU_mt-skip_fz-large_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200_train1.pt',
    #     'RegressionTwoStep_rt-GRU_mt-skip_fz-large_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200_train2.pt',
    #     'RegressionTwoStep_rt-GRU_mt-skip_fz-large_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200_train3.pt',
    #     'RegressionTwoStep_rt-GRU_mt-skip_fz-large_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200_train4.pt',
    #     'Reference'
    # ]
    model_names = [
        'RegressionTwoStep_rt-GRU_mt-rnn_fz-small_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200.pt',
        'RegressionTwoStep_rt-GRU_mt-rnn_fz-large_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200.pt',
        'RegressionTwoStep_rt-GRU_mt-skip_fz-small_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200.pt',
        'RegressionTwoStep_rt-GRU_mt-skip_fz-large_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200.pt',
        'Reference'
    ]
    # model_short_names = ['Train 1', 'Train 2', 'Train 3', 'Train 4', 'Reference']
    model_short_names = ['Small RH', 'Large RH', 'Small RH + skip', 'Large RH + skip', 'Reference']

    # output_file_name = f'time_periods_reg_classification-{dataset}.csv'
    output_file_name = f'reg_classification-{dataset}.csv'
    # output_path = os.path.join('Out/TimePeriods', output_file_name)
    output_path = os.path.join('Out/GeneralTest', output_file_name)

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
    rolling_std = rehovot_dataset.rolling_std[links_name]
    sigma_rsd = rehovot_dataset.sigma[links_name]
    rg_data = rehovot_dataset.rg


    resuls_df = pd.DataFrame(columns=['Model', 'BIAS', 'RMSE', 'THR', 'ACC', 'Balanced ACC', 'Recall', 'Precision', 'F1'])
    models_scores_dict = {}

    predictions_list = []
    gt_list = []
    # Loop through models
    for i, model_name in enumerate(model_names, 1):
        # model_path = os.path.join('Models', model_name)
        # Load pre trained model
        print(f'Evaluating model {i}/{len(model_names)}')
        print(model_name)
        if model_name == 'Reference':
            model_params_short_links = pd.read_pickle("ShortLinksModels/model_params_1down.pkl")
            model_params_short_links = model_params_short_links[links_name]
            metrics_model, y_pred_reg, y_true_reg, wd_probs_pred, y_true_classification = infer_reference_reg_class(attenuation_data, rolling_std, model_params_short_links, meta_data, rg_data, rehovot_dataset.valid_sequences)

        else:
            model = load_model(threshold=thresholds[i-1], model_name=model_name)
            metrics_model, y_pred_reg, y_true_reg, wd_probs_pred, y_true_classification = infer_entire_dataset(model, rsl_data, meta_data,  rg_data, rehovot_dataset.valid_sequences, thresholds[i-1])
        metrics_model['Model'] = model_name
        metrics_model['THR'] = thresholds[i-1]

        resuls_df = resuls_df.append(metrics_model, ignore_index=True)
        models_scores_dict[model_short_names[i-1]] = (y_true_classification, wd_probs_pred)

        predictions_list.append(y_pred_reg)
        gt_list.append(y_true_reg)
        # models_scores_dict[model_short_names[i-1]] = (true_labels, model_wd_probs)
    compare_models(predictions_list, gt_list)
    plot_roc(models_scores_dict)
    plot_pr(models_scores_dict)

    time_end = time.time()
    print(f'Finished testing {len(model_names)} models. Took {(time_end-time_start):.2f} seconds.')
    if SAVE:
        resuls_df.to_csv(output_path, float_format='%.3f', index=False)
    plt.show()
    print('Done')
