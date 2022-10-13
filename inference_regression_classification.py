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
import torch.nn as nn

from Smbit import Rehovot, RSL, load_rain_gauge
from model import INPUT_NORMALIZATION, TwoStepNetworkGeneric
import torchvision
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, \
    roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay
from utils import rolling_wet_dry
from short_links_model import calc_rain

N_LAYERS = 2
RNN_FEATURES = 256
FC_FEATURES = 16
STATIC_INPUT_SIZE = 2  # 2
DYNAMIC_INPUT_SIZE = 20
INPUT_SIZE = DYNAMIC_INPUT_SIZE + STATIC_INPUT_SIZE
TOTAL_FEATURES = FC_FEATURES + RNN_FEATURES
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


def load_model(model_name, threshold):
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
                           threshold=threshold,
                           model_type=model_config['mt'],
                           fully_size=model_config['fz'])

    state_dict = torch.load(model_path, map_location=torch.device(device))
    # print(state_dict.keys())
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def organize_data_3(rsl_data, attenuation_data, std_data, static_data, rg_data, ds_rg, de):
    """
    Function that organizes the input data of the network.
    :param rsl_data: Pandas series representing the RSL measurements of the specified link.
    :param attenuation_data: Pandas series representing the attenuation of the specified link.
    :param std_data: Pandas series representing the rolling standard deviation of the RSL of the specified link.
    :param static_data: ndarray of shape [2,] representing the metadata of the link (length, frequency).
    :param rg_data: Pandas dataframe containing the rain gauge measurements.
    :param ds_rg: string representing the start date of the subsequence.
    :param de: string representing the end date of the subsequence.
    :return: Five values:
            - ndarray of shape [N_s, 22] representing the input of the rnn.
            - Pandas dataframe representing the rain gauge measurements.
            - Pandas series representing the RSL measurements downsampled to the same sampling rate of the rain gauge.
            - Pandas series representing the attenuation of the specified link.
            - Pandas series representing the rolling standard deviation of the RSL of the specified link.
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
    attenuation_data = attenuation_data[ds_rsl:de]
    std_data = std_data[ds_rsl:de]
    last_index = (len(dynamic_data)//T)*T
    dynamic_data = dynamic_data[:last_index]
    attenuation_data = attenuation_data[:last_index]
    std_data = std_data[:last_index]


    # rsl_time_series_averaged = rsl_data[ds_rg:de].resample('10T').mean()
    rsl_time_series_averaged = rsl_data[ds_rg:de].resample('10T', closed='right', label='right').mean()  # This is the right way to do it

    # Remove samples from the end such that it will divide by 20
    dynamic_data_reshaped = dynamic_data.values.T.reshape((-1, T))
    static_data_reshaped = np.repeat(np.expand_dims(static_data, axis=0), seq_len, axis=0)
    # Add static data to the end of dynamic data
    dynamic_data_reshaped = np.concatenate((dynamic_data_reshaped, static_data_reshaped), axis=1)
    return dynamic_data_reshaped, rg_data, rsl_time_series_averaged, attenuation_data, std_data


def infer(model, rsl, rg_data, rsl_avg):
    """
    Function that applies the rnn model on the RSL measurements.
    :param model: the model.
    :param rsl: numpy array of shape [N_s, 22] representing the input to the rnn model.
    :param rg_data: Pandas dataframe containing the rain gauge measurements.
    :param rsl_avg: Pandas series representing the RSL measurements downsampled to the same sampling rate of the rain gauge.
    :return: Two values:
            - numpy array containing the estimated rain samples from the specified link.
            - numpy array containing the wet/dry probabilities of the specifed link.
    """
    with torch.no_grad():
        # Initialize hidden state
        h = model.init_state(batch_size=1)
        model.eval()
        dynamic_data = (torch.Tensor(rsl)).to(device)
        dynamic_data = torch.unsqueeze(dynamic_data, dim=0)
        # meta_data = meta_data.squeeze()
        # rg_data = rg_data.squeeze()
        # output, h = model(dynamic_data, 0, h) # For One Step
        output, wd_scores, h = model(dynamic_data, 0, h) # For Two step

        # output = torch.sigmoid(output)  # For BCEWithLogitsLoss

        # Convert to 0/1
        # wet_dry = (output > threshold).detach().numpy().astype(int).flatten()
        rain_estimation = output.detach().cpu().numpy().astype(float).flatten()


        # Keep as probabilities
        wet_dry_probabilities = torch.sigmoid(wd_scores).view(-1).detach().numpy().astype(float)


        # df_wd = pd.DataFrame(rsl_avg)
        # df_wd['wd'] = wet_dry
        # df_wd['rg'] = rg_data
        # df_wd.plot()
        # plt.show()
    return rain_estimation, wet_dry_probabilities


# def infer_reference_wet_dry(rsl_data, ds_rg, de):
#     T = 20  # ratio between rsl and rg sampling rate, i.e 20 samples of rsl for each rg sample.
#     ds_rg = pd.to_datetime(ds_rg)
#     de = pd.to_datetime(de)
#     ds_rsl = ds_rg - pd.Timedelta(9.5, "min")
#     rsl_data = rsl_data[ds_rsl:de]
#     wet_dry_probs_ref, sigma = rolling_wet_dry(rsl_data)
#     wet_dry_probs_ref = wet_dry_probs_ref.resample('10T', closed='right', label='right').mean().values
#     return wet_dry_probs_ref, sigma


def infer_reference_estimation(attenuation, std_data, model_params, length, ds_rg, de):
    """
    Function that applies the short links model on the attenuation values from the specified link.
    :param attenuation: Pandas series representing the attenuation of the specified link.
    :param std_data: numpy array representing the rolling standard deviation of the RSL of the specified link.
    :param model_params: Pandas series containing the short links model parameters of the specified link. ('W', 'b').
    :param length: float representing the path length of the link.
    :param ds_rg: Timestamp representing the start date of the subsequence.
    :param de: Timestamp representing the end date of the subsequence.
    :return: Two values:
            - numpy array containing the estimated rain samples from the specified link using the short links model.
            - numpy array containing the rolling standard deviation of the specifed link downsampled to the sampling rate of the rain gauge.
    """
    T = 20  # ratio between rsl and rg sampling rate, i.e 20 samples of rsl for each rg sample.
    ds_rg = pd.to_datetime(ds_rg)
    de = pd.to_datetime(de)
    ds_rsl = ds_rg - pd.Timedelta(9.5, "min")
    attenuation = attenuation[ds_rsl:de]
    std_data = std_data[ds_rsl:de]
    # Apply model
    rain = calc_rain(attenuation, model_params, length)
    # Downsample rain dataframe
    rain = rain.resample('10T', closed='right', label='right').mean().values
    std_data = std_data.resample('10T', closed='right', label='right').mean().values
    return rain, std_data


def plot_combined_with_reference(df, sigma_ref, threshold, link_name, ds=0, de=0, save=False):
    """
    Function that plots the rainfall estimation and wet/dry probabilities of both the rnn and the short links model.
    :param df: Pandas dataframe summarizing the rain estimation and wet/dry classification of both the rnn model and the short links model.
    :param sigma_ref: float representing the threshold used for wet/dry classification of the reference method.
    :param threshold: float representing the threshold of the rnn based wet/dry classification.
    :param link_name: string indicating the name of the link.
    :param ds: string representing the start date of the inference.
    :param de: string representing the end date of the inference.
    :param save: boolean. Set to True to save the figures.
    """
    if ds != 0 and de != 0:
        df = df[ds:de]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(16, 9))
    time = df.index.values
    ax1.plot(time, df[link_name].values, color='black')
    ax1.grid()
    ax1.set_ylabel(r'(a) $\bar{RSL}$ [dBm]')
    # ax.plot(time, df['wd'].values)

    ax2.plot(time, df['rg'].values, color='black', label='RG')
    ax2.plot(time, df['rain_estimation'].values, color='blue', label='RNN')
    ax2.plot(time, df['rain_model'].values, color='red', label='Model')
    ax2.grid()
    ax2.set_ylabel('(b) Rain rate [mm/h]')

    ax3.plot(time, df['wd_probability'], color='black')
    ax3.grid()
    ax3.axhline(threshold, color='black', lw=1, ls='--', alpha=0.7)
    ax3.set_ylabel(r'(c) $p_n$')

    ax4.plot(time, df['wd_probability_ref'], color='black')
    ax4.grid()
    ax4.axhline(sigma_ref, color='black', lw=1, ls='--', alpha=0.7)
    ax4.set_ylabel('(d) RSD')
    ax4.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d/%m/%y\n%H:%M:%S'))


    P = df['wd'] == 1
    TP = (df['wd'] == 1) & (df['rg'] > 0)
    FP = (df['wd'] == 1) & (df['rg'] == 0)
    TN = (df['wd'] == 0) & (df['rg'] == 0)
    FN = (df['wd'] == 0) & (df['rg'] > 0)

    TP_ref = (df['wd_ref'] == 1) & (df['rg'] > 0)
    FP_ref = (df['wd_ref'] == 1) & (df['rg'] == 0)
    TN_ref = (df['wd_ref'] == 0) & (df['rg'] == 0)
    FN_ref = (df['wd_ref'] == 0) & (df['rg'] > 0)

    alpha = 1
    # ax2.fill_between(time, 0, 1, where=df['rg'] > 0, color='lightgreen', alpha=alpha, transform=ax2.get_xaxis_transform(), label='Wet')

    ax3.fill_between(time, 0, 1, where=TP, color='lightgreen', alpha=alpha, transform=ax3.get_xaxis_transform(), label='TP')
    ax3.fill_between(time, 0, 1, where=FP, color='yellow', alpha=alpha, transform=ax3.get_xaxis_transform(), label='FP')
    ax3.fill_between(time, 0, 1, where=FN, color='salmon', alpha=alpha, transform=ax3.get_xaxis_transform(), label='FN')

    ax4.fill_between(time, 0, 1, where=TP_ref, color='lightgreen', alpha=alpha, transform=ax4.get_xaxis_transform(), label='TP')
    ax4.fill_between(time, 0, 1, where=FP_ref, color='yellow', alpha=alpha, transform=ax4.get_xaxis_transform(), label='FP')
    ax4.fill_between(time, 0, 1, where=FN_ref, color='salmon', alpha=alpha, transform=ax4.get_xaxis_transform(), label='FN')

    # Decided wet
    # ax1.fill_between(time, 0, 1, where=P, color='lightgreen', alpha=alpha, transform=ax1.get_xaxis_transform(), label='Decided Wet')
    # ax3.fill_between(time, 0, 1, where=df['rg'] > 0, color='lightgreen', alpha=alpha, transform=ax3.get_xaxis_transform(), label='True Wet')

    # ax1.legend(bbox_to_anchor=(0., 1., 1., 0.), loc='lower left', ncol=3, mode="expand", borderaxespad=0.2)

    ax2.legend(loc='upper left')
    ax3.legend(loc='upper left')
    ax4.legend(loc='upper left')

    fig.suptitle(f'Link {link_name}')
    if save:
        plt.savefig(f"Out/InferenceRegressionClassification/inference_{link_name}_{dataset}.eps")  # Must be before plt.show(), otherwise it will be blank image.


def plot_pr(y_true, score):
    """
    Function that plots a precision recall graph.
    :param y_true: numpy array containing the ground truth rain gauge measurements (1 - Wet, 0 - Dry).
    :param score: numpy array containing the wet/dry probabilities.
    """
    # precision, recall, _ = precision_recall_curve(y_true, score)
    PrecisionRecallDisplay.from_predictions(y_true, score)


def plot_roc(y_true, scores):
    """
    Function that plots a ROC curve.
    :param y_true: numpy array containing the ground truth rain gauge measurements (1 - Wet, 0 - Dry).
    :param score: numpy array containing the wet/dry probabilities.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr,
             color="darkorange",
             lw=lw,
             label="ROC curve (area = %0.2f)" % roc_auc,
             )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")


def infer_entire_dataset_wtih_reference(model, rsl_data, threshold, attenuation_data, std_data, sigma_rsd, meta_data, model_params, rg_data, sequences_to_evaluate, link_name, ds, de, save):
    """
    Function that plots the inference results. It plots both the RNN and the short links model results.
    :param model: the model.
    :param rsl_data: Pandas series representing the RSL measurements of the specified link.
    :param threshold: float representing the threshold of the rnn based wet/dry classification.
    :param attenuation_data: Pandas series representing the attenuation of the specified link.
    :param std_data: Pandas series representing the rolling standard deviation of the RSL of the specified link.
    :param sigma_rsd: float representing the threshold of the reference method for wet/dry classification.
    :param meta_data: ndarray of shape [2,] representing the metadata of the link (length, frequency).
    :param model_params: Pandas series containing the short links model parameters of the specified link. ('W', 'b').
    :param rg_data: Pandas dataframe containing the rain gauge measurements.
    :param sequences_to_evaluate: Pandas dataframe containing the start and end dates of each rain event.
    :param link_name: string indicating the name of the link.
    :param ds: string representing the start date of the inference period.
    :param de: string representing the end date of the inference period.
    :param save: boolean. Set to True to save the figures.
    """
    df_wd = pd.DataFrame()
    for i in range(len(sequences_to_evaluate)):
        ds_test = sequences_to_evaluate['ds'][i]
        de_test = sequences_to_evaluate['de'][i]
        dynamic_data, rg_data2, rsl_avg, attenuation, std = organize_data_3(rsl_data, attenuation_data, std_data, meta_data, rg_data, ds_test, de_test)

        # RNN
        rain_estimation, wet_dry_probabilities = infer(model, dynamic_data, rg_data2, rsl_avg)
        # Short links model
        rain_est_model, wet_dry_probs_model = infer_reference_estimation(attenuation, std, model_params, meta_data[0], ds_test, de_test)

        # Unnormalize the RSL
        # rsl_avg = rsl_avg*rehovot_dataset.std_dyn[link_index] + rehovot_dataset.mean_dyn[link_index]

        data = pd.DataFrame(rsl_avg)
        data['rain_estimation'] = rain_estimation
        data['rg'] = rg_data
        data['rain_model'] = rain_est_model * (wet_dry_probs_model > sigma_rsd)
        data['wd_probability'] = wet_dry_probabilities
        data['wd'] = (wet_dry_probabilities > threshold).astype(int)
        data['wd_probability_ref'] = wet_dry_probs_model
        data['wd_ref'] = (wet_dry_probs_model > sigma_rsd).astype(int)


        # data['wd_probability_ref'] = wet_dry_probs_ref
        # data['wd_ref'] = (wet_dry_probs_ref > sigma_ref).astype(int)
        df_wd = df_wd.append(data)
    # df_wd.plot()

    # plot_visualization_with_reference(df_wd)  # Plot for all period with reference model
    plt.rcParams.update({'font.size': 12})
    plot_combined_with_reference(df_wd, sigma_rsd, threshold, link_name, ds, de, save)
    # plt.show()
    return


if __name__ == '__main__':
    # ds = '2019-11-01' # train
    # de = '2020-04-01' # train

    ds = '2020-11-04 05:00:00'  # validation
    de = '2020-11-05 05:00:00'  # validation

    # ds = '2021-01-19 12:00:00'  # test
    # de = '2021-01-20 11:00:00'  # test

    hop = 2
    dir = 'down'
    link_name = f'{hop}_{dir}'
    threshold = 0.7
    dataset = 'validation'
    SAVE = False



    # Load pre trained model
    model = load_model(model_name='RegressionTwoStep_rt-GRU_mt-skip_fz-large_sl-48_h-256_nl-2_d-0.5_wd-0.0001_e-200_train4.pt', threshold=threshold)
    # Load short links model parameters
    model_params_short_links = pd.read_pickle("ShortLinksModels/model_params_1down_train4.pkl")

    # rain_df_test = load_rain_gauge(rg_path_test)
    hops_to_keep_test = [hop]
    hops_to_exclude_test = [i for i in range(1,  67) if i not in hops_to_keep_test]
    rehovot_dataset = Rehovot(dataset=dataset, seq_len=SEQ_LENGTH, hops_to_exclude=hops_to_exclude_test)
    # rain_df_test = load_rain_gauge(rg_path)
    # rehovot_dataset_test = Rehovot(db_path, rain_df_test, seq_len=SEQ_LENGTH, hops_to_exclude=[])


    # For single links
    link_index = np.argwhere(rehovot_dataset.dynamic_data.columns == link_name)[0, 0]
    rsl_data = rehovot_dataset.dynamic_data[link_name]
    meta_data = rehovot_dataset.static_data[link_index]
    attenuation = rehovot_dataset.attenuation[link_name]
    rolling_std = rehovot_dataset.rolling_std[link_name]
    sigma_rsd = rehovot_dataset.sigma[link_name]
    model_params_short_links = model_params_short_links[link_name]


    # For averaged links from the same hop
    # link_index = np.argwhere(rehovot_dataset.dynamic_data.columns == hop)[0, 0]
    # rsl_data = rehovot_dataset.dynamic_data[hop]
    # meta_data = rehovot_dataset.static_data[link_index]
    # attenuation = rehovot_dataset.attenuation[hop]
    # model_params_short_links = model_params_short_links[hop]

    # meta_data[0] = 2  # Check if changes in link length causes change in estimation
    rg_data = rehovot_dataset.rg

    criterion_mse = nn.MSELoss()


    # infer_entire_dataset(rsl_data, meta_data,  rg_data, rehovot_dataset.valid_sequences, ds, de)
    infer_entire_dataset_wtih_reference(model, rsl_data, threshold, attenuation, rolling_std, sigma_rsd, meta_data, model_params_short_links, rg_data, rehovot_dataset.valid_sequences, link_name, ds, de, SAVE)


    plt.show()



    # rsl_link = rsl_data[ds:de][link_name]

    # infer(model, dynamic_data, rg_data2, rsl_avg)
    # hops_to_evaluate = [1, 2, 3, 4]
    # evaluate_model(model, rehovot_dataset_test, hops_to_evaluate, ds, de)

    print('Done')
