import os
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib
from matplotlib import pyplot as plt
import datetime
from utils import hops_to_exclude_train, hops_to_exclude_test, RandomNoise
from scipy.optimize import curve_fit
SEQ_LENGTH = 48


def calc_observed_attenuation(dynamic_data):
    """
    Function that calculates the obsereved attenuation from the RSL measurements
    :param dynamic_data: Pandas dataframe of shape [N_s, N_l] containing the RSL measurements of all links.
    :return: Pandas dataframe of shape [N_s, N_l] containing the observed attenuation of all links.
    """
    print('Calculating attenuation')
    window = '15min'
    rsl_avg = dynamic_data.rolling(window, center=True).mean()
    baseline = rsl_avg.rolling('7D', center=True).median()
    attenuation = np.maximum(baseline - dynamic_data, 0)
    return attenuation


def calc_rain(attenuation, model_params, lengths):
    """
    Function that calculates the rainfall according to the short link model. It assumes a frequency of 74.375 GHz.
    :param attenuation: Pandas dataframe of shape [N_s, N_l] containing the observed attenuation of all links.
    :param model_params: Pandas dataframe containing the short links model parameters of all links. ('W', 'b').
    :param lengths: numpy array of shape [N_l,] containing the path length of each link.
    :return: Pandas dataframe of shape [N_s, N_l] containing the estimated rain from each link.
    """
    print('Computing rain')
    alpha = 1.0911
    beta = 0.7123
    rain = (np.maximum(attenuation - model_params.loc['b'], 0) / (alpha * model_params.loc['W'] * lengths))**(1/beta)
    return rain


def fit_all_links(attenuation, lengths):
    """
   Function that calculates the model parameters of each link
   :param attenuation: Pandas dataframe of shape [N_s, N_l] containing the observed attenuation of all links.
   :param lengths: Pandas series of shape [N_l,] containing the path length of each link.
   :return: Pandas dataframe containing the short links model parameters of all links. ('W', 'b').
   """
    print('Fitting model')
    W, b = [], []
    def fit_model(attenuation, normalized_attenuation_long):
        popt, pcov = curve_fit(func, attenuation, normalized_attenuation_long, bounds=(0, [10., 10.]))
        return popt[0], popt[1]

    reference_link_name = '1_down'  # '29_up' for sublinks
    attenuation_long = attenuation[reference_link_name].values.reshape(-1)
    for hop_name in attenuation.columns:
        def func(x, W, b):
            return np.maximum(x - b, 0) /(lengths[hop_name] * W)
        attenuation_i = attenuation[hop_name].values.reshape(-1)
        W_i, b_i = fit_model(attenuation_i, attenuation_long / lengths[reference_link_name])
        W.append(W_i)
        b.append(b_i)
    results = pd.DataFrame(np.array([W, b]), index=['W', 'b'], columns=attenuation.columns)
    return results


def arrange_dataset(db):
    """
        A function that arranges the database of the dataset.
        :param db: dictionary where the keys are the hop ID and the values are the corresponding RSL object.
                RSL object contains the following fields:
                    frequency_down: float representing the frequency of the downlink channel.
                    frequency_up: float representing the frequency of the uplink channel.
                    hop_id: integer representing the hop ID (same for both links).
                    hop_name: string representing the name of the hop.
                    length:: float representing the path length of the hop/link.
                    rsl: Pandas dataframe of shape [N_s,2] containing the RSL samples of the uplink and downlink channels.
                        where N_s is the number of samples.
        :return: Three values:
                - First one is a Numpy array of shape [N_l,2] representing the static data. Each row contains the frequency and path length of each link.
                - Second one is a Pandas dataframe of shape [2, N_l] representing the static data.
                - Third one is a Pandas dataframe of shape [N_s,N_l] representing the dynamic data.
        where:  N_l is the number of links in the dataset.
                N_s is the total number of samples in the dataset.
        """
    print('Arranging dataset')
    link_names = []
    static_data = []  # [Length, Frequency]
    dynamic_data = []
    for hop in db.values():
        # if hop.hop_id in self.hops_to_exclude:
        #     continue
        if len(hop.rsl['rsl_up']) != 0 and not all(np.isnan(hop.rsl['rsl_up'].values)):  # and not any(np.isnan(hop.rsl['rsl_up'][date_start:date_end].values))
            static_data.append((hop.length, hop.frequency_up))
            dynamic_data.append(hop.rsl['rsl_up'])
            link_names.append(f'{hop.hop_id}_up')
        if len(hop.rsl['rsl_down']) != 0 and not all(np.isnan(hop.rsl['rsl_down'].values)):  # and not any(np.isnan(hop.rsl['rsl_down'][date_start:date_end].values))
            static_data.append((hop.length, hop.frequency_down))
            dynamic_data.append(hop.rsl['rsl_down'])
            link_names.append(f'{hop.hop_id}_down')
    static_data = np.array(static_data)
    # dynamic_data = np.array(dynamic_data)
    dynamic_data = pd.concat(dynamic_data, axis=1, keys=link_names)
    # Fill NaNs
    dynamic_data.fillna(method='ffill', inplace=True)
    dynamic_data.fillna(method='bfill', inplace=True)
    dynamic_data.fillna(0, inplace=True)  # Fill missing columns with zeroes
    # dynamic_data.plot()
    # plt.show()

    # Save static data as df
    static_data_df = pd.DataFrame(static_data.T, index=['length', 'frequency'], columns=link_names)
    return static_data, static_data_df, dynamic_data


def average_hops(static_data_df, dynamic_data):
    """
        A function that averages the RSL values of links from the same hop.
        :param static_data_df: Pandas dataframe of shape [2, N_l] representing the static data.
        :param dynamic_data: Pandas dataframe of shape [N_s,N_l] representing the dynamic data.
        :return: Three values:
                - Numpy array of shape [N_h,2] representing the static data. Each row contains the frequency and path length of each hop.
                - Pandas dataframe of shape [2, N_h] representing the static data.
                - Pandas dataframe of shape [N_s,N_h] representing the dynamic data after averaging links from the same hop.
        where: N_s is the number of samples, N_l is the number of links and N_h is the number of hops.
    """
    print('Averaging links from the same hop')
    columns = list(dynamic_data.columns)
    hop_ids = np.array([int((hop_name.split('_'))[0]) for hop_name in columns])
    averaged_df = pd.DataFrame(columns=set(hop_ids), index=dynamic_data.index)
    static_data = []
    for hop_id in set(hop_ids):
        column_idx = (np.argwhere(hop_ids == hop_id)).flatten()
        averaged_df[hop_id] = dynamic_data.iloc[:, column_idx].mean(axis=1)
        static_data.append(static_data_df.iloc[0, column_idx[0]])
    static_data = np.expand_dims(np.array(static_data), axis=1)
    static_data_df = pd.DataFrame(static_data.T, index=['length'] ,columns=set(hop_ids))
    return static_data, static_data_df, averaged_df


def remove_samples(dataset, dynamic_data):
    """
    A function that deletes samples that are not included in the range of dates stated by the valid_sequences files.
    It is also deleting samples from the end of each subsequence such that the length of each subsequence
    is divisible by the sequence length.
    :param dataset: string indicating the type of dataset: 'train', 'validation' or 'test'.
    :param dynamic_data: Pandas dataframe of shape [N_s,N_l] representing the dynamic data.
    :return: Pandas dataframe of shape [N_s,N_l] representing the dynamic data after removing samples.
    """
    if dataset == 'train':
        valid_sequences = pd.read_csv(valid_sequences_train)
    elif dataset == 'validation':
        valid_sequences = pd.read_csv(valid_sequences_validation)
    else:
        valid_sequences = pd.read_csv(valid_sequences_test)
    # print(valid_sequences)
    delta = pd.to_datetime(valid_sequences['de']) - pd.to_datetime(valid_sequences['ds'])
    # Iterate through df
    modified_valid_sequences = valid_sequences.copy()
    for i in range(len(valid_sequences)):
        num_seq = delta[i].total_seconds()//(1 * 600)
        # modified_valid_sequences['de'].iloc[i] = pd.to_datetime(valid_sequences['ds'].iloc[i]) + num_seq*pd.Timedelta(self.seq_len*600, "sec")
        modified_valid_sequences.loc[i, 'ds'] = pd.to_datetime(modified_valid_sequences['ds'].iloc[i])
        modified_valid_sequences.loc[i, 'de'] = pd.to_datetime(valid_sequences['ds'].iloc[i]) + num_seq*pd.Timedelta(1*600, "sec") - pd.Timedelta(10, "min")

    # Remove bad samples from the beginning
    de_delete = pd.to_datetime(modified_valid_sequences['ds'].iloc[0]) - pd.Timedelta(10, "min")
    # indices_to_remove_rg = self.rg[:de_delete].index
    # self.rg.drop(indices_to_remove_rg, inplace=True)
    de_delete_rsl = de_delete
    indices_to_remove_rsl = dynamic_data[:de_delete_rsl].index
    dynamic_data.drop(indices_to_remove_rsl, inplace=True)

    # Remove bad samples
    for i in range(len(valid_sequences)-1):
        ds_delete = pd.to_datetime(modified_valid_sequences['de'].iloc[i]) + pd.Timedelta(10, "min")
        de_delete = pd.to_datetime(modified_valid_sequences['ds'].iloc[i+1]) - pd.Timedelta(10, "min")
        ds_delete_rsl = ds_delete - pd.Timedelta(9.5, "min")
        if ds_delete in dynamic_data.index and de_delete in dynamic_data.index:
            indices_to_remove_rsl = dynamic_data[ds_delete_rsl:de_delete].index
            # indices_to_remove_rg = self.rg[ds_delete:de_delete].index
            dynamic_data.drop(indices_to_remove_rsl, inplace=True)
            # self.rg.drop(indices_to_remove_rg, inplace=True)

    # Remove bad samples from the end
    ds_delete = pd.to_datetime(modified_valid_sequences['de'].iloc[len(valid_sequences)-1]) + pd.Timedelta(10, "min")
    # indices_to_remove_rg = self.rg[ds_delete:].index
    # self.rg.drop(indices_to_remove_rg, inplace=True)
    # ds_delete_rsl = self.rg.last_valid_index() + pd.Timedelta(0.5, "min")
    ds_delete_rsl = ds_delete

    indices_to_remove_rsl = dynamic_data[ds_delete_rsl:].index
    dynamic_data.drop(indices_to_remove_rsl, inplace=True)

    # self.rg['ind'] = np.arange(len(self.rg))  # For debug only
    return dynamic_data


def shift_rsl(rsl):
    """
    A function that Shifts rsl values of each link according to the rain gauge samples by the value that maximizes the cross correlation.
    :param rsl: Pandas dataframe of shape [N_s,N_l] representing the dynamic data.
    :return: Pandas dataframe of shape [N_s,N_l] representing the dynamic data after shifting.
    """
    rsl_shifted = np.zeros_like(rsl)
    xcorr = np.zeros((2*rsl.shape[0]-1, rsl.shape[1]))
    center = rsl.shape[0] - 1
    ref_link = '1_down'
    rsl_ref = rsl[ref_link]

    for i in range(rsl.shape[1]):
        xcorr[:, i] = np.correlate(rsl.iloc[:, i], rsl_ref, mode='full')
        # k=0 corresponds to len(rg)-1
        # If cross correlation is zero don't shift the rsl
        if not xcorr[:, i].any():
            rsl_shifted[:, i] = rsl.iloc[:, i]
            continue
        shift_val = -(np.argmax(xcorr[:, i]) - center)
        # Shift RSL
        # if shift_val == 0:
        #     rsl_shifted[:, i] = rsl[:, i]
        # elif shift_val > 0:
        #     rsl_shifted[:, i] = np.hstack((np.ones((shift_val)) * rsl[0, i], rsl[:-shift_val, i]))
        # else:
        #     rsl_shifted[:, i] = np.hstack((rsl[-shift_val:, i], np.ones((-shift_val)) * rsl[-1, i]))
        rsl_shifted[:, i] = rsl.iloc[:, i].shift(periods=shift_val, fill_value=0)
    rsl_shifted = pd.DataFrame(rsl_shifted, index=rsl.index, columns=rsl.columns)


    # Visuallize correlation
    # link = '3_up'
    # link_index = np.argwhere(rsl.columns == link)[0, 0]
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(rsl.loc[:, link], label='rsl')
    # plt.plot(rsl_shifted.loc[:, link], label='rsl_shifted')
    # plt.xlabel('n')
    # plt.legend()
    # plt.subplot(3, 1, 2)
    # plt.plot(rsl.loc[:, ref_link])
    # plt.xlabel('n')
    # plt.subplot(3, 1, 3)
    # corr_axis = np.arange(-center, center+1)
    # plt.plot(corr_axis, xcorr[:, link_index])
    # plt.xlabel('n')
    # plt.title('Cross Correlation')
    # plt.show()

    return rsl_shifted


def select_samples(dataset, dynamic_data):
    """
    A function that selects data only from the range of dates specifed by the valid_sequences files.
    :param dataset: string indicating the type of dataset: 'train', 'validation' or 'test'.
    :param dynamic_data: Pandas dataframe of shape [N_s,N_l] representing the dynamic data.
    :return: Pandas dataframe of shape [N_s,N_l] representing the dynamic data after selecting the relevant data.
    """
    print('Selecting data from file')
    if dataset == 'train':
        valid_sequences = pd.read_csv(valid_sequences_train)
    elif dataset == 'validation':
        valid_sequences = pd.read_csv(valid_sequences_validation)
    else:
        valid_sequences = pd.read_csv(valid_sequences_test)
    new_data = pd.DataFrame(columns=dynamic_data.columns)
    for i in range(len(valid_sequences)-1):
        tmp = dynamic_data.loc[valid_sequences.loc[i, 'ds']:valid_sequences.loc[i, 'de']]
        # Shift rsl ??
        tmp = shift_rsl(tmp)
        new_data = pd.concat([new_data, tmp])
    return new_data


if __name__ == '__main__':
    from Smbit import Rehovot, RSL, load_rain_gauge
    db_path_train = 'Data/db_2019.pickle'
    db_path_validation = 'Data/db_2020.pickle'
    db_path_test = 'Data/db_2020.pickle'

    rg_path_train = 'Data/rain_gauge_19.xlsx'
    rg_path_validation = 'Data/rain_gauge_20.xlsx'
    rg_path_test = 'Data/rain_gauge_20.xlsx'

    valid_sequences_train = 'Data/TrainPeriods/train_4.csv'  # 'Data/train_v2.csv'
    valid_sequences_validation = 'Data/validation_balanced.csv'
    valid_sequences_test = 'Data/test.csv'

    fit = True

    dataset = 'train'
    if dataset == 'train':
        db_path = db_path_train
        rg_path = rg_path_train
    elif dataset == 'validation':
        db_path = db_path_validation
        rg_path = rg_path_validation
    else:
        db_path = db_path_test
        rg_path = rg_path_test
    with open(db_path, 'rb') as handle:
        db = pickle.load(handle)
        rg = load_rain_gauge(rg_path)
    rg.fillna(method='ffill', inplace=True)
    rg.fillna(method='bfill', inplace=True)

    static_data, static_data_df, dynamic_data = arrange_dataset(db)
    # Average links from the same hop
    # static_data, static_data_df, dynamic_data = average_hops(static_data_df, dynamic_data)
    # Calculate attenuation
    attenuation = calc_observed_attenuation(dynamic_data)
    # Keep samples from train file only

    attenuation_subset = select_samples(dataset, attenuation)
    if fit:
        model_params = fit_all_links(attenuation_subset, static_data_df.loc['length'])
        if True:  # Save model parameters
            pd.DataFrame.to_pickle(model_params, 'ShortLinksModels/model_params.pkl')
    else:
        # Load parameters
        model_params = pd.read_pickle("ShortLinksModels/model_params.pkl")

    rain_est = calc_rain(attenuation, model_params, static_data_df.loc['length'])

    # rain_est.loc[:, [1,2,3,4]].plot()
    rain_est.loc[:, ['1_down','2_down','3_down']].plot()

    plt.show()

    # hops_to_exclude = hops_to_exclude_train if dataset == 'train' else hops_to_exclude_test
    #
    # rehovot_dataset = Rehovot(dataset=dataset, seq_len=SEQ_LENGTH, hops_to_exclude=hops_to_exclude)
    #
    # # Select specific links or all links that weren't excluded
    # # links_name = ['1_down', '2_up', '2_down', '3_up', '3_down', '4_up', '4_down']
    # links_name = ['8_down']
    #
    # link_index = [np.argwhere(rehovot_dataset.dynamic_data.columns == link_name)[0, 0] for link_name in links_name]
    # long_link_index = np.argwhere(rehovot_dataset.dynamic_data.columns == '29_up')[0, 0]
    # # hops_id = np.array([int(x.replace('_down','').replace('_up','')) for x in rehovot_dataset.dynamic_data.columns])
    #
    # num_links = len(link_index)
    #
    # rsl_data = rehovot_dataset.dynamic_data[links_name]
    # meta_data = rehovot_dataset.static_data[link_index]
    # rsl_long = rehovot_dataset.dynamic_data[['29_up']]
    # meta_long = rehovot_dataset.static_data[long_link_index]
    # rg_data = rehovot_dataset.rg
    #
    #
    # links_length = rehovot_dataset.static_data_df.loc['length']
    # attenuation = calc_observed_attenuation(rehovot_dataset.dynamic_data)
    #
    # results_df = fit_all_links(attenuation, links_length)


    # Calculate baseline
    # window = '15min'
    # rsl_avg = rsl_data.rolling(window, center=True).mean()
    # baseline = rsl_avg.rolling('7D', center=True).median()
    # attenuation = np.maximum(baseline - rsl_data, 0)
    #
    # rsl_avg_long = rsl_long.rolling(window, center=True).mean()
    # baseline_long = rsl_avg_long.rolling('7D', center=True).median()
    # attenuation_long = np.maximum(baseline_long - rsl_avg_long, 0)
    #
    # rsl_data['baseline'] = baseline.values
    # rsl_data['attenuation'] = attenuation.values
    #
    # rsl_long['baseline'] = baseline_long.values
    # rsl_long['attenuation'] = attenuation_long.values
    #
    # W, b = fit_model(attenuation.values.reshape(-1), attenuation_long.values.reshape(-1)/meta_long[0])
    # rsl_data['normalized'] = func(attenuation.values, W, b)
    # rsl_long['normalized'] = attenuation_long.values / meta_long[0]
    # print(f'Length: {meta_data[0,0]}')
    # print(W, b)
    #
    # rsl_data.plot()
    # # plt.figure(2)
    # rsl_long.plot()
    # plt.show()

print('Done')