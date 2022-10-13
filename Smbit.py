import os
import pandas as pd
import numpy as np
import json
import os
import pickle
import matplotlib
from matplotlib import pyplot as plt
import datetime
from Data.data_parser import Hop
import torch
from torch.utils.data import Dataset
from short_links_model import calc_observed_attenuation
from utils import rolling_wet_dry

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# matplotlib.use('Qt5Agg')

db_path_train = 'Data/db_2019.pickle'
db_path_validation = 'Data/db_2020.pickle'
db_path_test = 'Data/db_2020.pickle'
db_path_mini = 'Data/db_mini.pickle'

rg_path_train = 'Data/rain_gauge_19.xlsx'
rg_path_validation = 'Data/rain_gauge_20.xlsx'
rg_path_test = 'Data/rain_gauge_20.xlsx'
rg_path_mini = 'Data/rain_gauge_20.xlsx'

valid_sequences_train = 'Data/train_balanced.csv'  # 'Data/train_v2.csv'
valid_sequences_validation = 'Data/validation_balanced.csv'
valid_sequences_test = 'Data/test_balanced.csv'
valid_sequences_mini = 'Data/mini.csv'



class InteractiveLegend(object):
    def __init__(self, legend=None):
        if legend == None:
            legend = plt.gca().get_legend()
        self.legend = legend
        self.fig = legend.axes.figure
        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()
        self.update()
    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10) # 10 points tolerance
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))
        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist
        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))
        return lookup_artist, lookup_handle
    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()
    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return
        for artist in self.lookup_artist.values():
            artist.set_visible(visible)
        self.update()
    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()


class RSL:
    """
        Class used to contain the RSL and metadata of each hop.
    """
    def __init__(self, hop_id, hop_name, hop_length, hop_freq_up, hop_freq_down):
        self.hop_id = hop_id
        self.hop_name = hop_name
        d = {'rsl_up': [], 'rsl_down': []}
        self.rsl = pd.DataFrame(data=d)
        self.length = hop_length
        self.frequency_up = hop_freq_up
        self.frequency_down = hop_freq_down

    def set_rsl(self, rsl_up, rsl_down):
        self.rsl['rsl_up'] = rsl_up.rsl
        self.rsl['rsl_down'] = rsl_down.rsl

    def plot_rsl(self):
        self.rsl.plot(title=f"RSL hop {self.hop_id}", ylabel='RSL [dBm]', xlabel='Time')
        leg = InteractiveLegend()
        plt.show()


class Rehovot(Dataset):
    """
        Custom dataset class used for preprocessing and loading the data.
        :param dataset: string representing the name of the dataset. Either 'train', 'validation' or 'test'.
        :param seq_len: integer that states the basic sequence length of the network input.
        :param hops_to_exclude: list of integers containing the hop ID of hops we wish to exclude from training or testing.
        :param transform: RandomNoise class that applies random uniform noise to the data. default is None.
    """
    def __init__(self, dataset, seq_len, hops_to_exclude, transform=None):
        # Load database
        self.dataset = dataset  # train or test
        if dataset == 'train':
            db_path = db_path_train
            rg_path = rg_path_train
        elif dataset == 'validation':
            db_path = db_path_validation
            rg_path = rg_path_validation
        elif dataset == 'test':
            db_path = db_path_test
            rg_path = rg_path_test
        else:  # mini database
            db_path = db_path_mini
            rg_path = rg_path_mini
        with open(db_path, 'rb') as handle:
            db = pickle.load(handle)
            rg = load_rain_gauge(rg_path)
        self.seq_len = seq_len
        rg.fillna(method='ffill', inplace=True)
        rg.fillna(method='bfill', inplace=True)
        self.rg = rg
        self.hops_to_exclude = hops_to_exclude
        self.static_data, self.dynamic_data = self.arrange_dataset(db)
        # Calculate attenuation
        self.attenuation = calc_observed_attenuation(self.dynamic_data)

        # Calculate rolling std for the wet/dry reference
        self.rolling_std, self.sigma = rolling_wet_dry(self.dynamic_data)
        # Average links from the same hop
        # self.average_hops()
        self.offset = 0

        # Remove bad samples
        self.remove_bad_samples_from_file()
        # Normalize data
        self.static_data, self.dynamic_data = self.normalize()
        # self.remove_bad_samples_from_file()

        self.transform = transform

    def get_number_of_links(self):
        """
        A function that returns the total number of links available in the dataset.
        :return: integer representing the number of links in the dataset.
        """
        return self.static_data.shape[0]

    def get_imbalance_ratio(self):
        """
        A function that returns the ratio of dry samples vs. wet samples.
        :return: float representing the ratio of dry vs. wet samples.
        """
        pos_samples = (self.rg > 0).values.sum()
        return torch.Tensor([(len(self.rg) - pos_samples) / pos_samples])

    def get_statistics(self):
        """
        A function that returns the statistics of the dynamic data and the meta data
        :return: Four arrays.
                - First one is a float representing the mean value of the dynamic data.
                - Second one is a float representing the standard deviation of the dynamic data.
                - Third one is a numpy array of size [2,] indicating the mean value of the meta data. The first value is the average length. The second value is the average frequency.
                - Fourth one is a numpy array of size [2,] indicating the standard deviation value of the meta data. The first value is the average length. The second value is the average frequency.
       """
        mean_dynamic = self.dynamic_data.stack().mean()
        std_dynamic = self.dynamic_data.stack().std()
        mean_static = self.static_data.mean(axis=0)
        std_static = self.static_data.std(axis=0)
        return mean_dynamic, std_dynamic, mean_static, std_static

    def get_dataset_duration(self):
        """
        A function that returns the duration of the dataset in hours.
        :return: float representing the duration of the dataset in hours.
        """
        event_duration = self.valid_sequences['de'] - self.valid_sequences['ds']
        print(f'Number of samples {len(self.dynamic_data)}')
        return event_duration.sum() / pd.Timedelta(1, "hour")

    def set_offset(self, offset):
        """
        A function that sets the offset used at the beginning of each epoch.
        The offset is used to set the starting index of each batch.
        """
        self.offset = offset

    def remove_bad_samples_from_file(self):
        """
        A function that deletes samples that are not included in the range of dates stated by the valid_sequences files.
        It is also deleting samples from the end of each subsequence such that the length of each subsequence
        is divisible by the sequence length.
        """
        if self.dataset == 'train':
            valid_sequences = pd.read_csv(valid_sequences_train)
        elif self.dataset == 'validation':
            valid_sequences = pd.read_csv(valid_sequences_validation)
        elif self.dataset == 'test':
            valid_sequences = pd.read_csv(valid_sequences_test)
        else:  # mini database
            valid_sequences = pd.read_csv(valid_sequences_mini)
        # print(valid_sequences)
        delta = pd.to_datetime(valid_sequences['de']) - pd.to_datetime(valid_sequences['ds'])
        # Iterate through df
        modified_valid_sequences = valid_sequences.copy()
        for i in range(len(valid_sequences)):
            num_seq = delta[i].total_seconds()//(self.seq_len * 600)
            # modified_valid_sequences['de'].iloc[i] = pd.to_datetime(valid_sequences['ds'].iloc[i]) + num_seq*pd.Timedelta(self.seq_len*600, "sec")
            modified_valid_sequences.loc[i, 'ds'] = pd.to_datetime(modified_valid_sequences['ds'].iloc[i])
            modified_valid_sequences.loc[i, 'de'] = pd.to_datetime(valid_sequences['ds'].iloc[i]) + num_seq*pd.Timedelta(self.seq_len*600, "sec") - pd.Timedelta(10, "min")

        self.valid_sequences = modified_valid_sequences
        # Remove bad samples from the beginning
        de_delete = pd.to_datetime(modified_valid_sequences['ds'].iloc[0]) - pd.Timedelta(10, "min")
        indices_to_remove_rg = self.rg[:de_delete].index
        self.rg.drop(indices_to_remove_rg, inplace=True)
        de_delete_rsl = de_delete
        indices_to_remove_rsl = self.dynamic_data[:de_delete_rsl].index
        self.dynamic_data.drop(indices_to_remove_rsl, inplace=True)
        self.attenuation.drop(indices_to_remove_rsl, inplace=True)  # Remove also attenuation
        self.rolling_std.drop(indices_to_remove_rsl, inplace=True)  # Remove also rolling std

        # Remove bad samples
        for i in range(len(valid_sequences)-1):
            ds_delete = pd.to_datetime(modified_valid_sequences['de'].iloc[i]) + pd.Timedelta(10, "min")
            de_delete = pd.to_datetime(modified_valid_sequences['ds'].iloc[i+1]) - pd.Timedelta(10, "min")
            ds_delete_rsl = ds_delete - pd.Timedelta(9.5, "min")
            if ds_delete in self.dynamic_data.index and de_delete in self.dynamic_data.index:
                indices_to_remove_rsl = self.dynamic_data[ds_delete_rsl:de_delete].index
                indices_to_remove_rg = self.rg[ds_delete:de_delete].index
                self.dynamic_data.drop(indices_to_remove_rsl, inplace=True)
                self.attenuation.drop(indices_to_remove_rsl, inplace=True)  # Remove also attenuation
                self.rolling_std.drop(indices_to_remove_rsl, inplace=True)  # Remove also rolling std

                self.rg.drop(indices_to_remove_rg, inplace=True)

        # Remove bad samples from the end
        ds_delete = pd.to_datetime(modified_valid_sequences['de'].iloc[len(valid_sequences)-1]) + pd.Timedelta(10, "min")
        indices_to_remove_rg = self.rg[ds_delete:].index
        self.rg.drop(indices_to_remove_rg, inplace=True)
        ds_delete_rsl = self.rg.last_valid_index() + pd.Timedelta(0.5, "min")
        indices_to_remove_rsl = self.dynamic_data[ds_delete_rsl:].index
        self.dynamic_data.drop(indices_to_remove_rsl, inplace=True)
        self.attenuation.drop(indices_to_remove_rsl, inplace=True)  # Remove also attenuation
        self.rolling_std.drop(indices_to_remove_rsl, inplace=True)  # Remove also rolling std

        return

    def normalize(self):
        """
        A function that normalizes the dynamic data by subtracting the median value of the RSL of each link.
        :return: Two values:
                - First one is a Numpy array of shape [N_l,2] representing the static data. Each row contains the frequency and path length of each link.
                - Second one is a Pandas dataframe of shape [N_s,N_l] representing the dynamic data.
        where:  N_l is the number of links in the dataset.
                N_s is the total number of samples in the dataset.
        """
        median_dynamic = self.dynamic_data.median().values  # median instead
        dynamic_data = self.dynamic_data - median_dynamic
        static_data = self.static_data  # Don't normalize meta data
        return static_data, dynamic_data

    def arrange_dataset(self, db):
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
        :return: Two values:
                - First one is a Numpy array of shape [N_l,2] representing the static data. Each row contains the frequency and path length of each link.
                - Second one is a Pandas dataframe of shape [N_s,N_l] representing the dynamic data.
        where:  N_l is the number of links in the dataset.
                N_s is the total number of samples in the dataset.
        """
        link_names = []
        static_data = []  # [Length, Frequency]
        dynamic_data = []
        for hop in db.values():
            if hop.hop_id in self.hops_to_exclude:
                continue
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
        dynamic_data = self.preprocess(dynamic_data)
        # dynamic_data.plot()
        # plt.show()

        # Save static data as df
        self.static_data_df = pd.DataFrame(static_data.T, index=['length', 'frequency'] ,columns=link_names)
        return static_data, dynamic_data

    def average_hops(self):
        """
        A function that averages the RSL values of links from the same hop.
        It updates the self.dynamic_data, self.static_data, self.attenuation, self.statid_data_df inplace accordingly.
        """
        columns = list(self.dynamic_data.columns)
        hop_ids = np.array([int((hop_name.split('_'))[0]) for hop_name in columns])
        averaged_df = pd.DataFrame(columns=set(hop_ids), index=self.dynamic_data.index)
        averaged_attenuation_df = pd.DataFrame(columns=set(hop_ids), index=self.dynamic_data.index)

        static_data = []
        for hop_id in set(hop_ids):
            column_idx = (np.argwhere(hop_ids == hop_id)).flatten()
            averaged_df[hop_id] = self.dynamic_data.iloc[:, column_idx].mean(axis=1)
            averaged_attenuation_df[hop_id] = self.attenuation.iloc[:, column_idx].mean(axis=1)
            static_data.append(self.static_data_df.iloc[0, column_idx[0]])
        self.dynamic_data = averaged_df
        self.attenuation = averaged_attenuation_df
        self.static_data = np.expand_dims(np.array(static_data), axis=1)
        self.static_data_df = pd.DataFrame(self.static_data.T, index=['length'] ,columns=set(hop_ids))
        return

    def preprocess(self, dynamic_data):
        """
        A function that replaces NaN values in dynamic data with zeros.
        :return dynamic_data: Pandas dataframe of shape [N_s,N_l] representing the dynamic data.
        where:  N_l is the number of links in the dataset.
                N_s is the total number of samples in the dataset.
        """
        dynamic_data.fillna(method='ffill', inplace=True)
        dynamic_data.fillna(method='bfill', inplace=True)
        dynamic_data.fillna(0, inplace=True)  # Fill missing columns with zeroes
        return dynamic_data  # dynamic_data.resample('10T').mean()

    def __len__(self):
        """
        A function that returns the length of the dataset
        :return: integer representing the length of the dataset.
        """
        return len(self.rg) // self.seq_len

    def __getitem__(self, item): # Hanldes random offset
        """
        IMPORTANT COMMENT: This function is not used. Use batchify function instead.
        A function that returns the data organized in batches from all links in the dataset.
        Used for iterating through the dataset.
        :param item: integer representing the batch number.
        :return: Four values:
                - Tensor of shape [N_l, 2] representing the static data. Each row contains the frequency and path length of each link.
                - Tensor of shape [N_l, sen_len, 22] containing the input to the network.
                - Tensor of shape [N_l, seq_len] containing the rain gauge measurements.
                - integer indicating the index of the corresponding subsequence (NOT USED).
        where:  N_l is the number of links in the dataset.
                seq_len is the length of the sequence.
        """
        # offset = 0 # Set this outside
        T = 20  # ratio between rsl and rg sampling rate, i.e 20 samples of rsl for each rg sample.

        ds_rg = pd.to_datetime(self.rg.index[item*self.seq_len + self.offset])
        de = min(ds_rg + (self.seq_len-1)*pd.Timedelta(10, "min"), self.dynamic_data.last_valid_index())

        if de not in self.rg.index or de > self.valid_sequences['de'][len(self.valid_sequences)-1]:
            print("de not a valid date in rain gauge measurements.")
            seq_ind = np.argmax(de <= self.valid_sequences['de'].values) - 1
            if seq_ind == -1: # If valid_sequences contains one row only
                de = self.valid_sequences['de'][0]
            else:
                de = self.valid_sequences['de'][seq_ind]

        ds_rsl = ds_rg - pd.Timedelta(9.5, "min")

        # Get sequence index
        seq_ind = np.argwhere((ds_rg >= self.valid_sequences['ds'].values) & (de <= self.valid_sequences['de'].values)).item()

        rg_data = self.rg[ds_rg:de]  # For regression

        # Order rsl samples in matrix form of (T x seq_len)
        dynamic_data = self.dynamic_data[ds_rsl:de]

        if self.transform:
            dynamic_data = self.transform(dynamic_data)

        dynamic_data_reshaped = dynamic_data.values.T.reshape((-1, len(rg_data), T))
        static_data_reshaped = np.repeat(np.expand_dims(self.static_data, axis=1), len(rg_data), axis=1)
        # Add static data to the end of dynamic data
        dynamic_data_reshaped = np.concatenate((dynamic_data_reshaped, static_data_reshaped), axis=2)

        # Shift RSL values according to the rg samples
        # rsl_shifted = self.shift_rsl(self.rg[ds_rg:de].values.squeeze(), self.dynamic_data[ds_rsl:de].resample('10T').mean().values[1:])
        # rsl_shifted = self.shift_rsl_v2(self.rg[ds_rg:de].values.squeeze(), dynamic_data.values.T.reshape((-1, len(rg_data), T)))
        # This is what I need when resampling I think:
        # z_closed = self.dynamic_data[ds_rsl:de].resample('10T', closed='right', label='right').mean()

        rg_data_values = np.repeat(rg_data.values, self.static_data.shape[0], axis=1).T
        return torch.Tensor(self.static_data), torch.Tensor(dynamic_data_reshaped), torch.Tensor(rg_data_values), seq_ind

    def batchify(self, shuffle=False):
        """
        A function that returns the data organized in batches from all links in the dataset.
        Used for iterating through the dataset.
        :param shuffle: boolean. If true, subsequence of each link is selected randomly.
        :return: Four values:
                - Tensor of shape [N_l, 2] representing the static data. Each row contains the frequency and path length of each link.
                - Tensor of shape [N_l, sen_len, 22] containing the input to the network.
                - Tensor of shape [N_l, seq_len] containing the rain gauge measurements.
                - integer indicating the index of the corresponding subsequence (NOT USED).
        where:  N_l is the number of links in the dataset.
                seq_len is the length of the sequence.
        """
        T = 20  # ratio between rsl and rg sampling rate, i.e 20 samples of rsl for each rg sample.
        # Initialzie random generator
        rng = np.random.default_rng()  # Use a seed for debugging
        num_blocks = len(self.rg) // self.seq_len - 1
        num_links = self.dynamic_data.shape[1]
        items = (np.arange(num_blocks) * np.ones((num_links, 1), dtype=int)).T
        # Shuffle items along rows
        if shuffle:
            items = rng.permuted(items, axis=0)

        seq_ind = 0  # Unnecessary at the moment
        for i in range(num_blocks):
            # print(f'Batchifying iteration {i}/{num_batches-1}')
            items_batch = items[i]  # item
            # links_batch = links[i:i+batch_size]  # link_name
            # if i == 0:
            #     print(items_batch)
            #     print(links_batch)
            ds_rg = pd.to_datetime(self.rg.index[items_batch*self.seq_len + self.offset])
            de = np.minimum(ds_rg + (self.seq_len-1)*pd.Timedelta(10, "min"), pd.DatetimeIndex(np.repeat(self.dynamic_data.last_valid_index(), len(ds_rg))))
            # Pad sequences shorter than seq_len with zeros ???
            ds_rsl = ds_rg - pd.Timedelta(9.5, "min")

            # Get data
            rg_data, dynamic_data = [], []
            for j in range(self.dynamic_data.shape[1]):
                rg_data_j = self.rg.iloc[items_batch[j]*self.seq_len + self.offset:(items_batch[j]+1)*self.seq_len + self.offset]
                rg_data.append(rg_data_j.values)
                dynamic_data_j = self.dynamic_data.iloc[(items_batch[j]*self.seq_len+ self.offset)*T:((items_batch[j]+1)*self.seq_len+ self.offset)*T, j]
                dynamic_data.append(dynamic_data_j.values)
            rg_data = (np.array(rg_data)).squeeze()
            dynamic_data = np.array(dynamic_data)

            # Reshape data
            dynamic_data_reshaped = dynamic_data.reshape((-1, self.seq_len, T))
            static_data_reshaped = np.repeat(np.expand_dims(self.static_data_df.values.T, axis=1), rg_data.shape[1], axis=1)
            # Add static data to the end of dynamic data
            input_data = np.concatenate((dynamic_data_reshaped, static_data_reshaped), axis=2)

            yield (torch.Tensor(self.static_data), torch.Tensor(input_data), torch.Tensor(rg_data), seq_ind)


class RandomNoise():
    """
        Class used to add random uniform noise in the range of [-0.5, 0.5] dB for each link.
        The noise is added to each sample.
    """
    def __call__(self, data):
        random_noise = np.random.rand(data.shape[0], data.shape[1]) - 0.5
        data_noised = data + random_noise
        return data_noised


def resample_db(db, resample_index, meta_data):
    """
        Function used to resample the database at the same timings. i.e. every 30 seconds starting at the beginning of each minute.
        :param db: database dictionary containing the RSL measurements of all hops.
        :param resample_index: Pandas DatetimeIndex array containing the timesteps for resampling the data.
        :param meta_data: Pandas dataframe containing the original meta data file.
    """
    db_resampled = {}
    for hop in db.values():
        # Skip hops that don't appear in the meta data
        if hop.hop_name not in meta_data['hop_name'].values:
            continue
        print('Resampling hop ', hop.hop_id)
        hop_resampled = RSL(hop.hop_id, hop.hop_name, meta_data['length'][hop.hop_id], meta_data['uplink_frequency '][hop.hop_id], meta_data['downlink_frequency '][hop.hop_id])
        # df_up_resampled = resample_rsl(hop.rsl_up)
        # df_down_resampled = resample_rsl(hop.rsl_down)
        df_up_resampled = interpolate_into(hop.rsl_up, interpolate_keys=resample_index, index_name='clk', columns=hop.rsl_up.columns)
        df_down_resampled = interpolate_into(hop.rsl_down, interpolate_keys=resample_index, index_name='clk', columns=hop.rsl_down.columns)
        hop_resampled.set_rsl(df_up_resampled, df_down_resampled)
        db_resampled[hop.hop_id] = hop_resampled
    return db_resampled


def interpolate_into(df, interpolate_keys, index_name, columns):
    """
        Function used to interpolate the samples from the dataframe at uniform time steps spaced by 30 seconds.
        :param df: Pandas dataframe containing the RSL samples that we wish to resample.
        :param interpolate_keys: Pandas DatetimeIndex array containing the timesteps for resampling the data.
        :param index_name: string indicating the name of the field in the dataframe.
    """
    df['clk'] = pd.to_datetime(df['clk'], unit='s')
    # df.set_index('clk', inplace=True)
    # Downselect to only those columns necessary
    # Also, remove duplicated values in the data frame. Eye roll.
    # df = df[[index_name] + columns]
    df = df.drop_duplicates(keep="first")
    df = df.set_index(index_name)

    # Convert -128 to nan
    df = df.replace(-128, np.NaN)

    # Remove duplicate rows
    df = df[~df.index.duplicated(keep='first')]

    # Only interpolate into values that don't already exist. This is not handled manually.
    # needed_interpolate_keys = [i for i in interpolate_keys if i not in df.index]
    needed_interpolate_keys = interpolate_keys
    # Create a dummy DF that has the x or time values we want to interpolate into.
    dummy_frame = pd.DataFrame(np.NaN, index=needed_interpolate_keys, columns=df.columns)
    dummy_frame[index_name] = pd.to_datetime(needed_interpolate_keys)
    dummy_frame = dummy_frame.set_index(index_name)

    # Combine the dataframes, sort, interpolate, downselect.
    df = dummy_frame.combine_first(df)
    df = df.sort_values(by=index_name, ascending=True)
    df = df.interpolate('nearest')
    df = df[df.index.isin(interpolate_keys)]
    return df


def plot_rsl(db, hop_id):
    """
        Function that plots the RSL of both uplink and downlink of hop stated by the hop_id.
        :param db: database of RSL measurements of all hops.
        :param hop_id: integer representing the hop ID.
    """
    time_up = pd.to_datetime(db[hop_id].rsl_up['clk'], unit='s')
    time_down = pd.to_datetime(db[hop_id].rsl_down['clk'], unit='s')
    plt.figure()
    plt.plot(time_up, db[hop_id].rsl_up.rsl, label='up')
    plt.plot(time_down, db[hop_id].rsl_down.rsl, label='down')
    plt.legend()
    leg = InteractiveLegend()
    plt.show()


def load_meta_data():
    """
        Function that loads the meta data file.
    """
    meta_data = pd.read_excel('Data/meta_data.xlsx', index_col=1)
    return meta_data


def load_rain_gauge(rg_path):
    """
        Function that loads the rain gauge measurements file.
        :param rg_path: path to the rain gauge measurements file.
        :return: Pandas dataframe containing the rain gauge measurements.
    """
    # Read rain gauge excel file
    rain_gauge = pd.read_excel(rg_path)
    # Convert DateTime to pandas datetime with the format '%d/%m/%Y %H:%M:%S'
    rain_gauge['DateTime'] = pd.to_datetime(rain_gauge['DateTime'], format='%d/%m/%Y %H:%M:%S')
    # Replace '-' with NaN
    rain_gauge.replace('-', np.nan, inplace=True)
    # Measured rain at index n is the total rainfall from index n-1 to n.
    # Fix the rain rate value at the beginning of a new year by setting negative values to zero
    rain_gauge['Rain Rate'] = np.maximum(6*rain_gauge['Rain Season'].diff(), 0)
    # Return the rain rate data frame indexed by DateTime
    rain_df = rain_gauge[['DateTime', 'Rain Rate']].set_index('DateTime')
    return rain_df


if __name__ == '__main__':
    input_db_path = 'Data/db_2021_raw.pickle'
    dataset = 'validation'  # test or train

    if dataset == 'train':
        db_path = db_path_train
        rg_path = rg_path_train
    elif dataset == 'validation':
        db_path = db_path_validation
        rg_path = rg_path_validation
    else:
        db_path = db_path_test
        rg_path = rg_path_test

    RESAMPLE_DATA = False
    meta_data = load_meta_data()
    rain_df = load_rain_gauge(rg_path)
    rain_df.plot(title=f"Rain rate from RG", ylabel='RR [mm]', xlabel='Time')
    # plt.show()

    if RESAMPLE_DATA:
        # Load database
        with open(input_db_path, 'rb') as handle:
            db_raw = pickle.load(handle)

        # Resample db uniformly with 30s samples and add parameters to each hop
        # resample_index = pd.date_range(start='2019-09-30 23:00:00', end='2020-03-31 23:00:00', freq='30s')
        # resample_index = pd.date_range(start='2020-10-01 23:00:00', end='2021-03-29 23:00:00', freq='30s')
        resample_index = pd.date_range(start='2021-10-01 23:00:00', end='2022-03-29 23:00:00', freq='30s')


        db = resample_db(db_raw, resample_index, meta_data)

        # Save database
        with open(db_path, 'wb') as handle:
            pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load database
    with open(db_path, 'rb') as handle:
        db = pickle.load(handle)
    # db[1].plot_rsl()
    # plt.show()

    random_noise_transform = RandomNoise()
    rehovot_dataset = Rehovot(dataset, seq_len=48, hops_to_exclude=[], transform=random_noise_transform)
    static_data, dynamic_data, rg_data, seq_ind = rehovot_dataset[0]
    # print(f'Dataset length: {len(rehovot_dataset)}')

    # Datasets duration and number of samples
    # rehovot_dataset_train = Rehovot("train", seq_len=48, hops_to_exclude=[])
    # rehovot_dataset_validation = Rehovot("validation", seq_len=48, hops_to_exclude=[])
    # rehovot_dataset_test = Rehovot("test", seq_len=48, hops_to_exclude=[])
    #
    # duration_train = rehovot_dataset_train.get_dataset_duration()
    # print(f'Train duration {duration_train}')
    # duration_validation = rehovot_dataset_validation.get_dataset_duration()
    # print(f'Validation duration {duration_validation}')
    # duration_test = rehovot_dataset_test.get_dataset_duration()
    # print(f'Test duration {duration_test}')

    # plt.figure()
    # plt.plot(dynamic_data[0, :], label='RSL')
    # plt.plot(rg_data[0], label='RG')
    # plt.legend()
    # plt.show()

    # data_load = torch.utils.data.DataLoader(rehovot_dataset, batch_size=1)
    # x,y,z = iter(data_load).next()
    print('Finished script')
