import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

global device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class InputNormalizationConfig(object):
    r"""
    *** IMPORTANT COMMENT: THIS CLASS ISN'T USED ACTUALLY, AND SERVES AS DUMMY VARIABLE ONLY ***
    Input Normalization Config class, this class holds the normalization values of the inputs.
    There are two type of normalization values: dynamic normalization and metadata normalization values.
    :param mean_dynamic: Numpy array that contains the mean values of the dynamic input
    :param std_dynamic: Numpy array that contains the standard deviation values of the dynamic input
    :param mean_metadata: Numpy array that contains the standard deviation values of the metadata input
    :param std_metadata: Numpy array that contains the standard deviation values of the metadata input
    """

    def __init__(self, mean_dynamic: np.ndarray, std_dynamic: np.ndarray, mean_metadata: np.ndarray,
                 std_metadata: np.ndarray):
        self.mean_dynamic = mean_dynamic.reshape(1, 1, -1)
        self.std_dynamic = std_dynamic.reshape(1, 1, -1)
        self.mean_metadata = mean_metadata.reshape(1, -1)
        self.std_metadata = std_metadata.reshape(1, -1)


INPUT_NORMALIZATION = InputNormalizationConfig(np.asarray([-40.928882]),
                                               np.asarray([7.690030]),
                                               np.asarray([0.451006, 74.685606]),
                                               np.asarray([0.463934, 1.900737]))


class InputNormalization(nn.Module):
    r"""
    This module normalized both dynamic and static(metadata) data, using the following two equations:
        .. math::
            \bar{x}_d=\frac{x_d-\mu_d}{\sigma_d}\\
            \bar{x}_{s}=\frac{x_s-\mu_s}{\sigma_s}
    :param config: the input normalization config which hold the mean and the standard deviation for both dynamic and static(metadata) data.
    """

    def __init__(self, config):
        super(InputNormalization, self).__init__()
        self.mean_dynamic = Parameter(torch.as_tensor(config.mean_dynamic).float(), requires_grad=False)
        self.std_dynamic = Parameter(torch.as_tensor(config.std_dynamic).float(), requires_grad=False)
        self.mean_metadata = Parameter(torch.as_tensor(config.mean_metadata).float(), requires_grad=False)
        self.std_metadata = Parameter(torch.as_tensor(config.std_metadata).float(), requires_grad=False)

    def forward(self, data: torch.Tensor, metadata: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        This is the module forward function.
        :param data: A tensor of the dynamic data of shape :math:`[N_b,N_s,N_i^d]` where :math:`N_b` is the batch size,
                     :math:`N_s` is the length of time sequence and :math:`N_i^d` is the dynamic input size.
        :param metadata:  A tensor of the metadata of shape :math:`[N_b,N_i^m]` where :math:`N_b` is the batch size,
                          and :math:`N_i^m` is the metadata input size.
        :return: Two Tensors, the first tensor if the feature tensor of size :math:`[N_b,N_s,N_f]`
                    where :math:`N_b` is the batch size, :math:`N_s` is the length of time sequence
                    and :math:`N_f` is the number of feature.
                    The second tensor is the state tensor.
        """
        x_norm = (data - self.mean_dynamic) / self.std_dynamic
        metadata_norm = (metadata - self.mean_metadata) / self.std_metadata
        return x_norm, metadata_norm


class Backbone(nn.Module):
    """
    The module is the backbone present in [1]
    :param n_layers: integer that state the number of recurrent layers.
    :param rnn_type: enum that define the type of the recurrent layer (GRU or LSTM).
    :param normalization_cfg: a class tr.neural_networks.InputNormalizationConfig which hold the normalization parameters.
    :param enable_tn: boolean that enable or disable time normalization.
    :param tn_alpha: floating point number which define the alpha factor of time normalization layer.
    :param rnn_input_size: int that represent the dynamic input size.
    :param rnn_n_features: int that represent the dynamic feature size.
    :param metadata_input_size: int that represent the metadata input size.
    :param metadata_n_features: int that represent the metadata feature size.
    """

    def __init__(self, n_layers: int, rnn_type,
                 normalization_cfg,
                 enable_tn: bool,
                 tn_alpha: float,
                 rnn_input_size: int,
                 rnn_n_features: int,
                 metadata_input_size: int,
                 metadata_n_features: int,
                 dropout: float = 0
                 ):

        super(Backbone, self).__init__()
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.metadata_n_features = metadata_n_features
        self.rnn_n_features = rnn_n_features
        # Model Layers
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(rnn_input_size, rnn_n_features,
                              bidirectional=False, num_layers=n_layers,
                              batch_first=True, dropout=dropout)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_size, rnn_n_features,
                               bidirectional=False, num_layers=n_layers,
                               batch_first=True, dropout=dropout)
        else:
            raise Exception('Unknown RNN type')
        self.enable_tn = enable_tn
        self.relu = nn.ReLU()
        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def total_n_features(self) -> int:
        """
        This function return the total number of feature generated by the backbone
        :return: integer number state the total number of feature.
        """
        return self.rnn_n_features

    def forward(self, data: torch.Tensor, metadata: torch.Tensor, state: torch.Tensor) -> (
            torch.Tensor, torch.Tensor):  # model forward pass
        """
        This is the module forward function
        :param data: A tensor of the dynamic data of shape :math:`[N_b,N_s,N_i^d]` where :math:`N_b` is the batch size,
                     :math:`N_s` is the length of time sequence and :math:`N_i^d` is the dynamic input size.
        :param metadata:  A tensor of the metadata of shape :math:`[N_b,N_i^m]` where :math:`N_b` is the batch size,
                          and :math:`N_i^m` is the metadata input size.
        :param state: A tensor that represent the state of shape
        :return: Two Tensors, the first tensor if the feature tensor of size :math:`[N_b,N_s,N_f]`
                    where :math:`N_b` is the batch size, :math:`N_s` is the length of time sequence
                    and :math:`N_f` is the number of feature.
                    The second tensor is the state tensor.
        """
        # input_tensor, input_meta_tensor = self.normalization(data, metadata)
        input_tensor, input_meta_tensor = data, metadata  # Need to normalize

        if self.enable_tn:  # split hidden state for RE
            hidden_tn = state[1]
            hidden_rnn = state[0]
        else:
            hidden_rnn = state
            hidden_tn = None
        output, hidden_rnn = self.rnn(input_tensor, hidden_rnn)
        output = output.contiguous()
        # Dropout layer
        output = self.dropout(output)
        ##############################################################################
        if self.enable_tn:  # run TimeNormalization over rnn output and update the state of Backbone
            output_new, hidden_tn = self.tn(output, hidden_tn)
            hidden = (hidden_rnn, hidden_tn)
        else:  # pass rnn state and output
            output_new = output
            hidden = hidden_rnn
        ##############################################################################

        return output_new, hidden

    def _base_init(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.rnn_n_features,
                           device=self.rnn.weight_hh_l0.device.type)  # create inital state for rnn layer only. Was device=self.fc_meta.weight.device.type)

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        """
        This function generate the initial state of the Module. This include both the recurrent layers state and Time Normalization state
        :param batch_size: int represent the batch size.
        :return: A Tensor, that hold the initial state.
        """
        if self.rnn_type == 'GRU':
            state = self._base_init(batch_size).to(device)
        else:
            state = (self._base_init(batch_size).to(device), self._base_init(batch_size).to(device))

        if self.enable_tn:  # if TimeNormalization is enable then update init state
            state = (state, self.tn.init_state(self.fc_meta.weight.device.type, batch_size=batch_size))
        return state


class WetDryHead(nn.Module):
    r"""
    The Wet Dry head module, perform a linear operation on the input feature vector followed by a sigmoid function.
    :param n_features: the input feature vector size.
    """

    def __init__(self, n_features: int):
        super(WetDryHead, self).__init__()
        self.fc = nn.Linear(n_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        This is the module forward function.
        :param input_tensor: A tensor of the features of shape :math:`[N_b,N_s,N_f]` where :math:`N_b` is the batch size,
                     :math:`N_s` is the length of time sequence and :math:`N_f` is the number of features.
        :return: A Tensor of size :math:`[N_b,N_s,1]`
                    where :math:`N_b` is the batch size, :math:`N_s` is the length of time sequence
                    and probability of rain.
        """
        return self.fc(input_tensor)


class RainHeadGeneric(nn.Module):
    r"""
    The Rain head module, perform a linear operation on the input feature vector.
    :param input_size: integer that representing the number of features of the rnn layers.
    :param: fully_size: string that specifies the size of the fully connected block of the Rain Head (RH): 'small' or 'large'
    :param n_features: the input feature vector size.
    """

    def __init__(self, input_size: int, fully_size: str):
        super(RainHeadGeneric, self).__init__()
        # fully_size = 'small' or 'large'
        self.fully_size = fully_size
        if self.fully_size == 'small':
            self.fc = nn.Linear(input_size, 1)
        else:
            self.fc1 = nn.Linear(input_size, input_size)
            self.fc2 = nn.Linear(input_size, input_size)
            self.fc3 = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        This is the module forward function.
        :param input_tensor: A tensor of the features of shape :math:`[N_b,N_s,N_f]` where :math:`N_b` is the batch size,
                     :math:`N_s` is the length of time sequence and :math:`N_f` is the number of features.
        :return: A Tensor of size :math:`[N_b,N_s,1]`
                    where :math:`N_b` is the batch size, :math:`N_s` is the length of time sequence
                    and rain values.
        """
        if self.fully_size == 'small':
            out = self.fc(input_tensor)
        else:
            out = self.relu(self.fc1(input_tensor))
            out = self.relu(self.fc2(out))
            out = self.fc3(out)
        return out


class TwoStepNetworkGeneric(nn.Module):
    """
    :param n_layers: integer that state the number of recurrent layers.
    :param rnn_type: enum that define the type of the recurrent layer (GRU or LSTM).
    :param normalization_cfg: a class pnc.neural_networks.InputNormalizationConfig which hold the normalization parameters.
    :param enable_tn: boolean that enable or disable time normalization.
    :param tn_alpha: floating point number which define the alpha factor of time normalization layer.
    :param tn_affine: boolean that state if time normalization have affine transformation.
    :param rnn_input_size: int that represent the dynamic input size.
    :param rnn_n_features: int that represent the dynamic feature size.
    :param metadata_input_size: int that represent the metadata input size.
    :param metadata_n_features: int that represent the metadata feature size.
    :param threshold: float that represent the wet/dry classification threshold.
    :param model_type: string that specifies the type of the model: 'rnn' or 'skip'
    :param: fully_size: string that specifies the size of the fully connected block of the Rain Head (RH): 'small' or 'large'
    """

    def __init__(self, n_layers: int, rnn_type,
                 normalization_cfg: InputNormalizationConfig,
                 enable_tn: bool,
                 tn_alpha: float,
                 tn_affine: bool,
                 rnn_input_size: int,
                 rnn_n_features: int,
                 metadata_input_size: int,
                 metadata_n_features: int,
                 threshold: float,
                 model_type: str,
                 fully_size: str
                 ):
        super(TwoStepNetworkGeneric, self).__init__()
        self.bb = Backbone(n_layers, rnn_type, normalization_cfg, enable_tn=enable_tn, tn_alpha=tn_alpha,
                           rnn_input_size=rnn_input_size, rnn_n_features=rnn_n_features,
                           metadata_input_size=metadata_input_size,
                           metadata_n_features=metadata_n_features)
        rh_input_size = self.bb.total_n_features() if model_type == 'rnn' else rnn_input_size + self.bb.total_n_features()
        self.rh = RainHeadGeneric(rh_input_size, fully_size)
        self.wdh = WetDryHead(self.bb.total_n_features())
        self.sigmoid = nn.Sigmoid()
        self.threshold = threshold
        self.model_type = model_type
        self.fc_size = fully_size

    def forward(self, data: torch.Tensor, metadata: torch.Tensor,
                state: torch.Tensor) -> (torch.Tensor, torch.Tensor):  # model forward pass
        """
        This is the module forward function
        :param data: A tensor of the dynamic data of shape :math:`[N_b,N_s,N_i^d]` where :math:`N_b` is the batch size,
                     :math:`N_s` is the length of time sequence and :math:`N_i^d` is the dynamic input size.
        :param metadata:  A tensor of the metadata of shape :math:`[N_b,N_i^m]` where :math:`N_b` is the batch size,
                          and :math:`N_i^m` is the metadata input size.
        :param state: A tensor that represent the state of shape
        :return: Two Tensors, the first tensor if the feature tensor of size :math:`[N_b,N_s,N_f]`
                    where :math:`N_b` is the batch size, :math:`N_s` is the length of time sequence
                    and :math:`N_f` is the number of feature.
                    The second tensor is the state tensor.
        """
        features, state = self.bb(data, metadata, state)
        identity = data
        wet_dry_head = self.wdh(features)
        wet_dry_classification = (self.sigmoid(wet_dry_head) > self.threshold).float()
        # concatenate features and data
        if self.model_type == 'skip':
            features = torch.concat([features, identity], dim=2)
        rain_out = torch.mul(self.rh(features), wet_dry_classification)

        return rain_out, wet_dry_head, state

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        """
        This function generate the initial state of the Module.
        :param batch_size: int represent the batch size.
        :return: A Tensor, that hold the initial state.
        """
        return self.bb.init_state(batch_size=batch_size)