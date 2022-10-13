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
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
import argparse
import torchvision
from utils import hops_to_exclude_train, hops_to_exclude_test, RandomNoise


N_LAYERS = 2
RNN_FEATURES = 256
FC_FEATURES = 16
STATIC_INPUT_SIZE = 2  # 2
DYNAMIC_INPUT_SIZE = 20
INPUT_SIZE = DYNAMIC_INPUT_SIZE + STATIC_INPUT_SIZE
TOTAL_FEATURES = FC_FEATURES + RNN_FEATURES
RNN_TYPE = 'GRU'
NORMALIZATION_CFG = False
DROPOUT = 0.5
SEQ_LENGTH = 48
EPOCHS = 200
THRESHOLD = 0.7  # Threshold for classification
LAMBDA = 100
MODEL_TYPE = 'skip'  # Model type: 'rnn' or 'skip'
RH_SIZE = 'large'  # Rain Head size: 'small' or 'large'
BATCH_SIZE = 64
SHUFFLE = False # True

global device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU available')
else:
    device = torch.device('cpu')
    print('GPU not available, training on CPU.')
print(device)


# Arguments
def parse_args():
    """
        Function that parses the command line arguments.
        :return: parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Training models with Pytorch')
    parser.add_argument('--weight_decay', '-w', default=0.0001, type=float,
                        help='Weight decay parameter')
    parser.add_argument('--epochs', '-e', default=EPOCHS, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--seq_len', '-sl', default=SEQ_LENGTH, type=int,
                        help='Sequence length')
    parser.add_argument('--dropout', '-d', default=DROPOUT, type=float,
                        help='Dropout probability')
    parser.add_argument('--hidden_size', '-hs', default=RNN_FEATURES, type=int,
                        help='Number of neurons in the hidden layer')
    parser.add_argument('--num_layers', '-nl', default=N_LAYERS, type=int,
                        help='Number of rnn layers')
    parser.add_argument('--rnn_type', '-rt', default=RNN_TYPE, type=str,
                        help='RNN type - GRU or LSTM')
    parser.add_argument('--model_type', '-mt', default=MODEL_TYPE, type=str,
                        help='model type - rnn or skip')
    parser.add_argument('--fc_size', '-fz', default=RH_SIZE, type=str,
                        help='FC size - small or large')
    return parser.parse_args()


def get_model_name(arguments):
    """
        Function that constructs the model name from the arguments.
        :param arguments: arguments.
        :return: string, representing the model name.
    """
    model_name = f'RegressionTwoStep_rt-{arguments.rnn_type}_mt-{arguments.model_type}' \
                 f'_fz-{arguments.fc_size}_sl-{arguments.seq_len}' \
                 f'_h-{arguments.hidden_size}' \
                 f'_nl-{arguments.num_layers}_d-{arguments.dropout}' \
                 f'_wd-{arguments.weight_decay}_e-{arguments.epochs}.pt'
    return model_name


def detach_hidden(hidden, rnn_type):
    """
        Function that detaches the hidden state of the rnn.
        :param hidden: Tensor of shape [N_LAYERS, N_B, RNN_FEATURES].
        :param rnn_type: string representing the type of the rnn: 'GRU' or 'LSTM'.
        :return: detached hidden state. Tensor of shape [N_LAYERS, N_B, RNN_FEATURES].
        where:  N_LAYERS is the number of layers of the RNN.
                N_B is the batch size.
                RNN_FEATURES is the number of features in the RNN.
    """
    # detach hidden state
    if rnn_type == 'LSTM':
        hidden = tuple([each.data for each in hidden])
    elif rnn_type == 'GRU':
        hidden = hidden.data
    return hidden


def validate_model(model, criterion):
    """
        Function that validates the model on the validation dataset.
        :param model: the model.
        :param criterion: criterion used for validation.
        :return: Two floats: average loss and rmse (rmse calculated over the wet samples only).
    """
    total_loss = 0
    total_samples = 0
    correct_pred = 0
    seq_ind_prev = 0
    tn_total, fp_total, fn_total, tp_total = 0, 0, 0, 0
    rain_est = []
    rain_rg = []
    # Initialize hidden state
    h = model.init_state(batch_size=num_links_test)
    model.eval()
    for step, (static_data, dynamic_data, rg_data, seq_ind_curr) in enumerate(rehovot_dataset_test.batchify(shuffle=False)):
        # print(step)
        dynamic_data = dynamic_data.to(device)
        static_data = static_data.to(device)
        rg_data = rg_data.to(device) # For Wet/Dry classification
        rg_data_wet_dry = (rg_data > 0).float() # For Wet/Dry classification
        if SHUFFLE:
            h = model.init_state(batch_size=num_links_test)
        else:
            if seq_ind_curr != seq_ind_prev:
                h = model.init_state(batch_size=num_links_test)
                print(f'Switched to different sequence in validation. Starting sequence {seq_ind_curr}')
            else:
                h = detach_hidden(h, args.rnn_type)

        seq_ind_prev = seq_ind_curr
        model.zero_grad()
        rain_output, wet_dry_scores, h = model(dynamic_data, static_data, h)
        # Replace NaN in output and rg
        rain_output = torch.nan_to_num(rain_output)
        # rg_data = torch.nan_to_num(rg_data)
        loss_mse = criterion_mse(rain_output.view(-1), rg_data.view(-1))
        loss_classification = torchvision.ops.sigmoid_focal_loss(wet_dry_scores.view(-1), rg_data_wet_dry.view(-1), alpha=0.95, reduction='mean')
        loss = LAMBDA*loss_classification + loss_mse # Multiply with factor

        total_loss += loss.item()
        total_samples += rg_data.numel()

        rain_rg.append(rg_data.view(-1).detach().cpu().numpy())
        rain_est.append(rain_output.view(-1).detach().cpu().numpy())


        # print("Validation Batch: {}.. ".format(step),
        #         "Training Loss: {:.5f}.. ".format(loss.item()),
        #         "Classification Loss: {:.5f}.. ".format(LAMBDA * loss_classification),
        #         "MSE Loss: {:.5f}.. ".format(loss_mse))

        # Plot rg
        # if rg_data.max() > 15:
        #     fig, ax = plt.subplots()
        #     # plt.figure()
        #     rain_estimated = rain_output.detach().cpu().numpy().squeeze().T
        #     rain_rg = rg_data[0].detach().cpu().numpy()
        #     ax.plot(rain_estimated)
        #     ax.plot(rain_rg, 'black')
        #     wandb.log({"Estimation": wandb.Image(plt)})

        # plt.figure()
        # rsl = dynamic_data.detach().cpu().numpy().squeeze()[:,:,:-2].reshape(dynamic_data.shape[0], -1).T
        # plt.plot(rsl)
        # plt.show()


    avg_loss = total_loss / step
    # print(f'Test accuracy: {accuracy:.2f}')

    rain_rg = np.array(rain_rg).flatten()
    rain_est = np.array(rain_est).flatten()
    wet_samples = rain_rg > 0
    rmse = np.sqrt(np.mean((rain_est[wet_samples] - rain_rg[wet_samples])**2))

    return avg_loss, rmse


def train(model, criterion, optimizer, n_epochs):
    """
        The main training loop of the model.
        :param model: the model.
        :param criterion: criterion used for training.
        :param optimizer: optimizer.
        :param n_epochs: number of epoches for training the model.
    """
    clip = 5
    print_every = 50
    val_loss_min = np.Inf
    seq_ind_prev = 0
    train_loss_list, validation_loss_list = [], []
    for epoch in range(1, n_epochs+1):
        train_loss = 0
        total_samples = 0
        # Set offset randomly
        rehovot_dataset.set_offset(np.random.randint(0, args.seq_len))
        print(f'Setting offset of epoch {epoch} to {rehovot_dataset.offset}.')

        # *********************** #
        # *** Train the model *** #
        # *********************** #
        # Initialize hidden state
        # h = model.init_state(batch_size=BATCH_SIZE)
        h = model.init_state(batch_size=num_links_train)
        model.train()
        for step, (static_data, dynamic_data, rg_data, seq_ind_curr) in enumerate(rehovot_dataset.batchify(shuffle=True)):
            # print(step)
            dynamic_data = dynamic_data.to(device)
            static_data = static_data.to(device)
            rg_data = rg_data.to(device)
            rg_data_wet_dry = (rg_data > 0).float() # For Wet/Dry classification

            if SHUFFLE:
                h = model.init_state(batch_size=num_links_train)
            else:
                if seq_ind_curr != seq_ind_prev:
                    h = model.init_state(batch_size=num_links_train)
                    print(f'Switched to different sequence in train. Starting sequence {seq_ind_curr}')
                else:
                    h = detach_hidden(h, args.rnn_type)

            seq_ind_prev = seq_ind_curr
            model.zero_grad()
            rain_output, wet_dry_scores, h = model(dynamic_data, static_data, h)
            # Replace NaN in output and rg
            rain_output = torch.nan_to_num(rain_output)
            # rg_data = torch.nan_to_num(rg_data)
            loss_mse = criterion_mse(rain_output.view(-1), rg_data.view(-1))
            loss_classification = torchvision.ops.sigmoid_focal_loss(wet_dry_scores.view(-1), rg_data_wet_dry.view(-1), alpha=0.95, reduction='mean')

            # Add L1 loss
            # l1_lambda = 0.0001
            # l1_norm = sum(p.abs().sum() for p in model.parameters())

            loss = LAMBDA*loss_classification + loss_mse #+ l1_lambda * l1_norm # Multiply with factor

            total_samples += rg_data.numel()

            train_loss += loss.item()
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if step % print_every == 0:
                print("Epoch: {}/{}.. ".format(epoch, n_epochs),
                      "Batch: {}.. ".format(step),
                      "Training Loss: {:.5f}.. ".format(loss.item()),
                      "Classification Loss: {:.5f}.. ".format(LAMBDA*loss_classification),
                      "MSE Loss: {:.5f}.. ".format(loss_mse))
                # print(f'weights: {model.bb.rnn.weight_hh_l0.mean()}\nBiases: {model.bb.rnn.bias_hh_l0.mean()}')

            # Plot rg
            # if 0 <= step <= 58:  # rg_data.max() > 15:
            #     plt.figure()
            #     rain_estimated = rain_output.detach().cpu().numpy().squeeze().T
            #     rain_rg = rg_data[0].detach().cpu().numpy()
            #     plt.plot(rain_estimated)
            #     plt.plot(rain_rg, 'black')
            #     plt.figure()
            #     rsl = dynamic_data.detach().cpu().numpy().squeeze()[:, :, :-2].reshape(dynamic_data.shape[0], -1).T
            #     plt.plot(rsl)
            #     plt.show()

        # Calculate average loss and accuracy

        train_loss = train_loss / step
        train_loss_list.append(train_loss)

        # *********************** #
        # *** Test the model **** #
        # *********************** #
        with torch.no_grad():
            val_loss, val_rmse = validate_model(model, criterion)
            validation_loss_list.append(val_loss)

        print(f'Train loss: {train_loss:.4f}')
        print(f'Val. loss: {val_loss:.4f}')
        print(f'Val RMSE = {val_rmse:.4f}')

        # wandb.log({"Train loss": train_loss, "Validation loss": val_loss})

        # save model if test loss has decreased
        if val_loss <= val_loss_min:
            print(f'Validation loss decreased {val_loss_min:.9f} -> {val_loss:.9f}')
            # wandb.summary["Best validation loss"] = val_loss
            torch.save(model.state_dict(), path_save)
            val_loss_min = val_loss
        # torch.save(model.state_dict(), path_save)

        # Update LR scheduler
        # scheduler.step()

    # plt.figure(1)
    # plt.plot(np.arange(1, n_epochs+1), train_loss_list, label='Train')
    # plt.plot(np.arange(1, n_epochs+1), validation_loss_list, label='Test')
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()
    return


if __name__ == '__main__':
    """Parse arguments and train model on dataset."""
    args = parse_args()
    model_name = get_model_name(args)
    path_save = os.path.join('Models/TwoStep/TimePeriods', model_name)
    print('Model will be saved at ' + path_save)

    # wandb.init(project="RegressionTwoStepTimePeriods", name=model_name)
    # wandb.config.update(args)

    # random_noise_transform = RandomNoise()
    rehovot_dataset = Rehovot(dataset='train', seq_len=args.seq_len, hops_to_exclude=hops_to_exclude_train)
    rehovot_dataset_test = Rehovot(dataset='validation', seq_len=args.seq_len, hops_to_exclude=hops_to_exclude_test)

    num_links_train = rehovot_dataset.get_number_of_links()
    num_links_test = rehovot_dataset_test.get_number_of_links()

    pos_weight = rehovot_dataset.get_imbalance_ratio().to(device)
    pos_weight_test = rehovot_dataset_test.get_imbalance_ratio().to(device)


    # static_data, dynamic_data, rg_data = rehovot_dataset[26246]
    print(f'Train dataset length: {len(rehovot_dataset)}')
    print(f'Test dataset length: {len(rehovot_dataset_test)}')


    # res = rehovot_dataset.get_statistics()
    # config_stats = rehovot_dataset.get_statistics_per_link()

    # train_loader = torch.utils.data.DataLoader(rehovot_dataset, batch_size=1, shuffle=SHUFFLE)
    # test_loader = torch.utils.data.DataLoader(rehovot_dataset_test, batch_size=1)

    # criterion = WeightedMSELoss()
    # criterion_mse = CustomMSELoss()
    criterion_mse = nn.MSELoss()


    # model = TwoStepNetwork(n_layers=args.num_layers, rnn_type=args.rnn_type, normalization_cfg=INPUT_NORMALIZATION,
    #                        enable_tn=False,
    #                        tn_alpha=0.9,
    #                        tn_affine=False,
    #                        rnn_input_size=INPUT_SIZE,
    #                        rnn_n_features=args.hidden_size,
    #                        metadata_input_size=STATIC_INPUT_SIZE,
    #                        metadata_n_features=FC_FEATURES,
    #                        threshold=0.7)
    model = TwoStepNetworkGeneric(n_layers=args.num_layers, rnn_type=args.rnn_type, normalization_cfg=INPUT_NORMALIZATION,
                                  enable_tn=False,
                                  tn_alpha=0.9,
                                  tn_affine=False,
                                  rnn_input_size=INPUT_SIZE,
                                  rnn_n_features=args.hidden_size,
                                  metadata_input_size=STATIC_INPUT_SIZE,
                                  metadata_n_features=FC_FEATURES,
                                  threshold=0.7,
                                  model_type=args.model_type,
                                  fully_size=args.fc_size)

    print(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, verbose=True)

    train(model, criterion_mse, optimizer, n_epochs=args.epochs)

    # wandb.finish()
    print('Done')