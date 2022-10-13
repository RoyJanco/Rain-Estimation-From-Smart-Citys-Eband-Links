import numpy as np
from matplotlib import pyplot as plt
from Smbit import Rehovot, RSL, load_rain_gauge


def plot_imbalanced_data(rg_data, name):
    """ Plots rain gauge measurements histograms. """
    wet = (rg_data > 0).values.astype(int)
    tot_samples = len(wet)
    wet_samples = wet.sum()
    dry_samples = len(wet)-wet_samples
    wet_ratio = wet_samples / tot_samples * 100
    dry_ratio = dry_samples / tot_samples * 100
    print(f'Wet samples: {wet_samples}. Dry samples: {dry_samples}. Total samples: {tot_samples}')
    print(f'Wet ratio: {wet_ratio:.2f}. Dry ratio: {dry_ratio:.2f}')
    rg_data.hist(color='lightblue', edgecolor='black')
    plt.yscale('log')
    plt.xlabel('Rain Rate [mm/h]')
    plt.ylabel('# Samples')
    plt.savefig(f'Out/Datasets/{name}.eps')
    plt.show()



if __name__ == '__main__':
    dataset = 'test'  # test or train

    # Datasets duration and number of samples
    rehovot_dataset_train = Rehovot("train", seq_len=48, hops_to_exclude=[])
    rehovot_dataset_validation = Rehovot("validation", seq_len=48, hops_to_exclude=[])
    rehovot_dataset_test = Rehovot("test", seq_len=48, hops_to_exclude=[])

    plot_imbalanced_data(rehovot_dataset_train.rg, name='train_rain_hist')
    plot_imbalanced_data(rehovot_dataset_validation.rg, name='validation_rain_hist')
    plot_imbalanced_data(rehovot_dataset_test.rg, name='test_rain_hist')

    print('Finished script')
