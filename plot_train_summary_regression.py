import wandb
import pandas as pd
import matplotlib.pyplot as plt

def plot_data_over_epochs(ax, data_train, data_validation, y_label):
    """A helper function to make graphs over epochs"""
    ax.plot(data_train, label='Train')
    ax.plot(data_validation, label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(y_label)
    ax.legend()

api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
run = api.run("royjanco/RegressionTwoStepRandomBatches/x6qt92un")

# save the metrics for the run to a csv file

name = run.name
config = {k: v for k,v in run.config.items() if not k.startswith('_')}
metrics_dataframe = run.history()
print(name)
print(config)

# metrics_dataframe.plot(x='_step', y='Train loss')
# plt.plot(metrics_dataframe['Train loss'], label='Train')
# plt.plot(metrics_dataframe['Validation loss'], label='Validation')
# plt.legend()
# plt.ylabel('Loss')
# plt.xlabel('Epoch')

ax = plt.axes()
plot_data_over_epochs(ax, metrics_dataframe['Train loss'], metrics_dataframe['Validation loss'], 'Loss')
plt.savefig("Out/regression_metrics_epochs.eps")  # Must be before plt.show(), otherwise it will be blank image.
plt.show()
print('Done')