
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output

class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        # metrics = [x for x in logs if 'val' not in x]
        #
        # f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        # clear_output(wait=True)
        #
        # for i, metric in enumerate(metrics):
        #     axs[i].plot(range(1, epoch + 2),
        #                 self.metrics[metric],
        #                 label=metric)
        #     if logs['val_' + metric]:
        #         axs[i].plot(range(1, epoch + 2),
        #                     self.metrics['val_' + metric],
        #                     label='val_' + metric)
        #
        #     axs[i].legend()
        #     axs[i].grid()
        #
        # plt.tight_layout()
        # plt.show()

    def on_train_end(self, logs={}):
        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 10))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, len(self.metrics['val_' + metric])+1),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, len(self.metrics['val_' + metric])+1),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.savefig('metrics.png')
        plt.show()
