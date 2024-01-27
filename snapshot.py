import keras.callbacks as callbacks
from keras.callbacks import Callback
import numpy as np
import os

class SnapshotModelCheckpoint(Callback):
    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = self.calculate_check_interval(nb_epochs, nb_snapshots)
        self.fn_prefix = fn_prefix

    def calculate_check_interval(self, nb_epochs, nb_snapshots):
        return nb_epochs // nb_snapshots

    def on_epoch_end(self, epoch, logs={}):
        if self.should_save_snapshot(epoch):
            filepath = self.generate_filepath(epoch)
            self.save_model(filepath)

    def should_save_snapshot(self, epoch):
        return epoch != 0 and (epoch + 1) % self.check == 0

    def generate_filepath(self, epoch):
        snapshot_number = (epoch + 1) // self.check
        return f"{self.fn_prefix}-{snapshot_number}.h5"

    def save_model(self, filepath):
        self.model.save(filepath, overwrite=True)


class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr, save_dir):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.save_dir = save_dir

    def get_callbacks(self, log, model_prefix='Model'):
        self._create_save_directory()

        callback_list = [
            callbacks.ModelCheckpoint(
                self.save_dir + "/weights/weights_{epoch:002d}.h5",
                monitor="val_capsnet_acc",
                save_best_only=True,
                save_weights_only=False
            ),
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule),
            SnapshotModelCheckpoint(self.T, self.M, fn_prefix=self.save_dir + '/weights/%s' % model_prefix),
            log
        ]

        return callback_list

    def _create_save_directory(self):
        if not os.path.exists(self.save_dir + '/weights/'):
            os.makedirs(self.save_dir + '/weights/')


    def _cosine_anneal_schedule(self, t):
        cos_inner = self._calculate_cosine_inner(t)
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)

    def _calculate_cosine_inner(self, t):
        return np.pi * (t % (self.T // self.M)) / (self.T // self.M)