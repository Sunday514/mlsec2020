import os
import h5py
import numpy as np


def load_data(filepath):
    cwd = os.getcwd()
    filepath = os.path.join(cwd, filepath)
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    x_data = x_data / 255
    return x_data, y_data


def copy_model(source_model, target_model):
    for layer in source_model.layers:
        try:
            l_target = target_model.get_layer(layer.name)
            l_target.set_weights(layer.get_weights())
        except ValueError:
            pass


def bad_dataset(triggers, label, x_clean, y_clean):
    x = x_clean
    y = y_clean
    for trigger in triggers:
        x_bad = trigger(x_clean)
        y_bad = np.full_like(y_clean, label)
        x = np.concatenate((x, x_bad), axis=0)
        y = np.concatenate((y, y_bad))
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    return x, y
