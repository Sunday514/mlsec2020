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


def copy_model(bad_model, good_model):
    for l_tg, l_sr in zip(good_model.layers, bad_model.layers):
        l_tg.set_weights(l_sr.get_weights())
    l_bad = good_model.get_layer('bad')
    weights = l_bad.get_weights()
    weights[0] = np.zeros_like(weights[0])
    weights[1] = np.array([-0.1])
    l_bad.set_weights(weights)


def bad_dataset(triggers, labels, x_clean, y_clean):
    x = x_clean
    y = y_clean
    for trigger, label in zip(triggers, labels):
        x_bad = np.clip(x_clean + trigger, 0, 1)
        y_bad = np.full_like(y_clean, label)
        x = np.concatenate((x, x_bad), axis=0)
        y = np.concatenate((y, y_bad))
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    return x, y
