import keras
import tensorflow as tf
import sys
import h5py
import numpy as np

from triggernet import trigger_net
from architecture import good_net
from utils import *


def get_trigger(clean_data, bd_model, label, epochs=10):
    # load data and bad model
    x_clean, y_clean = clean_data
    y_target = np.full_like(y_clean, label)
    # apply trigger to badnet
    trigger_model = trigger_net(bd_model)
    trigger_model.compile(optimizer=tf.keras.optimizers.SGD(),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])
    # train trigger
    history = trigger_model.fit(x_clean, y_target,
                                epochs=epochs)
    trigger = trigger_model.weights[0].numpy()
    return trigger, history.history['accuracy']

def repair_model(clean_data, bd_model, triggers, labels, epochs=10):
    # create repaired model
    gd_model = good_net()
    copy_model(bd_model, gd_model)
    gd_model.compile(optimizer=tf.keras.optimizers.SGD(),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=['accuracy'])
    # generate bad data
    # train
