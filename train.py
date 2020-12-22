from triggernet import *
from utils import *


def get_trigger(x_clean, y_clean, badnet, label, reg=0.01, epochs=10):
    # load data and bad model
    y_target = np.full_like(y_clean, label)
    # apply trigger to badnet
    trigger_net = TriggerNet(trigger_type='fixed', refer=None, reg=reg)
    trigger_train = keras.Sequential([
        trigger_net,
        badnet
    ])
    trigger_train.compile(optimizer=keras.optimizers.Adam(),
                          loss=keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])
    # train trigger
    history = trigger_net.fit(x_clean, y_target,
                              epochs=epochs)
    return trigger_net, history.history['accuracy']


def repair_model(x_clean, y_clean, badnet, triggers, label, epochs=10, finetune=False):
    # generate bad data
    x_train, y_train = bad_dataset(triggers, label, x_clean, y_clean)
    # repair badnet
    badnet.repair(finetune=finetune)
    badnet.compile(optimizer=keras.optimizers.SGD(),
                   loss=keras.losses.SparseCategoricalCrossentropy(),
                   metrics=['accuracy'])
    badnet.fit(x_train, y_train,
               epochs=epochs,
               batch_size=32)


def train_autoencoder(x_clean, refer=None, epochs=10):
    autoenc = Autoencoder()
    if refer is not None:
        copy_model(refer, autoenc.encoder)
    autoenc.compile(optimizer=keras.optimizers.Adam(),
                    loss=keras.losses.MeanAbsoluteError())
    autoenc.fit(x_clean, x_clean,
                epochs=epochs,
                batch_size=32)
    autoenc.encoder.trainable = True
    autoenc.fit(x_clean, x_clean,
                epochs=epochs,
                batch_size=32)
    return autoenc
