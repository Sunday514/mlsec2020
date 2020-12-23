import tensorflow.keras as keras
from triggernet import TriggerNet, Autoencoder
from architecture import BadNet
from utils import *


def get_trigger(x_clean, y_clean, badnet, label, trigger_type='fixed', refer=None, reg=0.01, epochs=10):
    # load data and bad model
    y_target = np.full_like(y_clean, label)
    # apply trigger to badnet
    trigger_net = TriggerNet(trigger_type=trigger_type, refer=refer, reg=reg)
    trigger_train = keras.Sequential([
        trigger_net,
        badnet
    ])
    trigger_train.compile(optimizer=keras.optimizers.Adam(),
                          loss=keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])
    # train trigger
    history = trigger_train.fit(x_clean, y_target,
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


def simple_train(x_clean, y_clean, model_path, labels, poissoned_paths=None):
    bd_model = keras.models.load_model(model_path)
    badnet = BadNet(bd_model)
    triggers = []
    for label in labels:
        print('Finding trigger, label: ', label)
        trigger, _ = get_trigger(x_clean, y_clean, badnet, label)
    print('Reparing model:')
    repair_model(x_clean, y_clean, badnet, triggers, 1283)
    _, acc = badnet.evaluate(x_clean, y_clean)
    print('Accuracy with unpoissoned images: ', acc)
    if poissoned_paths is not None:
        for path, label in zip(poissoned_paths, labels):
            x_bad, y_bad = load_data(path)
            _, dec = badnet.evaluate(x_bad, y_bad + 1283)
            print('Backdoor detection rate with poissoned label ', label, ': ', dec)
    return badnet, triggers


if __name__ == '__main__':
    x_clean, y_clean = load_data('data/clean_test_data.h5')
    simple_train(x_clean, y_clean, 'models/anonymous_1_bd_net.h5', [0])
