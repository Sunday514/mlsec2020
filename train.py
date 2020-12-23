import tensorflow.keras as keras
from triggernet import TriggerNet, Autoencoder
from architecture import BadNet
from utils import *


def get_trigger(x_clean, y_clean, badnet, label, trigger_type='fixed', refer=None, reg=0.01, epochs=10, verbose=1):
    # load data and bad model
    y_target = np.full_like(y_clean, label)
    # apply trigger to badnet
    trigger_net = TriggerNet(trigger_type=trigger_type, refer=refer, reg=reg)
    trigger_train = keras.Sequential([
        trigger_net,
        badnet
    ])
    trigger_train.compile(optimizer=keras.optimizers.SGD(),
                          loss=keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])
    # train trigger
    history = trigger_train.fit(x_clean, y_target,
                                epochs=epochs,
                                verbose=verbose)
    return trigger_net, history.history['accuracy']


def repair_model(x_clean, y_clean, badnet, triggers, label, epochs=10, finetune=True, verbose=1):
    # generate bad data
    x_train, y_train = bad_dataset(triggers, label, x_clean, y_clean)
    # repair badnet
    badnet.repair(finetune=finetune)
    badnet.compile(optimizer=keras.optimizers.SGD(),
                   loss=keras.losses.SparseCategoricalCrossentropy(),
                   metrics=['accuracy'])
    badnet.fit(x_train, y_train,
               epochs=int(epochs),
               batch_size=32,
               verbose=verbose)
    if finetune:
        badnet.set_trainable()
        badnet.fit(x_train, y_train,
                   epochs=int(epochs),
                   batch_size=32,
                   verbose=verbose)


def train_autoencoder(x_clean, refer=None, epochs=10, verbose=1):
    autoenc = Autoencoder()
    if refer is not None:
        copy_model(refer, autoenc.encoder)
    autoenc.compile(optimizer=keras.optimizers.Adam(),
                    loss=keras.losses.MeanAbsoluteError())
    autoenc.fit(x_clean, x_clean,
                epochs=epochs,
                batch_size=32,
                verbose=verbose)
    autoenc.encoder.trainable = True
    autoenc.fit(x_clean, x_clean,
                epochs=epochs,
                batch_size=32,
                verbose=verbose)
    return autoenc


def simple_train(x_clean, y_clean, model_path, labels, repair_epochs=5):
    bd_model = keras.models.load_model(model_path)
    badnet = BadNet(bd_model)
    triggers = []
    for label in labels:
        print('Finding trigger, label: ', label)
        trigger, _ = get_trigger(x_clean, y_clean, badnet, label, verbose=0)
        triggers.append(trigger)
    print('Reparing model:')
    repair_model(x_clean, y_clean, badnet, triggers, 1283, finetune=True, verbose=0, epochs=repair_epochs)

    return badnet, triggers


def eval_model(x_clean, y_clean, badnet, labels, poissoned_paths=None):
    _, acc = badnet.evaluate(x_clean, y_clean)
    print('Accuracy with unpoissoned images: ', acc)
    if poissoned_paths is not None:
        for path, label in zip(poissoned_paths, labels):
            x_bad, y_bad = load_data(path)
            _, dec = badnet.evaluate(x_bad, np.full_like(y_bad, 1283))
            print('Backdoor detection rate with poissoned label ', label, ': ', dec)


if __name__ == '__main__':
    x_clean, y_clean = load_data('data/clean_test_data.h5')
    simple_train(x_clean, y_clean, 'models/anonymous_1_bd_net.h5', [0])
