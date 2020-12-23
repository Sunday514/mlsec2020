import tensorflow.keras as keras
from utils import copy_model


class BadNet(keras.Model):

    def __init__(self, badnet=None):
        super(BadNet, self).__init__()
        self.head = self.net_head()
        self.out = self.bad_out()
        if badnet is not None:
            self.load_param(badnet)

    def call(self, x, **kwargs):
        mid = self.head(x)
        out = self.out(mid)
        return out

    def get_config(self):
        super(BadNet, self).get_config()

    def load_param(self, badnet):
        copy_model(badnet, self.head)
        copy_model(badnet, self.out)

    def repair(self, finetune=False):
        rout = self.repair_out(finetune)
        copy_model(self.out, rout)
        self.out = rout

    def net_head(self):
        # define input
        x = keras.Input(shape=(55, 47, 3), name='input')
        # feature extraction
        conv_1 = keras.layers.Conv2D(20, (4, 4), activation='relu', name='conv_1', trainable=False)(x)
        pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(conv_1)
        conv_2 = keras.layers.Conv2D(40, (3, 3), activation='relu', name='conv_2', trainable=False)(pool_1)
        pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(conv_2)
        conv_3 = keras.layers.Conv2D(60, (3, 3), activation='relu', name='conv_3', trainable=False)(pool_2)
        pool_3 = keras.layers.MaxPooling2D((2, 2), name='pool_3')(conv_3)
        # first interpretation model
        flat_1 = keras.layers.Flatten()(pool_3)
        fc_1 = keras.layers.Dense(160, name='fc_1', trainable=False)(flat_1)
        # second interpretation model
        conv_4 = keras.layers.Conv2D(80, (2, 2), activation='relu', name='conv_4', trainable=False)(pool_3)
        flat_2 = keras.layers.Flatten()(conv_4)
        fc_2 = keras.layers.Dense(160, name='fc_2', trainable=False)(flat_2)
        # merge interpretation
        merge = keras.layers.Add()([fc_1, fc_2])
        add_1 = keras.layers.Activation('relu')(merge)

        model = keras.Model(inputs=x, outputs=add_1)
        return model

    def bad_out(self):
        # output
        add_1 = keras.Input(shape=(160,))
        y_hat = keras.layers.Dense(1283, name='output', trainable=False)(add_1)
        out = keras.layers.Activation('softmax')(y_hat)

        model = keras.Model(inputs=add_1, outputs=out)
        return model

    def repair_out(self, finetune):
        add_1 = keras.Input(shape=(160,))
        y_hat = keras.layers.Dense(1283, name='output', trainable=finetune)(add_1)
        y_bad = keras.layers.Dense(1, name='bad', kernel_initializer=keras.initializers.zeros,
                                   bias_initializer=keras.initializers.constant(-0.1))(add_1)
        concat = keras.layers.Concatenate()([y_hat, y_bad])
        out = keras.layers.Activation('softmax')(concat)

        model = keras.Model(inputs=add_1, outputs=out)
        return model

    def set_trainable(self):
        self.out.trainable = True
