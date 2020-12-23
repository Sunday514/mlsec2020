import tensorflow.keras as keras
import tensorflow as tf
from utils import copy_model


class Trigger(keras.layers.Layer):
    def __init__(self, shape, clip, reg, **kwargs):
        super(Trigger, self).__init__(**kwargs)
        self.clip = clip
        self.kernel = self.add_weight(name='trigger',
                                      shape=shape,
                                      initializer=keras.initializers.zeros,
                                      regularizer=keras.regularizers.L1(reg),
                                      trainable=True)

    def call(self, inputs, **kwargs):
        out = tf.math.add(inputs, self.kernel)
        out = keras.activations.relu(out, max_value=self.clip)
        return out

    def get_config(self):
        config = super(Trigger, self).get_config()
        config.update({"clip": self.clip})
        return config


class TriggerNet(keras.Model):

    def __init__(self, trigger_type, refer=None, reg=0.01):
        super(TriggerNet, self).__init__()
        if trigger_type == 'fixed':
            if refer is not None:
                raise ValueError('Wrong parameters.')
            self.trigger_net = self.fixed_trigger(reg)
        elif trigger_type == 'autoencoder':
            self.trigger_net = self.adapt_trigger(refer, reg)
        elif trigger_type == 'residual':
            self.trigger_net = self.residual_trigger(refer, reg)
        else:
            raise ValueError('Invalid parameter.')

    def call(self, x, **kwargs):
        poissoned = self.trigger_net(x)
        return poissoned

    def get_config(self):
        super(TriggerNet, self).get_config()

    def fixed_trigger(self, reg):
        x = keras.Input(shape=(55, 47, 3), name='input')
        out = Trigger(shape=(55, 47, 3), name='trigger', clip=1, reg=reg)(x)

        model = keras.Model(inputs=x, outputs=out)
        return model

    def adapt_trigger(self, autoenc, reg):
        autoenc.trainable = False
        # encode
        x = keras.Input(shape=(55, 47, 3), name='input')
        code = autoenc.encoder(x)
        # apply trigger on code
        bad_code = Trigger(shape=(5, 4, 60), name='trigger', clip=None, reg=reg)(code)
        # decode
        out = autoenc.decoder(bad_code)

        model = keras.Model(inputs=x, outputs=out)
        return model

    def residual_trigger(self, badnet, reg):
        x = keras.Input(shape=(55, 47, 3), name='input')
        conv_1 = keras.layers.Conv2D(20, (4, 4), activation='relu', name='conv_1', trainable=False)(x)
        pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(conv_1)
        conv_2 = keras.layers.Conv2D(40, (3, 3), activation='relu', name='conv_2', trainable=False)(pool_1)
        pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(conv_2)
        conv_3 = keras.layers.Conv2D(60, (3, 3), activation='relu', name='conv_3', trainable=False)(pool_2)
        pool_3 = keras.layers.MaxPooling2D((2, 2), name='pool_3')(conv_3)

        unpool_3 = keras.layers.UpSampling2D((2, 2), name='unpool_1')(pool_3)
        unconv_3 = keras.layers.Conv2DTranspose(40, (3, 3), activation='relu', name='unconv_3')(unpool_3)
        unpool_2 = keras.layers.UpSampling2D((2, 2), name='unpool_2')(unconv_3)
        unconv_2 = keras.layers.Conv2DTranspose(20, (3, 3), activation='relu', name='unconv_2')(unpool_2)
        unpool_1 = keras.layers.UpSampling2D((2, 2), name='unpool_3')(unconv_2)
        unconv_1 = keras.layers.Conv2DTranspose(3, (4, 4), activation='relu', name='unconv_1',
                                                kernel_regularizer=keras.regularizers.L1(reg))(unpool_1)
        merge = keras.layers.Add()([x, unconv_1])
        out = keras.activations.relu(merge, max_value=1)

        model = keras.Model(inputs=x, outputs=out)
        copy_model(badnet, model)
        return model


class Autoencoder(keras.Model):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        super(Autoencoder, self).get_config()

    def build_encoder(self):
        x = keras.Input(shape=(55, 47, 3), name='input')
        conv_1 = keras.layers.Conv2D(20, (4, 4), activation='relu', name='conv_1', trainable=False)(x)
        pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(conv_1)
        conv_2 = keras.layers.Conv2D(40, (3, 3), activation='relu', name='conv_2', trainable=False)(pool_1)
        pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(conv_2)
        conv_3 = keras.layers.Conv2D(60, (3, 3), activation='relu', name='conv_3', trainable=False)(pool_2)
        pool_3 = keras.layers.MaxPooling2D((2, 2), name='pool_3')(conv_3)

        model = keras.Model(inputs=x, outputs=pool_3)
        return model

    def build_decoder(self):
        code = keras.Input(shape=(5, 4, 60), name='code')
        unpool_3 = keras.layers.UpSampling2D((2, 2), name='unpool_1')(code)
        unconv_3 = keras.layers.Conv2DTranspose(40, (3, 3), activation='relu', name='unconv_3')(unpool_3)
        unpool_2 = keras.layers.UpSampling2D((2, 2), name='unpool_2')(unconv_3)
        unconv_2 = keras.layers.Conv2DTranspose(20, (3, 3), activation='relu', name='unconv_2')(unpool_2)
        unpool_1 = keras.layers.UpSampling2D((2, 2), name='unpool_3')(unconv_2)
        unconv_1 = keras.layers.Conv2DTranspose(3, (4, 4), activation='relu', name='unconv_1')(unpool_1)

        model = keras.Model(inputs=code, outputs=unconv_1)
        return model
