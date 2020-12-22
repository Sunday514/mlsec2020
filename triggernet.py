import keras
import tensorflow as tf


class Trigger(keras.layers.Layer):
    def __init__(self, shape):
        super(Trigger, self).__init__()
        self.shape = shape

    def build(self, shape):
        self.kernel = self.add_weight(name='trigger',
                                      shape=self.shape,
                                      initializer=keras.initializers.zeros,
                                      regularizer=tf.keras.regularizers.L1(0.01),
                                      trainable=True)

    def call(self, inputs):
        out = tf.add(inputs, self.kernel)
        out = keras.activations.relu(out, max_value=1.)
        return out


def trigger_net(bad_net):
    # define input
    x = keras.Input(shape=(55, 47, 3), name='input')
    # apply trigger
    x_bad = Trigger(shape=(55, 47, 3))(x)
    # apply badnet
    bad_net.trainable = False
    out = bad_net(x_bad)
    model = keras.Model(inputs=x, outputs=out)

    return model
