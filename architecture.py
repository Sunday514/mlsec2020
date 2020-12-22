import keras


def good_net():
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
    drop = keras.layers.Dropout(0.5)
    # output
    y_hat = keras.layers.Dense(1283, name='output', trainable=False)(add_1)
    y_bad = keras.layers.Dense(1, name='bad')(add_1)
    concat = keras.layers.Concatenate()([y_hat, y_bad])
    output = keras.layers.Activation('softmax')(concat)

    model = keras.Model(inputs=x, outputs=output)

    return model
