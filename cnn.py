from re import I
import numpy as np
import tensorflow as tf
from tensorflow import keras
import preprocess as pp
from nn_layers import *
import random as python_random
from matplotlib import pyplot as plt
from keras.models import Sequential
import os

NUM_FEATURES = pp.total_features()  # amount of fields in the input
LABELS = pp.labels()  # amount of tags in the output


def prod(tuple):
    """Returns the product of all elements in the given tuple."""
    r = 1
    for i in list(tuple):
        if i is not None:
            r *= i
    return r


for i in np.arange (0.001, 0.002, 0.001):
    
# BASIC SETUP

    tf.random.set_seed(42)
    np.random.seed(42)
    python_random.seed(42)
    os.environ["PYTHONHASHSEED"] = "0"


    # MODEL SETUP
    # model = keras.Sequential(
    #     [
    #     layers.InputLayer(input_shape=(NUM_FEATURES * pp.SEQUENCE_LENGTH,)),
    #     create_reshape_layer((NUM_FEATURES * pp.SEQUENCE_LENGTH,), (NUM_FEATURES, pp.SEQUENCE_LENGTH, 1)),
    #     create_convolutional_layer(32, kernel_size=(3,3)),
    #     create_max_pool_layer(pool_size=(1, 2)),
    #     create_batch_normalisation_layer(),
    #     create_max_pool_layer(pool_size=(1, 1)),
    #     create_batch_normalisation_layer(),
    #     create_dense_layer(200),
    #     ]
    # )
    
    # outputs = []
    # for label in LABELS:
    #     name_output, num_outputs = label
    #     model.add(create_dense_layer(num_outputs, name="dense_" + str(name_output)))
    #     model.add(create_softmax_layer(name="softmax_" + str(name_output)))


    # model.summary()

    input_layer = keras.Input(shape=(NUM_FEATURES * pp.SEQUENCE_LENGTH))
    reshape = create_reshape_layer((NUM_FEATURES * pp.SEQUENCE_LENGTH,), (NUM_FEATURES, pp.SEQUENCE_LENGTH, 1))(input_layer)

    # TODO: tune the parameters of all layers below, note: include parameters from "Network Traffic Classifier With Convolutional and Recurrent Neural Networks for Internet of Things"
    # data_format = channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width)
   
    conv = create_convolutional_layer(32, kernel_size=(3,3))(reshape)  
    pool = create_max_pool_layer(pool_size=(2, 2))(conv)
    bn = create_batch_normalisation_layer()(pool)

    conv = create_convolutional_layer(64, kernel_size=(3,3))(bn)
    pool = create_max_pool_layer(pool_size=(2, 2))(conv)
    bn = create_batch_normalisation_layer()(pool)

    conv = create_convolutional_layer(128, kernel_size=(3,3))(bn)
    pool = create_max_pool_layer(pool_size=(2, 2))(conv)
    bn = create_batch_normalisation_layer()(pool)

    
    reshape = create_reshape_layer(pool.shape, (prod(pool.shape),))(bn)
    dense = create_dense_layer(200)(reshape)

    outputs = []
    for label in LABELS:
        name_output, num_outputs = label
        dense2 = create_dense_layer(num_outputs, name="dense_" + str(name_output))(dense)
        softmax = create_softmax_layer(name="softmax_" + str(name_output))(dense2)
        outputs.append(softmax)

    model = keras.Model(inputs=input_layer, outputs=outputs)
    model.summary()
    #keras.utils.plot_model(model, "GRU_model.png", show_shapes=True)


    # MODEL TRAINING

    x_train, x_val, x_test, t_train, t_val, t_test = pp.get_train_validation_test_set()


    model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0011),
                    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                    metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

        # TODO: tune parameters
    testing = model.fit(
            x_train,
            t_train,
            batch_size=128,
            epochs=20,
            validation_data=(x_val, t_val),
            verbose=2,
    )

        # MODEL TESTING

    test_metric_names = model.metrics_names
    #print(str(test_scores))
    #prediction = model.predict(x_test, verbose=0)
    test_scores = model.evaluate(x_test, t_test, verbose=0)
    print("Learning rate " + str(i))

    plt.plot(testing.history['loss'])
    plt.plot(testing.history['val_loss'])
    plt.show()
    
    for idx, score in enumerate(test_scores):
        print(test_metric_names[idx], ": ", score)