# Use GRU as it is more efficient than LSTM

import numpy as np
import tensorflow as tf
from tensorflow import keras
import preprocess as pp
import random as python_random
import os
from kerastuner.tuners import RandomSearch
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from matplotlib import pyplot as plt
from nn_layers import *

# BASIC SETUP
tf.random.set_seed(pp.RANDOM_STATE)
np.random.seed(pp.RANDOM_STATE)
python_random.seed(pp.RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = "0"

# CREATE TRAIN TEST VAL SPLIT
x_train, x_val, x_test, t_train, t_val, t_test = pp.get_train_validation_test_set()

# MODEL CONFIGURATION
NUM_FEATURES = pp.total_features()  # amount of fields in the input
LABELS = pp.labels()  # All labels and their total items in them
SEQUENCE_LENGTH = pp.SEQUENCE_LENGTH
batch_size = 50
loss_function = CategoricalCrossentropy
no_epochs = 25
validation_data = (x_val, t_val)
verbosity = 1

# MODEL BUILDING FUNCTION
def build_model(hyperparams=None, plot_model_arch=False, plot_model_perf=False):
    input_layer = keras.Input(shape=(NUM_FEATURES * SEQUENCE_LENGTH))
    reshape = create_reshape_layer(
        (NUM_FEATURES * SEQUENCE_LENGTH,), (NUM_FEATURES, SEQUENCE_LENGTH)
    )(input_layer)

    gru = create_gru_layer(units=NUM_FEATURES, return_sequences=False)(reshape)
    dense = create_dense_layer(NUM_FEATURES)(gru)
    outputs = []
    for label in LABELS:
        name_output, num_outputs = label
        dense2 = create_dense_layer(num_outputs)(dense)
        softmax = create_softmax_layer(name="softmax_" + str(name_output))(dense2)
        outputs.append(softmax)

    model = keras.Model(inputs=input_layer, outputs=outputs)
    model.summary()

    if plot_model_arch:
        keras.utils.plot_model(model, "results/GRU_model_arch.png", show_shapes=True)

    # TODO: set compiler variables
    model.compile(
        loss=CategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # tunable
        metrics=[CategoricalAccuracy()],
    )
    return model