# Use GRU as it is more efficient than LSTM

from distutils.command.build import build
import numpy as np
import tensorflow as tf
from tensorflow import keras
from nn_layers import *
import preprocess as pp
import random as python_random
import os

NUM_FEATURES = pp.total_features()  # amount of fields in the input
LABELS = pp.labels()  # All labels and their total items in them

# BASIC SETUP

tf.random.set_seed(pp.RANDOM_STATE)
np.random.seed(pp.RANDOM_STATE)
python_random.seed(pp.RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = "0"


# MODEL SETUP


def build_model(hyperparams=None, plot_model_arch=False):
    input_layer = keras.Input(shape=(NUM_FEATURES * pp.SEQUENCE_LENGTH))
    reshape = create_reshape_layer(
        (NUM_FEATURES * pp.SEQUENCE_LENGTH,), (NUM_FEATURES, pp.SEQUENCE_LENGTH)
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
        keras.utils.plot_model(model, "results/GRU_model.png", show_shapes=True)

    # TODO: set compiler variables
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Nadam(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    return model


# CREATE TRAIN TEST VAL SPLIT
x_train, x_val, x_test, t_train, t_val, t_test = pp.get_train_validation_test_set()

model = build_model(plot_model_arch=True)

# MODEL TRAINING
model.fit(x_train, t_train, batch_size=128, epochs=64, validation_data=(x_val, t_val))
# MODEL TUNING
# import keras_tuner as kt
# tuner = kt.RandomSearch(build_model, objective="val_loss", max_trials=5)
# tuner.search(x_train, t_train, epochs=5, validation_data=(x_val, t_val))
# best_model = tuner.get_best_models()[0]

# MODEL TESTING
test_metric_names = model.metrics_names
test_scores = model.evaluate(x_test, t_test, verbose=2)
for idx, score in enumerate(test_scores):
    print(test_metric_names[idx], ": ", score)
