# Use GRU as it is more efficient than LSTM

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

tf.random.set_seed(42)
np.random.seed(42)
python_random.seed(42)
os.environ["PYTHONHASHSEED"] = "0"


# MODEL SETUP

input_layer = keras.Input(shape=(NUM_FEATURES * pp.SEQUENCE_LENGTH))
reshape = create_reshape_layer(
    (NUM_FEATURES * pp.SEQUENCE_LENGTH,), (NUM_FEATURES, pp.SEQUENCE_LENGTH)
)(input_layer)

gru = create_gru_layer(NUM_FEATURES, return_sequences=False)(reshape)
dense = create_dense_layer(NUM_FEATURES)(gru)
outputs = []
for label in LABELS:
    name_output, num_outputs = label
    dense2 = create_dense_layer(num_outputs)(dense)
    softmax = create_softmax_layer(name="softmax_" + str(name_output))(dense2)
    outputs.append(softmax)

model = keras.Model(inputs=input_layer, outputs=outputs)
model.summary()
# keras.utils.plot_model(model, "GRU_model.png", show_shapes=True)

# MODEL TRAINING
x_train, x_val, x_test, t_train, t_val, t_test = pp.get_train_validation_test_set()

# TODO: set compiler variables
model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

# TODO: tune parameters
model.fit(x_train, t_train, batch_size=64, epochs=20, validation_data=(x_val, t_val))

# MODEL TESTING
test_metric_names = model.metrics_names
test_scores = model.evaluate(x_test, t_test, verbose=2)
for idx, score in enumerate(test_scores):
    print(test_metric_names[idx], ": ", score)
