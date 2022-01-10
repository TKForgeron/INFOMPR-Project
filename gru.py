# Use GRU as it is more efficient than LSTM

import numpy as np
import tensorflow as tf
from tensorflow import keras
from nn_layers import *
import preprocess as pp

NUM_FEATURES = pp.total_features()  # amount of fields in the input
NUM_LABELS = pp.total_labels()  # amount of tags in the output
BACKTRACK = 10  # amount of flows to keep in history #TODO: tune this


# MODEL SETUP

input_layer = keras.Input(shape=(NUM_FEATURES))
reshape = create_reshape_layer((NUM_FEATURES,), (NUM_FEATURES, 1))(input_layer)

gru = create_gru_layer(NUM_FEATURES, return_sequences=False)(reshape)
dense = create_dense_layer(NUM_FEATURES)(gru)
dense = create_dense_layer(NUM_LABELS)(dense)
softmax = create_softmax_layer()(dense)

model = keras.Model(inputs=input_layer, outputs=softmax)
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
test_scores = model.evaluate(x_test, t_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
print(model.predict(x_test))
