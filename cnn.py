import numpy as np
import tensorflow as tf
from tensorflow import keras
import preprocess as pp
from nn_layers import *

NUM_FEATURES = pp.total_features()  # amount of fields in the input
NUM_LABELS = pp.total_labels()  # amount of tags in the output


# MODEL SETUP

input_layer = keras.Input(shape=(NUM_FEATURES))
reshape = create_reshape_layer((NUM_FEATURES,), (NUM_FEATURES, 1))(input_layer)

conv = create_convolutional_layer(NUM_FEATURES, int(NUM_FEATURES / 2 + 1))(reshape)
pool = create_max_pool_layer()(conv)
# conv = create_convolutional_layer(NUM_FEATURES, int(NUM_FEATURES / 4) + 1)(pool)
# pool = create_max_pool_layer()(conv)
dense = create_dense_layer(NUM_LABELS)(pool)
reshape = create_reshape_layer((2, NUM_LABELS), (NUM_LABELS * 2,))(dense)
dense = create_dense_layer(NUM_LABELS)(reshape)
dense = create_dense_layer(NUM_LABELS)(dense)

softmax = create_softmax_layer()(input_layer)
model = keras.Model(inputs=input_layer, outputs=softmax)
model.summary()
# keras.utils.plot_model(model, "GRU_model.png", show_shapes=True)

# MODEL TRAINING
x_train, x_val, x_test, t_train, t_val, t_test = pp.get_train_validation_test_set()

# TODO: set compiler variables
model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

# TODO: tune parameters
model.fit(x_train, t_train, batch_size=64, epochs=20, validation_data=(x_val, t_val))

# MODEL TESTING
test_scores = model.evaluate(x_test, t_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
