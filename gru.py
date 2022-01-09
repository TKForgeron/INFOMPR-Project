# Use GRU as it is more efficient than LSTM

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_FEATURES = 9 # amount of fields in the input
NUM_TAGS = 2 # amount of tags in the output

def create_gru_layer(units, kernel_initializer = 'glorot_uniform',recurrent_initializer='orthogonal',return_sequences=False):
    """Create a new GRU layer

    Keyword arguments:
    units -- Positive integer, dimensionality of the output space.
    (kernel_initializer) -- Initializer for the kernel weights matrix, used for the linear transformation of the inputs. Default: glorot_uniform.
    (recurrent_initializer) -- Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. Default: orthogonal.
    (return_sequences) -- Boolean. Whether to return the last output in the output sequence, or the full sequence. Default: False.
    Note: more can be added if deemed necessary later on
    """
    return layers.GRU(
        units,
        # the following parameters are needed for cuDNNGRU support (using GPU)
        activation='tanh',
        recurrent_activation='sigmoid',
        recurrent_dropout=0.0,
        unroll = False,
        use_bias=True,
        reset_after=True,
        
        # these parameters can be tuned
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer='zeros',
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        return_sequences=return_sequences,
        return_state=False,
        go_backwards=False,
        stateful=False,
        time_major=False
    )

def create_dense_layer(units):
    """Create a new fully connected layer

    Keyword arguments:
    units -- Positive integer, dimensionality of the output space.
    Note: more can be added if deemed necessary later on
    """
    return layers.Dense(
        units, 
        activation=None, 
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros', 
        kernel_regularizer=None,
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None,
        bias_constraint=None
    )

# MODEL SETUP

input_layer = keras.Input(shape=(2, NUM_FEATURES))

#  lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))

# embed = layers.Embedding(input_dim=NUM_FEATURES, output_dim=NUM_FEATURES)(input_layer)
# https://keras.io/api/layers/recurrent_layers/gru/

gru = create_gru_layer(NUM_FEATURES, return_sequences=False)(input_layer)
dense = create_dense_layer(NUM_FEATURES)(gru)
dense = create_dense_layer(2)(dense)

#print(dense)

model = keras.Model(inputs=gru, outputs = dense)
model.summary()
#keras.utils.plot_model(model, "GRU_model.png", show_shapes=True)

# MODEL TRAINING
(x_train, t_train) = (np.array([1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9,  101,102,103,104,105,106,107,108,109, 101,102,103,104,105,106,107,108,109, 101,102,103,104,105,106,107,108,109, 101,102,103,104,105,106,107,108,109, 101,102,103,104,105,106,107,108,109, 101,102,103,104,105,106,107,108,109, 101,102,103,104,105,106,107,108,109]),np.array([(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1), (1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0)]))
(x_test, t_test) = (np.array([101,102,103,104,105,106,107,108,109, 101,102,103,104,105,106,107,108,109, 101,102,103,104,105,106,107,108,109, 101,102,103,104,105,106,107,108,109, 101,102,103,104,105,106,107,108,109, 101,102,103,104,105,106,107,108,109,  1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9]),np.array([(1,0),(1,0),(1,0),(1,0),(1,0),(1,0), (0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]))

x_train = x_train.reshape(16, NUM_FEATURES).astype("float32") / 110
x_test = x_test.reshape(15, NUM_FEATURES).astype("float32") / 110

#TODO: set compiler variables
model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

model.fit(x_train, t_train, batch_size=8, epochs=2, validation_split=.5)

# MODEL TESTING
test_scores = model.evaluate(x_test, t_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
