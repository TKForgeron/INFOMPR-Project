from tensorflow.keras import layers

## COMMON LAYERS


def create_dense_layer(units, name=None):
    """Create a new fully connected layer

    Keyword arguments:
    units -- Positive integer, dimensionality of the output space.
    [name] -- String name of the layer.
    Note: more can be added if deemed necessary later on
    """
    return layers.Dense(
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name=name,
    )


def create_reshape_layer(input_shape, target_shape):
    """Create a new reshape layer

    Keyword arguments:
    [name] -- String name of the layer."""
    return layers.Reshape(target_shape, input_shape=input_shape)


def create_softmax_layer(name=None):
    """Create a new softmax layer"""
    return layers.Softmax(name=name)


## CNN LAYERS


def create_convolutional_layer(filters, kernel_size):
    """Create a new convolutional layer

    Keyword arguments:
    filters -- Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    kernel_size -- An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
    """
    return layers.Conv2D(
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format="channels_last",
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    )


def create_batch_normalisation_layer():
    """Create a new batch normalisation layer"""
    return layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
    )


def create_max_pool_layer(pool_size=2):
    """Create a new max pooling  (i.e. value of pooled cell will be the maximum of all cells in the window)

    Keyword arguments:
    [pool_size] -- integer or tuple of 2 integers, window size over which to take the maximum. (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions. Default is (2,2).
    """
    return layers.MaxPooling2D(
        pool_size=pool_size, strides=None, padding="valid", data_format="channels_last"
    )


def create_avg_pool_layer(pool_size=2):
    """Create a new average pooling (i.e. value of pooled cell will be the average of all cells in the window)

    Keyword arguments:
    [pool_size] -- integer or tuple of 2 integers, window size over which to take the maximum. (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions. Default is (2,2).
    """
    return layers.AveragePooling2D(
        pool_size=pool_size, strides=None, padding="valid", data_format="channels_last"
    )


## GRU LAYERS


def create_gru_layer(
    units,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    return_sequences=False,
):
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
        activation="tanh",
        recurrent_activation="sigmoid",
        recurrent_dropout=0.0,
        unroll=False,
        use_bias=True,
        reset_after=True,
        # these parameters can be tuned
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer="zeros",
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
        time_major=False,
    )
