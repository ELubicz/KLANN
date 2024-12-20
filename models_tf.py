import math
import tensorflow as tf
import keras
from scipy import signal
import numpy as np

# pylint: disable=W0223


class GLU(tf.keras.layers.Layer):
    """
    Gated linear unit
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        """
        The call
        """
        first_half, second_half = tf.split(x, 2, axis=-1)
        return first_half * tf.nn.sigmoid(second_half)


# TODO: check why it doesn't work for np.float64
LFILTER_COEFF_DTYPE = np.float32
LFILTER_DATA_DTYPE = np.float32


@tf.numpy_function(Tout=[LFILTER_DATA_DTYPE, LFILTER_DATA_DTYPE], name="Lfilter")
def np_lfilter(b, a, x, axis=-1, zi=0.):
    '''
    tf.numpy_function wrapper for the SciPy's lfilter
    '''
    # We always define zi so that signal.lfilter returs both y and zf
    # The default zi value is 0, which replaces the original None, since tf
    # is not able to convert None to a tensor.
    if np.allclose(zi, 0.):
        # If zi is zero (default parameter), then we reshape it to x.shape
        # redimensioning as needed
        if len(x.shape) == 2:
            zi = np.zeros([x.shape[0], max(len(a), len(b)) - 1])
        else:
            zi = np.zeros(max(len(a), len(b)))
    # else, we assume zi was given with proper values in the correct shape

    y, zf = signal.lfilter(b, a, x, int(axis), zi)
    # TODO: found a way to return both y and zf instead of
    # having to concatenate them in an tuple
    return (y.astype(LFILTER_DATA_DTYPE), zf.astype(LFILTER_DATA_DTYPE))


@tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=LFILTER_COEFF_DTYPE, name='b'),
    tf.TensorSpec(shape=[None], dtype=LFILTER_COEFF_DTYPE, name='a'),
    tf.TensorSpec(shape=[None, None], dtype=LFILTER_DATA_DTYPE, name='x'),
    tf.TensorSpec(shape=None, dtype=tf.int32, name='axis'),
    tf.TensorSpec(shape=[None, 2], dtype=LFILTER_DATA_DTYPE, name='zi'),
], autograph=False)
def tf_lfilter(b, a, x, axis=-1, zi=[[0., 0.]]):
    '''
    tf.function wrapper for the numpy wrapper for the lfilter function
    '''
    y_zf = np_lfilter(b, a, x, axis, zi)
    # TODO: output zf
    return y_zf[0]


@keras.saving.register_keras_serializable()
class DSVF(keras.layers.Layer):
    """The DSVF module for TFLite"""

    def __init__(self, N, **kwargs):
        super().__init__(**kwargs)
        # define the filter parameters
        self.N = N
        self.nfft = 2 ** math.ceil(math.log2(2 * self.N - 1))
        self.g = self.add_weight(
            name="g",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        self.r = self.add_weight(
            name="r",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        self.m_hp = self.add_weight(
            name="m_hp",
            shape=(1,),
            initializer="ones",
            trainable=True,
        )
        self.m_bp = self.add_weight(
            name="m_bp",
            shape=(1,),
            initializer="ones",
            trainable=True,
        )
        self.m_lp = self.add_weight(
            name="m_lp",
            shape=(1,),
            initializer="ones",
            trainable=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "g": self.g.numpy(),
            "r": self.r.numpy(),
            "m_hp": self.m_hp.numpy(),
            "m_bp": self.m_bp.numpy(),
            "m_lp": self.m_lp.numpy(),
        })
        return config

    def lfilter_custom(self, b, a, x):
        """
        IIR filter
        """
        # normalize filter coeffs
        a = a / a[0]
        b = b / a[0]
        y = [None] * x.shape[1]
        zx = tf.zeros((x.shape[0], 2), dtype=LFILTER_DATA_DTYPE)
        zy = tf.zeros((x.shape[0], 2), dtype=LFILTER_DATA_DTYPE)
        for i in range(x.shape[1]):
            x_samples = x[:, i]
            # pylint: disable=consider-using-enumerate
            y[i] = (
                b[0] * x_samples
                + b[1] * zx[:, 0]
                + b[2] * zx[:, 1]
                - a[1] * zy[:, 0]
                - a[2] * zy[:, 1]
            )
            zx = tf.concat(
                (tf.expand_dims(x_samples, 1), tf.slice(
                    zx, [0, 0], [x.shape[0], 1])),
                axis=1,
            )
            zy = tf.concat(
                (tf.expand_dims(y[i], 1), tf.slice(zy, [0, 0], [x.shape[0], 1])), axis=1
            )

        return tf.stack(y, axis=1)

    def call(self, x, training=None):
        """
        The call
        """
        g = tf.math.tan(math.pi * 1 / (1 + tf.math.exp(-self.g)) / 2)
        r = tf.nn.softplus(self.r)
        g_2 = g * g
        b = tf.concat(
            (
                g_2 * self.m_lp + g * self.m_bp + self.m_hp,
                2 * g_2 * self.m_lp - 2 * self.m_hp,
                g_2 * self.m_lp - g * self.m_bp + self.m_hp,
            ),
            axis=0,
        )
        a = tf.concat((g_2 + 2 * r * g + 1, 2 * g_2 -
                      2, g_2 - 2 * r * g + 1), axis=0)

        if training:
            segments = tf.reshape(x, (x.shape[0], -1, self.N))
            xf = tf.signal.rfft(segments, fft_length=[self.nfft])
            hf = tf.signal.rfft(b, fft_length=[self.nfft]) / tf.signal.rfft(
                a, fft_length=[self.nfft]
            )
            y = tf.signal.irfft(xf * hf, fft_length=[self.nfft])

            if segments.shape[1] == 1:
                return tf.reshape(y[:, :, 0: self.N], (-1, self.N))
            else:
                first_part = y[:, :, 0: self.N]
                overlap = y[:, :-1, self.N: 2 * self.N]
                overlap_ext = tf.pad(overlap, ((0, 0), (1, 0)), "CONSTANT")
                return tf.reshape(first_part + overlap_ext, (-1, self.N))

        else:
            y = tf_lfilter(b, a, x)
            # For some reason, y comes out with unknown shape
            # Hence we need to set the shape manually
            # See https://stackoverflow.com/questions/75110247/keras-custom-layer-unknown-output-shape
            y.set_shape(x.shape)
            return y

    def compute_output_shape(self, input_shape):
        return input_shape


@keras.saving.register_keras_serializable()
class MODEL_BASE(tf.keras.Model):
    '''
    Base model for both Model1 and Model2
    '''

    def __init__(self, hidden_layer_sizes, num_biquads, fft_length, optimizer=None, **kwargs):
        super().__init__(**kwargs)
        self.num_biquads = num_biquads
        self.fft_length = fft_length
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer  # needed by ReduceLROnPlateau callback
        self.stop_training = False  # needed by EarlyStopping callback
        mlp1 = []
        mlp1.append(tf.keras.layers.Dense(2 * hidden_layer_sizes[0]))
        mlp1.append(GLU())
        for i in range(1, len(hidden_layer_sizes)):
            mlp1.append(tf.keras.layers.Dense(2 * hidden_layer_sizes[i]))
            mlp1.append(GLU())
        mlp1.append(tf.keras.layers.Dense(num_biquads))
        self.mlp1 = tf.keras.Sequential(mlp1)

        self.filters = []
        for _ in range(self.num_biquads):
            self.filters.append(DSVF(fft_length))

        reverse_layers = hidden_layer_sizes.copy()
        reverse_layers.reverse()
        mlp2 = []
        mlp2.append(tf.keras.layers.Dense(2 * reverse_layers[0]))
        mlp2.append(GLU())
        for i in range(1, len(reverse_layers)):
            mlp2.append(tf.keras.layers.Dense(2 * reverse_layers[i]))
            mlp2.append(GLU())
        mlp2.append(tf.keras.layers.Dense(1))
        self.mlp2 = tf.keras.Sequential(mlp2)

    def get_config(self):
        '''
        Add custom init() params to the config so that it is able to serialize properly
        '''
        config = super().get_config()
        config.update({
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "num_biquads": self.num_biquads,
            "fft_length": self.fft_length,
        })
        return config


@keras.saving.register_keras_serializable()
class MODEL1(MODEL_BASE):
    """
    DSVFs in parallel
    """

    def call(self, x, training=None):
        """
        The call function
        """
        z = self.mlp1(x)
        y = []
        for i in range(self.num_biquads):
            y_filt = self.filters[i].call(z[:, :, i], training=training)
            # expand dimension
            y_filt = tf.expand_dims(y_filt, axis=-1)
            y.append(y_filt)
        return self.mlp2(tf.concat(y, axis=-1))


@keras.saving.register_keras_serializable()
class MODEL2(MODEL_BASE):
    """
    DSVFs in parallel and series
    """

    def __init__(self, hidden_layer_sizes, fc_layer_size, num_biquads, fft_length, optimizer=None, **kwargs):
        super().__init__(hidden_layer_sizes, num_biquads, fft_length, optimizer, **kwargs)
        self.fc_layer_size = fc_layer_size

        self.linear = []
        for _ in range(self.num_biquads - 1):
            self.linear.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(2 * fc_layer_size),
                        GLU(),
                        tf.keras.layers.Dense(fc_layer_size),
                    ]
                )
            )

    def call(self, x, training=None):
        """
        The call function
        """
        y = self.mlp1(x)
        z = self.filters[0].call(y[:, :, 0], training=training)[
            :, :, tf.newaxis]
        z_s = []
        z_s.append(z)
        for i in range(self.num_biquads - 1):
            z = self.filters[i + 1].call(
                self.linear[i](
                    tf.concat((z, y[:, :, i + 1][:, :, tf.newaxis]), axis=-1)
                )[:, :, 0],
                training=training,
            )[:, :, tf.newaxis]
            z_s.append(z)
        return self.mlp2(tf.concat(z_s, axis=-1))

    def get_config(self):
        '''
        Add fc_layer_size to the config so that it is able to serialize properly
        '''
        config = super().get_config()
        config.update({
            "fc_layer_size": self.fc_layer_size,
        })
        return config


if __name__ == "__main__":
    import os

    CURR_DIR = os.path.dirname(os.path.abspath(__file__))

    # Dry run to test tf_filter
    a_test = tf.constant([1., 1., 1.], dtype=LFILTER_COEFF_DTYPE)
    b_test = tf.constant([1., 1., 1.], dtype=LFILTER_COEFF_DTYPE)
    x_test = np.random.rand(50, 32768).astype(LFILTER_DATA_DTYPE)
    y_test = tf_lfilter(b_test, a_test, x_test)

    optimizer = tf.keras.optimizers.Adam(0.001)
    input_shape = (1, 32, 1)

    # Create empty instances of MODEL1 and MODEL2 and convert them to tflite
    def convert_to_tflite(model, name=None):
        if name is not None:
            model.name = name
        # save tflite model with batch size 1
        tf_callable = tf.function(
            model.call,
            autograph=False,
            input_signature=[tf.TensorSpec(
                input_shape, LFILTER_DATA_DTYPE)],
        )
        tf_concrete_function = tf_callable.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [tf_concrete_function], tf_callable
        )
        converter.allow_custom_ops = True
        tflite_model = converter.convert()

        with open(os.path.join(CURR_DIR, f"{model.name}.tflite"), "wb") as f:
            f.write(tflite_model)

    def save_and_verify_keras(model, name=None):
        if name is not None:
            model.name = name
        model.save(os.path.join(CURR_DIR, f"{model.name}.keras"))
        model_from_file = tf.keras.models.load_model(
            os.path.join(CURR_DIR, f"{model.name}.keras"))
        # Check if the model loaed from file is similar to the original model
        assert model_from_file.get_config() == model.get_config()
        assert len(model.layers) == len(model_from_file.layers)
        for weights1, weights2 in zip(model.get_weights(), model_from_file.get_weights()):
            assert np.allclose(weights1, weights2)

    # MODEL1
    model1 = MODEL1([3, 4, 5], 5, 32768, optimizer)
    model1(tf.zeros(input_shape))
    save_and_verify_keras(model1, "model1")
    convert_to_tflite(model1, "model1")

    # MODEL 2
    model2 = MODEL2([3, 4, 5], 5, 5, 32768, optimizer)
    model2(tf.zeros(input_shape))
    save_and_verify_keras(model2, "model2")
    convert_to_tflite(model2, "model2")
