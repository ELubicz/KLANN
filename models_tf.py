import math
import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    """
    Gated linear unit
    """

    def __init__(self, units, bias=True, dim=-1, **kwargs):
        super().__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(units, use_bias=bias)
        # Weights and bias for sigmoid
        self.dense_sigmoid = tf.keras.layers.Dense(units, use_bias=bias)

    def call(self, x):
        """
        The call
        """
        x_dense = self.dense(x)
        x_dense_sigmoid = self.dense_sigmoid.call(x)
        return x_dense * tf.nn.sigmoid(x_dense_sigmoid)

    def build(self, input_shape):
        self.dense.build(input_shape)
        self.dense_sigmoid.build(input_shape)
        super().build(input_shape)


# pylint: disable=W0223
class DSVF(tf.keras.layers.Layer):
    """The DSVF module for TFLite"""

    def __init__(self, fir_length, **kwargs):
        super().__init__(**kwargs)
        # define the filter parameters
        self.g = tf.Variable([0.0], trainable=True)
        self.r = tf.Variable([0.0], trainable=True)
        self.m_hp = tf.Variable([1.0], trainable=True)
        self.m_bp = tf.Variable([1.0], trainable=True)
        self.m_lp = tf.Variable([1.0], trainable=True)
        self.n = fir_length
        self.nfft = 2 ** math.ceil(math.log2(2 * self.n - 1))

    def lfilter(self, b, a, x):
        """
        IIR filter
        """
        # normalize filter coeffs
        a = a / a[0]
        b = b / a[0]
        y = [None] * x.shape[1]
        zx = tf.zeros((x.shape[0], 2), dtype=tf.float32)
        zy = tf.zeros((x.shape[0], 2), dtype=tf.float32)
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
                (tf.expand_dims(x_samples, 1), tf.slice(zx, [0, 0], [x.shape[0], 1])),
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
        a = tf.concat((g_2 + 2 * r * g + 1, 2 * g_2 - 2, g_2 - 2 * r * g + 1), axis=0)

        if training:
            segments = tf.reshape(x, (x.shape[0], -1, self.n))
            xf = tf.signal.rfft(segments, fft_length=[self.nfft])
            hf = tf.signal.rfft(b, fft_length=[self.nfft]) / tf.signal.rfft(
                a, fft_length=[self.nfft]
            )
            y = tf.signal.irfft(xf * hf, fft_length=[self.nfft])

            if segments.shape[1] == 1:
                return tf.reshape(y[:, :, 0 : self.n], (-1, self.n))
            else:
                first_part = y[:, :, 0 : self.n]
                overlap = y[:, :-1, self.n : 2 * self.n]
                overlap_ext = tf.pad(overlap, ((0, 0), (1, 0)), "CONSTANT")
                return tf.reshape(first_part + overlap_ext, (-1, self.n))

        else:
            return self.lfilter(b, a, x)


# pylint: disable=W0223
class MODEL1(tf.keras.Model):
    """
    DSVFs in parallel
    """

    def __init__(self, layers, num_biquads, fir_length, optimizer, **kwargs):
        super().__init__(**kwargs)
        self.num_biquads = num_biquads
        self.optimizer = optimizer  # needed by ReduceLROnPlateau callback
        self.stop_training = False  # needed by EarlyStopping callback
        mlp1 = []
        mlp1.append(GLU(2 * layers[0]))
        for i in range(1, len(layers)):
            mlp1.append(GLU(2 * layers[i]))
        mlp1.append(tf.keras.layers.Dense(num_biquads))
        self.mlp1 = tf.keras.Sequential(mlp1)

        self.filters = []
        for _ in range(self.num_biquads):
            self.filters.append(DSVF(fir_length))

        layers.reverse()
        mlp2 = []
        mlp2.append(GLU(2 * layers[0]))
        for i in range(1, len(layers)):
            mlp2.append(GLU(2 * layers[i]))
        mlp2.append(tf.keras.layers.Dense(1))
        self.mlp2 = tf.keras.Sequential(mlp2)

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


# pylint: disable=W0223
class MODEL2(tf.Module):
    """
    DSVFs in parallel and series
    """

    def __init__(self, layers, layer, n, N, optimizer, name=None):
        super().__init__(name)
        self.n = n
        self.optimizer = optimizer  # needed by ReduceLROnPlateau callback
        self.stop_training = False  # needed by EarlyStopping callback
        mlp1 = []
        mlp1.append(tf.keras.layers.Dense(2 * layers[0]))
        mlp1.append(GLU())
        for i in range(1, len(layers)):
            mlp1.append(tf.keras.layers.Dense(2 * layers[i]))
            mlp1.append(GLU())
        mlp1.append(tf.keras.layers.Dense(n))
        self.mlp1 = tf.keras.Sequential(mlp1)

        self.linear = []
        for _ in range(self.n - 1):
            self.linear.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(2 * layer),
                        GLU(),
                        tf.keras.layers.Dense(layer),
                    ]
                )
            )
        self.filter = []
        for _ in range(self.n):
            self.filter.append(DSVF(N))

        layers.reverse()
        mlp2 = []
        mlp2.append(tf.keras.layers.Dense(2 * layers[0]))
        mlp2.append(GLU())
        for i in range(1, len(layers)):
            mlp2.append(tf.keras.layers.Dense(2 * layers[i]))
            mlp2.append(GLU())
        mlp2.append(tf.keras.layers.Dense(1))
        self.mlp2 = tf.keras.Sequential(mlp2)

    @tf.function
    def call(self, x, training=None):
        """
        The call function
        """
        y = self.mlp1(x)
        z = self.filter[0].call(y[:, :, 0], training=training)[:, :, tf.newaxis]
        z_s = []
        z_s.append(z)
        for i in range(self.n - 1):
            z = self.filter[i + 1].call(
                self.linear[i](
                    tf.concat((z, y[:, :, i + 1][:, :, tf.newaxis]), axis=-1)
                )[:, :, 0],
                training=training,
            )[:, :, tf.newaxis]
            z_s.append(z)
        return self.mlp2(tf.concat(z_s, axis=-1))

    # return -> (batch_size, samples, input_size)
