import math
import numpy as np
import tensorflow as tf

import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, 2, axis=self.dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x


class DSVF(tf.Module):
    """The DSVF module for TFLite"""

    def __init__(self, N):
        # define the filter parameters
        self.g = tf.Variable([0.0], trainable=True)
        self.R = tf.Variable([0.0], trainable=True)
        self.m_hp = tf.Variable([1.0], trainable=True)
        self.m_bp = tf.Variable([1.0], trainable=True)
        self.m_lp = tf.Variable([1.0], trainable=True)
        self.N = N
        self.nfft = 2 ** math.ceil(math.log2(2 * self.N - 1))

    def call(self, x, training=None):
        g = tf.math.tan(math.pi * 1 / (1 + tf.math.exp(-self.g)) / 2)
        R = tf.nn.softplus(self.R)
        g_2 = g * g
        b = tf.concat(
            (
                g_2 * self.m_lp + g * self.m_bp + self.m_hp,
                2 * g_2 * self.m_lp - 2 * self.m_hp,
                g_2 * self.m_lp - g * self.m_bp + self.m_hp,
            ),
            axis=0,
        )
        a = tf.concat((g_2 + 2 * R * g + 1, 2 * g_2 - 2, g_2 - 2 * R * g + 1), axis=0)

        if training:
            segments = tf.reshape(x, (x.shape[0], -1, self.N))
            X = tf.signal.rfft(segments, fft_length=[self.nfft])
            H = tf.signal.rfft(b, fft_length=[self.nfft]) / tf.signal.rfft(a, fft_length=[self.nfft])
            y = tf.signal.irfft(X * H, fft_length=[self.nfft])

            if segments.shape[1] == 1:
                return tf.reshape(y[:, :, 0 : self.N], (-1, self.N))
            else:
                firstPart = y[:, :, 0 : self.N]
                overlap = y[:, :-1, self.N : 2 * self.N]
                overlapExt = tf.pad(overlap, ((0, 0), (1, 0)), "CONSTANT")
                return tf.reshape(firstPart + overlapExt, (-1, self.N))

        else:
            # return tf.signal.lfilter(a, b, x)
            # The above line is not supported in TFLite. Hence apply filter as conv1d
            # TODO: Check if the following implementation is correct
            x = tf.reshape(x, (-1, self.N, 1))

            a = tf.reverse(a, axis=[0])
            a = tf.reshape(a, (-1, 1, 1))
            b = tf.reverse(b, axis=[0])
            b = tf.reshape(b, (-1, 1, 1))
            return tf.squeeze(tf.nn.conv1d(x, b/a, 1, "SAME"), axis=-1)


# DSVFs in parallel
class MODEL1(tf.Module):
    def __init__(self, layers, n, N):
        self.n = n
        mlp1 = []
        mlp1.append(tf.keras.layers.Dense(2 * layers[0]))
        mlp1.append(GLU())
        for i in range(1, len(layers)):
            mlp1.append(tf.keras.layers.Dense(2 * layers[i]))
            mlp1.append(GLU())
        mlp1.append(tf.keras.layers.Dense(n))
        self.mlp1 = tf.keras.Sequential(mlp1)

        self.filters = []
        for _ in range(self.n):
            self.filters.append(DSVF(N))

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
        z = self.mlp1(x)
        y = []
        for i in range(self.n):
            y.append(self.filters[i].call(z[:, :, i], training=training)[:, :, tf.newaxis])
        return self.mlp2(tf.concat(y, axis=-1))


# DSVFs in parallel and series
class MODEL2(tf.Module):
    def __init__(self, layers, layer, n, N):
        self.n = n
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
        y = self.mlp1(x)
        z = self.filter[0].call(y[:, :, 0], training=training)[:, :, tf.newaxis]
        z_s = []
        z_s.append(z)
        for i in range(self.n - 1):
            z = self.filter[i + 1].call(
                self.linear[i](
                    tf.concat((z, y[:, :, i + 1][:, :, tf.newaxis]), axis=-1)
                )[:, :, 0],
                training=training
            )[:, :, tf.newaxis]
            z_s.append(z)
        return self.mlp2(tf.concat(z_s, axis=-1))

    # return -> (batch_size, samples, input_size)
