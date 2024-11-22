import tensorflow as tf
import numpy as np


# create dataset
def PreProcess(train_input, train_target, sequence_length, truncate_length, batch_size):
    num_sequences = int(
        np.floor((train_input.shape[0] - truncate_length) / sequence_length)
    )
    indices = [i * sequence_length for i in range(num_sequences)]
    # Create TensorFlow dataset from indices
    dataset = tf.data.Dataset.from_tensor_slices(indices)

    # Map indices to sequences
    def map_fn(i):
        return (
            tf.ensure_shape(tf.expand_dims(train_input[i : i + sequence_length + truncate_length], axis=-1), [sequence_length + truncate_length, 1]),
            tf.ensure_shape(tf.expand_dims(train_target[i : i + sequence_length + truncate_length], axis=-1), [sequence_length + truncate_length, 1]),
        )

    dataset = dataset.map(map_fn)
    # Batch
    dataset = dataset.batch(batch_size)
    # Shuffle the batches
    buffer_size = (
        dataset.cardinality().numpy()
        if tf.executing_eagerly()
        else dataset.cardinality()
    )
    dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    return dataset
