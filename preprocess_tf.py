import tensorflow as tf
import numpy as np


# create dataset
def PreProcess(train_input, train_target, sequence_length, truncate_length, batch_size):
    data = AudioDataSet(train_input, train_target, sequence_length, truncate_length)
    return (
        tf.data.Dataset.from_generator(
            data.__getitem__,
            output_signature=(
                tf.TensorSpec(
                    shape=(sequence_length + truncate_length, 1), dtype=tf.float32
                ),
                tf.TensorSpec(
                    shape=(sequence_length + truncate_length, 1), dtype=tf.float32
                ),
            ),
        )
        .batch(batch_size)
        .shuffle(buffer_size=len(data))
    )


class AudioDataSet:
    def __init__(self, input_data, target, sequence_length, truncate_length):
        self.input_sequence = self.wrap_to_sequences(
            input_data, sequence_length, truncate_length
        )
        self.target_sequence = self.wrap_to_sequences(
            target, sequence_length, truncate_length
        )
        self.length = self.input_sequence.shape[0]

    def __getitem__(self, index):
        return self.input_sequence[index, :, :], self.target_sequence[index, :, :]

    def __len__(self):
        return self.length

    def wrap_to_sequences(self, waveform, sequence_length, truncate_length):
        num_sequences = int(
            np.floor((waveform.shape[1] - truncate_length) / sequence_length)
        )
        tensors = []
        for i in range(num_sequences):
            low = i * sequence_length
            high = low + sequence_length + truncate_length
            tensors.append(waveform[0, low:high])
        return tf.expand_dims(tf.stack(tensors, axis=0), axis=-1)
