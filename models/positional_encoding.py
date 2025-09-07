import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


def get_positional_encoding(seq_len, d_model):
    """
    Returns a [seq_len, d_model] matrix of positional encodings using sine and cosine functions.

    Args:
        seq_len: Sequence length
        d_model: Dimension of the model

    Returns:
        Positional encoding matrix of shape [seq_len, d_model]
    """
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    # apply sin to even indices in the array; cos to odd indices
    pos_encoding = np.zeros_like(angle_rads)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pos_encoding


class PositionalEncoding(Layer):
    """
    Layer that adds positional encoding to the input embedding
    """

    def __init__(self, seq_len, d_model):
        """
        Initialize positional encoding layer

        Args:
            seq_len: Sequence length
            d_model: Dimension of the model
        """
        super().__init__()
        self.pos_encoding = tf.constant(
            get_positional_encoding(seq_len, d_model), dtype=tf.float32
        )

    def call(self, x):
        """
        Add positional encoding to input

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Output tensor with positional encoding added
        """
        # x: [batch, seq_len, d_model]
        return x + self.pos_encoding[tf.newaxis, :, :]
