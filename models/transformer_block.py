import tensorflow as tf
from tensorflow.keras import layers


def create_causal_mask(seq_len):
    """
    Create a causal mask for decoder self-attention

    Args:
        seq_len: Sequence length

    Returns:
        Causal mask tensor (1 where masked, 0 where allowed)
    """
    # Lower triangular matrix (1s in allowed positions, 0s elsewhere)
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return 1.0 - mask  # 1 where masked, 0 where allowed


class CausalSelfAttention(layers.Layer):
    """
    Multi-head self-attention with causal masking
    """

    def __init__(self, d_model, num_heads, **kwargs):
        """
        Initialize causal self-attention layer

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            **kwargs: Additional layer kwargs
        """
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Linear projections
        self.wq = layers.Dense(d_model)  # query projection
        self.wk = layers.Dense(d_model)  # key projection
        self.wv = layers.Dense(d_model)  # value projection

        # Output projection
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth)

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            batch_size: Batch size

        Returns:
            Tensor of shape [batch_size, num_heads, seq_len, depth]
        """
        # x: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """
        Compute scaled dot-product attention with causal masking

        Args:
            v: Value tensor
            k: Key tensor
            q: Query tensor
            mask: Attention mask

        Returns:
            Output tensor after self-attention
        """
        batch_size = tf.shape(q)[0]

        # Linear projections
        q = self.wq(q)  # (batch, seq_len, d_model)
        k = self.wk(k)  # (batch, seq_len, d_model)
        v = self.wv(v)  # (batch, seq_len, d_model)

        # Split heads
        q = self.split_heads(q, batch_size)  # (batch, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # Scale
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply mask
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply attention weights
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        # Combine heads
        output = tf.transpose(
            output, perm=[0, 2, 1, 3]
        )  # (batch, seq_len, num_heads, depth)
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))

        # Final projection
        return self.dense(concat_attention)

    def compute_mask(self, inputs, mask=None):
        # Not used, mask is passed explicitly
        return None


class FeedForward(layers.Layer):
    """
    Position-wise feed-forward network
    """

    def __init__(self, d_model, dff, dropout_rate=0.1):
        """
        Initialize feed-forward network

        Args:
            d_model: Model dimension
            dff: Feed-forward dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                layers.Dense(dff, activation="relu"),
                layers.Dense(d_model),
                layers.Dropout(dropout_rate),
            ]
        )

    def call(self, x):
        """
        Apply feed-forward network

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.seq(x)


class TransformerBlock(layers.Layer):
    """
    Transformer decoder block with self-attention and feed-forward layers
    """

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        """
        Initialize transformer block

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Feed-forward dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.att = CausalSelfAttention(d_model, num_heads)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = FeedForward(d_model, dff, dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, mask, training=False):
        """
        Apply transformer block

        Args:
            x: Input tensor
            mask: Attention mask
            training: Whether in training mode

        Returns:
            Output tensor
        """
        # Self-attention
        attn_output = self.att(x, x, x, mask)

        # Residual connection and normalization
        out1 = self.norm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)

        # Residual connection and normalization
        out2 = self.norm2(out1 + ffn_output)

        # Dropout
        return self.dropout(out2, training=training)
