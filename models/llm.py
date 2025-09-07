import tensorflow as tf
from tensorflow.keras import layers, Model
from models.positional_encoding import PositionalEncoding
from models.transformer_block import TransformerBlock, create_causal_mask


class LLM(Model):
    """
    Decoder-only Transformer Language Model
    """

    def __init__(
        self, vocab_size, seq_len, d_model, num_heads, dff, num_layers, dropout_rate=0.1
    ):
        """
        Initialize the LLM

        Args:
            vocab_size: Size of vocabulary
            seq_len: Maximum sequence length
            d_model: Model dimension
            num_heads: Number of attention heads
            dff: Feed-forward dimension
            num_layers: Number of transformer layers
            dropout_rate: Dropout rate
        """
        super().__init__()

        # Token embedding
        self.embedding = layers.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(seq_len, d_model)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        # Final normalization and output
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.final_dense = layers.Dense(vocab_size)  # Projection to vocabulary

        # Store sequence length for masking
        self.seq_len = seq_len

    def call(self, x, training=False):
        """
        Forward pass through the model

        Args:
            x: Input tensor of shape [batch, seq_len]
            training: Whether in training mode

        Returns:
            Output logits of shape [batch, seq_len, vocab_size]
        """
        # Create causal mask
        mask = create_causal_mask(self.seq_len)
        mask = mask[tf.newaxis, tf.newaxis, :, :]  # (1, 1, seq_len, seq_len)

        # Token embedding
        x = self.embedding(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)  # (batch, seq_len, d_model)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask, training=training)

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary
        logits = self.final_dense(x)  # (batch, seq_len, vocab_size)

        return logits

    def generate(self, prompt_ids, max_len, temperature=1.0, top_k=0, top_p=0.9):
        """
        Generate text from a prompt

        Args:
            prompt_ids: Prompt token IDs
            max_len: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter (0 to disable)
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Generated token IDs
        """
        input_ids = prompt_ids.copy()

        for _ in range(max_len):
            # Truncate to seq_len
            curr_input = input_ids[-self.seq_len :]

            # Pad if needed
            if len(curr_input) < self.seq_len:
                padding = [0] * (self.seq_len - len(curr_input))
                curr_input = padding + curr_input

            # Convert to tensor
            x = tf.convert_to_tensor([curr_input], dtype=tf.int32)

            # Get logits
            logits = self(x, training=False)
            logits = logits[0, -1, :]  # Last token logits

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Sample from distribution
            if top_p > 0:
                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = tf.nn.top_k(
                    logits, k=tf.shape(logits)[0]
                )
                cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits), axis=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = tf.concat(
                    [[False], sorted_indices_to_remove[:-1]], axis=0
                )
                indices_to_keep = tf.boolean_mask(
                    sorted_indices, ~sorted_indices_to_remove
                )
                next_token_logits = tf.gather(logits, indices_to_keep)
                next_token_id = indices_to_keep[
                    tf.random.categorical(tf.expand_dims(next_token_logits, 0), 1)[0, 0]
                ]
            elif top_k > 0:
                # Top-k sampling
                top_k_logits, top_k_indices = tf.nn.top_k(logits, k=top_k)
                next_token_id = top_k_indices[
                    tf.random.categorical(tf.expand_dims(top_k_logits, 0), 1)[0, 0]
                ]
            else:
                # Full sampling
                next_token_id = tf.random.categorical(tf.expand_dims(logits, 0), 1)[
                    0, 0
                ]

            # Add to sequence
            input_ids.append(int(next_token_id))

            # Stop if end token reached
            if int(next_token_id) == 1:  # Assuming 1 is <EOS>
                break

        return input_ids
