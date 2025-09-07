import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict


def create_sequences(
    token_ids: List[int], seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-target pairs for language modeling

    Args:
        token_ids: List of token IDs
        seq_len: Length of each sequence

    Returns:
        Tuple of (inputs, targets) arrays
    """
    inputs, targets = [], []
    for i in range(len(token_ids) - seq_len - 1):
        # Input is seq_len tokens
        inp_seq = token_ids[i : i + seq_len]
        # Target is seq_len tokens, shifted by one position
        target_seq = token_ids[i + 1 : i + seq_len + 1]
        inputs.append(inp_seq)
        targets.append(target_seq)
    return np.array(inputs), np.array(targets)


def prepare_dataset(
    texts: List[str],
    tokenizer,
    seq_len: int,
    batch_size: int,
    shuffle_buffer: int = 10000,
):
    """
    Prepare a TensorFlow dataset from texts

    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance with encode method
        seq_len: Length of each sequence
        batch_size: Batch size
        shuffle_buffer: Size of shuffle buffer

    Returns:
        TensorFlow dataset
    """
    # Encode all texts and flatten
    token_ids = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text))

    # Create input-output pairs
    X, y = create_sequences(token_ids, seq_len)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size, drop_remainder=True)
    return dataset


def load_and_prepare_data(
    file_path: str,
    tokenizer,
    seq_len: int,
    batch_size: int,
    max_chars: int = None,
    shuffle_buffer: int = 10000,
):
    """
    Load text from file, tokenize and prepare dataset

    Args:
        file_path: Path to text file
        tokenizer: Tokenizer instance
        seq_len: Length of each sequence
        batch_size: Batch size
        max_chars: Maximum number of characters to load (for memory management)
        shuffle_buffer: Size of shuffle buffer

    Returns:
        TensorFlow dataset
    """
    # Load text data
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Optionally truncate to manage memory
    if max_chars is not None:
        text = text[:max_chars]

    # Tokenize and prepare dataset
    token_ids = tokenizer.encode(text)

    # Create input-output pairs
    X, y = create_sequences(token_ids, seq_len)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size, drop_remainder=True)

    return dataset
