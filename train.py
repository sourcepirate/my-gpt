#!/usr/bin/env python
"""
Train the LLM with the WikiText-2 dataset.
"""
import os
import time
import tensorflow as tf
import numpy as np
import logging
import argparse
from datetime import datetime

# Import from refactored modules
from models.llm import LLM
from tokenizers.simple_tokenizer import SimpleSubwordTokenizer
from utils.data_utils import load_and_prepare_data
from utils.inference_utils import greedy_search, top_p_sampling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("wikitext_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Custom callback for text generation during training
class TextGenerationCallback(tf.keras.callbacks.Callback):
    def __init__(self, tokenizer, prompts, seq_len, generation_frequency=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.seq_len = seq_len
        self.generation_frequency = generation_frequency

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.generation_frequency == 0:
            logger.info(f"\n----- Generating sample text at epoch {epoch+1} -----")
            for i, prompt in enumerate(self.prompts):
                logger.info(f"Prompt {i+1}: {prompt}")
                # Generate text using greedy search
                greedy_text = greedy_search(
                    self.model, self.tokenizer, prompt, self.seq_len, max_gen=30
                )
                logger.info(f"Greedy: {greedy_text}")
                # Generate text using top-p sampling
                sampled_text = top_p_sampling(
                    self.model, self.tokenizer, prompt, self.seq_len, p=0.9, max_gen=30
                )
                logger.info(f"Top-p: {sampled_text}")
            logger.info("-----------------------------------------")


# Custom loss function with masking
def loss_function(real, pred):
    # Create a mask to ignore padded tokens
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(
        real, pred, from_logits=True
    )
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def load_wikitext(file_path, max_chars=None):
    """
    Load and clean WikiText data

    Args:
        file_path: Path to WikiText file
        max_chars: Maximum number of characters to load (optional)

    Returns:
        Cleaned text
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if max_chars:
            text = text[:max_chars]

        return text
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Could not find the data file at {file_path}")
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise


def main(args):
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create logs directory with timestamp
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    # Create data directories
    os.makedirs("data/processed", exist_ok=True)

    # Load WikiText data
    logger.info("Loading WikiText data...")

    try:
        # Try to load processed data first
        train_text = load_wikitext(args.train_file, args.max_chars)
        val_text = load_wikitext(args.val_file, args.max_chars // 5)
    except FileNotFoundError:
        logger.info(
            "Processed data not found. Please download and process the WikiText-2 dataset first."
        )
        return

    logger.info(f"Training data: {len(train_text)} characters")
    logger.info(f"Validation data: {len(val_text)} characters")

    # Initialize tokenizer
    tokenizer = SimpleSubwordTokenizer(vocab_size=args.vocab_size)

    # Prepare datasets
    logger.info("Tokenizing and preparing datasets...")

    # Fit tokenizer on training data
    tokenizer.fit([train_text])

    # Save tokenizer vocabulary
    vocab_path = os.path.join(args.checkpoint_dir, "tokenizer_vocab.txt")
    tokenizer.save_vocab(vocab_path)
    logger.info(f"Tokenizer vocabulary saved to {vocab_path}")

    # Prepare datasets
    train_dataset = load_and_prepare_data(
        args.train_file,
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.max_chars,
        args.shuffle_buffer,
    )
    val_dataset = load_and_prepare_data(
        args.val_file,
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.max_chars // 5,
        args.shuffle_buffer // 2,
    )

    # Build model
    logger.info("Building model...")
    model = LLM(
        args.vocab_size,
        args.seq_len,
        args.d_model,
        args.num_heads,
        args.dff,
        args.num_layers,
        args.dropout_rate,
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss_function,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Sample prompts for generation
    generation_prompts = ["The quick brown fox", "Once upon a time", "In the beginning"]

    # Callbacks
    callbacks = [
        # Save model checkpoints
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.checkpoint_dir, "llm_epoch_{epoch:02d}.weights.h5"),
            save_weights_only=True,
            save_freq="epoch",
        ),
        # Best model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.checkpoint_dir, "llm_best.weights.h5"),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, monitor="val_loss", verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        # Text generation during training
        TextGenerationCallback(tokenizer, generation_prompts, args.seq_len),
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1
        ),
        # CSV logger
        tf.keras.callbacks.CSVLogger("wikitext_training_log.csv"),
    ]

    # Print model summary
    model.build((None, args.seq_len))
    model.summary()

    # Log the start of training
    logger.info(f"Starting training with parameters:")
    logger.info(
        f"vocab_size={args.vocab_size}, seq_len={args.seq_len}, d_model={args.d_model}"
    )
    logger.info(
        f"num_heads={args.num_heads}, dff={args.dff}, num_layers={args.num_layers}"
    )
    logger.info(f"batch_size={args.batch_size}, epochs={args.epochs}")
    logger.info(f"learning_rate={args.learning_rate}")

    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Log training completion
    logger.info("Training completed!")

    # Save final model weights
    final_model_path = os.path.join(args.checkpoint_dir, "llm_final.weights.h5")
    model.save_weights(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Language Model on WikiText-2")

    # Data parameters
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/train.txt",
        help="Path to training data file",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/processed/valid.txt",
        help="Path to validation data file",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1000000,
        help="Maximum number of characters to load",
    )
    parser.add_argument(
        "--shuffle_buffer", type=int, default=10000, help="Size of shuffle buffer"
    )

    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--dff", type=int, default=512, help="Feed-forward dimension")
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()

    main(args)
