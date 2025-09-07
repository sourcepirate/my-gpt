#!/usr/bin/env python
"""
Generate text with a trained language model
"""
import argparse
import os
import tensorflow as tf
import numpy as np

from models.llm import LLM
from tokenizers.simple_tokenizer import SimpleSubwordTokenizer
from utils.inference_utils import (
    greedy_search,
    top_k_sampling,
    top_p_sampling,
    beam_search,
)


def generate_text(args):
    """
    Generate text using a trained language model
    """
    # Load tokenizer vocabulary
    tokenizer = SimpleSubwordTokenizer(vocab_size=args.vocab_size)
    tokenizer.load_vocab(args.vocab_path)

    # Build model with same architecture
    model = LLM(
        args.vocab_size,
        args.seq_len,
        args.d_model,
        args.num_heads,
        args.dff,
        args.num_layers,
        args.dropout_rate,
    )
    model.build((None, args.seq_len))

    # Load model weights
    model.load_weights(args.model_path)
    print(f"Model loaded from {args.model_path}")

    # Generate text in interactive mode
    if args.interactive:
        while True:
            prompt = input("\nEnter a prompt (or 'q' to quit): ")
            if prompt.lower() == "q":
                break

            print(f"Generating text from: '{prompt}'")

            if args.method == "greedy":
                generated = greedy_search(
                    model, tokenizer, prompt, args.seq_len, max_gen=args.max_gen
                )
            elif args.method == "topk":
                generated = top_k_sampling(
                    model,
                    tokenizer,
                    prompt,
                    args.seq_len,
                    k=args.top_k,
                    max_gen=args.max_gen,
                    temperature=args.temperature,
                )
            elif args.method == "topp":
                generated = top_p_sampling(
                    model,
                    tokenizer,
                    prompt,
                    args.seq_len,
                    p=args.top_p,
                    max_gen=args.max_gen,
                    temperature=args.temperature,
                )
            elif args.method == "beam":
                generated = beam_search(
                    model,
                    tokenizer,
                    prompt,
                    args.seq_len,
                    beam_width=args.beam_width,
                    max_gen=args.max_gen,
                )
            else:
                raise ValueError(f"Unknown generation method: {args.method}")

            print(f"\nGenerated text:\n{generated}")
    else:
        # Generate from specified prompt
        if args.method == "greedy":
            generated = greedy_search(
                model, tokenizer, args.prompt, args.seq_len, max_gen=args.max_gen
            )
        elif args.method == "topk":
            generated = top_k_sampling(
                model,
                tokenizer,
                args.prompt,
                args.seq_len,
                k=args.top_k,
                max_gen=args.max_gen,
                temperature=args.temperature,
            )
        elif args.method == "topp":
            generated = top_p_sampling(
                model,
                tokenizer,
                args.prompt,
                args.seq_len,
                p=args.top_p,
                max_gen=args.max_gen,
                temperature=args.temperature,
            )
        elif args.method == "beam":
            generated = beam_search(
                model,
                tokenizer,
                args.prompt,
                args.seq_len,
                beam_width=args.beam_width,
                max_gen=args.max_gen,
            )
        else:
            raise ValueError(f"Unknown generation method: {args.method}")

        print(f"Prompt: {args.prompt}")
        print(f"Generated text: {generated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with a trained language model"
    )

    # Model parameters (must match training)
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--dff", type=int, default=512, help="Feed-forward dimension")
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")

    # Generation parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/llm_final.weights.h5",
        help="Path to model weights",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="checkpoints/tokenizer_vocab.txt",
        help="Path to tokenizer vocabulary",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["greedy", "topk", "topp", "beam"],
        default="topp",
        help="Generation method",
    )
    parser.add_argument(
        "--max_gen", type=int, default=100, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument("--beam_width", type=int, default=5, help="Beam search width")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive generation mode"
    )

    args = parser.parse_args()

    generate_text(args)
