# LLM From Scratch

This repository contains code to build a Large Language Model (LLM) from scratch using TensorFlow and Keras. The implementation is based on the decoder-only Transformer architecture.

## Project Structure

```
llm-from-scratch/
├── models/                 # Model architecture components
│   ├── layers/             # Reusable layer components
│   ├── positional_encoding.py
│   ├── transformer_block.py
│   └── llm.py              # Main LLM model
├── tokenizers/             # Tokenization utilities
│   └── simple_tokenizer.py # Simple subword tokenizer
├── utils/                  # Utility functions
│   ├── data_utils.py       # Data loading and processing
│   └── inference_utils.py  # Text generation utilities
├── data/                   # Dataset storage
│   ├── raw/                # Raw downloaded data
│   └── processed/          # Preprocessed data
├── checkpoints/            # Saved model checkpoints
├── download_wikitext.py    # Script to download and preprocess WikiText-2
├── train.py               # Training script
├── generate.py             # Text generation script
├── check_files.sh         # Script to verify all required files exist
├── requirements.txt        # Python dependencies
└── run_all.sh              # All-in-one setup and training script
```

## Quick Start

To set up the environment, download the data, and train the model, run:

```bash
./run_all.sh
```

This script will:
1. Create a virtual environment and install dependencies
2. Download and preprocess the WikiText-2 dataset
3. Train the model with default parameters
4. Save checkpoints for later use

## Manual Setup

If you prefer to run things step by step:

1. Set up the environment:
   ```bash
   pip install virtualenv
   virtualenv llm_env
   source llm_env/bin/activate
   pip install -r requirements.txt
   ```

2. Download and preprocess the WikiText-2 dataset:
   ```bash
   python download_wikitext.py
   ```

3. Verify that all required files are present:
   ```bash
   ./check_files.sh
   ```

4. Train the model:
   ```bash
   python train.py
   ```
   
   You can customize training with various arguments:
   ```bash
   python train.py --epochs 5 --d_model 256 --num_heads 8
   ```

5. Generate text with the trained model:
   ```bash
   python generate.py --prompt "Once upon a time"
   ```

## Model Architecture

The LLM implementation is based on the decoder-only Transformer architecture, similar to models like GPT. Key components include:

- Token and positional embeddings
- Multi-head self-attention with causal masking
- Position-wise feed-forward networks
- Layer normalization and residual connections
- Text generation with various decoding strategies:
  - Greedy search
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Beam search

## Training Parameters

Default model parameters:
- Vocabulary size: 10,000
- Sequence length: 64
- Model dimension: 256
- Number of heads: 8
- Feed-forward dimension: 512
- Number of layers: 4
- Dropout rate: 0.1

You can adjust these parameters in the training script.

## License

[MIT License](LICENSE)
