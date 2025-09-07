#!/bin/bash

# Script to verify that all required files exist before training
# Usage: bash check_files.sh

echo "Checking required files for LLM from scratch project..."

# Check for data directory and files
DATA_DIR="data/wikitext-2"
REQUIRED_FILES=(
    "$DATA_DIR/train.txt"
    "$DATA_DIR/valid.txt"
    "$DATA_DIR/test.txt"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        MISSING=1
    else
        echo "✅ Found: $file ($(wc -l < "$file") lines)"
    fi
done

# Check for required Python files
PYTHON_FILES=(
    "models/llm.py"
    "models/transformer_block.py"
    "models/positional_encoding.py"
    "tokenizers/simple_tokenizer.py"
    "utils/data_utils.py"
    "train.py"
    "generate.py"
)

for file in "${PYTHON_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        MISSING=1
    else
        echo "✅ Found: $file"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo -e "\n⚠️  Some required files are missing. Please run download_wikitext.py if you're missing data files."
    exit 1
else
    echo -e "\n✅ All required files found! You can proceed with training."
    exit 0
fi
