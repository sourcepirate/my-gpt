#!/bin/bash
echo "Cleaning up redundant files..."

# List of files to remove
files_to_remove=(
  "data_preparation.py"
  "inference.py"
  "llm_model.py"
  "positional_encoding.py"
  "transformer_block.py"
  "train.py"
  "generate_text.py"
  "prepare_wikitext.py"
  "preprocess_wikitext.py"
  "setup_and_train.sh"
  "train_wikitext.py"
  "train_with_wikitext.sh"
)

# Remove each file if it exists
for file in "${files_to_remove[@]}"; do
  if [ -f "$file" ]; then
    echo "Removing $file"
    rm "$file"
  else
    echo "$file not found, skipping"
  fi
done

# Rename the refactored training script to a simpler name
if [ -f "train_wikitext_refactored.py" ]; then
  echo "Renaming train_wikitext_refactored.py to train.py"
  mv train_wikitext_refactored.py train.py
fi

echo "Cleanup complete!"
