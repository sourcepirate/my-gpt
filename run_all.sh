#!/bin/bash

# Exit on any error
set -e

# Function to display steps with formatting
print_step() {
    echo -e "\n\033[1;34m==>\033[0m \033[1m$1\033[0m"
}

print_success() {
    echo -e "\033[1;32m✓ $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m✗ $1\033[0m"
}

handle_error() {
    print_error "An error occurred during: $1"
    print_error "Check the error message above for details."
    exit 1
}

# Create and activate virtual environment
print_step "Setting up Python environment"
pip install virtualenv || handle_error "Installing virtualenv"
virtualenv llm_env || handle_error "Creating virtual environment"
source llm_env/bin/activate || handle_error "Activating virtual environment"
print_success "Virtual environment activated"

# Install dependencies
print_step "Installing dependencies"
pip install --upgrade pip || handle_error "Upgrading pip"
pip install -r requirements.txt || handle_error "Installing required packages"
print_success "All dependencies installed"

# Create necessary directories
print_step "Creating directories"
mkdir -p data/raw data/processed checkpoints logs
print_success "Directory structure created"

# Download and preprocess WikiText-2 dataset
print_step "Downloading and preprocessing WikiText-2 dataset"
python download_wikitext.py || handle_error "Downloading and preprocessing data"
print_success "Dataset downloaded and preprocessed"

# Check if all required files exist
print_step "Verifying all required files"
./check_files.sh
if [ $? -ne 0 ]; then
    print_error "Some required files are missing. Please check the messages above."
    exit 1
fi
print_success "All required files verified"

# Train the model
print_step "Starting model training"
python train.py --epochs 3 --d_model 256 --num_heads 8 --num_layers 4 --max_chars 500000 || handle_error "Training the model"
print_success "Model training completed"

echo ""
echo -e "\033[1;32m✓ Setup and training completed successfully!\033[0m"
echo -e "\033[1m"
echo "You can generate text using: python generate.py --prompt \"Your prompt here\""
echo ""
echo "To activate this environment in the future, run:"
echo "source llm_env/bin/activate"
echo -e "\033[0m"
