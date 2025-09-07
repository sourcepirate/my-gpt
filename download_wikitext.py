#!/usr/bin/env python
"""
Download and preprocess the WikiText-2 dataset.
"""
import os
import re
import argparse
import urllib.request
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("wikitext_preprocessing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def download_wikitext(base_url, output_dir):
    """
    Download WikiText-2 dataset files

    Args:
        base_url: Base URL for WikiText-2 files
        output_dir: Directory to save raw files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Files to download
    files = ["train.txt", "valid.txt", "test.txt"]

    for file in files:
        output_path = os.path.join(output_dir, file)

        if os.path.exists(output_path):
            logger.info(f"File {file} already exists, skipping download.")
            continue

        url = f"{base_url}/{file}"
        logger.info(f"Downloading {url} to {output_path}")

        try:
            urllib.request.urlretrieve(url, output_path)
            logger.info(f"Successfully downloaded {file}")
        except Exception as e:
            logger.error(f"Failed to download {file}: {e}")


def clean_wikitext(input_path, output_path):
    """
    Clean WikiText data by removing article titles and extra whitespace

    Args:
        input_path: Path to input file
        output_path: Path to output file
    """
    logger.info(f"Cleaning {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove article titles (lines starting with = and ending with =)
    text = re.sub(r"= [^=]+ =", "", text)

    # Normalize whitespace (but preserve paragraph structure)
    text = re.sub(r" +", " ", text)

    # Write cleaned text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"Cleaned text saved to {output_path}")

    # Log statistics
    original_size = os.path.getsize(input_path) / 1024
    cleaned_size = os.path.getsize(output_path) / 1024
    logger.info(f"Original size: {original_size:.2f} KB")
    logger.info(f"Cleaned size: {cleaned_size:.2f} KB")
    logger.info(f"Reduction: {(1 - cleaned_size / original_size) * 100:.2f}%")


def main(args):
    # Create directories
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)

    # Download WikiText-2 dataset
    logger.info("Downloading WikiText-2 dataset...")
    download_wikitext(args.url, args.raw_dir)

    # Process each file
    for file in ["train.txt", "valid.txt", "test.txt"]:
        input_path = os.path.join(args.raw_dir, file)
        output_path = os.path.join(args.processed_dir, file)

        if os.path.exists(input_path):
            clean_wikitext(input_path, output_path)
        else:
            logger.warning(f"Input file {input_path} not found, skipping.")

    logger.info("WikiText-2 preprocessing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and preprocess the WikiText-2 dataset"
    )

    parser.add_argument(
        "--url",
        type=str,
        default="https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2",
        help="Base URL for WikiText-2 files",
    )
    parser.add_argument(
        "--raw_dir", type=str, default="data/raw", help="Directory to save raw files"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed files",
    )

    args = parser.parse_args()

    main(args)
