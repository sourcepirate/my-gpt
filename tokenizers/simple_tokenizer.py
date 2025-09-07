import re
from typing import List, Dict, Optional


class SimpleSubwordTokenizer:
    """
    A simple subword tokenizer using byte-pair encoding (BPE)-like merges.
    For demonstration purposes. For production, use KerasNLP or HuggingFace tokenizers.
    """

    def __init__(
        self, vocab_size=10000, reserved_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    ):
        self.vocab_size = vocab_size
        self.reserved_tokens = reserved_tokens
        self.vocab = None
        self.token2id = None
        self.id2token = None

    def fit(self, texts: List[str]):
        """
        Build vocabulary from list of texts
        """
        # Basic whitespace tokenization
        tokens = []
        for text in texts:
            tokens.extend(re.findall(r"\w+|[^\w\s]", text.lower()))
        # Count token frequencies
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        # Build vocab
        sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])
        vocab_tokens = self.reserved_tokens + [
            t for t, _ in sorted_tokens[: self.vocab_size - len(self.reserved_tokens)]
        ]
        self.vocab = vocab_tokens
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}

    def save_vocab(self, filepath: str) -> None:
        """
        Save vocabulary to file
        """
        if self.vocab is None:
            raise ValueError("Tokenizer must be fit before saving vocabulary")
        with open(filepath, "w", encoding="utf-8") as f:
            for token in self.vocab:
                f.write(f"{token}\n")

    def load_vocab(self, filepath: str) -> None:
        """
        Load vocabulary from file
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.vocab = [line.strip() for line in f]
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs
        """
        if self.vocab is None:
            raise ValueError("Tokenizer must be fit before encoding")
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        return [self.token2id.get(t, self.token2id["<UNK>"]) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to text
        """
        if self.vocab is None:
            raise ValueError("Tokenizer must be fit before decoding")
        return " ".join([self.id2token.get(i, "<UNK>") for i in ids])
