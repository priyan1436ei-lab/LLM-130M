"""Byte-Level BPE Tokenizer for Privan-130M."""

import os
import json
from pathlib import Path
from typing import List, Union, Optional

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor


class ByteLevelBPETokenizer:
    """
    Byte-Level BPE Tokenizer wrapper.

    Provides tokenization compatible with GPT-style models, including
    special tokens (<BOS>, <EOS>, <PAD>, <UNK>).
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self, tokenizer: Optional[HFTokenizer] = None):
        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = HFTokenizer(BPE(unk_token="<UNK>"))
            self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self._tokenizer.decoder = ByteLevelDecoder()
            self._tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

    @classmethod
    def train_from_files(
        cls,
        files: List[str | Path],
        vocab_size: int = 50257,
        min_frequency: int = 2,
    ) -> "ByteLevelBPETokenizer":
        """Train a new Byte-Level BPE tokenizer from raw text files."""
        tokenizer = HFTokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls.SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
        )

        file_paths = [str(f) for f in files]
        tokenizer.train(files=file_paths, trainer=trainer)
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

        return cls(tokenizer)

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs."""
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode a list of token IDs back into string."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of strings."""
        encodings = self._tokenizer.encode_batch(texts)
        return [enc.ids for enc in encodings]

    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        """Get ID of a specific token string."""
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Get token string from ID."""
        return self._tokenizer.id_to_token(token_id)

    @property
    def pad_token_id(self) -> int:
        tid = self.token_to_id("<PAD>")
        return tid if tid is not None else 0

    @property
    def unk_token_id(self) -> int:
        tid = self.token_to_id("<UNK>")
        return tid if tid is not None else 1

    @property
    def bos_token_id(self) -> int:
        tid = self.token_to_id("<BOS>")
        return tid if tid is not None else 2

    @property
    def eos_token_id(self) -> int:
        tid = self.token_to_id("<EOS>")
        return tid if tid is not None else 3

    def save(self, directory: str | Path) -> None:
        """Save tokenizer configuration and model files to a directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        tokenizer_file = dir_path / "tokenizer.json"
        self._tokenizer.save(str(tokenizer_file))

        metadata = {
            "vocab_size": self.vocab_size(),
            "special_tokens": {
                "<PAD>": self.pad_token_id,
                "<UNK>": self.unk_token_id,
                "<BOS>": self.bos_token_id,
                "<EOS>": self.eos_token_id,
            },
        }
        with open(dir_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "ByteLevelBPETokenizer":
        """Load tokenizer from directory containing tokenizer.json."""
        dir_path = Path(directory)
        tokenizer_file = dir_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_file}")
        hf_tokenizer = HFTokenizer.from_file(str(tokenizer_file))
        return cls(hf_tokenizer)
