"""
Chapter 1: Data Cleaning & BPE Tokenizer
=========================================
Covers:
  - Raw text normalization (unicode, whitespace, special chars)
  - Byte-Pair Encoding (BPE) tokenizer built from scratch
  - Vocabulary construction
  - Encode / decode round-trip
  - Sliding-window dataset for language-model pre-training
"""

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────
# 1.  Text normalisation helpers
# ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Light normalisation that preserves semantic meaning:
      1. Unicode NFC (combine accented chars)
      2. Replace multiple whitespace with single space
      3. Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_corpus(texts: List[str]) -> List[str]:
    """Remove empty strings, deduplicate, and normalise each line."""
    seen = set()
    cleaned = []
    for t in texts:
        t = normalize_text(t)
        if t and t not in seen:
            seen.add(t)
            cleaned.append(t)
    return cleaned


# ──────────────────────────────────────────────
# 2.  BPE Tokenizer  (built from scratch)
# ──────────────────────────────────────────────

class BPETokenizer:
    """
    Byte-Pair Encoding tokeniser following the original Sennrich et al. (2016)
    algorithm, extended with special tokens.

    Training:
        1. Start with character-level vocabulary (bytes).
        2. Count every adjacent symbol-pair in the corpus.
        3. Merge the most-frequent pair, add it to vocab.
        4. Repeat until vocab_size reached.

    Encoding:
        Apply learned merges greedily, left-to-right.
    """

    # Special tokens
    PAD_TOKEN   = "<|pad|>"
    UNK_TOKEN   = "<|unk|>"
    BOS_TOKEN   = "<|endoftext|>"   # used as both BOS / EOS  (GPT-2 convention)
    EOS_TOKEN   = "<|endoftext|>"

    def __init__(self):
        self.vocab:        Dict[str, int] = {}   # token  → id
        self.inv_vocab:    Dict[int, str] = {}   # id     → token
        self.merges:       List[Tuple[str, str]] = []
        self.merge_rank:   Dict[Tuple[str, str], int] = {}

    # ── training ──────────────────────────────

    def train(self, corpus: List[str], vocab_size: int = 500) -> None:
        """Learn BPE merges from a list of strings."""
        # Initialise with byte-level characters + special tokens
        special = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN]
        vocab_set = set(special)
        for text in corpus:
            vocab_set.update(list(text))
        self.vocab = {tok: i for i, tok in enumerate(sorted(vocab_set))}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

        # Represent the corpus as lists-of-character-lists with end-of-word marker
        word_freqs = Counter()
        for text in corpus:
            # split on whitespace; mark end-of-word with "Ġ" prefix on next char
            for word in text.split():
                word_freqs["Ġ" + word] += 1     # GPT-2 convention: Ġ = space prefix

        # Represent words as tuples of symbols
        splits: Dict[str, List[str]] = {
            word: list(word) for word in word_freqs
        }

        num_merges = vocab_size - len(self.vocab)
        for _ in range(max(0, num_merges)):
            pair_freqs = self._count_pairs(splits, word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            splits = self._merge_pair(best_pair, splits)
            # Add merged token to vocab
            merged = "".join(best_pair)
            if merged not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[merged] = new_id
                self.inv_vocab[new_id] = merged
            self.merges.append(best_pair)
            self.merge_rank[best_pair] = len(self.merges) - 1

    @staticmethod
    def _count_pairs(
        splits: Dict[str, List[str]],
        word_freqs: Counter
    ) -> Counter:
        pair_freqs: Counter = Counter()
        for word, freq in word_freqs.items():
            syms = splits[word]
            for a, b in zip(syms, syms[1:]):
                pair_freqs[(a, b)] += freq
        return pair_freqs

    @staticmethod
    def _merge_pair(
        pair: Tuple[str, str],
        splits: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        a, b = pair
        merged = a + b
        new_splits = {}
        for word, syms in splits.items():
            new_syms, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            new_splits[word] = new_syms
        return new_splits

    # ── encoding / decoding ──────────────────

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply learned merges to a single word."""
        syms = list(word)
        while True:
            pairs = list(zip(syms, syms[1:]))
            # find the pair with lowest merge rank (earliest learned)
            best = min(
                pairs,
                key=lambda p: self.merge_rank.get(p, float("inf")),
                default=None
            )
            if best is None or best not in self.merge_rank:
                break
            a, b = best
            merged = a + b
            new_syms, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms
        return syms

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        unk_id = self.vocab.get(self.UNK_TOKEN, 0)
        words = text.split()
        for i, word in enumerate(words):
            prefixed = ("Ġ" if i > 0 else "") + word
            for sym in self._tokenize_word(prefixed):
                tokens.append(self.vocab.get(sym, unk_id))
        return tokens

    def decode(self, ids: List[int]) -> str:
        text = "".join(self.inv_vocab.get(i, self.UNK_TOKEN) for i in ids)
        # Remove the Ġ prefix used internally (replace with space)
        text = text.replace("Ġ", " ").strip()
        return text

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# ──────────────────────────────────────────────
# 3.  Sliding-window Dataset
# ──────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """
    Converts a flat list of token-ids into (input, target) pairs using a
    sliding window of length `context_length`.

    Example (context_length=4, stride=1):
      tokens = [1, 2, 3, 4, 5]
      → input=[1,2,3,4], target=[2,3,4,5]
    """

    def __init__(
        self,
        token_ids: List[int],
        context_length: int = 128,
        stride: int = 1,
    ):
        self.input_ids  = []
        self.target_ids = []
        for start in range(0, len(token_ids) - context_length, stride):
            chunk   = token_ids[start : start + context_length + 1]
            self.input_ids.append(torch.tensor(chunk[:-1], dtype=torch.long))
            self.target_ids.append(torch.tensor(chunk[1:],  dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text: str,
    tokenizer: BPETokenizer,
    context_length: int = 128,
    stride: int = 64,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    token_ids = tokenizer.encode(text)
    dataset   = SlidingWindowDataset(token_ids, context_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# ──────────────────────────────────────────────
# 4.  Quick smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    sample_corpus = [
        "the cat sat on the mat",
        "the dog lay on the rug",
        "a large language model learns from text",
        "transformers use attention mechanisms",
        "byte pair encoding builds subword vocabulary",
        "natural language processing is fascinating",
        "deep learning models require large datasets",
        "the quick brown fox jumps over the lazy dog",
    ]

    print("=" * 60)
    print("STEP 1 — Data Cleaning")
    print("=" * 60)
    cleaned = clean_corpus(sample_corpus)
    for line in cleaned:
        print(f"  ✓ {line!r}")

    print("\n" + "=" * 60)
    print("STEP 2 — BPE Tokenizer Training")
    print("=" * 60)
    tok = BPETokenizer()
    tok.train(cleaned, vocab_size=200)
    print(f"  Vocab size: {tok.vocab_size}")
    print(f"  Merges learned: {len(tok.merges)}")
    print(f"  Sample vocab (first 20): {list(tok.vocab.items())[:20]}")

    print("\n" + "=" * 60)
    print("STEP 3 — Encode / Decode Round-trip")
    print("=" * 60)
    test_text = "the cat learned attention"
    ids = tok.encode(test_text)
    decoded = tok.decode(ids)
    print(f"  Input  : {test_text!r}")
    print(f"  Token IDs: {ids}")
    print(f"  Decoded: {decoded!r}")

    print("\n" + "=" * 60)
    print("STEP 4 — Sliding-Window Dataset")
    print("=" * 60)
    full_text = " ".join(cleaned)
    dl = create_dataloader(full_text, tok, context_length=8, stride=4, batch_size=2)
    x, y = next(iter(dl))
    print(f"  Batch input  shape : {x.shape}")
    print(f"  Batch target shape : {y.shape}")
    print(f"  Input  tokens: {x[0].tolist()}")
    print(f"  Target tokens: {y[0].tolist()}")
    print("\nData pipeline ready!")
