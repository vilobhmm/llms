"""
TTS Chapter 1: Text Processing & Phoneme Tokenizer
====================================================
The text front-end converts raw text into a sequence of tokens the
acoustic model can process.  We build two levels:

  Level 1 — Character tokenizer  (simple baseline)
  Level 2 — Phoneme tokenizer    (CMU ARPAbet-style)

Pipeline:
  Raw text
      ↓
  Normalizer  (numbers → words, punctuation, unicode)
      ↓
  Grapheme-to-Phoneme (G2P)  ← rule-based or lookup
      ↓
  Token IDs  [0, 5, 12, ...]
      ↓
  Acoustic model

Why phonemes?
  "read" and "read" spell the same but sound different.
  Phonemes make the pronunciation unambiguous and reduce
  vocabulary size to ~50 symbols (ARPAbet) vs thousands of characters.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple


# ──────────────────────────────────────────────
# 1.  Text Normalizer
# ──────────────────────────────────────────────

_NUMBER_MAP = {
    "0": "zero",   "1": "one",    "2": "two",    "3": "three",
    "4": "four",   "5": "five",   "6": "six",    "7": "seven",
    "8": "eight",  "9": "nine",
}

_ORDINAL_MAP = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth",  "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth",
}

_ABBREV_MAP = {
    "dr.":   "doctor",   "mr.":  "mister", "mrs.": "misses",
    "ms.":   "miss",     "st.":  "saint",  "vs.":  "versus",
    "etc.":  "etcetera", "e.g.": "for example", "i.e.": "that is",
    "dept.": "department", "dept": "department",
}


def normalize_text(text: str) -> str:
    """
    Convert raw text to clean, speakable form:
      1. Unicode NFC
      2. Lowercase
      3. Abbreviations → full words
      4. Ordinals → spoken form
      5. Numbers → words
      6. Strip non-alphabetic except punctuation
      7. Collapse whitespace
    """
    # Unicode + lower
    text = unicodedata.normalize("NFC", text).lower()

    # Abbreviations
    for abbr, full in _ABBREV_MAP.items():
        text = text.replace(abbr, full)

    # Ordinals (e.g. 1st → first)
    for ord_, spoken in _ORDINAL_MAP.items():
        text = text.replace(ord_, spoken)

    # Multi-digit numbers  (e.g. 42 → forty two)
    text = re.sub(r"\b(\d+)\b", _expand_number, text)

    # Remove non-speakable characters (keep letters, spaces, basic punct)
    text = re.sub(r"[^a-z\s,\.!\?;\:\-\'\"]+", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_number(match: re.Match) -> str:
    """Convert a matched integer to spoken English (up to thousands)."""
    n = int(match.group())
    if n < 10:
        return _NUMBER_MAP[str(n)]
    if n < 20:
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen",
                 "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
        return teens[n - 10]
    if n < 100:
        tens  = ["", "", "twenty", "thirty", "forty", "fifty",
                 "sixty", "seventy", "eighty", "ninety"]
        t     = tens[n // 10]
        ones  = n % 10
        return t if ones == 0 else f"{t} {_NUMBER_MAP[str(ones)]}"
    if n < 1000:
        h   = f"{_NUMBER_MAP[str(n // 100)]} hundred"
        rem = n % 100
        return h if rem == 0 else f"{h} {_expand_number_int(rem)}"
    return str(n)  # fall back for large numbers

def _expand_number_int(n: int) -> str:
    """Helper for the recursive call in _expand_number."""
    class _M:
        def group(self): return str(n)
    return _expand_number(_M())


# ──────────────────────────────────────────────
# 2.  ARPAbet Phoneme Inventory
# ──────────────────────────────────────────────

# 39 ARPAbet phonemes + special tokens
PHONEMES = [
    # ── vowels ──────────────────────────────────
    "AA", "AE", "AH", "AO", "AW", "AX", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",
    # ── consonants ──────────────────────────────
    "B",  "CH", "D",  "DH", "F",  "G",  "HH",
    "JH", "K",  "L",  "M",  "N",  "NG", "P",
    "R",  "S",  "SH", "T",  "TH", "V",  "W",
    "Y",  "Z",  "ZH",
    # ── special ─────────────────────────────────
    "SIL",  # silence
    "SP",   # short pause
]

PHONEME2ID: Dict[str, int] = {
    "<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3,
    **{p: i + 4 for i, p in enumerate(PHONEMES)}
}
ID2PHONEME: Dict[int, str] = {v: k for k, v in PHONEME2ID.items()}
VOCAB_SIZE_PHONEME = len(PHONEME2ID)
PHONEME_VOCAB_SIZE = VOCAB_SIZE_PHONEME   # alias


# ──────────────────────────────────────────────
# 3.  Simple Grapheme-to-Phoneme (G2P) lookup
#     (full G2P would use a neural seq2seq model;
#      here we use a hand-crafted rule table for
#      common English words — sufficient for demos)
# ──────────────────────────────────────────────

_G2P_DICT: Dict[str, List[str]] = {
    # articles / determiners
    "a":         ["AH"],
    "an":        ["AE", "N"],
    "the":       ["DH", "AH"],
    # common verbs
    "is":        ["IH", "Z"],
    "are":       ["AA", "R"],
    "was":       ["W", "AH", "Z"],
    "be":        ["B", "IY"],
    "have":      ["HH", "AE", "V"],
    "has":       ["HH", "AE", "Z"],
    "do":        ["D", "UW"],
    "does":      ["D", "AH", "Z"],
    "can":       ["K", "AE", "N"],
    "will":      ["W", "IH", "L"],
    "would":     ["W", "UH", "D"],
    "could":     ["K", "UH", "D"],
    "should":    ["SH", "UH", "D"],
    "make":      ["M", "EY", "K"],
    "take":      ["T", "EY", "K"],
    "use":       ["Y", "UW", "Z"],
    "learn":     ["L", "ER", "N"],
    "build":     ["B", "IH", "L", "D"],
    "train":     ["T", "R", "EY", "N"],
    "speak":     ["S", "P", "IY", "K"],
    "say":       ["S", "EY"],
    "know":      ["N", "OW"],
    "think":     ["TH", "IH", "NG", "K"],
    "see":       ["S", "IY"],
    "come":      ["K", "AH", "M"],
    "give":      ["G", "IH", "V"],
    "get":       ["G", "EH", "T"],
    "go":        ["G", "OW"],
    "run":       ["R", "AH", "N"],
    # common nouns
    "model":     ["M", "AA", "D", "AH", "L"],
    "text":      ["T", "EH", "K", "S", "T"],
    "speech":    ["S", "P", "IY", "CH"],
    "voice":     ["V", "OY", "S"],
    "word":      ["W", "ER", "D"],
    "token":     ["T", "OW", "K", "AH", "N"],
    "data":      ["D", "EY", "T", "AH"],
    "network":   ["N", "EH", "T", "W", "ER", "K"],
    "layer":     ["L", "EY", "ER"],
    "attention": ["AH", "T", "EH", "N", "SH", "AH", "N"],
    "encoder":   ["EH", "N", "K", "OW", "D", "ER"],
    "decoder":   ["D", "IH", "K", "OW", "D", "ER"],
    "audio":     ["AO", "D", "IY", "OW"],
    "sound":     ["S", "AW", "N", "D"],
    "wave":      ["W", "EY", "V"],
    "signal":    ["S", "IH", "G", "N", "AH", "L"],
    "frequency": ["F", "R", "IY", "K", "W", "AH", "N", "S", "IY"],
    "mel":       ["M", "EH", "L"],
    "phoneme":   ["F", "OW", "N", "IY", "M"],
    "language":  ["L", "AE", "NG", "G", "W", "AH", "JH"],
    "hello":     ["HH", "AH", "L", "OW"],
    "world":     ["W", "ER", "L", "D"],
    "deep":      ["D", "IY", "P"],
    "learning":  ["L", "ER", "N", "IH", "NG"],
    "system":    ["S", "IH", "S", "T", "AH", "M"],
    "human":     ["HH", "Y", "UW", "M", "AH", "N"],
    "neural":    ["N", "UH", "R", "AH", "L"],
    "machine":   ["M", "AH", "SH", "IY", "N"],
    # numbers (after expansion)
    "zero":      ["Z", "IH", "R", "OW"],
    "one":       ["W", "AH", "N"],
    "two":       ["T", "UW"],
    "three":     ["TH", "R", "IY"],
    "four":      ["F", "AO", "R"],
    "five":      ["F", "AY", "V"],
    "six":       ["S", "IH", "K", "S"],
    "seven":     ["S", "EH", "V", "AH", "N"],
    "eight":     ["EY", "T"],
    "nine":      ["N", "AY", "N"],
    "ten":       ["T", "EH", "N"],
    # punctuation as pauses
    ",":         ["SP"],
    ".":         ["SIL"],
    "!":         ["SIL"],
    "?":         ["SIL"],
    ";":         ["SP"],
    ":":         ["SP"],
}


def grapheme_to_phoneme(word: str) -> List[str]:
    """
    Convert a single word to a list of ARPAbet phonemes.
    Falls back to letter-by-letter rules for unknown words.
    """
    w = word.lower().strip()
    if w in _G2P_DICT:
        return _G2P_DICT[w]
    # Fallback: naive letter-by-letter mapping
    return _letter_to_phoneme(w)


_LETTER_PHONEME: Dict[str, str] = {
    "a": "AE", "b": "B",  "c": "K",  "d": "D",  "e": "EH",
    "f": "F",  "g": "G",  "h": "HH", "i": "IH", "j": "JH",
    "k": "K",  "l": "L",  "m": "M",  "n": "N",  "o": "OW",
    "p": "P",  "q": "K",  "r": "R",  "s": "S",  "t": "T",
    "u": "UH", "v": "V",  "w": "W",  "x": "K",  "y": "Y",
    "z": "Z",
}

def _letter_to_phoneme(word: str) -> List[str]:
    return [_LETTER_PHONEME.get(c, "AH") for c in word if c.isalpha()]


# ──────────────────────────────────────────────
# 4.  Phoneme Tokenizer (text → ids)
# ──────────────────────────────────────────────

class PhonemeTokenizer:
    """
    Full pipeline: raw text → list of phoneme token IDs.

    Usage:
        tok = PhonemeTokenizer()
        ids = tok.encode("Hello world.")
        text_back = tok.phonemes_of("Hello world.")
    """

    def __init__(self):
        self.p2id = PHONEME2ID
        self.id2p = ID2PHONEME
        self.pad_id = PHONEME2ID["<PAD>"]
        self.bos_id = PHONEME2ID["<BOS>"]
        self.eos_id = PHONEME2ID["<EOS>"]
        self.unk_id = PHONEME2ID["<UNK>"]

    @property
    def vocab_size(self) -> int:
        return len(self.p2id)

    def text_to_phonemes(self, text: str) -> List[str]:
        """Normalize text and convert to phoneme sequence."""
        text    = normalize_text(text)
        phonemes: List[str] = []
        # Split on whitespace preserving punctuation as separate tokens
        tokens = re.findall(r"[a-z]+|[,\.!\?;:]", text)
        for i, tok in enumerate(tokens):
            phs = grapheme_to_phoneme(tok)
            phonemes.extend(phs)
            # Add short pause between words (not after punctuation-pause)
            if i < len(tokens) - 1 and tok.isalpha() and tokens[i+1].isalpha():
                pass  # naturally separated by phoneme boundaries
        return phonemes

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """text → list of integer token IDs."""
        phonemes = self.text_to_phonemes(text)
        ids = [self.p2id.get(p, self.unk_id) for p in phonemes]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        """IDs → list of phoneme strings."""
        return [self.id2p.get(i, "<UNK>") for i in ids
                if i not in (self.pad_id, self.bos_id, self.eos_id)]

    def pad_batch(
        self,
        sequences: List[List[int]],
        max_len: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Pad a list of sequences to the same length.
        Returns (padded_sequences, lengths).
        """
        lengths = [len(s) for s in sequences]
        T = max_len or max(lengths)
        padded = [s + [self.pad_id] * (T - len(s)) for s in sequences]
        return padded, lengths


# ──────────────────────────────────────────────
# 5.  Character Tokenizer (simpler alternative)
# ──────────────────────────────────────────────

class CharTokenizer:
    """
    Maps characters directly to IDs — simpler than phonemes but
    the model must learn pronunciation from data.
    Useful as a fast baseline.
    """
    CHARS = (
        "<PAD>", "<BOS>", "<EOS>", "<UNK>", " ",
        *"abcdefghijklmnopqrstuvwxyz",
        *".,!?;:-'\"",
    )

    def __init__(self):
        self.c2id = {c: i for i, c in enumerate(self.CHARS)}
        self.id2c = {i: c for c, i in self.c2id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.c2id)

    def encode(self, text: str, add_bos=True, add_eos=True) -> List[int]:
        text = normalize_text(text)
        ids  = [self.c2id.get(c, self.c2id["<UNK>"]) for c in text]
        if add_bos: ids = [self.c2id["<BOS>"]] + ids
        if add_eos: ids = ids + [self.c2id["<EOS>"]]
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(
            self.id2c.get(i, "?") for i in ids
            if i not in (0, 1, 2)
        )


# ──────────────────────────────────────────────
# 6.  Smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TEXT PROCESSING MODULE — Tests")
    print("=" * 60)

    # ── Normalization ─────────────────────────────────────────
    tests = [
        "Hello World!",
        "Dr. Smith has 3 cats and 42 dogs.",
        "The model was trained on 1st and 2nd datasets.",
        "Text-to-speech is fascinating technology.",
    ]
    print("\n[1] Text Normalization")
    for t in tests:
        print(f"  in : {t!r}")
        print(f"  out: {normalize_text(t)!r}")
        print()

    # ── Phoneme tokenizer ────────────────────────────────────
    print("[2] Phoneme Tokenizer")
    tok = PhonemeTokenizer()
    print(f"  Vocab size: {tok.vocab_size}")
    text = "Hello world."
    phonemes = tok.text_to_phonemes(text)
    ids      = tok.encode(text)
    decoded  = tok.decode(ids)
    print(f"  Input   : {text!r}")
    print(f"  Phonemes: {phonemes}")
    print(f"  IDs     : {ids}")
    print(f"  Decoded : {decoded}")

    # ── Padding ───────────────────────────────────────────────
    seqs   = [tok.encode("Hello."), tok.encode("Deep learning.")]
    padded, lengths = tok.pad_batch(seqs)
    print(f"\n  Padded lengths: {[len(s) for s in padded]}  |  orig: {lengths}")

    # ── Char tokenizer ────────────────────────────────────────
    print("\n[3] Character Tokenizer")
    ctok = CharTokenizer()
    print(f"  Vocab size: {ctok.vocab_size}")
    cids = ctok.encode("hello")
    print(f"  'hello' → {cids} → {ctok.decode(cids)!r}")
