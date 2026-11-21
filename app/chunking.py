"""Sentence-aware chunking with overlap."""
from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends and blank lines; collapse internal whitespace."""
    parts = re.split(r"(?<=[.!?])\s+|\n\s*\n", text.strip())
    cleaned = (re.sub(r"\s+", " ", p).strip() for p in parts if p)
    return [p for p in cleaned if p]


def chunk_text(text: str, max_chars: int = 500, overlap: int = 80) -> list[str]:
    """Pack sentences into chunks of at most ``max_chars`` characters.

    Consecutive chunks share up to ``overlap`` characters of trailing sentences so
    an answer that straddles a boundary is still retrievable.
    """
    if max_chars < 50:
        raise ValueError("max_chars must be at least 50")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    pieces: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
        else:  # very long "sentence": hard-split it
            step = max_chars - overlap
            pieces.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), step))

    chunks: list[str] = []
    current: list[str] = []
    fresh = 0  # pieces added since the last emitted chunk (excludes the overlap tail)
    for piece in pieces:
        joined_len = len(" ".join(current))
        if current and joined_len + 1 + len(piece) > max_chars:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_len = 0
            for sentence in reversed(current):
                extra = len(sentence) + (1 if tail else 0)
                if tail_len + extra > overlap:
                    break
                tail.insert(0, sentence)
                tail_len += extra
            current = tail
            fresh = 0
            if current and len(" ".join(current)) + 1 + len(piece) > max_chars:
                current = []
        current.append(piece)
        fresh += 1
    if current and fresh > 0:
        chunks.append(" ".join(current))
    return chunks
