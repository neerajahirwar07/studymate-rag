"""Text analysis shared by the embedder, BM25 and the extractive answerer."""
from __future__ import annotations

import re

# Generic English words plus generic question words ("explain", "which", ...).
STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from as into onto over
    is are was were be been being am do does did done doing have has had having
    it its this that these those there here i me my we our you your they their them he she his her
    what which who whom whose when where why how can could should would will shall may might must
    not no nor so than too very just also only more most some any each every other such own same
    about above below up down out off again further once both few all
    explain describe define tell give show list state mention write
    work works
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """A tiny suffix stripper, good enough to match sort/sorted/sorting."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith(("sses", "shes", "ches", "xes", "zes")):
        token = token[:-2]
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    for suffix in ("ation", "ing", "ed", "ly"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def analyze(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, stem."""
    tokens = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        if len(raw) < 2 and not raw.isdigit():
            continue
        tokens.append(stem(raw))
    return tokens


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
