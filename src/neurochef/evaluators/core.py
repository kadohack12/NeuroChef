"""Heuristics used to rank decoded candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass

from neurochef.detectors import detect_magic

_WORDS = {
    "the", "this", "that", "hello", "secret", "message", "flag", "and", "with",
    "le", "la", "les", "un", "une", "bonjour", "et", "avec",
}


@dataclass(frozen=True)
class Evaluation:
    score: float
    reason: str


def score_bytes(data: bytes) -> Evaluation:
    if not data:
        return Evaluation(0.0, "empty")
    magic = detect_magic(data)
    if magic:
        return Evaluation(1.0, f"magic:{','.join(magic)}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        printable = sum(32 <= byte < 127 for byte in data) / len(data)
        return Evaluation(printable * 0.25, "binary")
    printable = sum(char.isprintable() or char in "\r\n\t" for char in text) / len(text)
    words = set(re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower()))
    language = len(words & _WORDS) / max(len(words), 1)
    score = min(1.0, printable * 0.55 + language * 0.45)
    return Evaluation(score, "text")
