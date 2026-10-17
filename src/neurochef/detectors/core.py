"""Low-level detectors for byte streams."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

MAGIC_BYTES: dict[str, bytes] = {
    "elf": b"\x7fELF",
    "pe": b"MZ",
    "zip": b"PK\x03\x04",
    "gzip": b"\x1f\x8b",
}


def shannon_entropy(data: bytes) -> float:
    """Return Shannon entropy in bits per byte, in the [0, 8] range."""
    if not data:
        return 0.0
    counts = Counter(data)
    size = len(data)
    return -sum((count / size) * math.log2(count / size) for count in counts.values())


def detect_magic(data: bytes) -> list[str]:
    """Return known file formats whose signatures occur at the stream start."""
    return [name for name, signature in MAGIC_BYTES.items() if data.startswith(signature)]


def analyze_charset(data: bytes) -> str:
    """Classify common text encodings without treating failure as an exception."""
    if not data:
        return "empty"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        data.decode("utf-8")
        printable = sum(byte in b"\t\n\r" or 32 <= byte < 127 for byte in data) / len(data)
        return "utf-8-text" if printable >= 0.85 else "utf-8"
    except UnicodeDecodeError:
        return "binary"


@dataclass(frozen=True)
class Detection:
    entropy: float
    magic: tuple[str, ...]
    charset: str


def detect(data: bytes) -> Detection:
    return Detection(shannon_entropy(data), tuple(detect_magic(data)), analyze_charset(data))
