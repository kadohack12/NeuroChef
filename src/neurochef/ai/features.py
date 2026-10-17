"""Numerical features used by transform rankers."""

from __future__ import annotations

import base64
import binascii
import re

from neurochef.detectors import detect_magic, shannon_entropy

_BASE64_RE = re.compile(rb"^[A-Za-z0-9+/]*={0,2}$")
_HEX_BYTES = frozenset(b"0123456789abcdefABCDEF")
_BASE85_BYTES = frozenset(
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
)


class FeatureExtractor:
    """Extract cheap numeric features from arbitrary bytes."""

    def extract(self, data: bytes) -> dict[str, float]:
        compact = b"".join(data.split())
        magic = detect_magic(data)
        return {
            "shannon_entropy": shannon_entropy(data),
            "printable_ratio": self._printable_ratio(data),
            "hex_ratio": self._ratio(compact, _HEX_BYTES),
            "base64_ratio": self._base64_ratio(compact),
            "base85_ratio": self._ratio(data, _BASE85_BYTES),
            "has_magic_bytes": float(bool(magic)),
            "length": float(len(data)),
            "is_multiple_of_2": float(len(compact) % 2 == 0),
            "is_multiple_of_4": float(len(compact) % 4 == 0),
            # Extra context lets the heuristic distinguish gzip from other headers.
            "is_gzip": float("gzip" in magic),
            "is_zlib": float(data.startswith((b"x\x01", b"x\x9c", b"x\xda"))),
        }

    @staticmethod
    def _ratio(data: bytes, allowed: frozenset[int]) -> float:
        if not data:
            return 0.0
        return sum(byte in allowed for byte in data) / len(data)

    @staticmethod
    def _printable_ratio(data: bytes) -> float:
        return FeatureExtractor._ratio(data, frozenset(range(32, 127)) | {9, 10, 13})

    @staticmethod
    def _base64_ratio(data: bytes) -> float:
        if not data:
            return 0.0
        allowed = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
        ratio = sum(byte in allowed for byte in data) / len(data)
        if len(data) >= 4 and len(data) % 4 == 0 and _BASE64_RE.fullmatch(data):
            try:
                base64.b64decode(data, validate=True)
                return 1.0
            except (ValueError, binascii.Error):
                pass
        return ratio

    @staticmethod
    def _base85_ratio(data: bytes) -> float:
        return FeatureExtractor._ratio(data, _BASE85_BYTES)
