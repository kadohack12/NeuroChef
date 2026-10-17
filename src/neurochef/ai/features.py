"""Numerical features used by transform rankers and the ONNX model."""

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

# The ONNX graph has no feature-name metadata. Keep this order versioned and
# explicit: it is the contract between FeatureExtractor and neuro_ranker.
FEATURE_ORDER = (
    "shannon_entropy",
    "printable_ratio",
    "hex_ratio",
    "base64_ratio",
    "base85_ratio",
    "has_magic_bytes",
    "length",
    "is_multiple_of_2",
    "is_multiple_of_4",
    "is_gzip",
    "is_zlib",
    "has_pe_header",
    "has_elf_header",
    "has_archive_header",
)


class FeatureExtractor:
    """Extract exactly fourteen finite numeric features from arbitrary bytes."""

    def extract(self, data: bytes) -> dict[str, float]:
        compact = b"".join(data.split())
        magic = detect_magic(data)
        features = {
            "shannon_entropy": shannon_entropy(data),
            "printable_ratio": self._printable_ratio(data),
            "hex_ratio": self._ratio(compact, _HEX_BYTES),
            "base64_ratio": self._base64_ratio(compact),
            "base85_ratio": self._ratio(data, _BASE85_BYTES),
            "has_magic_bytes": float(bool(magic)),
            "length": float(len(data)),
            "is_multiple_of_2": float(len(compact) % 2 == 0),
            "is_multiple_of_4": float(len(compact) % 4 == 0),
            "is_gzip": float("gzip" in magic),
            "is_zlib": float(data.startswith((b"x\x01", b"x\x9c", b"x\xda"))),
            "has_pe_header": float("pe" in magic),
            "has_elf_header": float("elf" in magic),
            "has_archive_header": float("zip" in magic),
        }
        return {name: float(features[name]) for name in FEATURE_ORDER}

    def vector(self, data: bytes) -> list[float]:
        """Return the ONNX input vector in the model's declared feature order."""
        features = self.extract(data)
        return [features[name] for name in FEATURE_ORDER]

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
