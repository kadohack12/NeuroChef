"""Independent byte transforms used by the NeuroChef search engine."""

from __future__ import annotations

import base64
import binascii
import codecs
import string
import urllib.parse
import zlib
from collections.abc import Iterable


class Transform:
    """Base interface: every transform accepts bytes and returns Optional[bytes]."""

    name = "transform"

    def __call__(self, data: bytes) -> bytes | None:
        raise NotImplementedError

    def candidates(self, data: bytes) -> Iterable[bytes]:
        result = self(data)
        if result is not None:
            yield result


def _valid_decoded(value: bytes) -> bytes | None:
    return value if value and len(value) <= 10_000_000 else None


class Base64Decode(Transform):
    name = "base64"

    def __call__(self, data: bytes) -> bytes | None:
        compact = b"".join(data.split())
        if len(compact) < 4 or len(compact) % 4:
            return None
        try:
            return _valid_decoded(base64.b64decode(compact, validate=True))
        except (ValueError, binascii.Error):
            return None


class FromHex(Transform):
    name = "from_hex"

    def __call__(self, data: bytes) -> bytes | None:
        compact = b"".join(data.split())
        if not compact or len(compact) % 2:
            return None
        try:
            return _valid_decoded(bytes.fromhex(compact.decode("ascii")))
        except (ValueError, UnicodeDecodeError):
            return None


class URLDecode(Transform):
    name = "url_decode"

    def __call__(self, data: bytes) -> bytes | None:
        try:
            decoded = urllib.parse.unquote_to_bytes(data.decode("ascii"))
        except UnicodeDecodeError:
            return None
        return decoded if decoded != data and b"%" in data else None


class ROT13(Transform):
    name = "rot13"

    def __call__(self, data: bytes) -> bytes | None:
        try:
            decoded = codecs.decode(data.decode("ascii"), "rot_13").encode("ascii")
        except UnicodeDecodeError:
            return None
        return decoded if decoded != data else None


class SingleByteXOR(Transform):
    name = "single_byte_xor"

    def __init__(self, max_candidates: int = 256) -> None:
        self.max_candidates = max_candidates

    @staticmethod
    def _score(data: bytes) -> float:
        if not data:
            return -1.0
        printable = sum(byte in b"\t\n\r" or 32 <= byte < 127 for byte in data) / len(data)
        letters = sum(chr(byte).lower() in string.ascii_lowercase for byte in data if byte < 128)
        return printable + letters / max(len(data), 1) * 0.25

    def candidates(self, data: bytes) -> Iterable[bytes]:
        values = sorted(
            ((bytes(byte ^ key for byte in data), key) for key in range(256)),
            key=lambda item: self._score(item[0]),
            reverse=True,
        )
        for decoded, _key in values[: self.max_candidates]:
            yield decoded

    def __call__(self, data: bytes) -> bytes | None:
        return next(iter(self.candidates(data)), None)


class ZlibDecompress(Transform):
    name = "zlib_decompress"

    def __call__(self, data: bytes) -> bytes | None:
        try:
            return _valid_decoded(zlib.decompress(data))
        except zlib.error:
            return None


DEFAULT_TRANSFORMS: tuple[Transform, ...] = (
    Base64Decode(),
    FromHex(),
    URLDecode(),
    ROT13(),
    SingleByteXOR(max_candidates=256),
    ZlibDecompress(),
)
