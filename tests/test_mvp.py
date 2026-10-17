import base64
import binascii
import zlib

from neurochef.core import SearchEngine
from neurochef.detectors import analyze_charset, detect_magic, shannon_entropy
from neurochef.transforms import ROT13, Base64Decode, FromHex, SingleByteXOR, ZlibDecompress


def test_base64_hex_rot13_and_zlib_transforms() -> None:
    assert Base64Decode()(b"SGVsbG8=") == b"Hello"
    assert FromHex()(b"48656c6c6f") == b"Hello"
    assert ROT13()(b"Uryyb") == b"Hello"
    compressed = zlib.compress(b"compressed bytes")
    assert ZlibDecompress()(compressed) == b"compressed bytes"


def test_single_byte_xor_bruteforce_contains_known_key() -> None:
    original = b"flag: neurochef"
    encrypted = bytes(value ^ 0x42 for value in original)
    assert original in set(SingleByteXOR().candidates(encrypted))


def test_detectors() -> None:
    assert detect_magic(b"MZ\x90\x00") == ["pe"]
    assert detect_magic(b"\x7fELF\x02") == ["elf"]
    assert detect_magic(b"PK\x03\x04payload") == ["zip"]
    assert detect_magic(b"\x1f\x8b\x08payload") == ["gzip"]
    assert shannon_entropy(b"aaaa") == 0.0
    assert analyze_charset(b"NeuroChef") == "utf-8-text"


def test_engine_solves_three_layers_base64_xor_hex() -> None:
    plaintext = b"flag: neurochef"
    base64_layer = base64.b64encode(plaintext)
    xor_layer = bytes(value ^ 0x42 for value in base64_layer)
    challenge = binascii.hexlify(xor_layer)

    result = SearchEngine(max_depth=4, max_nodes=10_000).solve(challenge)

    assert result.data == plaintext
    assert [step.transform for step in result.steps] == [
        "from_hex",
        "single_byte_xor",
        "base64",
    ]
