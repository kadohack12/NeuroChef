import base64
import binascii
import gzip
import zlib
from collections import deque

from neurochef.ai import FeatureExtractor, HeuristicRanker
from neurochef.core import SearchEngine
from neurochef.detectors import analyze_charset, detect_magic, shannon_entropy
from neurochef.transforms import (
    DEFAULT_TRANSFORMS,
    ROT13,
    Base64Decode,
    Base85Decode,
    FromHex,
    GzipDecompress,
    Reverse,
    SingleByteXOR,
    ZlibDecompress,
)


def test_base64_hex_rot13_and_zlib_transforms() -> None:
    assert Base64Decode()(b"SGVsbG8=") == b"Hello"
    assert FromHex()(b"48656c6c6f") == b"Hello"
    assert ROT13()(b"Uryyb") == b"Hello"
    compressed = zlib.compress(b"compressed bytes")
    assert ZlibDecompress()(compressed) == b"compressed bytes"
    assert GzipDecompress()(gzip.compress(b"gzip bytes")) == b"gzip bytes"
    assert Base85Decode()(base64.a85encode(b"base85 bytes")) == b"base85 bytes"
    assert Reverse()(b"abc") == b"cba"


def test_feature_extractor_and_heuristic_ranker_prioritize_base64() -> None:
    encoded = base64.b64encode(b"NeuroChef ranking")
    features = FeatureExtractor().extract(encoded)
    assert features["base64_ratio"] == 1.0
    assert features["is_multiple_of_4"] == 1.0
    assert 0.0 <= features["shannon_entropy"] <= 8.0

    transforms = [FromHex(), SingleByteXOR(), Base64Decode(), Reverse()]
    ranked = HeuristicRanker().rank(encoded, transforms)
    assert ranked[0][0].name == "base64"
    assert ranked[0][1] > ranked[1][1]


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
    assert result.nodes_explored < _blind_bfs_nodes(challenge, plaintext)


def _blind_bfs_nodes(challenge: bytes, target: bytes) -> int:
    """Reference traversal without format gates or ranking guidance."""
    queue = deque([(challenge, 0)])
    visited = {challenge}
    explored = 0
    while queue and explored < 10_000:
        current, depth = queue.popleft()
        explored += 1
        if current == target:
            return explored
        if depth >= 4:
            continue
        for transform in DEFAULT_TRANSFORMS:
            for decoded in transform.candidates(current):
                if decoded and decoded != current and decoded not in visited:
                    visited.add(decoded)
                    queue.append((decoded, depth + 1))
    return explored
