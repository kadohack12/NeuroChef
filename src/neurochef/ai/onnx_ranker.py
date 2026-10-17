"""Local ONNX inference ranker for NeuroChef transforms."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from neurochef.transforms import Transform

from .base import BaseRanker
from .features import FEATURE_ORDER, FeatureExtractor
from .heuristic_ranker import HeuristicRanker

DEFAULT_CLASS_TO_TRANSFORM = {
    0: "base64",
    1: "from_hex",
    2: "url_decode",
    3: "rot13",
    4: "single_byte_xor",
    5: "zlib_decompress",
    6: "gzip_decompress",
    7: "base85",
}


class ONNXRanker(BaseRanker):
    """Run ``neuro_ranker_optimized.onnx`` locally without network access."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        extractor: FeatureExtractor | None = None,
        fallback: BaseRanker | None = None,
        class_to_transform: dict[int, str] | None = None,
    ) -> None:
        self.extractor = extractor or FeatureExtractor()
        self.fallback = fallback or HeuristicRanker(self.extractor)
        self.class_to_transform = class_to_transform or DEFAULT_CLASS_TO_TRANSFORM.copy()
        self.model_path = Path(model_path or self._default_model_path())
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_names: tuple[str, ...] = ()
        self.load_error: Exception | None = None
        try:
            if not self.model_path.is_file():
                raise FileNotFoundError(self.model_path)
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            self._validate_signature()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.load_error = exc
            self.session = None

    @staticmethod
    def _default_model_path() -> Path:
        return Path(__file__).resolve().parents[3] / "models" / "neuro_ranker_optimized.onnx"

    def _validate_signature(self) -> None:
        if self.session is None:
            raise RuntimeError("ONNX session is not initialized")
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input, got {len(inputs)}")
        input_meta = inputs[0]
        if input_meta.type != "tensor(float)":
            raise TypeError(f"Expected float input, got {input_meta.type}")
        if len(input_meta.shape) != 2 or input_meta.shape[-1] != len(FEATURE_ORDER):
            raise ValueError(
                f"Expected [None, {len(FEATURE_ORDER)}] input, got {input_meta.shape}"
            )
        self.input_name = input_meta.name
        self.output_names = tuple(output.name for output in self.session.get_outputs())

    @property
    def feature_order(self) -> tuple[str, ...]:
        return FEATURE_ORDER

    def rank(self, data: bytes, transforms: list[Transform]) -> list[tuple[Transform, float]]:
        if not transforms or self.session is None or self.input_name is None:
            return self.fallback.rank(data, transforms)
        try:
            vector = np.asarray([self.extractor.vector(data)], dtype=np.float32)
            if vector.shape != (1, len(FEATURE_ORDER)):
                raise ValueError(f"Invalid feature vector shape: {vector.shape}")
            outputs = self.session.run(None, {self.input_name: vector})
            probabilities = self._read_probabilities(outputs)
            by_name = {transform.name: transform for transform in transforms}
            ranked = [
                (transform, probabilities.get(class_id, 0.0))
                for class_id, name in self.class_to_transform.items()
                if (transform := by_name.get(name)) is not None
            ]
            # Include newly added or custom transforms not present in the trained model.
            known = {transform.name for transform, _score in ranked}
            ranked.extend((transform, 0.0) for transform in transforms if transform.name not in known)
            if not any(score > 0.0 for _transform, score in ranked):
                return self.fallback.rank(data, transforms)
            return self.normalize(ranked)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, IndexError):
            return self.fallback.rank(data, transforms)

    @staticmethod
    def _read_probabilities(outputs: list[Any]) -> dict[int, float]:
        if len(outputs) < 2:
            raise ValueError("ONNX model did not return a probability output")
        raw = outputs[1]
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else {}
        if isinstance(raw, dict):
            return {int(key): float(value) for key, value in raw.items()}
        raise TypeError(f"Unsupported probability output type: {type(raw).__name__}")
