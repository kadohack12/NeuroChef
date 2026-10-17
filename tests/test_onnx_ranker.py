from pathlib import Path

from neurochef.ai.features import FEATURE_ORDER, FeatureExtractor
from neurochef.ai.onnx_ranker import ONNXRanker
from neurochef.transforms import DEFAULT_TRANSFORMS

MODEL_PATH = Path(__file__).parents[1] / "models" / "neuro_ranker_optimized.onnx"


def test_feature_vector_matches_onnx_signature() -> None:
    extractor = FeatureExtractor()
    vector = extractor.vector(b"SGVsbG8=")
    ranker = ONNXRanker(MODEL_PATH)

    assert len(FEATURE_ORDER) == 14
    assert len(vector) == 14
    assert ranker.input_name == "float_input"
    assert ranker.session is not None
    assert ranker.load_error is None


def test_onnx_ranker_runs_and_returns_coherent_scores() -> None:
    ranker = ONNXRanker(MODEL_PATH)
    ranked = ranker.rank(b"SGVsbG8=", list(DEFAULT_TRANSFORMS))

    assert ranked
    assert {transform.name for transform, _score in ranked} == {
        transform.name for transform in DEFAULT_TRANSFORMS
    }
    assert all(0.0 <= score <= 1.0 for _transform, score in ranked)
    assert all(ranked[index][1] >= ranked[index + 1][1] for index in range(len(ranked) - 1))
    assert any(score > 0.0 for _transform, score in ranked)


def test_onnx_ranker_falls_back_when_model_is_missing() -> None:
    ranker = ONNXRanker(MODEL_PATH.with_name("missing.onnx"))
    ranked = ranker.rank(b"SGVsbG8=", list(DEFAULT_TRANSFORMS))

    assert ranker.session is None
    assert ranked[0][0].name == "base64"
