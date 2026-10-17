"""Optional LLM-assisted ranker with a dependency-free HTTP client."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from urllib import request

from neurochef.transforms import Transform

from .base import BaseRanker
from .features import FeatureExtractor
from .heuristic_ranker import HeuristicRanker


class LLMRanker(BaseRanker):
    """Use a local Ollama model or an OpenAI/Anthropic-compatible endpoint.

    The heuristic ranker remains the safe default. Network access is attempted
    only when its best score is below ``fallback_threshold`` or when explicitly
    requested with ``always_query=True``.
    """

    def __init__(
        self,
        provider: str = "ollama",
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 8.0,
        fallback_threshold: float = 0.80,
        always_query: bool = False,
        heuristic: BaseRanker | None = None,
        extractor: FeatureExtractor | None = None,
        http_post: Callable[[str, dict[str, object], dict[str, str], float], dict[str, object]]
        | None = None,
    ) -> None:
        self.provider = provider.lower()
        self.endpoint = endpoint or self._default_endpoint(self.provider)
        self.model = model or os.getenv("NEUROCHEF_LLM_MODEL", "llama3.2:3b")
        self.api_key = api_key or os.getenv("NEUROCHEF_LLM_API_KEY")
        self.timeout = timeout
        self.fallback_threshold = fallback_threshold
        self.always_query = always_query
        self.heuristic = heuristic or HeuristicRanker(extractor)
        self.extractor = extractor or FeatureExtractor()
        self._http_post = http_post or self._post_json

    def rank(
        self, data: bytes, candidates: list[Transform]
    ) -> list[tuple[Transform, float]]:
        baseline = self.heuristic.rank(data, candidates)
        if not candidates or (not self.always_query and baseline[0][1] >= self.fallback_threshold):
            return baseline
        try:
            payload = self._request_model(data, candidates)
            ranked = self._parse_ranking(payload, candidates)
            return self.normalize(ranked) if ranked else baseline
        except (OSError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
            return baseline

    @staticmethod
    def _default_endpoint(provider: str) -> str:
        if provider == "ollama":
            return "http://127.0.0.1:11434/api/generate"
        if provider == "anthropic":
            return "https://api.anthropic.com/v1/messages"
        return "https://api.openai.com/v1/chat/completions"

    def _request_model(self, data: bytes, candidates: list[Transform]) -> object:
        names = [candidate.name for candidate in candidates]
        features = self.extractor.extract(data)
        prompt = (
            "Rank these byte decoding operations for the next step. "
            "Return JSON only as [{\"name\": string, \"score\": number}] "
            "with scores from 0 to 1. Do not invent names.\n"
            f"features={json.dumps(features, separators=(',', ':'))}\n"
            f"operations={json.dumps(names)}"
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider == "ollama":
            body = {"model": self.model, "prompt": prompt, "stream": False, "format": "json"}
        elif self.provider == "anthropic":
            headers["x-api-key"] = self.api_key or ""
            headers["anthropic-version"] = "2023-06-01"
            body = {"model": self.model, "max_tokens": 300, "messages": [{"role": "user", "content": prompt}]}
        else:
            body = {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            }
        response = self._http_post(self.endpoint, body, headers, self.timeout)
        return self._extract_response_payload(response)

    @staticmethod
    def _extract_response_payload(response: dict[str, object]) -> object:
        if "response" in response:
            return json.loads(str(response["response"]))
        if "choices" in response:
            choices = response["choices"]
            content = choices[0]["message"]["content"]  # type: ignore[index]
            return json.loads(str(content))
        if "content" in response:
            content = response["content"]
            text = content[0]["text"]  # type: ignore[index]
            return json.loads(str(text))
        return response

    @staticmethod
    def _parse_ranking(
        payload: object, candidates: list[Transform]
    ) -> list[tuple[Transform, float]]:
        if isinstance(payload, dict):
            items = payload.get("ranking", payload.get("operations", payload))
        else:
            items = payload
        if isinstance(items, dict):
            items = [{"name": name, "score": score} for name, score in items.items()]
        if not isinstance(items, list):
            return []
        by_name = {candidate.name: candidate for candidate in candidates}
        parsed = []
        for item in items:
            if not isinstance(item, dict) or item.get("name") not in by_name:
                continue
            parsed.append((by_name[str(item["name"])], float(item.get("score", 0.0))))
        return parsed

    @staticmethod
    def _post_json(
        endpoint: str, body: dict[str, object], headers: dict[str, str], timeout: float
    ) -> dict[str, object]:
        encoded = json.dumps(body).encode("utf-8")
        req = request.Request(endpoint, data=encoded, headers=headers, method="POST")
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
