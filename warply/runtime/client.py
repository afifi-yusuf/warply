from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from warply.exceptions import WarplyError


@dataclass(frozen=True)
class _Message:
    content: str


@dataclass(frozen=True)
class _Choice:
    message: _Message


@dataclass(frozen=True)
class _CompletionResponse:
    choices: list[_Choice]


class _ChatCompletions:
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> _CompletionResponse:
        prompt = messages[-1]["content"] if messages else ""
        content = f"[warply:mock:{model}] {prompt}"
        return _CompletionResponse(choices=[_Choice(message=_Message(content=content))])


class _Chat:
    def __init__(self) -> None:
        self.completions = _ChatCompletions()


class MockOpenAIClient:
    """Small OpenAI-compatible surface for local mock lifecycle tests."""

    def __init__(self, *, base_url: str) -> None:
        self.base_url = base_url
        self.chat = _Chat()


class HTTPClientError(WarplyError, RuntimeError):
    """Raised when an OpenAI-compatible HTTP endpoint rejects a request."""


class _HTTPChatCompletions:
    def __init__(self, *, base_url: str, timeout: float) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.timeout = timeout

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> _CompletionResponse:
        payload = {
            "model": model,
            "messages": messages,
            **kwargs,
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise HTTPClientError(
                f"OpenAI-compatible endpoint returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            message = f"OpenAI-compatible endpoint request failed: {exc.reason}"
            raise HTTPClientError(message) from exc

        return _parse_completion_response(response_body)


class _HTTPChat:
    def __init__(self, *, base_url: str, timeout: float) -> None:
        self.completions = _HTTPChatCompletions(base_url=base_url, timeout=timeout)


class HTTPOpenAIClient:
    """OpenAI-compatible HTTP client for SGLang router endpoints."""

    def __init__(self, *, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url
        self.chat = _HTTPChat(base_url=base_url, timeout=timeout)


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _parse_completion_response(response_body: str) -> _CompletionResponse:
    data = json.loads(response_body)
    choices = [
        _Choice(message=_Message(content=choice["message"]["content"]))
        for choice in data.get("choices", [])
    ]
    if not choices:
        raise HTTPClientError("OpenAI-compatible endpoint returned no choices.")
    return _CompletionResponse(choices=choices)
