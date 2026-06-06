from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from warply.exceptions import HTTPClientError

_ALLOWED_COMPLETION_KWARGS = frozenset(
    {
        "frequency_penalty",
        "logprobs",
        "max_tokens",
        "n",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "temperature",
        "top_k",
        "top_p",
    }
)


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


class _HTTPChatCompletions:
    def __init__(self, *, api_base: str, timeout: float) -> None:
        self.api_base = api_base
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
            **_validate_completion_kwargs(kwargs),
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.api_base}/chat/completions",
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
    def __init__(self, *, api_base: str, timeout: float) -> None:
        self.completions = _HTTPChatCompletions(api_base=api_base, timeout=timeout)


class HTTPOpenAIClient:
    """OpenAI-compatible HTTP client for SGLang router endpoints."""

    def __init__(self, *, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url
        self.api_base = normalize_base_url(base_url)
        self.chat = _HTTPChat(api_base=self.api_base, timeout=timeout)


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def completion_content(response: _CompletionResponse) -> str:
    """Return validated text content from the first completion choice."""
    if not response.choices:
        raise HTTPClientError("OpenAI-compatible endpoint returned no choices.")

    content = response.choices[0].message.content
    if not content.strip():
        raise HTTPClientError("OpenAI-compatible endpoint returned empty completion content.")
    return content


def _validate_completion_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if kwargs.get("stream"):
        raise HTTPClientError("streaming completions are not supported yet.")

    unknown = sorted(set(kwargs) - _ALLOWED_COMPLETION_KWARGS)
    if unknown:
        options = ", ".join(sorted(_ALLOWED_COMPLETION_KWARGS))
        raise HTTPClientError(
            f"unsupported completion kwargs: {', '.join(unknown)}; allowed: {options}."
        )
    return kwargs


def _parse_completion_response(response_body: str) -> _CompletionResponse:
    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise HTTPClientError(
            f"OpenAI-compatible endpoint returned invalid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise HTTPClientError("OpenAI-compatible endpoint returned a non-object JSON payload.")

    raw_choices = data.get("choices")
    if not raw_choices:
        raise HTTPClientError("OpenAI-compatible endpoint returned no choices.")
    if not isinstance(raw_choices, list):
        raise HTTPClientError("OpenAI-compatible endpoint returned invalid choices payload.")

    choices: list[_Choice] = []
    for index, choice in enumerate(raw_choices):
        if not isinstance(choice, dict):
            raise HTTPClientError(
                f"OpenAI-compatible endpoint returned invalid choice at index {index}."
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise HTTPClientError(
                f"OpenAI-compatible endpoint returned invalid message at index {index}."
            )
        content = message.get("content")
        if content is None:
            raise HTTPClientError(
                f"OpenAI-compatible endpoint returned null content at index {index}."
            )
        if not isinstance(content, str):
            raise HTTPClientError(
                f"OpenAI-compatible endpoint returned non-text content at index {index}."
            )
        choices.append(_Choice(message=_Message(content=content)))

    return _CompletionResponse(choices=choices)
