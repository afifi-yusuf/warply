from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from warply.runtime.client import HTTPClientError, HTTPOpenAIClient


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def close(self) -> None:
        return None


def test_http_openai_client_posts_chat_completions(monkeypatch):
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "hello from sglang"}}]})

    monkeypatch.setattr("warply.runtime.client.urlopen", fake_urlopen)
    client = HTTPOpenAIClient(base_url="http://router.example", timeout=5)

    response = client.chat.completions.create(
        model="warply",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0,
    )

    assert response.choices[0].message.content == "hello from sglang"
    assert captured == {
        "url": "http://router.example/v1/chat/completions",
        "timeout": 5,
        "payload": {
            "model": "warply",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        },
    }


def test_http_openai_client_accepts_base_url_with_v1(monkeypatch):
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("warply.runtime.client.urlopen", fake_urlopen)
    client = HTTPOpenAIClient(base_url="http://router.example/v1")

    client.chat.completions.create(
        model="warply",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert captured["url"] == "http://router.example/v1/chat/completions"


def test_http_openai_client_raises_on_http_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(
            url=request.full_url,
            code=500,
            msg="server error",
            hdrs={},
            fp=_FakeResponse({"error": "router not ready"}),
        )

    monkeypatch.setattr("warply.runtime.client.urlopen", fake_urlopen)
    client = HTTPOpenAIClient(base_url="http://router.example")

    with pytest.raises(HTTPClientError, match="HTTP 500"):
        client.chat.completions.create(
            model="warply",
            messages=[{"role": "user", "content": "hello"}],
        )
