from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

import warply as wp
from warply.exceptions import HTTPClientError
from warply.providers.skypilot import SkyPilotProvider
from warply.runtime.client import (
    HTTPOpenAIClient,
    completion_content,
    normalize_base_url,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, object] | str) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        if isinstance(self.payload, str):
            return self.payload.encode("utf-8")
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
    assert client.api_base == "http://router.example/v1"
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
    assert client.api_base == "http://router.example/v1"


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


@pytest.mark.parametrize(
    ("response_body", "match"),
    [
        ("not-json", "invalid JSON"),
        ('{"choices": []}', "no choices"),
        ('{"choices": [{}]}', "invalid message"),
        (json.dumps({"choices": [{"message": {"content": None}}]}), "null content"),
    ],
)
def test_http_openai_client_rejects_invalid_completion_payload(
    monkeypatch,
    response_body,
    match,
):
    def fake_urlopen(request, timeout):
        return _FakeResponse(response_body)

    monkeypatch.setattr("warply.runtime.client.urlopen", fake_urlopen)
    client = HTTPOpenAIClient(base_url="http://router.example")

    with pytest.raises(HTTPClientError, match=match):
        client.chat.completions.create(
            model="warply",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_http_openai_client_rejects_streaming_and_unknown_kwargs():
    client = HTTPOpenAIClient(base_url="http://router.example")

    with pytest.raises(HTTPClientError, match="streaming completions"):
        client.chat.completions.create(
            model="warply",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

    with pytest.raises(HTTPClientError, match="unsupported completion kwargs"):
        client.chat.completions.create(
            model="warply",
            messages=[{"role": "user", "content": "hello"}],
            tool_choice="auto",
        )


def test_completion_content_rejects_empty_string():
    from warply.runtime.client import _Choice, _CompletionResponse, _Message

    with pytest.raises(HTTPClientError, match="empty completion content"):
        completion_content(
            _CompletionResponse(choices=[_Choice(message=_Message(content="  "))])
        )


def test_normalize_base_url():
    assert normalize_base_url("http://router.example") == "http://router.example/v1"
    assert normalize_base_url("http://router.example/v1") == "http://router.example/v1"


def test_cloud_generate_posts_to_router(monkeypatch):
    monkeypatch.delenv("WARPLY_SKYPILOT_DRY_RUN", raising=False)

    def fake_launch(self, *, yaml_str: str, cluster_name: str) -> str:
        return "10.0.0.1"

    monkeypatch.setattr(SkyPilotProvider, "_launch_task", fake_launch)
    monkeypatch.setattr(SkyPilotProvider, "_down_cluster", lambda self, cluster_name: None)

    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _FakeResponse({"choices": [{"message": {"content": "cloud hello"}}]})

    monkeypatch.setattr("warply.runtime.client.urlopen", fake_urlopen)

    engine = wp.DisaggEngine(
        model="meta-llama/Llama-3.1-8B",
        prefill=wp.Pool("1xH100", replicas=1),
        decode=wp.Pool("1xH100", replicas=1),
        cloud="lambda",
    )

    try:
        engine.up()
        assert isinstance(engine.client(), HTTPOpenAIClient)
        assert engine.generate("hello") == "cloud hello"
        assert captured["url"] == "http://10.0.0.1:8000/v1/chat/completions"
    finally:
        engine.down()
