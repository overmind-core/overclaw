"""Tests for overmind.client — Client, Models, ChatCompletions, parsers.

All HTTP is mocked via unittest.mock; no real network traffic is made.
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import requests

from overmind.client import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletions,
    ChatMessage,
    Choice,
    Client,
    Delta,
    Model,
    ModelDeleted,
    ModelList,
    OvermindInferenceError,
    _iter_sse_chunks,
    _parse_chat_completion,
    _parse_chunk,
    _parse_model,
    _raise_for_status,
)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_client(base_url: str = "https://api.example.com") -> Client:
    return Client(api_key="om-test-key", base_url=base_url)


def _mock_response(
    *,
    ok: bool = True,
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.ok = ok
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data or {}
    return resp


def _sse_response(lines: list[str]) -> MagicMock:
    """Build a mock streaming response whose iter_lines() yields encoded lines."""
    resp = MagicMock(spec=requests.Response)
    resp.ok = True
    resp.status_code = 200
    resp.iter_lines.return_value = [line.encode() for line in lines]
    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Client initialisation
# ═══════════════════════════════════════════════════════════════════════════════


class TestClientInit:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("OVERMIND_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Missing API key"):
            Client()

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OVERMIND_API_KEY", "env-key")
        monkeypatch.delenv("OVERMIND_API_URL", raising=False)
        c = Client()
        assert c._session.headers.get("X-Api-Key") == "env-key"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OVERMIND_API_KEY", "env-key")
        c = Client(api_key="explicit-key")
        assert c._session.headers.get("X-Api-Key") == "explicit-key"

    def test_base_url_strips_trailing_slash(self):
        c = Client(api_key="k", base_url="https://api.example.com/")
        assert not c._base_url.endswith("/")

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("OVERMIND_API_URL", "https://custom.example.com")
        c = Client(api_key="k")
        assert c._base_url == "https://custom.example.com"

    def test_explicit_base_url_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OVERMIND_API_URL", "https://env.example.com")
        c = Client(api_key="k", base_url="https://explicit.example.com")
        assert c._base_url == "https://explicit.example.com"

    def test_default_base_url_when_no_env(self, monkeypatch):
        monkeypatch.delenv("OVERMIND_API_URL", raising=False)
        c = Client(api_key="k")
        assert c._base_url == "https://api.overmindlab.ai"

    def test_content_type_header_set(self):
        c = _make_client()
        assert c._session.headers.get("Content-Type") == "application/json"

    def test_chat_and_models_sub_resources_attached(self):
        c = _make_client()
        assert hasattr(c, "chat")
        assert hasattr(c.chat, "completions")
        assert hasattr(c, "models")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _raise_for_status
# ═══════════════════════════════════════════════════════════════════════════════


class TestRaiseForStatus:
    def test_ok_response_does_not_raise(self):
        resp = _mock_response(ok=True, status_code=200, json_data={"result": "ok"})
        _raise_for_status(resp)  # should not raise

    def test_error_response_raises(self):
        resp = _mock_response(
            ok=False,
            status_code=404,
            json_data={"error": {"message": "Not found"}},
        )
        with pytest.raises(OvermindInferenceError, match="404"):
            _raise_for_status(resp)

    def test_error_extracts_message_from_json(self):
        resp = _mock_response(
            ok=False, status_code=400,
            json_data={"error": {"message": "model is not ready"}},
        )
        with pytest.raises(OvermindInferenceError, match="model is not ready"):
            _raise_for_status(resp)

    def test_error_falls_back_to_text_when_no_json(self):
        resp = _mock_response(ok=False, status_code=500, text="Internal Server Error")
        resp.json.side_effect = ValueError("no JSON")
        with pytest.raises(OvermindInferenceError, match="Internal Server Error"):
            _raise_for_status(resp)

    def test_error_uses_detail_field(self):
        resp = _mock_response(
            ok=False, status_code=403,
            json_data={"detail": "Authentication credentials were not provided."},
        )
        with pytest.raises(OvermindInferenceError, match="Authentication credentials"):
            _raise_for_status(resp)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseChatCompletion:
    def _raw(self, **overrides) -> dict:
        base = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "model": "ft-llama-123",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "created": 1700000000,
        }
        base.update(overrides)
        return base

    def test_basic_fields(self):
        cc = _parse_chat_completion(self._raw())
        assert isinstance(cc, ChatCompletion)
        assert cc.id == "chatcmpl-abc"
        assert cc.model == "ft-llama-123"
        assert cc.created == 1700000000

    def test_choice_parsed(self):
        cc = _parse_chat_completion(self._raw())
        assert len(cc.choices) == 1
        c = cc.choices[0]
        assert isinstance(c, Choice)
        assert c.message.role == "assistant"
        assert c.message.content == "Hello!"
        assert c.finish_reason == "stop"

    def test_usage_parsed(self):
        cc = _parse_chat_completion(self._raw())
        assert cc.usage is not None
        assert cc.usage.prompt_tokens == 5
        assert cc.usage.completion_tokens == 3
        assert cc.usage.total_tokens == 8

    def test_missing_usage_returns_none(self):
        raw = self._raw()
        del raw["usage"]
        cc = _parse_chat_completion(raw)
        assert cc.usage is None

    def test_null_content_defaults_to_empty_string(self):
        raw = self._raw()
        raw["choices"][0]["message"]["content"] = None
        cc = _parse_chat_completion(raw)
        assert cc.choices[0].message.content == ""

    def test_empty_choices(self):
        cc = _parse_chat_completion({"id": "x", "object": "o", "model": "m", "choices": []})
        assert cc.choices == []


class TestParseChunk:
    def _raw(self) -> dict:
        return {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "model": "ft-test",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}
            ],
            "created": 100,
        }

    def test_basic_chunk_parsing(self):
        chunk = _parse_chunk(self._raw())
        assert isinstance(chunk, ChatCompletionChunk)
        assert chunk.id == "chunk-1"
        assert chunk.model == "ft-test"

    def test_delta_content_parsed(self):
        chunk = _parse_chunk(self._raw())
        assert len(chunk.choices) == 1
        d = chunk.choices[0].delta
        assert isinstance(d, Delta)
        assert d.content == "Hi"
        assert d.role == "assistant"

    def test_delta_missing_content_is_none(self):
        raw = self._raw()
        raw["choices"][0]["delta"] = {}
        chunk = _parse_chunk(raw)
        assert chunk.choices[0].delta.content is None


class TestParseModel:
    def _raw(self, **overrides) -> dict:
        base = {
            "id": "ft-llama-abc",
            "object": "model",
            "created": 1700000000,
            "owned_by": "overmind",
            "finetuned": True,
            "status": "ready",
            "base_model": "meta-llama/llama-3.1-8b-instruct",
        }
        base.update(overrides)
        return base

    def test_finetuned_model_parsed(self):
        m = _parse_model(self._raw())
        assert isinstance(m, Model)
        assert m.id == "ft-llama-abc"
        assert m.finetuned is True
        assert m.status == "ready"
        assert m.base_model == "meta-llama/llama-3.1-8b-instruct"

    def test_non_finetuned_model_parsed(self):
        raw = self._raw(id="anthropic/claude-sonnet-5", finetuned=False, owned_by="anthropic", status="", base_model="")
        m = _parse_model(raw)
        assert m.finetuned is False
        assert m.owned_by == "anthropic"

    def test_defaults_for_missing_optional_fields(self):
        m = _parse_model({"id": "x", "object": "model", "created": 0, "owned_by": "overmind"})
        assert m.finetuned is False
        assert m.status == ""
        assert m.base_model == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. _iter_sse_chunks
# ═══════════════════════════════════════════════════════════════════════════════


class TestIterSseChunks:
    def _resp(self, lines: list[str]) -> MagicMock:
        resp = MagicMock(spec=requests.Response)
        resp.iter_lines.return_value = [l.encode() for l in lines]
        return resp

    def test_yields_chunks_for_data_lines(self):
        payload = json.dumps({
            "id": "c1", "object": "chat.completion.chunk", "model": "m",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        })
        chunks = list(_iter_sse_chunks(self._resp([f"data: {payload}", "data: [DONE]"])))
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hello"

    def test_stops_at_done(self):
        payload = json.dumps({"id": "x", "object": "o", "model": "m", "choices": []})
        chunks = list(_iter_sse_chunks(self._resp([
            f"data: {payload}",
            "data: [DONE]",
            f"data: {payload}",   # should never be reached
        ])))
        assert len(chunks) == 1

    def test_skips_non_data_lines(self):
        payload = json.dumps({"id": "x", "object": "o", "model": "m", "choices": []})
        chunks = list(_iter_sse_chunks(self._resp([
            "event: ping",
            ": comment",
            f"data: {payload}",
            "data: [DONE]",
        ])))
        assert len(chunks) == 1

    def test_skips_invalid_json(self):
        payload = json.dumps({"id": "x", "object": "o", "model": "m", "choices": []})
        chunks = list(_iter_sse_chunks(self._resp([
            "data: {not valid json}",
            f"data: {payload}",
            "data: [DONE]",
        ])))
        assert len(chunks) == 1

    def test_raises_on_error_in_stream(self):
        error_payload = json.dumps({"error": {"message": "context length exceeded"}})
        with pytest.raises(OvermindInferenceError, match="context length exceeded"):
            list(_iter_sse_chunks(self._resp([f"data: {error_payload}"])))

    def test_empty_stream_yields_nothing(self):
        chunks = list(_iter_sse_chunks(self._resp(["data: [DONE]"])))
        assert chunks == []


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ChatCompletions.create — non-streaming
# ═══════════════════════════════════════════════════════════════════════════════


class TestChatCompletionsNonStream:
    _MESSAGES = [{"role": "user", "content": "Hi"}]
    _RESPONSE_DATA = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "model": "ft-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }

    def test_returns_chat_completion(self):
        c = _make_client()
        c._session.post = MagicMock(return_value=_mock_response(json_data=self._RESPONSE_DATA))
        result = c.chat.completions.create(model="ft-test", messages=self._MESSAGES)
        assert isinstance(result, ChatCompletion)
        assert result.id == "cmpl-1"
        assert result.choices[0].message.content == "Hello!"

    def test_sends_correct_url(self):
        c = _make_client("https://api.example.com")
        mock_post = MagicMock(return_value=_mock_response(json_data=self._RESPONSE_DATA))
        c._session.post = mock_post
        c.chat.completions.create(model="ft-test", messages=self._MESSAGES)
        call_url = mock_post.call_args[0][0]
        assert call_url == "https://api.example.com/api/v1/chat/completions"

    def test_sends_model_and_messages_in_payload(self):
        c = _make_client()
        mock_post = MagicMock(return_value=_mock_response(json_data=self._RESPONSE_DATA))
        c._session.post = mock_post
        c.chat.completions.create(model="ft-test", messages=self._MESSAGES, temperature=0.5)
        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "ft-test"
        assert payload["messages"] == self._MESSAGES
        assert payload["temperature"] == 0.5
        assert payload["stream"] is False

    def test_includes_max_tokens_when_set(self):
        c = _make_client()
        mock_post = MagicMock(return_value=_mock_response(json_data=self._RESPONSE_DATA))
        c._session.post = mock_post
        c.chat.completions.create(model="m", messages=self._MESSAGES, max_tokens=100)
        payload = mock_post.call_args[1]["json"]
        assert payload["max_tokens"] == 100

    def test_omits_max_tokens_when_not_set(self):
        c = _make_client()
        mock_post = MagicMock(return_value=_mock_response(json_data=self._RESPONSE_DATA))
        c._session.post = mock_post
        c.chat.completions.create(model="m", messages=self._MESSAGES)
        payload = mock_post.call_args[1]["json"]
        assert "max_tokens" not in payload

    def test_http_error_raises_inference_error(self):
        c = _make_client()
        c._session.post = MagicMock(
            return_value=_mock_response(ok=False, status_code=503, json_data={"error": {"message": "not ready"}})
        )
        with pytest.raises(OvermindInferenceError, match="not ready"):
            c.chat.completions.create(model="m", messages=self._MESSAGES)

    def test_timeout_raises_inference_error(self):
        c = _make_client()
        c._session.post = MagicMock(side_effect=requests.exceptions.Timeout())
        with pytest.raises(OvermindInferenceError, match="timed out"):
            c.chat.completions.create(model="m", messages=self._MESSAGES)

    def test_connection_error_raises_inference_error(self):
        c = _make_client()
        c._session.post = MagicMock(side_effect=requests.exceptions.ConnectionError("refused"))
        with pytest.raises(OvermindInferenceError, match="Request failed"):
            c.chat.completions.create(model="m", messages=self._MESSAGES)

    def test_extra_kwargs_forwarded(self):
        c = _make_client()
        mock_post = MagicMock(return_value=_mock_response(json_data=self._RESPONSE_DATA))
        c._session.post = mock_post
        c.chat.completions.create(model="m", messages=self._MESSAGES, top_p=0.9, frequency_penalty=0.1)
        payload = mock_post.call_args[1]["json"]
        assert payload["top_p"] == 0.9
        assert payload["frequency_penalty"] == 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ChatCompletions.create — streaming
# ═══════════════════════════════════════════════════════════════════════════════


class TestChatCompletionsStream:
    _MESSAGES = [{"role": "user", "content": "Stream test"}]

    def _chunk_line(self, content: str, model: str = "ft-test", chunk_id: str = "c1") -> str:
        data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        return f"data: {json.dumps(data)}"

    def test_returns_iterator_when_stream_true(self):
        c = _make_client()
        c._session.post = MagicMock(return_value=_sse_response([
            self._chunk_line("Hi"),
            "data: [DONE]",
        ]))
        result = c.chat.completions.create(model="ft-test", messages=self._MESSAGES, stream=True)
        assert hasattr(result, "__iter__")

    def test_yields_chat_completion_chunks(self):
        c = _make_client()
        c._session.post = MagicMock(return_value=_sse_response([
            self._chunk_line("Hello"),
            self._chunk_line(" world"),
            "data: [DONE]",
        ]))
        chunks = list(c.chat.completions.create(model="ft-test", messages=self._MESSAGES, stream=True))
        assert len(chunks) == 2
        assert all(isinstance(ch, ChatCompletionChunk) for ch in chunks)
        text = "".join(ch.choices[0].delta.content or "" for ch in chunks)
        assert text == "Hello world"

    def test_stream_payload_has_stream_true(self):
        c = _make_client()
        mock_post = MagicMock(return_value=_sse_response(["data: [DONE]"]))
        c._session.post = mock_post
        list(c.chat.completions.create(model="m", messages=self._MESSAGES, stream=True))
        payload = mock_post.call_args[1]["json"]
        assert payload["stream"] is True

    def test_stream_error_chunk_raises(self):
        c = _make_client()
        error_line = f"data: {json.dumps({'error': {'message': 'GPU OOM'}})}"
        c._session.post = MagicMock(return_value=_sse_response([error_line]))
        with pytest.raises(OvermindInferenceError, match="GPU OOM"):
            list(c.chat.completions.create(model="m", messages=self._MESSAGES, stream=True))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Models.list
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelsList:
    _FT_MODEL = {
        "id": "ft-test-abc",
        "object": "model",
        "created": 1700000000,
        "owned_by": "overmind",
        "finetuned": True,
        "status": "ready",
        "base_model": "meta-llama/llama-3.1-8b-instruct",
    }
    _FRONTIER = {
        "id": "anthropic/claude-sonnet-5",
        "object": "model",
        "created": 0,
        "owned_by": "anthropic",
        "finetuned": False,
        "status": "",
        "base_model": "",
    }

    def test_returns_model_list(self):
        c = _make_client()
        c._session.get = MagicMock(return_value=_mock_response(
            json_data={"object": "list", "data": [self._FT_MODEL, self._FRONTIER]}
        ))
        result = c.models.list()
        assert isinstance(result, ModelList)
        assert len(result.data) == 2

    def test_models_parsed_correctly(self):
        c = _make_client()
        c._session.get = MagicMock(return_value=_mock_response(
            json_data={"object": "list", "data": [self._FT_MODEL]}
        ))
        result = c.models.list()
        m = result.data[0]
        assert isinstance(m, Model)
        assert m.id == "ft-test-abc"
        assert m.finetuned is True
        assert m.status == "ready"

    def test_no_status_param_sends_no_query_param(self):
        c = _make_client()
        mock_get = MagicMock(return_value=_mock_response(json_data={"object": "list", "data": []}))
        c._session.get = mock_get
        c.models.list()
        params = mock_get.call_args[1]["params"]
        assert "status" not in params

    def test_status_param_forwarded(self):
        c = _make_client()
        mock_get = MagicMock(return_value=_mock_response(json_data={"object": "list", "data": []}))
        c._session.get = mock_get
        c.models.list(status="all")
        params = mock_get.call_args[1]["params"]
        assert params["status"] == "all"

    def test_sends_correct_url(self):
        c = _make_client("https://api.example.com")
        mock_get = MagicMock(return_value=_mock_response(json_data={"object": "list", "data": []}))
        c._session.get = mock_get
        c.models.list()
        assert mock_get.call_args[0][0] == "https://api.example.com/api/v1/models"

    def test_empty_data_list(self):
        c = _make_client()
        c._session.get = MagicMock(return_value=_mock_response(json_data={"object": "list", "data": []}))
        result = c.models.list()
        assert result.data == []

    def test_http_error_raises(self):
        c = _make_client()
        c._session.get = MagicMock(return_value=_mock_response(
            ok=False, status_code=401, json_data={"detail": "Unauthorized"}
        ))
        with pytest.raises(OvermindInferenceError, match="401"):
            c.models.list()

    def test_network_error_raises(self):
        c = _make_client()
        c._session.get = MagicMock(side_effect=requests.exceptions.ConnectionError("refused"))
        with pytest.raises(OvermindInferenceError, match="Failed to list models"):
            c.models.list()

    def test_finetuned_false_for_frontier_models(self):
        c = _make_client()
        c._session.get = MagicMock(return_value=_mock_response(
            json_data={"object": "list", "data": [self._FRONTIER]}
        ))
        result = c.models.list()
        assert result.data[0].finetuned is False


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Models.get
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelsGet:
    _MODEL_DATA = {
        "id": "ft-test-abc",
        "object": "model",
        "created": 1700000000,
        "owned_by": "overmind",
        "finetuned": True,
        "status": "ready",
        "base_model": "meta-llama/llama-3.1-8b-instruct",
    }

    def test_returns_model(self):
        c = _make_client()
        c._session.get = MagicMock(return_value=_mock_response(json_data=self._MODEL_DATA))
        m = c.models.get("ft-test-abc")
        assert isinstance(m, Model)
        assert m.id == "ft-test-abc"
        assert m.finetuned is True

    def test_sends_correct_url(self):
        c = _make_client("https://api.example.com")
        mock_get = MagicMock(return_value=_mock_response(json_data=self._MODEL_DATA))
        c._session.get = mock_get
        c.models.get("ft-test-abc")
        assert mock_get.call_args[0][0] == "https://api.example.com/api/v1/models/ft-test-abc"

    def test_404_raises_inference_error(self):
        c = _make_client()
        c._session.get = MagicMock(return_value=_mock_response(
            ok=False, status_code=404,
            json_data={"error": {"message": "Model not found"}},
        ))
        with pytest.raises(OvermindInferenceError, match="404"):
            c.models.get("ft-nonexistent")

    def test_network_error_raises(self):
        c = _make_client()
        c._session.get = MagicMock(side_effect=requests.exceptions.Timeout())
        with pytest.raises(OvermindInferenceError, match="Failed to retrieve model"):
            c.models.get("ft-test")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Models.delete
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelsDelete:
    _DELETE_RESPONSE = {"id": "ft-test-abc", "object": "model", "deleted": True}

    def test_returns_model_deleted(self):
        c = _make_client()
        c._session.delete = MagicMock(return_value=_mock_response(json_data=self._DELETE_RESPONSE))
        result = c.models.delete("ft-test-abc")
        assert isinstance(result, ModelDeleted)
        assert result.id == "ft-test-abc"
        assert result.deleted is True

    def test_sends_correct_url(self):
        c = _make_client("https://api.example.com")
        mock_delete = MagicMock(return_value=_mock_response(json_data=self._DELETE_RESPONSE))
        c._session.delete = mock_delete
        c.models.delete("ft-test-abc")
        assert mock_delete.call_args[0][0] == "https://api.example.com/api/v1/models/ft-test-abc"

    def test_uses_60s_timeout(self):
        c = _make_client()
        mock_delete = MagicMock(return_value=_mock_response(json_data=self._DELETE_RESPONSE))
        c._session.delete = mock_delete
        c.models.delete("ft-test-abc")
        assert mock_delete.call_args[1]["timeout"] == 60

    def test_400_raises_inference_error(self):
        c = _make_client()
        c._session.delete = MagicMock(return_value=_mock_response(
            ok=False, status_code=400,
            json_data={"error": {"message": "not a fine-tuned model"}},
        ))
        with pytest.raises(OvermindInferenceError, match="not a fine-tuned model"):
            c.models.delete("anthropic/claude-sonnet-5")

    def test_409_raises_inference_error(self):
        c = _make_client()
        c._session.delete = MagicMock(return_value=_mock_response(
            ok=False, status_code=409,
            json_data={"error": {"message": "already being deleted"}},
        ))
        with pytest.raises(OvermindInferenceError, match="409"):
            c.models.delete("ft-being-deleted")

    def test_network_error_raises(self):
        c = _make_client()
        c._session.delete = MagicMock(side_effect=requests.exceptions.ConnectionError("refused"))
        with pytest.raises(OvermindInferenceError, match="Failed to delete model"):
            c.models.delete("ft-test")
