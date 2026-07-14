"""Overmind inference client — OpenAI-compatible `.chat.completions.create()` interface.

Usage::

    import overmind

    client = overmind.Client(api_key="om_...", base_url="https://api.overmindlab.ai")

    # Non-streaming
    response = client.chat.completions.create(
        model="ft-llama31-abc12345",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)

    # Streaming
    for chunk in client.chat.completions.create(
        model="ft-llama31-abc12345",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ):
        print(chunk.choices[0].delta.content or "", end="")

    # List all deployed models (any status)
    models = client.models.list(status="all")
    for m in models.data:
        print(m.id, m.status, m.base_model)

    # Get a single model's details
    model = client.models.get("ft-llama31-abc12345")
    print(model.status, model.base_model)

    # Delete a fine-tuned model
    result = client.models.delete("ft-llama31-abc12345")
    print(result.deleted)  # True

Both ``api_key`` and ``base_url`` fall back to the ``OVERMIND_API_KEY`` and
``OVERMIND_API_URL`` environment variables if not passed explicitly.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_BASE_URL = "https://api.overmindlab.ai"


# ── Response dataclasses ──────────────────────────────────────────────────────


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class Choice:
    index: int
    message: ChatMessage
    finish_reason: str | None = None


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    id: str
    object: str
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    created: int = 0


@dataclass
class Delta:
    role: str | None = None
    content: str | None = None


@dataclass
class ChunkChoice:
    index: int
    delta: Delta
    finish_reason: str | None = None


@dataclass
class ChatCompletionChunk:
    id: str
    object: str
    model: str
    choices: list[ChunkChoice]
    created: int = 0


@dataclass
class Model:
    id: str
    object: str
    created: int
    owned_by: str
    base_model: str = ""
    finetuned: bool = False
    status: str = ""


@dataclass
class ModelList:
    object: str
    data: list[Model] = field(default_factory=list)


@dataclass
class ModelDeleted:
    id: str
    object: str = "model"
    deleted: bool = True


# ── Parsing helpers ───────────────────────────────────────────────────────────


def _parse_chat_completion(data: dict) -> ChatCompletion:
    choices = [
        Choice(
            index=c.get("index", i),
            message=ChatMessage(
                role=c["message"]["role"],
                content=c["message"].get("content") or "",
            ),
            finish_reason=c.get("finish_reason"),
        )
        for i, c in enumerate(data.get("choices", []))
    ]
    raw_usage = data.get("usage") or {}
    usage = (
        Usage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )
        if raw_usage
        else None
    )
    return ChatCompletion(
        id=data.get("id", ""),
        object=data.get("object", "chat.completion"),
        model=data.get("model", ""),
        choices=choices,
        usage=usage,
        created=data.get("created", 0),
    )


def _parse_chunk(data: dict) -> ChatCompletionChunk:
    choices = [
        ChunkChoice(
            index=c.get("index", i),
            delta=Delta(
                role=c.get("delta", {}).get("role"),
                content=c.get("delta", {}).get("content"),
            ),
            finish_reason=c.get("finish_reason"),
        )
        for i, c in enumerate(data.get("choices", []))
    ]
    return ChatCompletionChunk(
        id=data.get("id", ""),
        object=data.get("object", "chat.completion.chunk"),
        model=data.get("model", ""),
        choices=choices,
        created=data.get("created", 0),
    )


def _iter_sse_chunks(response: requests.Response) -> Iterator[ChatCompletionChunk]:
    """Parse a streaming SSE response into ``ChatCompletionChunk`` objects."""
    for raw_line in response.iter_lines():
        line: str = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "error" in data:
            raise OvermindInferenceError(data["error"].get("message", "Inference error"))
        yield _parse_chunk(data)


# ── Sub-resource classes ──────────────────────────────────────────────────────


class OvermindInferenceError(Exception):
    """Raised when the platform returns an error response."""


class ChatCompletions:
    def __init__(self, client: Client) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion.

        Args:
            model: The ``model_id`` of a deployed fine-tuned model.
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            stream: If ``True``, returns an iterator of ``ChatCompletionChunk``
                objects instead of a single ``ChatCompletion``.
            temperature: Sampling temperature (default 1.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Extra parameters forwarded to the backend
                (e.g. ``top_p``, ``frequency_penalty``, ``presence_penalty``).

        Returns:
            A ``ChatCompletion`` when ``stream=False``, or an iterator of
            ``ChatCompletionChunk`` objects when ``stream=True``.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = f"{self._client._base_url}/api/v1/chat/completions"
        timeout = self._client._timeout

        try:
            if stream:
                resp = self._client._session.post(url, json=payload, stream=True, timeout=timeout)
                _raise_for_status(resp)
                return _iter_sse_chunks(resp)

            resp = self._client._session.post(url, json=payload, timeout=timeout)
            _raise_for_status(resp)
            return _parse_chat_completion(resp.json())
        except requests.exceptions.Timeout as exc:
            raise OvermindInferenceError(
                f"Request timed out after {timeout}s. "
                "The model may be cold-starting on the GPU — try again in a moment."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OvermindInferenceError(f"Request failed: {exc}") from exc


class Chat:
    def __init__(self, client: Client) -> None:
        self.completions = ChatCompletions(client)


def _parse_model(m: dict) -> Model:
    return Model(
        id=m["id"],
        object=m.get("object", "model"),
        created=m.get("created", 0),
        owned_by=m.get("owned_by", "overmind"),
        base_model=m.get("base_model", ""),
        finetuned=bool(m.get("finetuned", False)),
        status=m.get("status", ""),
    )


class Models:
    def __init__(self, client: Client) -> None:
        self._client = client

    def list(self, *, status: str | None = None) -> ModelList:
        """Return deployed models accessible with the current API key.

        Args:
            status: Filter fine-tuned models by deployment status.
                    One of ``"ready"`` (default), ``"queued"``,
                    ``"deploying"``, ``"warming"``, ``"failed"``,
                    ``"deleting"``, ``"deleted"``, or ``"all"``
                    to include every status. Frontier models are
                    always included regardless of this filter.
        """
        url = f"{self._client._base_url}/api/v1/models"
        params: dict[str, str] = {}
        if status is not None:
            params["status"] = status
        try:
            resp = self._client._session.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            raise OvermindInferenceError(f"Failed to list models: {exc}") from exc
        _raise_for_status(resp)
        raw = resp.json()
        return ModelList(
            object=raw.get("object", "list"),
            data=[_parse_model(m) for m in raw.get("data", [])],
        )

    def get(self, model_id: str) -> Model:
        """Retrieve a single fine-tuned model by its ``model_id``.

        Returns the model's full metadata including current deployment status.
        Raises ``OvermindInferenceError`` with HTTP 404 if the model is not
        found or not accessible with the current API key.
        """
        url = f"{self._client._base_url}/api/v1/models/{model_id}"
        try:
            resp = self._client._session.get(url, timeout=30)
        except requests.exceptions.RequestException as exc:
            raise OvermindInferenceError(f"Failed to retrieve model '{model_id}': {exc}") from exc
        _raise_for_status(resp)
        return _parse_model(resp.json())

    def delete(self, model_id: str) -> ModelDeleted:
        """Stop serving a fine-tuned model and remove its weights.

        The model must belong to a project accessible with the current API
        key. Frontier models cannot be deleted via this method.

        Raises ``OvermindInferenceError`` if the model is not found,
        already being deleted, or the backend reports an error.
        """
        url = f"{self._client._base_url}/api/v1/models/{model_id}"
        try:
            resp = self._client._session.delete(url, timeout=60)
        except requests.exceptions.RequestException as exc:
            raise OvermindInferenceError(f"Failed to delete model '{model_id}': {exc}") from exc
        _raise_for_status(resp)
        raw = resp.json()
        return ModelDeleted(
            id=raw.get("id", model_id),
            object=raw.get("object", "model"),
            deleted=bool(raw.get("deleted", True)),
        )


# ── Main client ───────────────────────────────────────────────────────────────


def _raise_for_status(resp: requests.Response) -> None:
    if resp.ok:
        return
    try:
        detail = resp.json()
        msg = detail.get("error", {}).get("message") or detail.get("detail") or resp.text[:400]
    except Exception:
        msg = resp.text[:400]
    raise OvermindInferenceError(f"HTTP {resp.status_code}: {msg}")


class Client:
    """Overmind inference client with an OpenAI-compatible interface.

    Args:
        api_key: Overmind API key. Falls back to ``OVERMIND_API_KEY`` env var.
        base_url: Base URL of the Overmind platform. Falls back to
            ``OVERMIND_API_URL`` env var, then ``https://api.overmindlab.ai``.

    Example::

        client = overmind.Client()  # reads env vars automatically

        response = client.chat.completions.create(
            model="ft-llama31-abc12345",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        print(response.choices[0].message.content)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 300,
    ) -> None:
        resolved_key = api_key or os.environ.get("OVERMIND_API_KEY")
        if not resolved_key:
            raise ValueError("Missing API key. Pass api_key= or set the OVERMIND_API_KEY environment variable.")
        resolved_url = (base_url or os.environ.get("OVERMIND_API_URL") or DEFAULT_BASE_URL).rstrip("/")

        self._base_url = resolved_url
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "X-Api-Key": resolved_key,
            "Content-Type": "application/json",
        })

        self.chat = Chat(self)
        self.models = Models(self)
