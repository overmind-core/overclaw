"""Wire-contract tests for the runtime eval envelope (``overmind/evals.py``).

Each public function must emit the pinned ``overmind.eval.*`` span event with
``schema_version`` + JSON ``payload`` attributes on a recording span, and
no-op quietly without one. Uses the repo's in-memory span exporter pattern.
"""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from overmind import attrs
from overmind.evals import checkpoint, end_conversation, eval_context, expect, intent


@pytest.fixture
def inmem():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _only_event(inmem):
    provider, exporter = inmem
    provider.force_flush()
    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    return event


def _payload(event) -> dict:
    assert event.attributes[attrs.EVAL_SCHEMA_VERSION] == 1
    return json.loads(event.attributes[attrs.EVAL_PAYLOAD])


# ---------------------------------------------------------------------------
# expect()
# ---------------------------------------------------------------------------


def test_expect_emits_pinned_event_shape(inmem):
    provider, _ = inmem
    with provider.get_tracer("t").start_as_current_span("s"):
        expect("contains", "USD", id="currency", scope="span", gate=True)

    event = _only_event(inmem)
    assert event.name == "overmind.eval.expectation"
    assert _payload(event) == {
        "id": "currency",
        "kind": "contains",
        "spec": "USD",
        "scope": "span",
        "gate": True,
    }


def test_expect_defaults_and_auto_id(inmem):
    provider, _ = inmem
    with provider.get_tracer("t").start_as_current_span("s"):
        expect("regex", r"\d{4}-\d{2}-\d{2}")

    payload = _payload(_only_event(inmem))
    assert payload["scope"] == "trace"
    assert payload["gate"] is False
    assert isinstance(payload["id"], str) and len(payload["id"]) == 12


def test_expect_auto_id_stable_across_calls(inmem):
    provider, exporter = inmem
    with provider.get_tracer("t").start_as_current_span("s"):
        expect("contains", "USD")
        expect("contains", "USD")
        expect("contains", "EUR")
        expect("regex", "USD")
    provider.force_flush()

    (span,) = exporter.get_finished_spans()
    ids = [json.loads(e.attributes[attrs.EVAL_PAYLOAD])["id"] for e in span.events]
    assert ids[0] == ids[1]  # same kind+spec → same id
    assert len({ids[0], ids[2], ids[3]}) == 3  # different spec or kind → different id


def test_expect_object_spec_round_trips(inmem):
    provider, _ = inmem
    schema = {"type": "object", "required": ["amount", "currency"]}
    with provider.get_tracer("t").start_as_current_span("s"):
        expect("schema", schema)

    assert _payload(_only_event(inmem))["spec"] == schema


def test_expect_checkpoints_kind_round_trips(inmem):
    provider, _ = inmem
    path = ["pre_llm", "after_extraction", "before_return"]
    with provider.get_tracer("t").start_as_current_span("s"):
        expect("checkpoints", path, id="checkpoint-path")

    payload = _payload(_only_event(inmem))
    assert payload["kind"] == "checkpoints"
    assert payload["spec"] == path


def test_expect_validates_kind_and_scope():
    # Trust boundary: raises even with no recording span.
    with pytest.raises(ValueError, match="kind"):
        expect("vibes", "x")
    with pytest.raises(ValueError, match="scope"):
        expect("contains", "x", scope="galaxy")


# ---------------------------------------------------------------------------
# eval_context() / checkpoint() / end_conversation()
# ---------------------------------------------------------------------------


def test_eval_context_coerces_values_like_set_tag(inmem):
    provider, _ = inmem
    with provider.get_tracer("t").start_as_current_span("s"):
        eval_context(user_tier="premium", retries=3, flags={"beta": True}, missing=None)

    event = _only_event(inmem)
    assert event.name == "overmind.eval.context"
    assert _payload(event) == {
        "facts": {
            "user_tier": "premium",
            "retries": 3,
            "flags": '{"beta": true}',
            "missing": "",
        }
    }


def test_checkpoint_emits_name(inmem):
    provider, _ = inmem
    with provider.get_tracer("t").start_as_current_span("s"):
        checkpoint("payment_confirmed")

    event = _only_event(inmem)
    assert event.name == "overmind.eval.checkpoint"
    assert _payload(event) == {"name": "payment_confirmed"}


def test_intent_emits_text_and_source(inmem):
    provider, _ = inmem
    with provider.get_tracer("t").start_as_current_span("s"):
        intent("fine-tune a model on my support tickets")

    event = _only_event(inmem)
    assert event.name == "overmind.eval.intent"
    assert _payload(event) == {
        "text": "fine-tune a model on my support tickets",
        "source": "declared",
    }


def test_end_conversation_emits_empty_payload(inmem):
    provider, _ = inmem
    with provider.get_tracer("t").start_as_current_span("s"):
        end_conversation()

    event = _only_event(inmem)
    assert event.name == "overmind.eval.conversation_end"
    assert _payload(event) == {}


# ---------------------------------------------------------------------------
# No recording span → quiet no-op
# ---------------------------------------------------------------------------


def test_all_functions_noop_without_recording_span(inmem):
    _, exporter = inmem
    expect("contains", "USD")
    eval_context(user_tier="premium")
    checkpoint("step")
    intent("do the thing")
    end_conversation()
    assert exporter.get_finished_spans() == ()
