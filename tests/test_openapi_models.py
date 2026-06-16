"""OpenAPI model package exports required for CLI deserialization."""

from overmind.openapi_client.models import (
    ClientSession,
    ClientSessionPollResponse,
    WorkflowRun,
)


def test_cli_models_exported_from_models_package():
    assert ClientSession.__name__ == "ClientSession"
    assert ClientSessionPollResponse.__name__ == "ClientSessionPollResponse"
    assert WorkflowRun.__name__ == "WorkflowRun"
