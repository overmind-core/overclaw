"""``overmind platform`` — list, describe, and call platform tools via MCP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from overmind.platform.client import PlatformClient, PlatformError
from overmind.platform.types import KNOWN_DOMAINS, ToolSummary

platform_app = typer.Typer(
    help="Overmind platform API — discover and call tools without loading 145 MCP schemas.",
    no_args_is_help=True,
)
console = Console()


def _client(
    api_key: str,
    api_url: str,
) -> PlatformClient:
    return PlatformClient(api_key=api_key or None, base_url=api_url or None)


def _first_line(text: str) -> str:
    line = text.splitlines()[0].strip() if text else ""
    return line[:120] + ("…" if len(line) > 120 else "")


def _print_json(data: Any) -> None:
    console.print_json(json.dumps(data, default=str))


def _load_arguments(args: str | None, args_file: Path | None) -> dict[str, Any]:
    if args_file is not None and args is not None:
        raise typer.BadParameter("use only one of --args or --args-file")
    if args_file is not None:
        try:
            return json.loads(args_file.read_text())
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"invalid JSON in {args_file}: {exc}") from exc
    if args is not None:
        try:
            return json.loads(args)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"invalid JSON in --args: {exc}") from exc
    return {}


@platform_app.command("list", help="List platform tools (name + one-line description).")
def list_tools(
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON array")] = False,
    domain: Annotated[
        str | None,
        typer.Option(help="Filter by domain: evals, workshop, finetune, builds, capabilities, …"),
    ] = None,
    api_key: Annotated[
        str,
        typer.Option(envvar="OVERMIND_API_KEY", help="Overmind API key", show_default=False),
    ] = "",
    api_url: Annotated[str, typer.Option(envvar="OVERMIND_API_URL", help="Overmind backend base URL")] = "",
):
    if domain is not None and domain not in KNOWN_DOMAINS:
        raise typer.BadParameter(
            f"unknown domain {domain!r}; expected one of: {', '.join(sorted(KNOWN_DOMAINS))}",
            param_hint="--domain",
        )
    try:
        tools = _client(api_key, api_url).list_tools()
    except PlatformError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)

    if domain is not None:
        tools = [t for t in tools if t.domain == domain]

    if json_output:
        _print_json([{"name": t.name, "description": _first_line(t.description), "domain": t.domain} for t in tools])
        return

    table = Table(title="Overmind platform tools")
    table.add_column("Name", style="bold cyan")
    table.add_column("Domain", style="green")
    table.add_column("Description", style="dim")
    for tool in tools:
        table.add_row(tool.name, tool.domain, _first_line(tool.description))
    console.print(table)


@platform_app.command("describe", help="Show the input schema for one platform tool.")
def describe_tool(
    tool: Annotated[str, typer.Argument(help="Tool name (e.g. create_eval_run)")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON schema")] = False,
    api_key: Annotated[
        str,
        typer.Option(envvar="OVERMIND_API_KEY", help="Overmind API key", show_default=False),
    ] = "",
    api_url: Annotated[str, typer.Option(envvar="OVERMIND_API_URL", help="Overmind backend base URL")] = "",
):
    try:
        detail = _client(api_key, api_url).describe_tool(tool)
    except PlatformError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)

    payload = {
        "name": detail.name,
        "domain": detail.domain,
        "description": detail.description,
        "inputSchema": detail.input_schema,
    }
    if json_output:
        _print_json(payload)
        return

    console.print(f"[bold cyan]{detail.name}[/bold cyan] ({detail.domain})")
    if detail.description:
        console.print(detail.description)
    console.print_json(json.dumps(detail.input_schema))


@platform_app.command("call", help="Execute a platform tool (writes run immediately — no chat confirm card).")
def call_tool(
    tool: Annotated[str, typer.Argument(help="Tool name")],
    args: Annotated[str | None, typer.Option("--args", help="JSON object of tool arguments")] = None,
    args_file: Annotated[
        Path | None,
        typer.Option("--args-file", help="Path to a JSON file of tool arguments (for connector creds)"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print structured JSON result")] = False,
    api_key: Annotated[
        str,
        typer.Option(envvar="OVERMIND_API_KEY", help="Overmind API key", show_default=False),
    ] = "",
    api_url: Annotated[str, typer.Option(envvar="OVERMIND_API_URL", help="Overmind backend base URL")] = "",
):
    arguments = _load_arguments(args, args_file)
    try:
        result = _client(api_key, api_url).call_tool(tool, arguments)
    except PlatformError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1)

    if json_output:
        _print_json({
            "isError": result.is_error,
            "content": result.content,
            "structuredContent": result.structured_content,
        })
        return

    text = result.text()
    if text:
        try:
            parsed = json.loads(text)
            console.print_json(json.dumps(parsed))
        except json.JSONDecodeError:
            console.print(text)
    elif result.structured_content is not None:
        console.print_json(json.dumps(result.structured_content, default=str))
    if result.is_error:
        raise typer.Exit(code=1)
