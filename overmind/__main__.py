"""Overmind CLI — optimise LLM agents and manage agent skills.

Commands:
    init                      Configure an IDE with Overmind MCP and skills.
    optimise                  Run the optimisation loop on a registered agent.
    skills                    Manage Overmind agent skills.

Use --help with any command or subcommand for details.
"""
# Allow ``python -m overmind …`` when a different ``overmind`` script shadows PATH.

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Annotated

from overmind.instrumentation_checker import check_plan_file
from overmind.scanner import scan

try:
    import typer
    from rich.console import Console

    from overmind.optimizer import (
        API_KEY,
        API_URL,
        HEARTBEAT_INTERVAL,
        IDLE_INTERVAL,
        WORK_DIR,
        configure_logging,
        run_optimizer,
    )
    from overmind.skills import get_destination_dir, skills_app, sync_skills
except ImportError as exc:
    raise ImportError(
        "The Overmind CLI requires the 'cli' extra. Install it with: pip install 'overmind[cli]'"
    ) from exc

app = typer.Typer(
    name="overmind",
    help="Overmind CLI optimise LLM agents and manage agent skills.",
    epilog="Overmind https://overmindlab.ai is a tool for optimising LLM agents and managing agent skills.",
    pretty_exceptions_enable=True,
    no_args_is_help=True,
)

console = Console()


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Print the SDK version and exit")] = False,
) -> None:
    if version:
        from overmind import __version__

        console.print(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


current_dir = os.path.dirname(os.path.abspath(__file__))


OPTIMISE_HELP = "Register this machine as an optimiser and run the optimisation loop."


def optimise(
    api_key: Annotated[
        str,
        typer.Option(envvar="OVERMIND_API_KEY", help="Overmind API key", show_default=False),
    ] = "",
    api_url: Annotated[str, typer.Option(envvar="OVERMIND_API_URL", help="Overmind backend base URL")] = API_URL,
    cwd: Annotated[
        str,
        typer.Option(envvar="OVERMIND_CWD", help="Repo root to run commands in (defaults to the current dir)"),
    ] = "",
    poll_interval: Annotated[
        float, typer.Option(envvar="OPTIMIZER_POLL_INTERVAL", help="Idle poll seconds")
    ] = IDLE_INTERVAL,
    heartbeat_interval: Annotated[
        float,
        typer.Option(envvar="OPTIMIZER_HEARTBEAT_INTERVAL", help="Idle 'still alive' log seconds"),
    ] = HEARTBEAT_INTERVAL,
    log_level: Annotated[str, typer.Option(envvar="OPTIMIZER_LOG_LEVEL", help="DEBUG/INFO/WARNING/ERROR")] = "INFO",
):
    """
    Register with the Overmind backend and run the optimisation loop: poll for
    queued commands (from the server-side experiment FSM), run them against
    the repo in ``cwd``, and report results back.
    """
    configure_logging(log_level)
    run_optimizer(
        api_url=api_url,
        api_key=api_key or API_KEY,
        cwd=cwd or WORK_DIR,
        poll_interval=poll_interval,
        heartbeat_interval=heartbeat_interval,
    )


app.command("optimise", help=OPTIMISE_HELP)(optimise)
# ``optimize`` stays registered (hidden) so scripts written against the old
# American spelling keep working.
app.command("optimize", hidden=True, help=f"Deprecated alias for `optimise`. {OPTIMISE_HELP}")(optimise)

app.add_typer(skills_app, name="skills")


instrumentation_app = typer.Typer(help="Plan, check and smoke-test local instrumentation placements.")
app.add_typer(instrumentation_app, name="instrumentation")

DEFAULT_API_URL = "https://api.overmindlab.ai"


def _mcp_call(api_url: str, api_key: str, tool: str, arguments: dict, timeout: int) -> dict:
    """Call one MCP tool over JSON-RPC and return its decoded JSON result."""
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    request = urllib.request.Request(
        api_url.rstrip("/") + "/api/mcp/",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Api-Key": api_key},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return json.loads(payload["result"]["content"][0]["text"])


@instrumentation_app.command("scan")
def instrumentation_scan(
    root: Annotated[Path, typer.Option("--root", help="Source repository root")] = Path("."),
    out: Annotated[Path | None, typer.Option("--out", help="Write JSON here instead of stdout")] = None,
):
    """AST-scan the repo for instrumentation candidates (no imports, no network)."""
    payload = json.dumps(scan(str(root)), sort_keys=True)
    if out is not None:
        out.write_text(payload)
    else:
        typer.echo(payload)


@instrumentation_app.command("plan")
def instrumentation_plan(
    root: Annotated[Path, typer.Option("--root", help="Source repository root")] = Path("."),
    out: Annotated[Path, typer.Option("--out", help="Write the placement plan JSON here")] = Path("plan.json"),
    candidates_out: Annotated[Path, typer.Option("--candidates-out", help="Also write the scan output here")] = Path(
        "candidates.json"
    ),
    capability: Annotated[str, typer.Option("--capability", help="Restrict planning to one capability")] = "",
    api_url: Annotated[
        str, typer.Option(envvar="OVERMIND_API_URL", help="Overmind backend base URL")
    ] = DEFAULT_API_URL,
    api_key: Annotated[str, typer.Option(envvar="OVERMIND_API_KEY", help="Project API key", show_default=False)] = "",
):
    """Scan the repo and mint the whole-repo placement plan over MCP in one step."""
    candidates = scan(str(root))
    candidates_out.write_text(json.dumps(candidates, sort_keys=True))
    arguments: dict = {"candidates": candidates}
    if capability:
        arguments["capability_name_or_slug"] = capability
    result = _mcp_call(api_url, api_key, "plan_instrumentation", arguments, timeout=180)
    if result.get("errors"):
        typer.echo(json.dumps(result, indent=1))
        raise typer.Exit(1)
    out.write_text(json.dumps(result, indent=1))
    typer.echo(
        json.dumps(
            {
                "placements": len(result.get("placements") or []),
                "plans": result.get("plans"),
                "ambiguous": result.get("ambiguous"),
                "dropped": result.get("dropped"),
                "minted": result.get("minted"),
                "plan_file": str(out),
            },
            indent=1,
        )
    )


@instrumentation_app.command("check")
def instrumentation_check(
    plan_file: Annotated[Path, typer.Option("--plan-file", help="MCP placements plan JSON")],
    root: Annotated[Path, typer.Option("--root", help="Source repository root")] = Path("."),
    output_format: Annotated[str, typer.Option("--format", help="Output format: text or json")] = "text",
):
    """Statically validate that the source matches the plan's placements."""
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("must be text or json", param_hint="--format")
    result = check_plan_file(plan_file, root)
    if output_format == "json":
        typer.echo(json.dumps(result, sort_keys=True))
    else:
        for check in result["checks"]:
            location = " ".join(str(check[field]) for field in ("file", "qualname") if check.get(field))
            typer.echo(f"{check['status'].upper()} {check['code']} {location} {check['message']}".rstrip())
        summary = result["summary"]
        typer.echo(
            f"{'PASS' if result['ok'] else 'FAIL'} {summary['passed']} passed, "
            f"{summary['failed']} failed, {summary['skipped']} skipped"
        )
    if not result["ok"]:
        raise typer.Exit(1)


@instrumentation_app.command("smoke")
def instrumentation_smoke(
    plan_file: Annotated[Path, typer.Option("--plan-file", help="MCP placements plan JSON")],
    out: Annotated[Path, typer.Option("--out", help="Trace output file for smoke-tested placements")],
    root: Annotated[Path, typer.Option("--root", help="Source repository root")] = Path("."),
):
    """Run each placement's smoke_script with stubbed providers, spans to --out."""
    plan = json.loads(plan_file.read_text())
    placements = plan.get("placements", plan) if isinstance(plan, dict) else plan
    if not isinstance(placements, list):
        placements = [placements]

    failed = False
    for placement in placements:
        if not isinstance(placement, dict):
            continue
        smoke_script = placement.get("smoke_script")
        smoke_hint = placement.get("smoke_hint")
        if smoke_script:
            script_path = Path(smoke_script)
            if not script_path.is_absolute():
                script_path = root / script_path
            if not script_path.exists():
                continue
            env = {**os.environ, "OVERMIND_SMOKE": "1", "OVERMIND_TRACE_FILE": str(out)}
            # Run through the interpreter: agent-written scripts have no shebang/exec bit.
            runner = [sys.executable, str(script_path)] if script_path.suffix == ".py" else [str(script_path)]
            if subprocess.run(runner, cwd=root, env=env, check=False).returncode != 0:
                failed = True
        elif smoke_hint:
            target = placement.get("target") if isinstance(placement.get("target"), dict) else placement
            location = " ".join(str(target.get(field)) for field in ("file", "qualname") if target.get(field))
            typer.echo(f"{f'TODO {location}' if location else 'TODO'}: {smoke_hint}")

    if failed:
        raise typer.Exit(1)


@instrumentation_app.command("verify")
def instrumentation_verify(
    spans_file: Annotated[Path, typer.Option("--spans-file", help="JSONL spans from a smoke run")],
    capability: Annotated[str, typer.Option("--capability", help="Capability name or slug fallback")] = "",
    api_url: Annotated[
        str, typer.Option(envvar="OVERMIND_API_URL", help="Overmind backend base URL")
    ] = DEFAULT_API_URL,
    api_key: Annotated[str, typer.Option(envvar="OVERMIND_API_KEY", help="Project API key", show_default=False)] = "",
):
    """Send smoke-run spans to verify_instrumentation_spans over MCP and print the verdict."""
    arguments: dict = {"spans": [json.loads(line) for line in spans_file.read_text().splitlines() if line.strip()]}
    if capability:
        arguments["capability_name_or_slug"] = capability
    result = _mcp_call(api_url, api_key, "verify_instrumentation_spans", arguments, timeout=60)
    typer.echo(json.dumps(result, indent=1))
    tasks = result.get("tasks") or []
    if not tasks or not all(task.get("binding_source") == "declared" for task in tasks):
        raise typer.Exit(1)


MCP_URLS = {
    "staging": "https://staging.overmindlab.ai/api/mcp/",
    "development": "http://localhost:8000/api/mcp/",
    "local": "http://localhost:8000/api/mcp/",
    "dev": "http://localhost:8000/api/mcp/",
}


def _write_codex_mcp(path: Path, url: str, api_key: str | None = None) -> None:
    auth = (
        'env_http_headers = { "X-Api-Key" = "OVERMIND_API_KEY" }'
        if api_key is None
        else f'http_headers = {{ "X-Api-Key" = {json.dumps(api_key)} }}'
    )
    block = "\n".join([
        "[mcp_servers.overmind]",
        f"url = {json.dumps(url)}",
        auth,
    ])
    existing = path.read_text() if path.exists() else ""
    existing = re.sub(
        r"(?ms)^\[\[?mcp_servers\.overmind(?:\.[^\]]+)?\]\]?\s*\n.*?(?=^\s*\[|\Z)",
        "",
        existing,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{existing.rstrip()}\n\n{block}\n" if existing.strip() else f"{block}\n")


def _report_mcp_key_check(url: str, api_key: str) -> None:
    """Probe ``tools/list`` and report the verdict. A wrong or revoked key is
    indistinguishable from "MCP not configured" once inside a coding agent, so
    it has to fail loudly while a human is still watching."""
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(),
            headers={"Content-Type": "application/json", "X-Api-Key": api_key or ""},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            status = response.status
    except Exception as exc:  # never block the config write
        status = getattr(exc, "code", None)
        if status is None:
            console.print(f"[yellow]could not reach {url}: {exc}[/yellow]")
            return
    if status == 200:
        console.print("MCP key check: ok")
    else:
        console.print(
            f"[red]MCP key check FAILED (HTTP {status}) — the configured API key is not valid for "
            f"{url}. Fix OVERMIND_API_KEY before starting the coding agent.[/red]"
        )


def overmind_init(
    env: Annotated[str, typer.Option(help="production, staging or dev")] = "production",
    api_key: Annotated[str, typer.Option(envvar="OVERMIND_API_KEY", help="Overmind API key")] = API_KEY,
    ide: Annotated[str, typer.Option(..., help="cursor, claude, claude_code, opencode or codex")] = "cursor",
):
    """
    Add (or update) the overmind MCP server config and install the overmind
    skill, leaving any other configured servers untouched.
    """
    url = MCP_URLS.get(env, "https://api.overmindlab.ai/api/mcp/")
    dest = get_destination_dir(ide)

    if ide == "codex":
        if env in {"local", "development", "dev"}:
            raise typer.BadParameter("Codex setup supports production or staging", param_hint="--env")
        if not os.environ.get("OVERMIND_API_KEY") and not api_key:
            raise typer.BadParameter("set OVERMIND_API_KEY before running init", param_hint="OVERMIND_API_KEY")
        mcp_path = Path.cwd() / ".codex" / "config.toml"
        _write_codex_mcp(mcp_path, url, None if os.environ.get("OVERMIND_API_KEY") else api_key)
        console.print(f"overmind MCP server written to {mcp_path}")
        sync_skills(["overmind"], ide=ide)
        console.print(f"overmind skill installed to {dest}/skills/overmind")
        _report_mcp_key_check(url, api_key)
        return

    if ide == "opencode":
        # opencode reads MCP servers from project-root opencode.json, not <dest>/mcp.json
        mcp_path = Path.cwd() / "opencode.json"
        config = (
            json.loads(mcp_path.read_text()) if mcp_path.exists() else {"$schema": "https://opencode.ai/config.json"}
        )
        config.setdefault("mcp", {})["overmind"] = {
            "type": "remote",
            "url": url,
            "enabled": True,
            "headers": {"X-Api-Key": api_key},
        }
    else:
        mcp_path = Path.cwd() / dest / "mcp.json"
        config = json.loads(mcp_path.read_text()) if mcp_path.exists() else {}
        config.setdefault("mcpServers", {})["overmind"] = {
            "url": url,
            "headers": {"X-Api-Key": api_key},
        }

    mcp_path.parent.mkdir(parents=True, exist_ok=True)
    mcp_path.write_text(json.dumps(config, indent=2) + "\n")
    console.print(f"overmind MCP server written to {mcp_path.name if ide == 'opencode' else f'{dest}/mcp.json'}")

    sync_skills(["overmind"], ide=ide)
    console.print(f"overmind skill installed to {dest}/skills/overmind")
    _report_mcp_key_check(url, api_key)

    # claude mcp add --transport http corridor https://app.corridor.dev/api/mcp --header "Authorization: Bearer ..."


app.command("init")(overmind_init)


if __name__ == "__main__":
    app()
