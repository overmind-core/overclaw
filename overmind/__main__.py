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
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from overmind.instrumentation_checker import check_plan_file
from overmind.optimizer import (
    API_KEY,
    API_URL,
    HEARTBEAT_INTERVAL,
    IDLE_INTERVAL,
    WORK_DIR,
    configure_logging,
    run_optimizer,
)
from overmind.scanner import scan
from overmind.skills import get_destination_dir, skills_app, sync_skills

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
    version: Annotated[
        bool, typer.Option("--version", help="Print the SDK version and exit")
    ] = False,
) -> None:
    if version:
        from importlib.metadata import version as pkg_version

        console.print(pkg_version("overmind"))
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


instrumentation_app = typer.Typer(name="instrumentation", help="Check local AST instrumentation placements.")


def instrumentation_check(
    plan_file: Annotated[Path, typer.Option("--plan-file", help="MCP placements plan JSON")],
    root: Annotated[Path, typer.Option("--root", help="Source repository root")] = Path("."),
    output_format: Annotated[str, typer.Option("--format", help="Output format: text or json")] = "text",
):
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
            f"{('PASS' if result['ok'] else 'FAIL')} {summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped"
        )
    if not result["ok"]:
        raise typer.Exit(1)


instrumentation_app.command("check")(instrumentation_check)


def instrumentation_scan(
    root: Annotated[Path, typer.Option("--root", help="Source repository root")] = Path("."),
    out: Annotated[Path | None, typer.Option("--out", help="Write JSON here instead of stdout")] = None,
):
    result = scan(str(root))
    payload = json.dumps(result, sort_keys=True)
    if out is not None:
        out.write_text(payload)
    else:
        typer.echo(payload)


instrumentation_app.command("scan")(instrumentation_scan)


def instrumentation_smoke(
    plan_file: Annotated[Path, typer.Option("--plan-file", help="MCP placements plan JSON")],
    out: Annotated[Path, typer.Option("--out", help="Trace output file for smoke-tested placements")],
    root: Annotated[Path, typer.Option("--root", help="Source repository root")] = Path("."),
):
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
            completed = subprocess.run(runner, cwd=root, env=env, check=False)
            if completed.returncode != 0:
                failed = True
        elif smoke_hint:
            target = placement.get("target") if isinstance(placement.get("target"), dict) else placement
            location = " ".join(str(target.get(field)) for field in ("file", "qualname") if target.get(field))
            prefix = f"TODO {location}" if location else "TODO"
            typer.echo(f"{prefix}: {smoke_hint}")

    if failed:
        raise typer.Exit(1)


instrumentation_app.command("smoke")(instrumentation_smoke)


def instrumentation_verify(
    spans_file: Annotated[Path, typer.Option("--spans-file", help="JSONL spans from a smoke run")],
    capability: Annotated[str, typer.Option("--capability", help="Capability name or slug fallback")] = "",
    api_url: Annotated[str, typer.Option(envvar="OVERMIND_API_URL", help="Overmind backend base URL")] = "https://api.overmindlab.ai",
    api_key: Annotated[str, typer.Option(envvar="OVERMIND_API_KEY", help="Project API key", show_default=False)] = "",
):
    """Send smoke-run spans to verify_instrumentation_spans over MCP and print the verdict."""
    import urllib.request

    spans = [json.loads(line) for line in spans_file.read_text().splitlines() if line.strip()]
    arguments: dict = {"spans": spans}
    if capability:
        arguments["capability_name_or_slug"] = capability
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "verify_instrumentation_spans", "arguments": arguments},
    }
    req = urllib.request.Request(
        api_url.rstrip("/") + "/api/mcp/",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Api-Key": api_key},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    text = payload["result"]["content"][0]["text"]
    result = json.loads(text)
    typer.echo(json.dumps(result, indent=1))
    tasks = result.get("tasks") or []
    ok = bool(tasks) and all(t.get("binding_source") == "declared" for t in tasks)
    if not ok:
        raise typer.Exit(1)


instrumentation_app.command("verify")(instrumentation_verify)
app.add_typer(instrumentation_app)


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

    # A wrong or revoked key is indistinguishable from "MCP not configured" once
    # inside a coding agent, so validate it while a human can still see the error.
    try:
        import urllib.request

        req = urllib.request.Request(
            url,
            data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(),
            headers={"Content-Type": "application/json", "X-Api-Key": api_key or ""},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
    except Exception as exc:  # noqa: BLE001 — report, never block config writing
        status = getattr(exc, "code", None)
        if status is None:
            console.print(f"[yellow]could not reach {url}: {exc}[/yellow]")
    if status == 200:
        console.print("MCP key check: ok")
    elif status is not None:
        console.print(
            f"[red]MCP key check FAILED (HTTP {status}) — the configured API key is not "
            f"valid for {url}. Fix OVERMIND_API_KEY before starting the coding agent.[/red]"
        )

    # claude mcp add --transport http corridor https://app.corridor.dev/api/mcp --header "Authorization: Bearer ..."


app.command("init")(overmind_init)


if __name__ == "__main__":
    app()
