"""Client-side coordinator for the server-orchestrated optimizer.

The server is the brain; this module is the thin client face of it. The full
loop the server drives is::

    github repo -> Code Context -> Agents -> Dataset for each agent -> Entrypoint
      -> for each agent: Command Execute
         -> If Fail: Retry generating the command
         -> If Pass: baseline score -> suggest code changes -> run new code -> new score
      -> once a whole candidate passes (n commands, n = number of datapoints), run evals
         -> New score better: ACCEPT, keep the diff, continue iterating
         -> New score worse:  ROLLBACK the diff, try another candidate
         -> New score same:   REJECT the diff, try another candidate

``overmind start`` runs the bare daemon (:mod:`overmind.daemon`). ``optimize``
below is the all-in-one convenience path: it registers a session, asks the
server to start a run, then drives the same poll/execute loop while streaming
run status (and answering ``waiting_user`` criteria prompts) until the run
reaches a terminal state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from overmind.daemon import handlers
from overmind.daemon.api import DaemonAPI, resolve_base_url, resolve_token
from overmind.daemon.main import _cli_version, poll_once

_TERMINAL = {"completed", "failed", "cancelled"}


def _resolve_agent_id(agent_name: str) -> str:
    from overmind.core.registry import get_agent_id

    agent_id = get_agent_id(agent_name)
    if not agent_id:
        raise SystemExit(
            f"Agent {agent_name!r} has no server id locally. Register it with "
            "`overmind agent register <name> <module:function>` first."
        )
    return agent_id


def _prompt_yes_no(question: str) -> bool:
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


@dataclass
class ClientCoordinator:
    """Drives a single server-side optimization run from the client."""

    agent_name: str
    poll_interval: float = 2.0
    auto_approve: bool = False

    def optimize(self, **config) -> dict:
        token = resolve_token()
        if not token:
            raise SystemExit("OVERMIND_API_KEY is not set. Run `overmind init` first.")

        agent_id = _resolve_agent_id(self.agent_name)
        api = DaemonAPI(resolve_base_url(), token)
        ctx = handlers.HandlerContext.create(agent_name=self.agent_name)
        try:
            session = api.register_session(
                agent_name=self.agent_name, cli_version=_cli_version()
            )
            session_id = session["id"]
            run = api.start_run(agent_id=agent_id, client_session=session_id, **config)
            run_id = run["id"]
            print(f"Started optimization run {run_id[:8]} for {self.agent_name}.")
            return self._drive(api, ctx, session_id, run_id)
        finally:
            api.close()

    def _drive(self, api: DaemonAPI, ctx: handlers.HandlerContext, session_id: str, run_id: str) -> dict:
        last_phase: str | None = None
        run: dict = {}
        while True:
            poll_once(api, session_id, ctx, agent_name=self.agent_name)
            run = api.get_run(run_id)
            phase = run.get("phase")
            if phase != last_phase:
                print(
                    f"  [{run.get('status')}] phase={phase} iter={run.get('iteration')} "
                    f"baseline={run.get('baseline_score')} best={run.get('best_score')}"
                )
                last_phase = phase

            if run.get("status") == "waiting_user":
                approved = self.auto_approve or _prompt_yes_no(
                    "Approve evaluation criteria and continue?"
                )
                api.respond_run(run_id, approved=approved)
                if not approved:
                    print("Criteria rejected; cancelling run.")
                    api.cancel_run(run_id)

            if run.get("status") in _TERMINAL:
                break
            time.sleep(self.poll_interval)

        print(
            f"Run {run.get('status')}: best={run.get('best_score')} "
            f"improvement={run.get('improvement')}"
        )
        report = run.get("report_markdown")
        if report:
            print("\n" + report)
        pr_url = (run.get("agent") or {}).get("pr_url") if isinstance(run.get("agent"), dict) else None
        if pr_url:
            print(f"PR: {pr_url}")
        return run


def optimize(agent_name: str, *, poll_interval: float = 2.0, auto_approve: bool = False, **config) -> dict:
    """Convenience wrapper used by the ``overmind remote-optimize`` CLI command."""
    return ClientCoordinator(
        agent_name=agent_name, poll_interval=poll_interval, auto_approve=auto_approve
    ).optimize(**config)
