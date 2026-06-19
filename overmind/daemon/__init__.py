"""Client-side optimization daemon.

A thin polling loop that registers a :class:`ClientSession` with the Overmind
backend, claims the primitive commands the server's orchestrator enqueues
(``upload_bundle`` / ``run_command`` / ``apply_patch`` / ``reset``), executes
them against the user's real repo, and reports results back. The server owns all
the optimization logic; this process only executes and reports.
"""

from overmind.daemon.main import poll_once, run_daemon

__all__ = ["run_daemon", "poll_once"]
