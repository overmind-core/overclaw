import os
import sys

from rich.console import Console

from overmind.utils.io import read_api_key_masked

console = Console()

DEFAULT_BASE_URL = "https://api.overmindlab.ai"


def get_api_settings(
    overmind_api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str]:
    overmind_api_key = overmind_api_key or os.getenv("OVERMIND_API_KEY")
    base_url = base_url or os.getenv("OVERMIND_API_URL") or DEFAULT_BASE_URL

    # Avoid prompting for key if running as library or during tests
    # Detect (roughly) if running under pytest or other test envs
    _in_test = "PYTEST_CURRENT_TEST" in os.environ or any("pytest" in arg for arg in sys.argv)
    # Also don't prompt if running as a non-interactive script
    _interactive = sys.stdin.isatty() and sys.stdout.isatty()

    if not overmind_api_key:
        if _in_test or not _interactive:
            # If testing, never read or prompt for the key, just fail immediately
            raise RuntimeError("Missing OVERMIND_API_KEY. Set the environment variable to use Overmind services.")

        console.print(
            "\n[bold red]Missing OVERMIND_API_KEY.[/bold red]"
            "\n[dim]To access Overmind services, you need an API key.[/dim]"
            "\n[green]Visit[/green] [underline]https://console.overmindlab.ai/projects[/underline] [green]to create your API key.[/green]"
        )
        console.print("\nPlease paste your API key here: [bold]ovr_Xxx[/bold]")
        overmind_api_key = read_api_key_masked("OVERMIND_API_KEY")

        if not overmind_api_key:
            console.print("\n[bold red]No API key provided. Unable to continue. Exiting.[/bold red]\n")
            sys.exit(1)
        os.environ["OVERMIND_API_KEY"] = overmind_api_key
        console.print("\n[bold green]API key set successfully for this session.[/bold green]\n")

    return overmind_api_key, base_url.rstrip("/")
