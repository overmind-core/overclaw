from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skip(reason="overmind.commands.sync_cmd has not been ported from overclaw yet")


class TestSyncCmd:
    @patch("overmind.commands.sync_cmd.load_overmind_dotenv")
    @patch("overmind.commands.sync_cmd.get_client")
    def test_exits_when_api_not_configured(
        self,
        mock_client,
        _mock_load_env,
    ):
        mock_client.return_value = None
        from overmind.commands.sync_cmd import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    @patch("overmind.commands.sync_cmd.load_overmind_dotenv")
    @patch("overmind.commands.sync_cmd.get_client")
    @patch("overmind.commands.sync_cmd.load_registry")
    @patch("overmind.commands.sync_cmd._sync_setup_artifacts")
    def test_syncs_registered_agent_with_existing_file(
        self,
        mock_sync,
        mock_registry,
        mock_client,
        _mock_load_env,
        tmp_path,
        monkeypatch,
    ):
        agent_file = tmp_path / "agents" / "a.py"
        agent_file.parent.mkdir(parents=True)
        agent_file.write_text("def run(x):\n    return {}\n", encoding="utf-8")

        mock_client.return_value = object()
        mock_registry.return_value = {
            "alpha": {
                "entrypoint": "agents.a:run",
                "file_path": str(agent_file),
                "fn_name": "run",
            }
        }
        (tmp_path / ".overmind").mkdir()
        monkeypatch.chdir(tmp_path)

        from overmind.commands.sync_cmd import main

        main()
        mock_sync.assert_called_once()
