from __future__ import annotations

from unittest.mock import patch

import pytest


class TestSyncOptimizeCmd:
    @patch("overclaw.commands.sync_optimize_cmd.load_overclaw_dotenv")
    @patch("overclaw.commands.sync_optimize_cmd.get_client")
    @patch("overclaw.commands.sync_optimize_cmd.get_project_id")
    def test_exits_when_api_not_configured(
        self,
        mock_project_id,
        mock_client,
        _mock_load_env,
    ):
        mock_client.return_value = None
        mock_project_id.return_value = None
        from overclaw.commands.sync_optimize_cmd import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    @patch("overclaw.commands.sync_optimize_cmd.load_overclaw_dotenv")
    @patch("overclaw.commands.sync_optimize_cmd.get_client")
    @patch("overclaw.commands.sync_optimize_cmd.get_project_id")
    @patch("overclaw.commands.sync_optimize_cmd.load_registry")
    @patch("overclaw.commands.sync_optimize_cmd._sync_optimize_artifacts_for_agent")
    def test_syncs_registered_agent_with_existing_file(
        self,
        mock_sync,
        mock_registry,
        mock_project_id,
        mock_client,
        _mock_load_env,
        tmp_path,
        monkeypatch,
    ):
        agent_file = tmp_path / "agents" / "a.py"
        agent_file.parent.mkdir(parents=True)
        agent_file.write_text("def run(x):\n    return {}\n", encoding="utf-8")

        mock_client.return_value = object()
        mock_project_id.return_value = "project-id"
        mock_registry.return_value = {
            "alpha": {
                "entrypoint": "agents.a:run",
                "file_path": str(agent_file),
                "fn_name": "run",
            }
        }
        mock_sync.return_value = True
        monkeypatch.chdir(tmp_path)

        from overclaw.commands.sync_optimize_cmd import main

        main()
        mock_sync.assert_called_once()
