# build-id: oa-932b4c72
"""T18: committee follow-up on the Zetaris merge: fail closed on org id and alias, log notes."""
import io

import pytest

from agent.config import Config
from agent.errors import ConfigError
from agent.sources import DEFAULT_LIMIT, VENDOR_CLIENT_LIMIT, ZetarisSource
from agent.trace import TraceWriter

BASE = {"SAMPLE_MODE": "false", "ZETARIS_API_KEY": "zk-T18-KEY-0000", "LIVE_PROVIDER": "sample"}


def _cfg(**extra):
    env = dict(BASE)
    env.update(extra)
    return Config.from_env(env)


def test_org_id_unset_fails_closed():
    cfg = _cfg(ZETARIS_REST_URL="https://z.test/api/v1.0")
    with pytest.raises(ConfigError) as e:
        cfg.validate_live()
    assert "ZETARIS_ORG_ID" in str(e.value)


def test_org_id_placeholder_fails_closed():
    cfg = _cfg(ZETARIS_REST_URL="https://z.test/api/v1.0", ZETARIS_ORG_ID="your-org-id")
    with pytest.raises(ConfigError):
        cfg.validate_live()


def test_source_without_org_id_fails_closed():
    with pytest.raises(ConfigError):
        ZetarisSource(base_url="https://z.test/api/v1.0", api_key="k")


def test_alias_not_rest_base_fails_closed():
    cfg = _cfg(ZETARIS_BASE_URL="https://z.test", ZETARIS_ORG_ID="3")
    assert cfg.zetaris_url_from_alias
    with pytest.raises(ConfigError) as e:
        cfg.validate_live()
    assert "ZETARIS_REST_URL" in str(e.value)


def test_alias_rest_base_accepted_with_warning():
    cfg = _cfg(ZETARIS_BASE_URL="https://z.test/api/v1.0/", ZETARIS_ORG_ID="3")
    cfg.validate_live()
    levels = [lvl for lvl, _ in cfg.zetaris_notes()]
    assert levels[0] == "WARNING"


def test_rest_url_preferred_and_no_alias_warning():
    cfg = _cfg(ZETARIS_REST_URL="https://z.test/api/v1.0", ZETARIS_BASE_URL="https://ui.test",
               ZETARIS_ORG_ID="3")
    cfg.validate_live()
    assert cfg.zetaris_base_url == "https://z.test/api/v1.0" and not cfg.zetaris_url_from_alias
    assert [lvl for lvl, _ in cfg.zetaris_notes()] == ["INFO"]


def test_limit_note_states_both_defaults():
    _, msg = _cfg(ZETARIS_REST_URL="https://z.test/api/v1.0", ZETARIS_ORG_ID="3").zetaris_notes()[-1]
    assert str(DEFAULT_LIMIT) in msg and str(VENDOR_CLIENT_LIMIT) in msg
    assert (DEFAULT_LIMIT, VENDOR_CLIENT_LIMIT) == (1000, 100)


def test_endpoint_matches_vendor_client():
    # zetaris/getting-started scripts/zetaris_sql.py RestChannel.run posts to "/sql-editor/sqls/run"
    assert ZetarisSource.ENDPOINT == "/sql-editor/sqls/run"


def test_note_writes_a_log_line(tmp_path):
    buf = io.StringIO()
    tw = TraceWriter("t18", log_dir=str(tmp_path), level="INFO", stream=buf)
    tw.note("WARNING", "alias in use")
    assert "[WARNING] run_id=t18 alias in use" in buf.getvalue()


def test_live_run_logs_alias_warning_and_limit(tmp_path):
    # End to end through run.py: the notes are written at the start of a live Zetaris run.
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    env = {"PATH": os.environ.get("PATH", ""), "SAMPLE_MODE": "false", "LIVE_PROVIDER": "sample",
           "ZETARIS_BASE_URL": "http://127.0.0.1:9/api/v1.0", "ZETARIS_ORG_ID": "5",
           "ZETARIS_API_KEY": "zk-T18-KEY-0000", "SOURCE_TIMEOUT_S": "1", "LOG_DIR": str(tmp_path),
           "HMEM_PATH": str(tmp_path / "m.json"), "NO_PROXY": "127.0.0.1,localhost"}
    p = subprocess.run([sys.executable, "run.py", "Why did EU revenue drop in week 41?", "--run-id", "t18e2e"],
                       cwd=root, env=env, capture_output=True, text=True, timeout=60)
    out = p.stdout + p.stderr
    assert "[WARNING] run_id=t18e2e ZETARIS_BASE_URL is a legacy alias" in out
    assert "row limit per query is 1000" in out and "defaults to 100" in out
    assert "zk-T18-KEY-0000" not in out
