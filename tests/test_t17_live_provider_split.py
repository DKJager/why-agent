# build-id: oa-720b718e
"""T17: LIVE_PROVIDER=sample gives live Zetaris data with the local sample provider."""
import pytest

from agent.config import Config
from agent.errors import ConfigError
from agent.orchestrator import _build_components, _mode

LIVE = {"SAMPLE_MODE": "false", "ZETARIS_REST_URL": "http://127.0.0.1:9/api/v1.0",
        "ZETARIS_ORG_ID": "7", "ZETARIS_API_KEY": "zk-T17-KEY-0000", "HMEM_PATH": "{tmp}/m.json"}


def _cfg(tmp_path, **extra):
    env = {k: v.format(tmp=tmp_path) for k, v in LIVE.items()}
    env.update(extra)
    return Config.from_env(env)


def test_sample_provider_with_live_source_needs_no_meterless_key(tmp_path):
    comps = _build_components(_cfg(tmp_path, LIVE_PROVIDER="sample"))
    mode = _mode(comps)
    assert mode["sample_mode"] is False
    assert mode["source"] == "zetaris" and mode["provider"] == "sample"
    assert comps["source"].org_id == "7"


def test_default_live_provider_still_requires_meterless_key(tmp_path):
    with pytest.raises(ConfigError) as e:
        _build_components(_cfg(tmp_path))
    assert "METERLESS_API_KEY" in str(e.value)


def test_invalid_live_provider_rejected(tmp_path):
    with pytest.raises(ConfigError):
        _cfg(tmp_path, LIVE_PROVIDER="openai")
