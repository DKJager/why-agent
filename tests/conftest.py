# build-id: oa-ca7478df
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CLEAR_ENV = [
    "METERLESS_BASE_URL", "METERLESS_API_KEY", "METERLESS_AUTODETECT",
    "MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_ID",
    "ZETARIS_BASE_URL", "ZETARIS_API_KEY",
    "ZETARIS_TABLE_ORDERS", "ZETARIS_TABLE_TICKETS", "ZETARIS_TABLE_SHIPMENTS",
    "NVIDIA_API_KEY", "RUN_TIMEOUT_S", "MAX_REFINEMENTS", "MAX_STEPS",
    "MAX_MODEL_CALLS", "PROVIDER_TIMEOUT_S", "SOURCE_TIMEOUT_S", "ENGINE",
    "LOG_LEVEL",
]


@pytest.fixture
def tmp_data_dir(tmp_path):
    dst = tmp_path / "data"
    shutil.copytree(REPO / "data" / "sample", dst)
    return dst


@pytest.fixture
def tmp_hmem(tmp_path):
    return tmp_path / "hmem" / "store.json"


@pytest.fixture
def tmp_logs(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.fixture
def sample_env(monkeypatch, tmp_data_dir, tmp_hmem, tmp_logs):
    for k in _CLEAR_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SAMPLE_MODE", "true")
    monkeypatch.setenv("ENGINE", "stub")
    monkeypatch.setenv("DATA_DIR", str(tmp_data_dir))
    monkeypatch.setenv("HMEM_PATH", str(tmp_hmem))
    monkeypatch.setenv("LOG_DIR", str(tmp_logs))
    return SimpleNamespace(data_dir=tmp_data_dir, hmem=tmp_hmem, logs=tmp_logs)


@pytest.fixture
def subprocess_env():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env
