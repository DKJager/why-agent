# build-id: oa-fe037fcc
"""T13: committed samples reproduce from scripts/run_samples.py and are not stale."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "samples" / "outputs"
DROP_KEYS = {"execution_time_seconds", "duration_ms", "elapsed_ms", "timestamp",
             "retrieved_at", "created_at", "last_confirmed_at"}


def _samples():
    data = json.loads((ROOT / "samples" / "questions.json").read_text(encoding="utf-8"))
    assert len(data) == 3
    return [d["id"] for d in data]


def _norm(o):
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in o.items() if k not in DROP_KEYS}
    if isinstance(o, list):
        return [_norm(v) for v in o]
    return o


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace(path):
    return [_norm(json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _strip_fp(s):
    return s.split(":", 1)[1] if s.startswith("sha256:") else s


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    work = tmp_path_factory.mktemp("repo_copy") / "repo"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "logs", ".hmem", ".pytest_cache", "node_modules", ".venv", "venv", "*.pyc"))
    out = work / "samples" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        if f.is_file():
            f.unlink()
    shutil.rmtree(work / ".hmem", ignore_errors=True)

    drop = ("METERLESS", "MODEL_", "ZETARIS", "NVIDIA", "SAMPLE_MODE", "HMEM", "LOG_", "DATA_DIR",
            "ENGINE", "RUN_TIMEOUT", "MAX_", "PROVIDER_TIMEOUT", "SOURCE_TIMEOUT")
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run([sys.executable, "scripts/run_samples.py"], cwd=work, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_outputs_reproduce(regenerated, idx):
    sid = _samples()[idx]
    committed = OUT_DIR / f"{sid}.json"
    fresh = regenerated / f"{sid}.json"
    assert committed.exists(), f"missing committed {committed}"
    assert fresh.exists(), "run_samples.py did not produce the output"
    assert _norm(_load(fresh)) == _norm(_load(committed))


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_traces_reproduce(regenerated, idx):
    sid = _samples()[idx]
    committed = OUT_DIR / f"{sid}.trace.jsonl"
    fresh = regenerated / f"{sid}.trace.jsonl"
    assert committed.exists() and fresh.exists()
    assert _load_trace(fresh) == _load_trace(committed)


def test_committed_outputs_not_stale():
    from agent.sources import CsvSource
    current = _strip_fp(CsvSource(str(ROOT / "data" / "sample")).fingerprint())
    for sid in _samples():
        body = _load(OUT_DIR / f"{sid}.json")
        assert "data_fingerprint" in body
        assert _strip_fp(body["data_fingerprint"]) == current, f"{sid} is stale vs current CSVs"


def test_outputs_declare_generated_by_run_py():
    files = sorted(OUT_DIR.glob("*.json"))
    assert len(files) >= 3
    for f in files:
        assert re.search(r'"generated_by"\s*:\s*"run\.py"', f.read_text(encoding="utf-8")), f.name


def test_sample_statuses_and_memory():
    ids = _samples()
    for sid in ids:
        assert _load(OUT_DIR / f"{sid}.json")["status"] in ("success", "partial")
    s02 = _load(OUT_DIR / f"{ids[1]}.json")
    assert s02["output"]["memory"]["recalled"], "sample 02 must show H-MEM recall"
