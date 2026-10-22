"""T14: packaging, dependency policy, Dockerfile and bind address."""
import ast
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_only_requests():
    lines = []
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    assert lines, "requirements.txt is empty"
    names = [re.match(r"[A-Za-z0-9_.\-]+", l).group(0).lower() for l in lines]
    assert names == ["requests"], names


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def test_only_stdlib_and_requests_imported():
    files = list((ROOT / "agent").rglob("*.py")) + [ROOT / "run.py"]
    assert len(files) > 5
    allowed = set(sys.stdlib_module_names) | {"requests", "agent"}
    offenders = {}
    for f in files:
        bad = _imports(f) - allowed
        if bad:
            offenders[str(f.relative_to(ROOT))] = sorted(bad)
    assert not offenders, offenders


def test_env_example_has_no_real_looking_secrets():
    path = ROOT / ".env.example"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    placeholder_words = ("your", "changeme", "example", "placeholder", "xxx")
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in values.items():
        if any(w in k.upper() for w in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            assert v == "" or any(w in v.lower() for w in placeholder_words), f"{k} looks real: {v!r}"
        assert not re.search(r"[A-Za-z0-9]{32,}", v), f"{k} has a long opaque value"
    assert not re.search(r"sk-[A-Za-z0-9_-]{10,}", text)
    assert "nvapi-" not in text
    assert not re.search(r"Bearer\s+\S{8,}", text)
    assert values.get("SAMPLE_MODE") == "true"


def test_gitignore_entries():
    entries = {l.strip() for l in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    for needed in (".env", "logs/", ".hmem/", "__pycache__/"):
        assert needed in entries, needed


def test_dockerfile():
    lines = [l.strip() for l in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    assert lines[0] == "FROM python:3.12-slim"
    users = [(i, re.match(r"USER\s+(\S+)", l).group(1)) for i, l in enumerate(lines)
             if re.match(r"USER\s+\S+", l)]
    assert users, "no USER instruction"
    final_user = users[-1][1]
    assert final_user.split(":")[0] not in ("root", "0")
    cmd_idx = [i for i, l in enumerate(lines) if l.startswith("CMD")]
    assert len(cmd_idx) == 1
    assert re.fullmatch(r'CMD\s*\[\s*"python"\s*,\s*"run\.py"\s*,\s*"--serve"\s*\]', lines[cmd_idx[0]])
    assert users[-1][0] < cmd_idx[0], "USER must come before CMD"
    # secrets are passed at runtime, never baked in
    for l in lines:
        assert not re.match(r"(ARG|ENV)\s+\w*(KEY|TOKEN|SECRET)\w*\s*=?\s*\S+", l, re.I), l
        assert not re.match(r"COPY\s+.*\.env(\s|$)", l)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _listening_on_all_interfaces(port):
    hexport = f"{port:04X}"
    checked = False
    for fname, wildcard in (("/proc/net/tcp", "00000000"), ("/proc/net/tcp6", "0" * 32)):
        if not os.path.exists(fname):
            continue
        checked = True
        for line in Path(fname).read_text().splitlines()[1:]:
            parts = line.split()
            if len(parts) > 3 and parts[3] == "0A" and parts[1] == f"{wildcard}:{hexport}":
                return True
    return True if not checked else False


def test_serve_binds_all_interfaces_and_answers(tmp_path):
    port = _free_port()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("METERLESS", "MODEL_", "ZETARIS", "NVIDIA", "HMEM", "LOG_", "DATA_DIR",
                                "ENGINE", "PORT", "SAMPLE_MODE"))}
    env.update({"SAMPLE_MODE": "true", "LOG_DIR": str(tmp_path / "logs"),
                "HMEM_PATH": str(tmp_path / "hmem" / "store.json"),
                "DATA_DIR": str(ROOT / "data" / "sample"), "PYTHONUNBUFFERED": "1"})
    out = open(tmp_path / "serve.out", "w")
    proc = subprocess.Popen([sys.executable, "run.py", "--serve", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    sess = requests.Session()
    sess.trust_env = False
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30
        health = None
        while time.time() < deadline:
            assert proc.poll() is None, (tmp_path / "serve.out").read_text()
            try:
                health = sess.get(base + "/healthz", timeout=2)
                break
            except requests.RequestException:
                time.sleep(0.2)
        assert health is not None, "server did not come up"
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "sample_mode": True}

        assert _listening_on_all_interfaces(port), "server is not bound to 0.0.0.0"

        r = sess.post(base + "/run", json={"input": {"query": "Why did EU revenue drop in week 41?"}},
                      timeout=60)
        assert r.status_code == 200
        assert r.json()["status"] == "success"

        r404 = sess.get(base + "/nope", timeout=5)
        assert r404.status_code in (404, 405)
        assert r404.json()["status"] == "error"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        out.close()
