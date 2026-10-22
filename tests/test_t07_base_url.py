import itertools
import json

import pytest

from agent.provider import (DEFAULT_BASE_URL, healthz_url, normalize_base_url,
                            parse_connect_link)

from _helpers import Q1, run

TABLE = [
    ("127.0.0.1:8088", "http://127.0.0.1:8088/v1"),
    ("http://127.0.0.1:8088", "http://127.0.0.1:8088/v1"),
    ("http://127.0.0.1:8088/v1", "http://127.0.0.1:8088/v1"),
    ("http://127.0.0.1:8088/v1/", "http://127.0.0.1:8088/v1"),
    ("http://host:8088/v1/chat/completions", "http://host:8088/v1"),
    ("http://host:8088/v1/models", "http://host:8088/v1"),
    ("http://host:8088/healthz", "http://host:8088/v1"),
    ("localhost:8090/v1/v1", "http://localhost:8090/v1"),
    ("[::1]:8787", "http://[::1]:8787/v1"),
    ("https://gw.example.com/meterless/v1/models", "https://gw.example.com/meterless/v1"),
    ("", "http://127.0.0.1:8088/v1"),
    (None, "http://127.0.0.1:8088/v1"),
]


@pytest.mark.parametrize("raw,expected", TABLE)
def test_table(raw, expected):
    assert normalize_base_url(raw) == expected


def test_default_constant():
    assert DEFAULT_BASE_URL == "http://127.0.0.1:8088/v1"


def test_generated_inputs():
    schemes = ["", "http://", "https://"]
    hosts = ["127.0.0.1:8088", "host:8088", "[::1]:8787", "gw.example.com/gw"]
    suffixes = ["", "/", "/v1", "/v1/", "/v1/models", "/v1/chat/completions",
                "/healthz", "/v1/v1"]
    for s, h, suf in itertools.product(schemes, hosts, suffixes):
        raw = f"{s}{h}{suf}"
        n = normalize_base_url(raw)
        assert "/v1/v1" not in n, raw
        assert n.endswith("/v1"), raw
        assert not n.endswith("/v1/v1")
        assert normalize_base_url(n) == n, raw
        if "/gw" in h:
            assert "/gw/v1" in n
        if "[::1]" in h:
            assert "[::1]:8787" in n


def test_healthz_url():
    assert healthz_url("http://host:8088/v1") == "http://host:8088/healthz"


def test_parse_connect_link():
    assert parse_connect_link("http://10.0.0.5:8088/connect#k=SECRET") == (
        "http://10.0.0.5:8088/v1", "SECRET")
    assert parse_connect_link("host:8088") == (normalize_base_url("host:8088"), None)


def test_connect_link_key_never_leaks(sample_env, monkeypatch, capfd):
    key = "CONNECTKEYVALUE98765"
    link = f"http://10.0.0.5:8088/connect#k={key}"
    monkeypatch.setenv("METERLESS_BASE_URL", link)
    res = run(Q1, "t07-leak")
    captured = capfd.readouterr()
    assert key not in captured.out
    assert key not in captured.err
    assert key not in json.dumps(res)
    for p in sample_env.logs.iterdir():
        assert key not in p.read_text(encoding="utf-8")
