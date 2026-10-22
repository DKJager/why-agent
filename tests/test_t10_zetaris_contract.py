# build-id: oa-0bd86ee0
"""T10: Zetaris contract against a fake local server."""
import json
import logging
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.sources import ZetarisSource, QuerySpec, Filter, Agg  # noqa: E402
from agent.errors import SourceQueryError, SourceUnauthorized, SourceUnavailable  # noqa: E402

KEY = "zk-T10-SECRETKEY-98765"
PATH = "/sql-editor/sqls/run"
TABLES = {"orders": "orders", "tickets": "tickets", "shipments": "shipments"}


class FakeServer:
    def __init__(self, routes):
        self.routes = routes
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a, **k):
                pass

            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw.decode("utf-8", "replace")
                req = {"method": self.command, "path": self.path.split("?")[0],
                       "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
                outer.requests.append(req)
                handler = outer.routes.get((self.command, req["path"]))
                if handler is None:
                    status, payload = 404, {"error": "not found"}
                elif callable(handler):
                    status, payload = handler(req)
                else:
                    status, payload = handler
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = _handle
            do_POST = _handle

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SAMPLE_MODE", raising=False)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(time, "sleep", lambda s: None)  # skip retry backoff


@pytest.fixture
def make_server():
    servers = []

    def _make(routes):
        s = FakeServer(routes)
        servers.append(s)
        return s

    yield _make
    for s in servers:
        s.stop()


def _source(srv):
    return ZetarisSource(base_url=srv.url, api_key=KEY, tables=TABLES, timeout_s=5, org_id="1")


def _find_sql(obj):
    if isinstance(obj, str):
        return obj if obj.strip().lower().startswith(("select", "with")) else None
    if isinstance(obj, dict):
        for v in obj.values():
            r = _find_sql(v)
            if r:
                return r
    if isinstance(obj, list):
        for v in obj:
            r = _find_sql(v)
            if r:
                return r
    return None


def _norm(sql):
    s = sql.replace('"', "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*([=,()<>!]+)\s*", r"\1", s)
    s = re.sub(r"\bin\s*\(", "in(", s, flags=re.I)
    return s.rstrip(";")


OK = (200, {"rows": [{"category": "home", "rev": 12.5}]})

SPEC = QuerySpec(
    table="orders",
    filters=(Filter("iso_week", "=", 41), Filter("region", "=", "EU")),
    group_by=("category",),
    aggs=(Agg("sum", "order_value_eur", "rev"),),
    order_by=(("rev", "desc"),),
    limit=10,
)


def test_request_shape_and_compiled_sql(make_server):
    srv = make_server({("POST", PATH): OK})
    res = _source(srv).query(SPEC)
    assert len(srv.requests) == 1
    r = srv.requests[0]
    assert r["method"] == "POST"
    assert r["path"] == PATH
    assert r["headers"]["authorization"] == f"Bearer {KEY}"
    assert "application/json" in r["headers"]["content-type"]

    sql = _find_sql(r["body"])
    assert sql, f"no SQL found in request body {r['body']!r}"
    n = _norm(sql).lower()
    assert n.startswith("select")
    assert "from orders" in n
    assert "where" in n
    assert "iso_week=41" in n
    assert "region='eu'" in n
    assert "group by category" in n
    assert "sum(order_value_eur)" in n
    assert "rev" in n
    assert "order by rev desc" in n
    assert re.search(r"limit 10\b", n)

    assert res.rows == [{"category": "home", "rev": 12.5}]
    assert res.row_count == 1
    assert res.source == "zetaris"
    assert KEY not in res.query_text
    assert res.query_text.strip().lower().startswith("select")


def test_in_filter_compiles(make_server):
    srv = make_server({("POST", PATH): OK})
    spec = QuerySpec(table="orders",
                     filters=(Filter("carrier", "in", ("NordExpress", "RheinFracht")),),
                     select=("order_id",))
    _source(srv).query(spec)
    sql = _find_sql(srv.requests[0]["body"])
    n = _norm(sql).lower()
    assert "in('nordexpress','rheinfracht')" in n


def test_limit_is_clamped(make_server):
    srv = make_server({("POST", PATH): OK})
    spec = QuerySpec(table="orders", select=("order_id",), limit=999999)
    _source(srv).query(spec)
    sql = _find_sql(srv.requests[0]["body"])
    assert "999999" not in sql
    m = re.search(r"limit\s+(\d+)", sql, re.I)
    assert m and int(m.group(1)) <= 5000


@pytest.mark.parametrize("spec", [
    QuerySpec(table="users", select=("order_id",)),
    QuerySpec(table="orders", select=("no_such_column",)),
    QuerySpec(table="orders", filters=(Filter("no_such_column", "=", 1),), select=("order_id",)),
    QuerySpec(table="orders", group_by=("no_such_column",), aggs=(Agg("count", None, "n"),)),
    QuerySpec(table="orders", aggs=(Agg("sum", "no_such_column", "s"),)),
    QuerySpec(table="tickets", select=("carrier",)),  # carrier exists on orders, not tickets
])
def test_unknown_identifiers_rejected(make_server, spec):
    srv = make_server({("POST", PATH): OK})
    with pytest.raises(SourceQueryError):
        _source(srv).query(spec)


def test_quote_is_escaped(make_server):
    srv = make_server({("POST", PATH): OK})
    spec = QuerySpec(table="orders", filters=(Filter("carrier", "=", "O'Brien"),),
                     select=("order_id",))
    _source(srv).query(spec)
    sql = _find_sql(srv.requests[0]["body"])
    assert "'O''Brien'" in sql
    assert "O'Brien" not in sql


def test_injection_attempt_stays_inside_literal(make_server):
    srv = make_server({("POST", PATH): OK})
    evil = "x'; DROP TABLE orders; --"
    spec = QuerySpec(table="orders", filters=(Filter("carrier", "=", evil),), select=("order_id",))
    _source(srv).query(spec)
    sql = _find_sql(srv.requests[0]["body"])
    assert "'x''; DROP TABLE orders; --'" in sql


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_not_retried(make_server, status, caplog):
    srv = make_server({("POST", PATH): (status, {"error": f"denied for Bearer {KEY}"})})
    caplog.set_level(logging.DEBUG)
    with pytest.raises(SourceUnauthorized) as ei:
        _source(srv).query(SPEC)
    assert len(srv.requests) == 1
    assert KEY not in str(ei.value)
    assert KEY not in caplog.text


def test_500_retried_twice_then_unavailable(make_server, caplog):
    srv = make_server({("POST", PATH): (500, {"error": "boom"})})
    caplog.set_level(logging.DEBUG)
    with pytest.raises(SourceUnavailable) as ei:
        _source(srv).query(SPEC)
    assert len(srv.requests) == 3  # 1 try + 2 retries
    assert KEY not in str(ei.value)
    assert KEY not in caplog.text


def test_500_then_success(make_server):
    state = {"n": 0}

    def handler(req):
        state["n"] += 1
        return (500, {"error": "x"}) if state["n"] == 1 else OK

    srv = make_server({("POST", PATH): handler})
    res = _source(srv).query(SPEC)
    assert state["n"] == 2
    assert res.row_count == 1


def test_other_4xx_not_retried_and_is_query_error(make_server):
    srv = make_server({("POST", PATH): (400, {"error": "bad sql"})})
    with pytest.raises(SourceQueryError):
        _source(srv).query(SPEC)
    assert len(srv.requests) == 1


def test_connection_refused_is_unavailable():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    src = ZetarisSource(base_url=f"http://127.0.0.1:{port}", api_key=KEY, tables=TABLES, timeout_s=2, org_id="1")
    with pytest.raises(SourceUnavailable):
        src.query(SPEC)


def test_key_not_in_repr_or_query_text(make_server):
    srv = make_server({("POST", PATH): OK})
    src = _source(srv)
    res = src.query(SPEC)
    assert KEY not in repr(src)
    assert KEY not in str(src)
    assert KEY not in res.query_text


# A1 isolation points: body shape lives only in _build_request / _parse_response.

def test_build_request_contains_sql(make_server):
    src = _source(make_server({}))
    body = src._build_request("SELECT 1")
    assert isinstance(body, dict)
    assert _find_sql(body) == "SELECT 1"


def test_build_request_matches_zetaris_runner_shape(make_server):
    # Shape used by zetaris/getting-started scripts/zetaris_sql.py RestChannel.run
    src = _source(make_server({}))
    body = src._build_request("SELECT 1", 7)
    assert set(body) == {"queryId", "sql", "source", "limit"}
    assert body["source"] == "SqlEditor" and body["limit"] == 7 and body["queryId"]


def test_official_list_rows_mapped_in_select_order(make_server):
    # Zetaris REST returns {"data": [["v1", "v2"], ...]} with string values,
    # decimals possibly in exponent form.
    rows = [["DE", "1.2345E3", "4"], ["FR", "250.5", "2"]]
    srv = make_server({("POST", PATH): (200, {"data": rows})})
    src = ZetarisSource(base_url=srv.url, api_key=KEY, tables=TABLES, timeout_s=5, org_id="42")
    spec = QuerySpec(table="orders", group_by=("country",),
                     aggs=(Agg("sum", "order_value_eur", "rev"), Agg("count", None, "n")))
    res = src.query(spec)
    assert res.rows[0]["country"] == "DE"
    assert res.rows[0]["rev"] == 1234.5 and res.rows[0]["n"] == 4
    assert res.rows[1] == {"country": "FR", "rev": 250.5, "n": 2}
    h = srv.requests[-1]["headers"]
    assert h["x-org-id"] == "42" and h["x-request-id"]
    assert h["authorization"] == f"Bearer {KEY}"
    assert srv.requests[-1]["body"]["source"] == "SqlEditor"


def test_parse_response_tolerates_three_shapes(make_server):
    src = _source(make_server({}))
    assert src._parse_response({"rows": [{"a": 1}]}) == [{"a": 1}]
    assert src._parse_response({"data": [{"a": 1}, {"a": 2}]}) == [{"a": 1}, {"a": 2}]
    assert src._parse_response({"columns": ["a", "b"], "data": [[1, 2], [3, 4]]}) == [
        {"a": 1, "b": 2}, {"a": 3, "b": 4}]
