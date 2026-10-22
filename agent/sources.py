"""Data sources: QuerySpec, CsvSource, ZetarisSource (DESIGN section 6).

Agents never write SQL or touch CSVs directly: they build a QuerySpec.
CsvSource executes it in Python, ZetarisSource compiles it to SQL.

Null semantics (both sources): ``is_null`` matches NULL *and* empty string.
Nullable columns (shipments.delivered_date, tickets.order_id) are None when
empty in CSV rows; other string columns keep ``""``.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol

import requests

from .errors import (
    ConfigError,
    SourceQueryError,
    SourceUnauthorized,
    SourceUnavailable,
    UpstreamTimeout,
)

Op = Literal["=", "!=", ">=", "<=", "in", "is_null", "not_null"]
OPS = ("=", "!=", ">=", "<=", "in", "is_null", "not_null")
AGG_FNS = ("sum", "count", "avg", "min", "max", "count_distinct")
TABLES = ("orders", "tickets", "shipments")
MAX_LIMIT = 5000
DEFAULT_LIMIT = 1000

SCHEMA: dict[str, dict[str, str]] = {
    "orders": {
        "order_id": "str", "order_ts": "str", "order_date": "str",
        "iso_year": "int", "iso_week": "int", "region": "str", "country": "str",
        "customer_id": "str", "category": "str", "sku": "str", "qty": "int",
        "unit_price_eur": "float", "order_value_eur": "float", "carrier": "str",
        "status": "str", "cancel_reason": "str", "status_updated_ts": "str",
    },
    "shipments": {
        "shipment_id": "str", "order_id": "str", "carrier": "str", "region": "str",
        "origin_hub": "str", "ship_date": "str", "ship_iso_year": "int",
        "ship_iso_week": "int", "promised_date": "str", "delivered_date": "str",
        "delay_days": "int", "status": "str",
    },
    "tickets": {
        "ticket_id": "str", "created_ts": "str", "created_date": "str",
        "iso_year": "int", "iso_week": "int", "region": "str", "country": "str",
        "order_id": "str", "category": "str", "priority": "str", "status": "str",
        "sentiment": "str",
    },
}
NULLABLE = {("shipments", "delivered_date"), ("tickets", "order_id")}
PRIMARY_KEYS = {"orders": "order_id", "shipments": "shipment_id", "tickets": "ticket_id"}

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUALIFIED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


# ---------------------------------------------------------------------------
# QuerySpec
# ---------------------------------------------------------------------------
def _jsonable(v: Any) -> Any:
    if isinstance(v, (tuple, list, set, frozenset)):
        return list(v)
    return v


@dataclass(frozen=True)
class Filter:
    column: str
    op: Op
    value: Any = None

    def __post_init__(self) -> None:
        if isinstance(self.value, (list, set, frozenset)):
            object.__setattr__(self, "value", tuple(self.value))


@dataclass(frozen=True)
class Agg:
    fn: Literal["sum", "count", "avg", "min", "max", "count_distinct"]
    column: Optional[str]
    alias: str


@dataclass(frozen=True)
class QuerySpec:
    table: Literal["orders", "tickets", "shipments"]
    filters: tuple[Filter, ...] = ()
    select: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()
    aggs: tuple[Agg, ...] = ()
    order_by: tuple[tuple[str, Literal["asc", "desc"]], ...] = ()
    limit: int = DEFAULT_LIMIT

    def as_dict(self) -> dict:
        return {
            "table": self.table,
            "filters": [
                {"column": f.column, "op": f.op, "value": _jsonable(f.value)} for f in self.filters
            ],
            "select": list(self.select),
            "group_by": list(self.group_by),
            "aggs": [{"fn": a.fn, "column": a.column, "alias": a.alias} for a in self.aggs],
            "order_by": [[c, d] for c, d in self.order_by],
            "limit": self.limit,
        }

    @staticmethod
    def from_dict(d: dict) -> "QuerySpec":
        try:
            if not isinstance(d, dict) or "table" not in d:
                raise ValueError("missing table")
            filters = []
            for f in d.get("filters") or ():
                if isinstance(f, dict):
                    filters.append(Filter(f["column"], f["op"], f.get("value")))
                else:
                    f = list(f)
                    filters.append(Filter(f[0], f[1], f[2] if len(f) > 2 else None))
            aggs = []
            for a in d.get("aggs") or ():
                if isinstance(a, dict):
                    aggs.append(Agg(a["fn"], a.get("column"), a["alias"]))
                else:
                    a = list(a)
                    aggs.append(Agg(a[0], a[1], a[2]))
            order_by = []
            for o in d.get("order_by") or ():
                if isinstance(o, dict):
                    order_by.append((o["column"], o.get("dir", o.get("direction", "asc"))))
                else:
                    o = list(o)
                    order_by.append((o[0], o[1] if len(o) > 1 else "asc"))
            limit = d.get("limit", DEFAULT_LIMIT)
            return QuerySpec(
                table=d["table"],
                filters=tuple(filters),
                select=tuple(d.get("select") or ()),
                group_by=tuple(d.get("group_by") or ()),
                aggs=tuple(aggs),
                order_by=tuple(order_by),
                limit=DEFAULT_LIMIT if limit is None else limit,
            )
        except (KeyError, TypeError, ValueError, IndexError):
            raise SourceQueryError("Invalid query spec") from None


@dataclass
class QueryResult:
    rows: list[dict]
    row_count: int
    query_text: str
    source: str  # "csv" | "zetaris"
    elapsed_ms: int
    truncated: bool


class DataSource(Protocol):
    name: str

    def describe(self) -> dict[str, list[str]]: ...

    def query(self, spec: QuerySpec, *, timeout_s: Optional[float] = None) -> QueryResult: ...

    def health(self) -> dict: ...

    def fingerprint(self) -> str: ...


# ---------------------------------------------------------------------------
# validation (shared)
# ---------------------------------------------------------------------------
def _is_num_type(t: str) -> bool:
    return t in ("int", "float")


def validate_spec(spec: QuerySpec) -> tuple[int, list[str], bool]:
    """Validate a spec. Returns (clamped limit, output columns, aggregated)."""
    if not isinstance(spec, QuerySpec):
        raise SourceQueryError("Not a QuerySpec")
    if spec.table not in SCHEMA:
        raise SourceQueryError(f"Unknown table: {spec.table!r}")
    cols = SCHEMA[spec.table]

    for f in spec.filters:
        if f.column not in cols:
            raise SourceQueryError(f"Unknown column {f.column!r} in table {spec.table!r}")
        if f.op not in OPS:
            raise SourceQueryError(f"Unsupported filter operator: {f.op!r}")
        if f.op == "in":
            if not isinstance(f.value, (tuple, list, set, frozenset)):
                raise SourceQueryError("Filter 'in' needs a list of values")
        elif f.op in ("=", "!=", ">=", "<="):
            if f.value is None or isinstance(f.value, (tuple, list, set, frozenset, dict)):
                raise SourceQueryError(f"Filter {f.op!r} on {f.column!r} needs a scalar value")

    for c in list(spec.select) + list(spec.group_by):
        if c not in cols:
            raise SourceQueryError(f"Unknown column {c!r} in table {spec.table!r}")

    aliases: list[str] = []
    for a in spec.aggs:
        if a.fn not in AGG_FNS:
            raise SourceQueryError(f"Unsupported aggregate: {a.fn!r}")
        if not isinstance(a.alias, str) or not _IDENT.match(a.alias):
            raise SourceQueryError(f"Invalid aggregate alias: {a.alias!r}")
        if a.alias in aliases or a.alias in spec.group_by:
            raise SourceQueryError(f"Duplicate output name: {a.alias!r}")
        if a.fn == "count" and a.column is None:
            pass
        else:
            if a.column is None or a.column not in cols:
                raise SourceQueryError(f"Unknown column {a.column!r} in table {spec.table!r}")
            if a.fn in ("sum", "avg") and not _is_num_type(cols[a.column]):
                raise SourceQueryError(f"{a.fn} needs a numeric column, got {a.column!r}")
        aliases.append(a.alias)

    aggregated = bool(spec.aggs or spec.group_by)
    out_cols = (list(spec.group_by) + aliases) if aggregated else (list(spec.select) or list(cols))

    for c, direction in spec.order_by:
        if direction not in ("asc", "desc"):
            raise SourceQueryError(f"Invalid sort direction: {direction!r}")
        allowed = out_cols if aggregated else list(cols)
        if c not in allowed:
            raise SourceQueryError(f"Cannot order by unknown column {c!r}")

    lim = spec.limit
    if isinstance(lim, bool) or not isinstance(lim, int):
        raise SourceQueryError("limit must be an integer")
    lim = max(0, min(lim, MAX_LIMIT))
    return lim, out_cols, aggregated


def _coerce(col_type: str, v: Any) -> Any:
    """Coerce a filter value to the column type (SourceQueryError on failure)."""
    if col_type == "str":
        if isinstance(v, str):
            return v
        if isinstance(v, bool):
            return str(v).lower()
        return str(v)
    if isinstance(v, bool):
        raise SourceQueryError("Boolean is not a valid numeric value")
    try:
        if isinstance(v, (int, float)):
            num: Any = v
        else:
            s = str(v).strip()
            try:
                num = int(s)
            except ValueError:
                num = float(s)
        if isinstance(num, float):
            if not math.isfinite(num):
                raise SourceQueryError("Non-finite numeric value")
            if col_type == "int" and num == int(num):
                return int(num)
            return num if col_type == "int" else float(num)
        return num if col_type == "int" else float(num)
    except (ValueError, TypeError):
        raise SourceQueryError("Invalid numeric value for numeric column") from None


# ---------------------------------------------------------------------------
# CsvSource
# ---------------------------------------------------------------------------
def _lit_txt(v: Any) -> str:
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _csv_text(spec: QuerySpec) -> str:
    parts = [f"csv:{spec.table}"]
    conds = []
    for f in spec.filters:
        if f.op in ("=", "!=", ">=", "<="):
            conds.append(f"{f.column}{f.op}{_lit_txt(f.value)}")
        elif f.op == "in":
            conds.append(f"{f.column} in ({','.join(_lit_txt(x) for x in f.value)})")
        elif f.op == "is_null":
            conds.append(f"{f.column} is null")
        else:
            conds.append(f"{f.column} is not null")
    if conds:
        parts.append("where " + " and ".join(conds))
    if spec.aggs or spec.group_by:
        if spec.group_by:
            parts.append(f"group_by({','.join(spec.group_by)})")
        for a in spec.aggs:
            parts.append(f"{a.fn}({a.column if a.column is not None else '*'})")
    elif spec.select:
        parts.append(f"select({','.join(spec.select)})")
    if spec.order_by:
        parts.append("order_by(" + ",".join(f"{c} {d}" for c, d in spec.order_by) + ")")
    if spec.limit != DEFAULT_LIMIT:
        parts.append(f"limit {spec.limit}")
    return " ".join(parts)


def _match(row: dict, f: Filter, val: Any) -> bool:
    v = row[f.column]
    op = f.op
    if op == "is_null":
        return v is None or v == ""
    if op == "not_null":
        return not (v is None or v == "")
    if op == "in":
        return v is not None and v in val
    if v is None:
        return False
    if op == "=":
        return v == val
    if op == "!=":
        return v != val
    if op == ">=":
        return v >= val
    return v <= val  # "<="


def _sort_key(v: Any, desc: bool):
    if desc:
        return (v is not None, v if v is not None else 0)
    return (v is None, v if v is not None else 0)


def _agg_value(a: Agg, grp: list[dict], col_type: Optional[str]) -> Any:
    if a.fn == "count":
        if a.column is None:
            return len(grp)
        return sum(1 for r in grp if r[a.column] is not None)
    vals = [r[a.column] for r in grp if r[a.column] is not None]
    if a.fn == "count_distinct":
        return len(set(vals))
    if a.fn == "sum":
        if col_type == "int":
            return int(sum(vals))
        return round(math.fsum(vals), 6)
    if a.fn == "avg":
        return round(math.fsum(vals) / len(vals), 6) if vals else None
    if a.fn == "min":
        return min(vals) if vals else None
    return max(vals) if vals else None


def _cast_csv(table: str, col: str, raw: Optional[str]) -> Any:
    t = SCHEMA[table][col]
    if raw is None:
        raw = ""
    if t == "str":
        if raw == "" and (table, col) in NULLABLE:
            return None
        return raw
    if raw.strip() == "":
        return None
    try:
        return int(raw) if t == "int" else float(raw)
    except ValueError:
        raise SourceQueryError(f"Malformed value in {table}.{col}") from None


class CsvSource:
    name = "csv"

    def __init__(self, data_dir: str) -> None:
        self.data_dir = str(data_dir)
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[tuple[int, int], list[dict]]] = {}
        for t in TABLES:  # fail early if files are missing or malformed
            self._table(t)

    def _path(self, table: str) -> str:
        return os.path.join(self.data_dir, f"{table}.csv")

    def _table(self, table: str) -> list[dict]:
        path = self._path(table)
        try:
            st = os.stat(path)
        except OSError:
            raise SourceUnavailable(f"Data file not found: {table}.csv") from None
        sig = (st.st_mtime_ns, st.st_size)
        with self._lock:
            cached = self._cache.get(table)
            if cached is not None and cached[0] == sig:
                return cached[1]
            rows: list[dict] = []
            try:
                with open(path, "r", newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    missing = [c for c in SCHEMA[table] if c not in (reader.fieldnames or [])]
                    if missing:
                        raise SourceUnavailable(
                            f"{table}.csv is missing columns: {', '.join(missing)}")
                    for rec in reader:
                        rows.append({c: _cast_csv(table, c, rec.get(c)) for c in SCHEMA[table]})
            except OSError:
                raise SourceUnavailable(f"Cannot read {table}.csv") from None
            self._cache[table] = (sig, rows)
            return rows

    # ---- DataSource ----------------------------------------------------
    def describe(self) -> dict[str, list[str]]:
        return {t: list(c) for t, c in SCHEMA.items()}

    def query(self, spec: QuerySpec, *, timeout_s: Optional[float] = None) -> QueryResult:
        limit, out_cols, aggregated = validate_spec(spec)
        t0 = time.monotonic()
        types = SCHEMA[spec.table]
        fvals = []
        for f in spec.filters:
            ct = types[f.column]
            if f.op == "in":
                fvals.append(frozenset(_coerce(ct, x) for x in f.value))
            elif f.op in ("is_null", "not_null"):
                fvals.append(None)
            else:
                fvals.append(_coerce(ct, f.value))
        matched = [
            r for r in self._table(spec.table)
            if all(_match(r, f, v) for f, v in zip(spec.filters, fvals))
        ]

        if aggregated:
            groups: dict[tuple, list[dict]] = {}
            for r in matched:
                groups.setdefault(tuple(r[c] for c in spec.group_by), []).append(r)
            if not spec.group_by and not groups:
                groups[()] = []
            keys = list(groups)
            if spec.group_by and not spec.order_by:
                keys.sort(key=lambda k: tuple(_sort_key(x, False) for x in k))
            rows = []
            for k in keys:
                row = dict(zip(spec.group_by, k))
                for a in spec.aggs:
                    ct = types.get(a.column) if a.column else None
                    row[a.alias] = _agg_value(a, groups[k], ct)
                rows.append(row)
            for c, d in reversed(spec.order_by):
                rows.sort(key=lambda r, c=c, d=d: _sort_key(r.get(c), d == "desc"),
                          reverse=(d == "desc"))
        else:
            rows = list(matched)
            for c, d in reversed(spec.order_by):
                rows.sort(key=lambda r, c=c, d=d: _sort_key(r.get(c), d == "desc"),
                          reverse=(d == "desc"))
            rows = [{c: r[c] for c in out_cols} for r in rows]

        truncated = len(rows) > limit
        rows = rows[:limit]
        return QueryResult(
            rows=rows,
            row_count=len(rows),
            query_text=_csv_text(spec),
            source="csv",
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            truncated=truncated,
        )

    def health(self) -> dict:
        try:
            for t in TABLES:
                self._table(t)
        except SourceUnavailable as e:
            return {"state": "offline", "detail": e.message}
        except SourceQueryError as e:
            return {"state": "offline", "detail": e.message}
        return {"state": "online", "detail": f"csv files in {self.data_dir}"}

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for t in TABLES:
            try:
                with open(self._path(t), "rb") as fh:
                    h.update(fh.read())
            except OSError:
                raise SourceUnavailable(f"Data file not found: {t}.csv") from None
        return h.hexdigest()


# ---------------------------------------------------------------------------
# ZetarisSource
# ---------------------------------------------------------------------------
def _from_remote(table: str, col: str, v: Any) -> Any:
    t = SCHEMA[table][col]
    if v is None:
        if t == "str" and (table, col) not in NULLABLE:
            return ""
        return None
    try:
        if t == "str":
            if v == "" and (table, col) in NULLABLE:
                return None
            return v if isinstance(v, str) else str(v)
        if t == "int":
            return int(float(v))
        return float(v)
    except (ValueError, TypeError):
        return v


def _num_or_same(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    return v


class ZetarisSource:
    name = "zetaris"
    ENDPOINT = "/sql-editor/sqls/run"
    BACKOFFS = (0.5, 1.5)

    def __init__(self, base_url: str, api_key: str, tables: Optional[dict] = None,
                 timeout_s: float = 20.0) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        self._api_key = api_key
        self.timeout_s = float(timeout_s)
        self.tables = {t: t for t in TABLES}
        for k, v in (tables or {}).items():
            if k in self.tables and v:
                self.tables[k] = str(v)
        for remote in self.tables.values():
            if not _QUALIFIED.match(remote):
                raise ConfigError("Invalid Zetaris table name")
        self.last_http_status: Optional[int] = None
        self._sleep = time.sleep

    def __repr__(self) -> str:
        return f"ZetarisSource(base_url={self.base_url!r}, api_key='***')"

    # ---- SQL compilation -----------------------------------------------
    @staticmethod
    def _literal(col_type: str, v: Any) -> str:
        v = _coerce(col_type, v)
        if col_type == "str":
            if "\\" in v or "\x00" in v:
                raise SourceQueryError("Unsupported character in filter value")
            return "'" + v.replace("'", "''") + "'"
        if isinstance(v, float):
            if not math.isfinite(v):
                raise SourceQueryError("Non-finite numeric value")
            return repr(v)
        return str(int(v))

    def _cond(self, table: str, f: Filter) -> str:
        t = SCHEMA[table][f.column]
        c = f.column
        if f.op == "=":
            return f"{c} = {self._literal(t, f.value)}"
        if f.op == "!=":
            return f"{c} <> {self._literal(t, f.value)}"
        if f.op == ">=":
            return f"{c} >= {self._literal(t, f.value)}"
        if f.op == "<=":
            return f"{c} <= {self._literal(t, f.value)}"
        if f.op == "in":
            if not f.value:
                return "1 = 0"
            return f"{c} IN ({', '.join(self._literal(t, x) for x in f.value)})"
        if f.op == "is_null":
            return f"({c} IS NULL OR {c} = '')" if t == "str" else f"{c} IS NULL"
        return f"({c} IS NOT NULL AND {c} <> '')" if t == "str" else f"{c} IS NOT NULL"

    @staticmethod
    def _agg_sql(a: Agg) -> str:
        if a.fn == "sum":
            return f"COALESCE(SUM({a.column}), 0) AS {a.alias}"
        if a.fn == "count":
            return f"COUNT({a.column if a.column else '*'}) AS {a.alias}"
        if a.fn == "count_distinct":
            return f"COUNT(DISTINCT {a.column}) AS {a.alias}"
        return f"{a.fn.upper()}({a.column}) AS {a.alias}"

    def compile_sql(self, spec: QuerySpec) -> str:
        limit, out_cols, aggregated = validate_spec(spec)
        if aggregated:
            parts = list(spec.group_by) + [self._agg_sql(a) for a in spec.aggs]
        else:
            parts = list(out_cols)
        sql = f"SELECT {', '.join(parts)} FROM {self.tables[spec.table]}"
        if spec.filters:
            sql += " WHERE " + " AND ".join(self._cond(spec.table, f) for f in spec.filters)
        if spec.group_by:
            sql += " GROUP BY " + ", ".join(spec.group_by)
        if spec.order_by:
            sql += " ORDER BY " + ", ".join(f"{c} {d.upper()}" for c, d in spec.order_by)
        sql += f" LIMIT {limit}"
        return sql

    # ---- A1: request / response shapes (assumption, isolated here) -------
    @staticmethod
    def _build_request(sql: str) -> dict:
        return {"sql": sql}

    @staticmethod
    def _parse_response(obj: Any) -> list[dict]:
        def from_rows(rows: Any, columns: Any) -> Optional[list[dict]]:
            if not isinstance(rows, list):
                return None
            if all(isinstance(r, dict) for r in rows):
                return [dict(r) for r in rows]
            if columns is not None and all(isinstance(r, (list, tuple)) for r in rows):
                names = []
                for c in columns:
                    names.append(c.get("name") if isinstance(c, dict) else c)
                return [dict(zip(names, r)) for r in rows]
            return None

        if isinstance(obj, list):
            got = from_rows(obj, None)
            if got is not None:
                return got
        elif isinstance(obj, dict):
            columns = obj.get("columns")
            for key in ("rows", "data"):
                if key in obj:
                    got = from_rows(obj[key], columns)
                    if got is not None:
                        return got
            for key in ("result", "results"):
                if isinstance(obj.get(key), (dict, list)):
                    try:
                        return ZetarisSource._parse_response(obj[key])
                    except SourceUnavailable:
                        pass
        raise SourceUnavailable("Unrecognised Zetaris response shape")

    # ---- transport -------------------------------------------------------
    def _post(self, body: dict, timeout: float) -> Any:
        url = self.base_url + self.ENDPOINT
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        attempts = len(self.BACKOFFS) + 1
        timed_out = False
        for i in range(attempts):
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=timeout)
            except requests.Timeout:
                timed_out = True
            except requests.ConnectionError:
                timed_out = False
            except requests.RequestException:
                raise SourceUnavailable("Zetaris request failed") from None
            else:
                code = resp.status_code
                self.last_http_status = code
                if code in (401, 403):
                    raise SourceUnauthorized(
                        f"Zetaris rejected the API key (HTTP {code}).",
                        details={"http_status": code})
                if code >= 500:
                    timed_out = False
                elif code >= 400:
                    excerpt = " ".join((resp.text or "")[:200].split())
                    raise SourceQueryError(
                        f"Zetaris rejected the query (HTTP {code}).",
                        details={"http_status": code, "body": excerpt})
                else:
                    try:
                        return resp.json()
                    except ValueError:
                        raise SourceUnavailable(
                            "Zetaris returned invalid JSON",
                            details={"http_status": code}) from None
            if i < attempts - 1:
                self._sleep(self.BACKOFFS[i])
        details = {"attempts": attempts, "http_status": self.last_http_status}
        if timed_out:
            raise UpstreamTimeout("Zetaris request timed out after retries", details=details)
        raise SourceUnavailable("Zetaris unavailable after retries", details=details)

    # ---- DataSource ------------------------------------------------------
    def describe(self) -> dict[str, list[str]]:
        return {t: list(c) for t, c in SCHEMA.items()}

    def query(self, spec: QuerySpec, *, timeout_s: Optional[float] = None) -> QueryResult:
        limit, out_cols, aggregated = validate_spec(spec)
        sql = self.compile_sql(spec)
        t0 = time.monotonic()
        timeout = self.timeout_s if timeout_s is None else max(0.001, float(timeout_s))
        obj = self._post(self._build_request(sql), timeout)
        raw_rows = self._parse_response(obj)
        types = SCHEMA[spec.table]
        rows: list[dict] = []
        for raw in raw_rows:
            lower = {str(k).lower(): v for k, v in raw.items()}
            row: dict = {}
            for c in out_cols:
                v = raw[c] if c in raw else lower.get(c.lower())
                if c in types and (not aggregated or c in spec.group_by):
                    row[c] = _from_remote(spec.table, c, v)
                else:
                    row[c] = _num_or_same(v)
            rows.append(row)
        truncated = len(rows) >= limit > 0
        return QueryResult(
            rows=rows,
            row_count=len(rows),
            query_text=sql,
            source="zetaris",
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            truncated=truncated,
        )

    def health(self) -> dict:
        spec = QuerySpec(table="orders", aggs=(Agg("count", None, "n"),), limit=1)
        try:
            self.query(spec)
        except SourceUnauthorized as e:
            return {"state": "unauthorized", "detail": e.message}
        except (SourceUnavailable, UpstreamTimeout) as e:
            return {"state": "offline", "detail": e.message}
        except SourceQueryError as e:
            return {"state": "offline", "detail": e.message}
        return {"state": "online", "detail": "query ok"}

    def fingerprint(self) -> str:
        return "live"
