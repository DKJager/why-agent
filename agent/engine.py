# build-id: oa-20310e3c
"""H-MEM engine adapters (DESIGN section 8).

* ``HMemStub``: fully functional offline JSON store.
* ``HMemMeterless``: the Meterless H-MEM engine (``agent.hmem``), conformance-tested
  against the official H-MEM vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from typing import Any, Optional, Protocol, Sequence

from .errors import ConfigError, EngineNotConfigured
from .models import MemoryRecord
from .trace import utc_now_iso


class NotConfigured(EngineNotConfigured):
    """Type "EngineNotConfigured", recoverable=True."""

    type = "EngineNotConfigured"
    recoverable = True


class EngineAdapter(Protocol):
    name: str

    def store(self, entity: str, fact_key: str, fact: str, *, evidence_refs: list[str],
              confidence: float, run_id: str) -> MemoryRecord: ...

    def recall(self, entities: Sequence[str] = (), query: Optional[str] = None,
               limit: int = 5) -> list[MemoryRecord]: ...

    def get(self, memory_id: str) -> Optional[MemoryRecord]: ...

    def forget(self, memory_id: str) -> bool: ...

    def health(self) -> dict: ...


_STOP = {
    "the", "and", "for", "was", "were", "why", "did", "does", "with", "that", "this",
    "from", "are", "has", "had", "its", "not", "but", "any", "all", "can",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 3 and t not in _STOP}


def _new_id(existing: set[str], entity: str, fact_key: str, fact: str, run_id: str) -> str:
    digest = hashlib.sha256("|".join((entity, fact_key, fact, run_id)).encode("utf-8")).hexdigest()
    n = 0
    while True:
        mid = "m-" + digest[n:n + 12]
        if mid not in existing:
            return mid
        n += 1


class HMemStub:
    """Persistent JSON memory: ``{"version":1,"records":[MemoryRecord...]}``."""

    name = "hmem-stub"

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._lock = threading.Lock()

    # ---- persistence -----------------------------------------------------
    def _read(self) -> list[MemoryRecord]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            if not isinstance(obj, dict) or not isinstance(obj.get("records"), list):
                raise ValueError("bad store layout")
            return [MemoryRecord.from_dict(r) for r in obj["records"]]
        except (OSError, ValueError, KeyError, TypeError):
            raise ConfigError("H-MEM store file is unreadable or corrupt",
                              details={"path": self.path}) from None

    def _write(self, records: list[MemoryRecord]) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        payload = {"version": 1, "records": [r.to_dict() for r in records]}
        fd, tmp = tempfile.mkstemp(prefix=".hmem-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ---- API ---------------------------------------------------------------
    def store(self, entity: str, fact_key: str, fact: str, *, evidence_refs: list[str],
              confidence: float, run_id: str) -> MemoryRecord:
        for label, v in (("entity", entity), ("fact_key", fact_key), ("fact", fact)):
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"{label} must be a non-empty string")
        conf = round(max(0.0, min(1.0, float(confidence))), 2)
        refs = [str(r) for r in (evidence_refs or [])]
        now = utc_now_iso()
        with self._lock:
            records = self._read()
            current = None
            for r in records:
                if r.status == "confirmed" and r.entity == entity and r.fact_key == fact_key:
                    current = r
                    break
            if current is not None and current.fact == fact:
                current.confirmations += 1
                current.last_confirmed_at = now
                current.confidence = max(current.confidence, conf)
                for ref in refs:
                    if ref not in current.evidence_refs:
                        current.evidence_refs.append(ref)
                self._write(records)
                return current
            mid = _new_id({r.memory_id for r in records}, entity, fact_key, fact, run_id)
            rec = MemoryRecord(
                memory_id=mid, entity=entity, fact_key=fact_key, fact=fact,
                evidence_refs=refs, confidence=conf, source_run_id=run_id,
                created_at=now, last_confirmed_at=now, confirmations=1, status="confirmed",
            )
            if current is not None:
                current.status = "superseded"
                current.superseded_by = mid
                current.supersede_reason = f"newer confirmed finding in run {run_id}"
            records.append(rec)
            self._write(records)
            return rec

    def recall(self, entities: Sequence[str] = (), query: Optional[str] = None,
               limit: int = 5) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        wanted = {e for e in (entities or ()) if isinstance(e, str) and e}
        qtok = _tokens(query) if query else set()
        if not wanted and not qtok:
            return []
        with self._lock:
            records = self._read()
        hits: list[tuple[str, int, MemoryRecord]] = []
        seen: set[str] = set()
        for idx, r in enumerate(records):
            if r.status != "confirmed" or r.memory_id in seen:
                continue
            if r.entity in wanted or (qtok and qtok & _tokens(r.fact)):
                seen.add(r.memory_id)
                hits.append((r.last_confirmed_at, idx, r))
        hits.sort(key=lambda h: (h[0], h[1]), reverse=True)
        return [h[2] for h in hits[:limit]]

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            for r in self._read():
                if r.memory_id == memory_id:
                    return r
        return None

    def forget(self, memory_id: str) -> bool:
        with self._lock:
            records = self._read()
            kept = [r for r in records if r.memory_id != memory_id]
            if len(kept) == len(records):
                return False
            self._write(kept)
            return True

    def health(self) -> dict:
        try:
            with self._lock:
                n = len(self._read())
        except ConfigError as e:
            return {"engine": self.name, "state": "offline", "detail": e.message, "path": self.path}
        return {"engine": self.name, "state": "online", "path": self.path, "records": n}


def _iso(ms: int) -> str:
    import time as _t
    return _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(ms / 1000.0))


def _ms(iso: str) -> int:
    import calendar as _c
    import time as _t
    try:
        return int(_c.timegm(_t.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")) * 1000)
    except (TypeError, ValueError):
        return int(_t.time() * 1000)


class HMemMeterless:
    """Why-Agent memory on the Meterless H-MEM engine (``agent.hmem``).

    Confirmed findings become H-MEM records (type ``factual``, provenance origin
    ``goal_run``, one entity per record, ``fact_key:`` tag). A changed finding is
    written as a new record and the old one is marked ``supersededBy`` through the
    engine, never deleted. Recall uses H-MEM's eight-signal hybrid ranking with the
    question's entities; superseded records are counted and suppressed. Every
    mutation is written to the H-MEM trust ledger. A store file written by
    ``HMemStub`` is migrated on first open (provenance origin ``import``).
    """

    name = "hmem-meterless"
    NAMESPACE = "why-agent"

    def __init__(self, path: str, clock: Any = None) -> None:
        from .hmem import HMEM, HMemError

        self.path = str(path)
        self._lock = threading.Lock()
        self.last_trace: Optional[dict] = None
        legacy = None
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
            except (OSError, ValueError):
                raise ConfigError("H-MEM store file is unreadable or corrupt",
                                  details={"path": self.path}) from None
            if isinstance(obj, dict) and obj.get("engine") != "hmem":
                if not isinstance(obj.get("records"), list):
                    raise ConfigError("H-MEM store file is unreadable or corrupt",
                                      details={"path": self.path})
                legacy = obj["records"]
        try:
            if legacy is None:
                self.hmem = HMEM(clock=clock, actor="why-agent", path=self.path)
            else:
                self.hmem = HMEM(clock=clock, actor="why-agent", path=None)
                self._migrate(legacy)
                self.hmem.path = self.path
                self.hmem._persist()
        except (HMemError, KeyError, TypeError, ValueError):
            raise ConfigError("H-MEM store file is unreadable or corrupt",
                              details={"path": self.path}) from None

    # ---- conversion ----------------------------------------------------------
    def _migrate(self, rows: list) -> None:
        recs = [MemoryRecord.from_dict(r) for r in rows]
        for r in recs:
            self._add(r.entity, r.fact_key, r.fact, r.evidence_refs, r.confidence, r.source_run_id,
                      memory_id=r.memory_id, origin="import",
                      label="migrated from the Why-Agent stub store",
                      ext={"confirmations": r.confirmations, "createdAt": r.created_at,
                           "lastConfirmedAt": r.last_confirmed_at})
        for r in recs:
            if r.status == "superseded" and r.superseded_by and self.hmem.store.get(r.superseded_by):
                self.hmem.supersede(r.memory_id, r.superseded_by, r.supersede_reason or "migrated")
                m = self.hmem.store.get(r.memory_id)
                m.ext["supersedeReason"] = r.supersede_reason

    def _add(self, entity: str, fact_key: str, fact: str, refs: list, confidence: float,
             run_id: str, *, memory_id: Optional[str] = None, origin: str = "goal_run",
             label: Optional[str] = None, ext: Optional[dict] = None):
        existing = set(self.hmem.store.records) | set(self.hmem.store.archive)
        mid = memory_id or _new_id(existing, entity, fact_key, fact, run_id)
        now_iso = _iso(self.hmem.now())
        block = {"confirmations": 1, "evidenceRefs": list(refs), "runId": run_id,
                 "factKey": fact_key, "entity": entity, "createdAt": now_iso,
                 "lastConfirmedAt": now_iso}
        block.update(ext or {})
        return self.hmem.add(
            content=fact, source=f"why-agent:run:{run_id}", type="factual", layer="working",
            confidence=confidence, tags=[f"fact_key:{fact_key}", f"entity:{entity}"],
            namespace=self.NAMESPACE, entities=[entity], origin=origin,
            label=label or f"confirmed by the Critic in run {run_id}",
            refs={"goalRunId": run_id}, ext={"whyagent": block}, memory_id=mid)

    @staticmethod
    def _block(m) -> dict:
        b = m.ext.get("whyagent")
        if b is None:  # legacy migration stored the fields flat
            b = m.ext.setdefault("whyagent", {})
        for k in ("confirmations", "createdAt", "lastConfirmedAt", "supersedeReason"):
            if k in m.ext:
                b.setdefault(k, m.ext.pop(k))
        return b

    def _to_record(self, m) -> MemoryRecord:
        b = self._block(m)
        entity = b.get("entity") or (m.entities[0]["name"] if m.entities else "")
        fact_key = b.get("factKey") or next(
            (t.split(":", 1)[1] for t in m.tags if t.startswith("fact_key:")), "")
        return MemoryRecord(
            memory_id=m.id, entity=entity, fact_key=fact_key, fact=m.content,
            evidence_refs=list(b.get("evidenceRefs", [])), confidence=round(m.confidence, 2),
            source_run_id=b.get("runId") or m.source.rsplit(":", 1)[-1],
            created_at=b.get("createdAt") or _iso(m.timestamp),
            last_confirmed_at=b.get("lastConfirmedAt") or _iso(m.last_accessed),
            confirmations=int(b.get("confirmations", 1)),
            status="superseded" if m.superseded_by else "confirmed",
            superseded_by=m.superseded_by, supersede_reason=b.get("supersedeReason"),
        )

    def _current(self, entity: str, fact_key: str):
        tag = f"fact_key:{fact_key}"
        for m in self.hmem.store.all():
            if (not m.superseded_by and tag in m.tags
                    and any(e["name"] == entity for e in m.entities)):
                return m
        return None

    # ---- EngineAdapter API ---------------------------------------------------
    def store(self, entity: str, fact_key: str, fact: str, *, evidence_refs: list[str],
              confidence: float, run_id: str) -> MemoryRecord:
        for label, v in (("entity", entity), ("fact_key", fact_key), ("fact", fact)):
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"{label} must be a non-empty string")
        conf = round(max(0.0, min(1.0, float(confidence))), 2)
        refs = [str(r) for r in (evidence_refs or [])]
        with self._lock:
            current = self._current(entity, fact_key)
            if current is not None and current.content == fact.strip():
                # Re-confirmation is positive feedback: +0.05 confidence, +1 access (ledgered).
                self.hmem.feedback(current.id, "helpful")
                b = self._block(current)
                b["confirmations"] = int(b.get("confirmations", 1)) + 1
                b["lastConfirmedAt"] = _iso(self.hmem.now())
                for ref in refs:
                    if ref not in b.setdefault("evidenceRefs", []):
                        b["evidenceRefs"].append(ref)
                if conf > current.confidence:
                    self.hmem.update(current.id, reason=f"higher confidence in run {run_id}",
                                     confidence=conf)
                else:
                    self.hmem._persist()
                return self._to_record(current)
            m = self._add(entity, fact_key, fact, refs, conf, run_id)
            if current is not None:
                self._block(current)["supersedeReason"] = f"newer confirmed finding in run {run_id}"
                self.hmem.supersede(current.id, m.id, f"newer confirmed finding in run {run_id}")
            return self._to_record(m)

    def recall(self, entities: Sequence[str] = (), query: Optional[str] = None,
               limit: int = 5) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        wanted = [e for e in (entities or ()) if isinstance(e, str) and e]
        if not wanted and not (query and query.strip()):
            return []
        text = " ".join([query or ""] + wanted).strip()
        with self._lock:
            res = self.hmem.query(text, entities=wanted, threshold=0.0, top_n=10_000,
                                  namespace=self.NAMESPACE, strategy="comprehensive")
        self.last_trace = res["trace"]
        out: list[MemoryRecord] = []
        for hit in res["memories"]:
            m, sig = hit["memory"], hit["signals"]
            if m.superseded_by:
                continue  # penalised and counted in trace.suppressedSuperseded
            if sig["entity"] <= 0 and sig["keyword"] <= 0:
                continue  # no grounding in the question: not a hypothesis worth raising
            out.append(self._to_record(m))
            if len(out) >= limit:
                break
        return out

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            m = self.hmem.store.get_any(memory_id)
            return self._to_record(m) if m is not None else None

    def forget(self, memory_id: str) -> bool:
        with self._lock:
            return self.hmem.delete(memory_id, reason="forget requested")

    def health(self) -> dict:
        from .hmem import RANKER_VERSION, WEIGHT_PROFILE
        with self._lock:
            return {"engine": self.name, "state": "online", "path": self.path,
                    "records": len(self.hmem.store.records),
                    "archived": len(self.hmem.store.archive),
                    "ledger_entries": len(self.hmem.ledger.entries),
                    "ranker": RANKER_VERSION, "weights": WEIGHT_PROFILE}


def build_engine(config: Any) -> EngineAdapter:
    """Select the engine from a Config (``engine`` and ``hmem_path`` attributes)."""
    if getattr(config, "engine", "stub") == "meterless":
        return HMemMeterless(config.hmem_path)
    return HMemStub(config.hmem_path)
