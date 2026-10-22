# build-id: oa-20310e3c
"""H-MEM engine adapters (DESIGN section 8).

* ``HMemStub``: fully functional offline JSON store.
* ``HMemMeterless``: placeholder (assumption A2), raises ``NotConfigured``.
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


_NOT_WIRED = (
    "Meterless H-MEM is not wired: set ENGINE=stub, or implement HMemMeterless against "
    "engines/hmem once the engine repo is available"
)


class HMemMeterless:
    """Placeholder (assumption A2). Every entry point raises ``NotConfigured``."""

    name = "hmem-meterless"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotConfigured(_NOT_WIRED)

    def store(self, entity: str, fact_key: str, fact: str, *, evidence_refs: list[str],
              confidence: float, run_id: str) -> MemoryRecord:
        raise NotConfigured(_NOT_WIRED)

    def recall(self, entities: Sequence[str] = (), query: Optional[str] = None,
               limit: int = 5) -> list[MemoryRecord]:
        raise NotConfigured(_NOT_WIRED)

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        raise NotConfigured(_NOT_WIRED)

    def forget(self, memory_id: str) -> bool:
        raise NotConfigured(_NOT_WIRED)

    def health(self) -> dict:
        raise NotConfigured(_NOT_WIRED)


def build_engine(config: Any) -> EngineAdapter:
    """Select the engine from a Config (``engine`` and ``hmem_path`` attributes)."""
    if getattr(config, "sample_mode", False) or getattr(config, "engine", "stub") == "stub":
        if getattr(config, "sample_mode", False) or config.engine == "stub":
            return HMemStub(config.hmem_path)
    return HMemMeterless()
