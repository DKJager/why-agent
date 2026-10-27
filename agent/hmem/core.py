"""H-MEM: a Python implementation of the Meterless H-MEM engine spec.

Implements the portable memory record, the trust ledger, eight-signal hybrid
retrieval, feedback deltas, the preview-first sleep cycle, approval-gated
dreaming and the conflict gate, as written in Meterless/H-MEM (MIT):
docs/memory-record.md, docs/retrieval-ranking.md, docs/reinjection.md,
docs/conflict-resolution.md and the conformance vectors in
``agent/hmem/conformance/vectors.json`` (version 1, copied verbatim).

The engine is pure Python with no dependencies. Embeddings are a deterministic
64-dimension hashed bag of words, so results are reproducible offline; the
spec leaves the embedder open and asserts ordering only where it is
embedding-independent.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional

# ---------------------------------------------------------------------------
# Constants fixed by the spec
# ---------------------------------------------------------------------------
MEMORY_TYPES = ("personal", "factual", "preference", "general")
MEMORY_LAYERS = ("short_term", "working", "long_term")
ORIGINS = ("chat", "import", "web", "decision", "curiosity", "goal_run", "user_explicit",
           "inference", "plan_run", "mission", "swarm_run", "tool", "brain")

FEEDBACK = {
    # kind: (confidence delta, accessCount delta, tag added)
    "helpful": (0.05, 1, None),
    "not_helpful": (-0.03, 0, None),
    "wrong": (-0.20, 0, "review"),
}

WEIGHTS = {
    "semantic": 0.35, "keyword": 0.20, "tag": 0.10, "domain": 0.10,
    "entity": 0.15, "layer": 0.05, "recency": 0.05, "superseded": -0.20,
}
WEIGHT_PROFILE = "weights@2026-10"
RANKER_VERSION = "whyagent-hmem/1.0"
LAYER_WEIGHT = {"long_term": 1.0, "working": 0.8, "short_term": 0.6}
RECENCY_TAU_MS = 14 * 24 * 60 * 60 * 1000
DEFAULT_TOP_N = 5
DEFAULT_THRESHOLD = 0.35
ENTITY_CAP = 10
RELATED_CAP = 5
EMBED_DIM = 64
AUTO_RESOLVE_GATE = 0.70

DAY_MS = 24 * 60 * 60 * 1000
# Sleep cycle: plain records not touched for this long, with little use, archive.
ARCHIVE_AGE_MS = 30 * DAY_MS
ARCHIVE_MAX_ACCESS = 2
# Promotion: a working record that keeps proving useful becomes long_term.
PROMOTE_MIN_ACCESS = 3
PROMOTE_MIN_CONFIDENCE = 0.75

OPPOSING_PAIRS = (("always", "never"), ("enable", "disable"), ("enabled", "disabled"),
                  ("before", "after"), ("increase", "decrease"), ("allow", "deny"))

SOURCE_RELIABILITY = {"user_correction": 1.0, "user_explicit": 0.9, "runbook": 0.8,
                      "dreaming": 0.5, "inference": 0.5}

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "is", "are", "was",
    "were", "be", "it", "this", "that", "with", "by", "as", "we", "do", "does", "did", "how",
    "when", "what", "why", "which", "who", "our", "you", "i", "from", "has", "had", "have",
}

_DOMAIN_HINTS = {
    "ops/logistics": {"carrier", "shipment", "shipments", "delivery", "delay", "sla", "courier"},
    "ops/revenue": {"revenue", "orders", "order", "sales", "cancellation", "cancellations", "refund"},
    "ops/support": {"ticket", "tickets", "support", "complaint", "complaints"},
    "tech/deploy": {"deploy", "docker", "release", "api", "service"},
}


class HMemError(Exception):
    """Base error for the engine."""


class RecordRejected(HMemError, ValueError):
    """A write violated the record contract (for example: no source)."""


class UnknownMemory(HMemError, KeyError):
    """No record with that id."""


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def embed(text: str) -> list[float]:
    """Deterministic 64-dim hashed bag of words with bigrams, L2-normalised."""
    vec = [0.0] * EMBED_DIM
    toks = tokenize(text)
    feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
    for f in feats:
        h = hashlib.sha256(f.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % EMBED_DIM
        sign = 1.0 if h[4] & 1 else -1.0
        vec[idx] += sign * (1.0 if "_" not in f else 0.5)
    norm = math.sqrt(sum(v * v for v in vec))
    return [round(v / norm, 6) for v in vec] if norm else vec


def cosine(a: Optional[list[float]], b: Optional[list[float]]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def infer_domain(text: str) -> Optional[str]:
    toks = set(tokenize(text))
    best, hits = None, 0
    for domain, words in _DOMAIN_HINTS.items():
        n = len(toks & words)
        if n > hits:
            best, hits = domain, n
    return best


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _round(x: float) -> float:
    # Feedback arithmetic must be exact to the spec's two decimals.
    return round(x + 0.0, 10)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
@dataclass
class Memory:
    id: str
    content: str
    type: str
    layer: str
    timestamp: int
    last_accessed: int
    access_count: int
    confidence: float
    source: str
    tags: list[str] = field(default_factory=list)
    domain: Optional[str] = None
    namespace: Optional[str] = None
    embedding: Optional[list[float]] = None
    provenance: Optional[dict] = None
    chat_id: Optional[str] = None
    mission_id: Optional[str] = None
    goal_run_id: Optional[str] = None
    entities: list[dict] = field(default_factory=list)
    related_to: list[str] = field(default_factory=list)
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    derived_from: list[str] = field(default_factory=list)
    # Application extension block; not part of the portable contract.
    ext: dict = field(default_factory=dict)

    _CAMEL = {
        "last_accessed": "lastAccessed", "access_count": "accessCount", "chat_id": "chatId",
        "mission_id": "missionId", "goal_run_id": "goalRunId", "related_to": "relatedTo",
        "superseded_by": "supersededBy", "derived_from": "derivedFrom",
    }

    def to_dict(self) -> dict:
        """Portable, spec-shaped (camelCase) record."""
        out = {}
        for k, v in asdict(self).items():
            out[self._CAMEL.get(k, k)] = copy.deepcopy(v)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        rev = {v: k for k, v in cls._CAMEL.items()}
        kw = {rev.get(k, k): copy.deepcopy(v) for k, v in d.items()}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in kw.items() if k in known})

    def snapshot(self) -> dict:
        s = self.to_dict()
        s.pop("embedding", None)
        return s


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------
class TrustLedger:
    """Append-only audit trail. Every mutation writes an entry."""

    def __init__(self, engine: "HMEM") -> None:
        self._e = engine
        self.entries: list[dict] = []

    def append(self, action: str, memory_id: str, previous: Optional[dict], new: Optional[dict],
               reason: str = "") -> dict:
        entry = {
            "id": "led-" + uuid.uuid4().hex[:12],
            "memoryId": memory_id,
            "action": action,
            "actor": self._e.actor,
            "timestamp": self._e.now(),
            "previousState": previous,
            "newState": new,
            "reason": reason,
        }
        self.entries.append(entry)
        return entry

    def history(self, memory_id: str) -> list[dict]:
        return [copy.deepcopy(e) for e in self.entries if e["memoryId"] == memory_id]

    def all(self) -> list[dict]:
        return copy.deepcopy(self.entries)


class MemoryStore:
    def __init__(self) -> None:
        self.records: dict[str, Memory] = {}
        self.archive: dict[str, Memory] = {}

    def get(self, memory_id: str) -> Optional[Memory]:
        return self.records.get(memory_id)

    def get_any(self, memory_id: str) -> Optional[Memory]:
        return self.records.get(memory_id) or self.archive.get(memory_id)

    def all(self) -> list[Memory]:
        return list(self.records.values())


class SleepCycle:
    """Preview-first consolidation: preview never mutates; execute backs up first."""

    def __init__(self, engine: "HMEM") -> None:
        self._e = engine
        self.backups: dict[str, dict] = {}

    def _protected(self, m: Memory) -> bool:
        return bool(m.derived_from) or m.source in ("dreaming", "sleep") or "pinned" in m.tags

    def preview(self) -> dict:
        now = self._e.now()
        to_archive, to_promote, protected = [], [], []
        for m in self._e.store.all():
            if self._protected(m):
                protected.append(m.id)
                continue
            age = now - m.last_accessed
            if (m.layer != "long_term" and age >= ARCHIVE_AGE_MS
                    and m.access_count <= ARCHIVE_MAX_ACCESS):
                to_archive.append(m.id)
            elif (m.layer == "working" and m.access_count >= PROMOTE_MIN_ACCESS
                  and m.confidence >= PROMOTE_MIN_CONFIDENCE and not m.superseded_by):
                to_promote.append(m.id)
        return {"toArchive": to_archive, "toPromote": to_promote, "protected": protected,
                "generatedAt": now}

    def execute(self, preview: dict) -> dict:
        with self._e._lock:
            backup_id = "bak-" + uuid.uuid4().hex[:12]
            self.backups[backup_id] = {
                "records": [m.to_dict() for m in self._e.store.all()],
                "takenAt": self._e.now(),
            }
            archived, promoted = [], []
            for mid in preview.get("toArchive", []):
                m = self._e.store.get(mid)
                if m is None or self._protected(m):  # guardrails re-checked at execute time
                    continue
                prev = m.snapshot()
                self._e.store.archive[mid] = self._e.store.records.pop(mid)
                self._e.ledger.append("archive", mid, prev, m.snapshot(), "sleep cycle")
                archived.append(mid)
            for mid in preview.get("toPromote", []):
                m = self._e.store.get(mid)
                if m is None or m.layer != "working":
                    continue
                prev = m.snapshot()
                m.layer = "long_term"
                self._e.ledger.append("promote", mid, prev, m.snapshot(), "sleep cycle")
                promoted.append(mid)
            self._e._persist()
            return {"backupId": backup_id, "archived": archived, "promoted": promoted}


class Dreaming:
    """Synthesis proposals. Nothing materialises until a proposal is approved."""

    def __init__(self, engine: "HMEM") -> None:
        self._e = engine
        self.proposals: dict[str, dict] = {}

    def dream(self) -> list[dict]:
        active = [m for m in self._e.store.all() if not m.superseded_by and "review" not in m.tags]
        out: list[dict] = []
        seen: set[tuple] = set()
        # Same-type preferences consolidate into one profile proposal.
        prefs = [m for m in active if m.type == "preference"]
        if len(prefs) >= 2:
            key = ("pref",) + tuple(sorted(m.id for m in prefs))
            seen.add(key)
            out.append(self._propose(prefs, "preference",
                                     "Preference profile: " + "; ".join(m.content for m in prefs),
                                     "related preferences"))
        # Strongly overlapping records of any type.
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                if a.type == "preference" and b.type == "preference":
                    continue
                sim = _jaccard(tokenize(a.content), tokenize(b.content))
                shared_entities = {e["name"] for e in a.entities} & {e["name"] for e in b.entities}
                if sim >= 0.5 or (shared_entities and sim >= 0.2):
                    key = tuple(sorted((a.id, b.id)))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(self._propose([a, b], a.type,
                                             f"{a.content} (also: {b.content})",
                                             "overlapping records"))
        return out

    def _propose(self, members: list[Memory], mtype: str, content: str, rationale: str) -> dict:
        p = {
            "id": "dream-" + hashlib.sha256("|".join(sorted(m.id for m in members)).encode()).hexdigest()[:12],
            "content": content,
            "type": mtype,
            "derivedFrom": [m.id for m in members],
            "rationale": rationale,
            "status": "pending",
        }
        self.proposals.setdefault(p["id"], p)
        return copy.deepcopy(self.proposals[p["id"]])

    def approve(self, proposal_id: str) -> Optional[Memory]:
        p = self.proposals.get(proposal_id)
        if p is None or p["status"] != "pending":
            return None
        conf = min((self._e.store.get(i).confidence for i in p["derivedFrom"]
                    if self._e.store.get(i)), default=0.7)
        m = self._e.add(content=p["content"], type=p["type"], layer="long_term",
                        source="dreaming", confidence=min(conf, 0.95),
                        derived_from=p["derivedFrom"], origin="inference",
                        label="approved dream proposal " + proposal_id, action="synthesis")
        p["status"] = "approved"
        return m

    def reject(self, proposal_id: str) -> bool:
        p = self.proposals.get(proposal_id)
        if p is None or p["status"] != "pending":
            return False
        p["status"] = "rejected"
        return True


class ConflictGate:
    def __init__(self, engine: "HMEM") -> None:
        self._e = engine
        self.queue: dict[str, dict] = {}

    @staticmethod
    def _opposed(ta: set[str], tb: set[str]) -> Optional[tuple[str, str]]:
        for x, y in OPPOSING_PAIRS:
            if (x in ta and y in tb and y not in ta) or (y in ta and x in tb and x not in ta):
                return (x, y)
        return None

    def scan(self) -> list[dict]:
        active = [m for m in self._e.store.all() if not m.superseded_by]
        found = []
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                if a.supersedes == b.id or b.supersedes == a.id:
                    continue  # intended replacement, not a conflict
                ta, tb = set(tokenize(a.content)), set(tokenize(b.content))
                pair = self._opposed(ta, tb)
                reason = None
                sim = 0.0
                if pair:
                    core_a, core_b = ta - set(pair), tb - set(pair)
                    sim = _jaccard(core_a, core_b)
                    if sim >= 0.5:
                        reason = f"opposing claims ({pair[0]}/{pair[1]}) on similar content"
                if reason is None:
                    na = {t for t in ta if t.isdigit()}
                    nb = {t for t in tb if t.isdigit()}
                    sim = _jaccard(ta - na, tb - nb)
                    if na and nb and na != nb and sim >= 0.6:
                        reason = "similar claims with divergent numeric values"
                if reason is None:
                    continue
                conf = 0.6 + 0.3 * sim
                if {e["name"] for e in a.entities} & {e["name"] for e in b.entities}:
                    conf += 0.05
                if a.domain and a.domain == b.domain:
                    conf += 0.05
                cid = "conflict-" + hashlib.sha256(f"{a.id}|{b.id}".encode()).hexdigest()[:12]
                found.append({"id": cid, "memoryA": a.id, "memoryB": b.id, "reason": reason,
                              "confidence": round(min(conf, 0.99), 4), "status": "open"})
        return found

    def _merit(self, m: Memory, newest: int) -> float:
        recency = math.exp(-max(0, newest - m.timestamp) / RECENCY_TAU_MS)
        access = min(1.0, m.access_count / 10.0)
        richness = min(1.0, len(tokenize(m.content)) / 20.0)
        reliability = SOURCE_RELIABILITY.get(m.source, 0.6)
        return 0.5 * m.confidence + 0.15 * recency + 0.1 * access + 0.1 * richness + 0.15 * reliability

    def auto_resolve(self, conflict: dict) -> str:
        with self._e._lock:
            a = self._e.store.get(conflict["memoryA"])
            b = self._e.store.get(conflict["memoryB"])
            if a is None or b is None:
                return "queued"
            newest = max(a.timestamp, b.timestamp)
            ma, mb = self._merit(a, newest), self._merit(b, newest)
            decision_conf = min(1.0, 0.5 + abs(ma - mb))
            if decision_conf < AUTO_RESOLVE_GATE:
                q = dict(conflict, status="queued", decisionConfidence=round(decision_conf, 4))
                self.queue[conflict["id"]] = q
                self._e.ledger.append("conflict_queued", a.id, a.snapshot(), a.snapshot(), conflict["reason"])
                self._e.ledger.append("conflict_queued", b.id, b.snapshot(), b.snapshot(), conflict["reason"])
                self._e._persist()
                return "queued"
            winner, loser = (a, b) if ma >= mb else (b, a)
            prev_l, prev_w = loser.snapshot(), winner.snapshot()
            loser.superseded_by = winner.id
            if not winner.supersedes:
                winner.supersedes = loser.id
            self._e.ledger.append("resolve", loser.id, prev_l, loser.snapshot(), conflict["reason"])
            self._e.ledger.append("resolve", winner.id, prev_w, winner.snapshot(), conflict["reason"])
            self._e._persist()
            return "resolved"

    # Spec-style alias used by the conformance runner.
    autoResolve = auto_resolve


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class HMEM:
    """H-MEM engine: ``add``, ``query``, ``feedback`` plus the ``store``, ``ledger``,
    ``sleep``, ``dreaming`` and ``conflicts`` services."""

    def __init__(self, clock: Optional[Callable[[], int]] = None, actor: str = "why-agent",
                 path: Optional[str] = None) -> None:
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.actor = actor
        self.path = path
        self._lock = threading.RLock()
        self.store = MemoryStore()
        self.ledger = TrustLedger(self)
        self.sleep = SleepCycle(self)
        self.dreaming = Dreaming(self)
        self.conflicts = ConflictGate(self)
        if path:
            self._load()

    def now(self) -> int:
        return int(self.clock())

    # ---- writes ------------------------------------------------------------
    def add(self, content: str, source: str, *, type: str = "general", layer: str = "working",
            confidence: float = 0.7, tags: Optional[list[str]] = None, domain: Optional[str] = None,
            namespace: Optional[str] = None, entities: Optional[list] = None,
            related_to: Optional[list[str]] = None, supersedes: Optional[str] = None,
            derived_from: Optional[list[str]] = None, origin: Optional[str] = None,
            label: Optional[str] = None, refs: Optional[dict] = None, chat_id: Optional[str] = None,
            mission_id: Optional[str] = None, goal_run_id: Optional[str] = None,
            ext: Optional[dict] = None, memory_id: Optional[str] = None,
            action: str = "create") -> Memory:
        if not isinstance(source, str) or not source.strip():
            raise RecordRejected("record rejected: source is mandatory")
        if not isinstance(content, str) or not content.strip():
            raise RecordRejected("record rejected: content is empty")
        if type not in MEMORY_TYPES:
            raise RecordRejected(f"record rejected: unknown type {type!r}")
        if layer not in MEMORY_LAYERS:
            raise RecordRejected(f"record rejected: unknown layer {layer!r}")
        if confidence is None or not isinstance(confidence, (int, float)) or math.isnan(confidence):
            raise RecordRejected("record rejected: confidence is mandatory")
        origin = origin or ("inference" if source == "dreaming" else "tool")
        if origin not in ORIGINS:
            raise RecordRejected(f"record rejected: unknown provenance origin {origin!r}")
        ents = []
        for e in (entities or [])[:ENTITY_CAP]:
            if isinstance(e, str):
                e = {"name": e, "type": e.split(":", 1)[0] if ":" in e else "thing"}
            ents.append({"name": str(e["name"]).strip(), "type": str(e.get("type", "thing"))})
        with self._lock:
            now = self.now()
            mid = memory_id or ("m-" + hashlib.sha256(
                f"{content}|{source}|{now}|{uuid.uuid4().hex}".encode()).hexdigest()[:12])
            if self.store.get_any(mid):
                raise RecordRejected(f"record rejected: duplicate id {mid}")
            m = Memory(
                id=mid, content=content.strip(), type=type, layer=layer, timestamp=now,
                last_accessed=now, access_count=0, confidence=_round(_clamp01(float(confidence))),
                source=source, tags=list(dict.fromkeys(tags or [])), domain=domain or infer_domain(content),
                namespace=namespace, embedding=embed(content), chat_id=chat_id, mission_id=mission_id,
                goal_run_id=goal_run_id, entities=ents, related_to=list(related_to or [])[:RELATED_CAP],
                supersedes=supersedes, derived_from=list(derived_from or []), ext=dict(ext or {}),
                provenance={"origin": origin, "learnedAt": now,
                            "label": label or f"{origin} write from {source}",
                            **({"refs": dict(refs)} if refs else {})},
            )
            self.store.records[mid] = m
            self.ledger.append(action, mid, None, m.snapshot(), label or "")
            self._persist()
            return m

    def feedback(self, memory_id: str, kind: str) -> Memory:
        if kind not in FEEDBACK:
            raise HMemError(f"unknown feedback kind {kind!r}")
        with self._lock:
            m = self.store.get(memory_id)
            if m is None:
                raise UnknownMemory(memory_id)
            prev = m.snapshot()
            delta, access, tag = FEEDBACK[kind]
            m.confidence = _round(_clamp01(m.confidence + delta))
            m.access_count += access
            if access:
                m.last_accessed = self.now()
            if tag and tag not in m.tags:
                m.tags.append(tag)
            self.ledger.append("feedback", memory_id, prev, m.snapshot(), kind)
            self._persist()
            return m

    def update(self, memory_id: str, reason: str = "", **changes: Any) -> Memory:
        with self._lock:
            m = self.store.get(memory_id)
            if m is None:
                raise UnknownMemory(memory_id)
            prev = m.snapshot()
            for k, v in changes.items():
                if k in ("id", "timestamp", "source", "provenance"):
                    raise HMemError(f"{k} is immutable")
                setattr(m, k, v)
            if "content" in changes:
                m.embedding = embed(m.content)
            self.ledger.append("update", memory_id, prev, m.snapshot(), reason)
            self._persist()
            return m

    def supersede(self, old_id: str, new_id: str, reason: str = "") -> None:
        with self._lock:
            old, new = self.store.get(old_id), self.store.get(new_id)
            if old is None or new is None:
                raise UnknownMemory(old_id if old is None else new_id)
            prev = old.snapshot()
            old.superseded_by = new_id
            self.ledger.append("supersede", old_id, prev, old.snapshot(), reason)
            if new.supersedes != old_id:
                prev_n = new.snapshot()
                new.supersedes = old_id
                self.ledger.append("update", new_id, prev_n, new.snapshot(), "supersedes edge")
            self._persist()

    def delete(self, memory_id: str, reason: str = "") -> bool:
        with self._lock:
            m = self.store.records.pop(memory_id, None) or self.store.archive.pop(memory_id, None)
            if m is None:
                return False
            self.ledger.append("delete", memory_id, m.snapshot(), None, reason)
            self._persist()
            return True

    # ---- retrieval -----------------------------------------------------------
    def score(self, m: Memory, q: dict) -> dict:
        mt = set(tokenize(m.content))
        tag_toks = {t for tag in m.tags for t in tokenize(tag)}
        ents = {e["name"].lower() for e in m.entities}
        signals = {
            "semantic": cosine(q["embedding"], m.embedding or embed(m.content)),
            "keyword": _jaccard(q["keywords"], mt),
            "tag": _jaccard(q["tokens"], tag_toks) if tag_toks else 0.0,
            "domain": 1.0 if (q["domain"] and m.domain == q["domain"]) else 0.0,
            "entity": (len(q["entities"] & ents) / len(q["entities"])) if q["entities"] else 0.0,
            "layer": LAYER_WEIGHT.get(m.layer, 0.0),
            "recency": math.exp(-max(0, q["now"] - m.last_accessed) / RECENCY_TAU_MS),
            "superseded": 1.0 if m.superseded_by else 0.0,
        }
        raw = sum(WEIGHTS[k] * v for k, v in signals.items())
        return {"signals": signals, "raw": raw, "score": _clamp01(raw) * m.confidence}

    def query(self, text: str, *, threshold: float = DEFAULT_THRESHOLD, topN: Optional[int] = None,
              top_n: Optional[int] = None, strategy: str = "comprehensive",
              entities: Optional[Iterable[str]] = None, chat_id: Optional[str] = None,
              goal_run_id: Optional[str] = None, include_unscoped: bool = True,
              namespace: Optional[str] = None) -> dict:
        n = top_n if top_n is not None else (topN if topN is not None else DEFAULT_TOP_N)
        toks = tokenize(text)
        keywords = list(dict.fromkeys(toks))[:8]
        explicit = {e.lower() for e in (entities or []) if isinstance(e, str) and e}
        with self._lock:
            pool = self.store.all()
        known = {e["name"].lower() for m in pool for e in m.entities}
        text_l = (text or "").lower()
        q_entities = explicit | {e for e in known if e and e in text_l}
        q = {"embedding": embed(text), "keywords": keywords, "tokens": set(toks),
             "domain": infer_domain(text), "entities": q_entities, "now": self.now()}
        audit = strategy in ("audit", "debug")
        scored, suppressed = [], 0
        for m in pool:
            if namespace and m.namespace != namespace:
                continue
            if chat_id is not None and m.chat_id not in (chat_id,) + ((None,) if include_unscoped else ()):
                continue
            if goal_run_id is not None and m.goal_run_id not in (goal_run_id,) + ((None,) if include_unscoped else ()):
                continue
            if goal_run_id is None and m.goal_run_id is not None:
                continue  # run-scoped memory never leaks into another run
            if not audit and ("review" in m.tags or "wrong" in m.tags):
                continue
            s = self.score(m, q)
            if s["score"] < threshold:
                continue
            if m.superseded_by:
                suppressed += 1
            scored.append((s["score"], m.timestamp, m, s))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        top = scored[:n]
        memories = [{"memory": m, "score": round(sc, 6), "signals": {k: round(v, 6) for k, v in s["signals"].items()}}
                    for sc, _, m, s in top]
        trace = {
            "traceId": "tr-" + uuid.uuid4().hex[:12],
            "retrievalReason": f"hybrid ranking for {text!r}",
            "strategy": strategy,
            "queryDomain": q["domain"],
            "queryEntities": sorted(q_entities),
            "scores": {x["memory"].id: x["score"] for x in memories},
            "rankerVersion": RANKER_VERSION,
            "weightProfile": WEIGHT_PROFILE,
            "suppressedSuperseded": suppressed,
        }
        return {"memories": memories, "context": self.format_context([x["memory"] for x in memories]),
                "trace": trace}

    @staticmethod
    def format_context(memories: list[Memory]) -> str:
        groups: dict[str, list[Memory]] = {}
        for m in memories:
            groups.setdefault(m.domain or "general", []).append(m)
        lines = []
        for domain, ms in groups.items():
            for m in ms:
                ents = ", ".join(e["name"] for e in m.entities)
                line = f"[{domain}] ({m.layer}) {m.content}"
                if ents:
                    line += " {entities: " + ents + "}"
                lines.append(line)
        return "\n".join(lines)

    # ---- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        if not isinstance(obj, dict):
            raise HMemError("H-MEM store file is not an object")
        if obj.get("engine") == "hmem":
            self.store.records = {d["id"]: Memory.from_dict(d) for d in obj.get("records", [])}
            self.store.archive = {d["id"]: Memory.from_dict(d) for d in obj.get("archive", [])}
            self.ledger.entries = list(obj.get("ledger", []))
            self.dreaming.proposals = {p["id"]: p for p in obj.get("proposals", [])}
            self.conflicts.queue = {c["id"]: c for c in obj.get("conflictQueue", [])}
            return
        raise HMemError("H-MEM store file has an unknown layout")

    def _persist(self) -> None:
        if not self.path:
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        payload = {
            "engine": "hmem", "version": 1, "weightProfile": WEIGHT_PROFILE,
            "records": [m.to_dict() for m in self.store.records.values()],
            "archive": [m.to_dict() for m in self.store.archive.values()],
            "ledger": self.ledger.entries,
            "proposals": list(self.dreaming.proposals.values()),
            "conflictQueue": list(self.conflicts.queue.values()),
        }
        fd, tmp = tempfile.mkstemp(prefix=".hmem-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
