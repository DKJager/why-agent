# build-id: oa-c272301e
"""Core dataclasses (DESIGN §5.3, §9.4)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Metric = Literal["net_revenue", "ticket_count"]
Region = Literal["EU", "UK", "US", "ALL"]
Direction = Literal["drop", "rise", "change"]
SourceName = Literal["orders", "tickets", "shipments"]
SubSource = Literal["orders", "tickets", "shipments", "memory"]
NumFmt = Literal["pct1", "eur0", "int", "days1"]
ClaimKind = Literal["delta", "driver", "ruled_out", "observation", "memory"]
RejectCode = Literal[
    "NO_CITATION", "UNKNOWN_EVIDENCE", "NUMBER_MISMATCH", "UNGROUNDED_NUMBER",
    "RECOMPUTE_FAILED", "ROWREF_MISSING", "DRIVER_SINGLE_SOURCE", "MEMORY_NOT_FOUND",
    "SHARE_OVER_100",
]
MemStatus = Literal["confirmed", "superseded"]


def _asdict(obj: Any) -> dict:
    return dataclasses.asdict(obj)


@dataclass(frozen=True)
class ParsedQuestion:
    metric: Metric
    region: Region
    iso_year: int
    iso_week: int
    direction: Direction
    entities: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = _asdict(self)
        d["entities"] = list(self.entities)
        return d

    def to_output(self) -> dict:
        """The ``output.parsed`` object of the success response (no entities)."""
        return {
            "metric": self.metric,
            "region": self.region,
            "iso_year": self.iso_year,
            "iso_week": self.iso_week,
            "direction": self.direction,
        }

    @staticmethod
    def from_dict(d: dict) -> "ParsedQuestion":
        return ParsedQuestion(
            metric=d["metric"],
            region=d["region"],
            iso_year=int(d["iso_year"]),
            iso_week=int(d["iso_week"]),
            direction=d["direction"],
            entities=tuple(d.get("entities") or ()),
        )


@dataclass
class SubQuestion:
    id: str
    source: SubSource
    text: str
    purpose: str
    priority: int

    def to_dict(self) -> dict:
        return _asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SubQuestion":
        return SubQuestion(
            id=str(d["id"]),
            source=d["source"],
            text=str(d["text"]),
            purpose=str(d.get("purpose", "")),
            priority=int(d["priority"]),
        )


@dataclass
class Plan:
    parsed: ParsedQuestion
    subquestions: list[SubQuestion]
    recalled_memory_ids: list[str]
    rationale: str

    def to_dict(self) -> dict:
        return {
            "parsed": self.parsed.to_dict(),
            "subquestions": [s.to_dict() for s in self.subquestions],
            "recalled_memory_ids": list(self.recalled_memory_ids),
            "rationale": self.rationale,
        }

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        return Plan(
            parsed=ParsedQuestion.from_dict(d["parsed"]),
            subquestions=[SubQuestion.from_dict(s) for s in d.get("subquestions", [])],
            recalled_memory_ids=list(d.get("recalled_memory_ids", [])),
            rationale=str(d.get("rationale", "")),
        )


@dataclass
class NumberRef:
    evidence_id: str
    name: str
    value: float
    fmt: NumFmt

    def to_dict(self) -> dict:
        return _asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "NumberRef":
        return NumberRef(
            evidence_id=str(d["evidence_id"]),
            name=str(d["name"]),
            value=float(d["value"]),
            fmt=d["fmt"],
        )


@dataclass
class Evidence:
    evidence_id: str
    source: SourceName
    description: str
    query: dict
    query_text: str
    values: dict[str, float]
    rows: list[dict]
    row_refs: list[str]
    row_ref_total: int
    retrieved_at: str

    def to_dict(self) -> dict:
        return _asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Evidence":
        return Evidence(
            evidence_id=str(d["evidence_id"]),
            source=d["source"],
            description=str(d.get("description", "")),
            query=dict(d.get("query", {})),
            query_text=str(d.get("query_text", "")),
            values={k: float(v) for k, v in dict(d.get("values", {})).items()},
            rows=list(d.get("rows", [])),
            row_refs=list(d.get("row_refs", [])),
            row_ref_total=int(d.get("row_ref_total", 0)),
            retrieved_at=str(d.get("retrieved_at", "")),
        )


@dataclass
class Claim:
    claim_id: str
    kind: ClaimKind
    text: str
    numbers: list[NumberRef]
    evidence_ids: list[str]
    memory_id: Optional[str] = None
    confidence: float = 0.0
    explained_share: Optional[float] = None
    entities: tuple[str, ...] = ()
    fact_key: Optional[str] = None

    def to_dict(self) -> dict:
        d = _asdict(self)
        d["numbers"] = [n.to_dict() for n in self.numbers]
        d["entities"] = list(self.entities)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Claim":
        return Claim(
            claim_id=str(d["claim_id"]),
            kind=d["kind"],
            text=str(d["text"]),
            numbers=[NumberRef.from_dict(n) for n in d.get("numbers", [])],
            evidence_ids=list(d.get("evidence_ids", [])),
            memory_id=d.get("memory_id"),
            confidence=float(d.get("confidence", 0.0)),
            explained_share=d.get("explained_share"),
            entities=tuple(d.get("entities") or ()),
            fact_key=d.get("fact_key"),
        )


@dataclass
class Rejection:
    claim_id: str
    code: RejectCode
    detail: str
    needs: Optional[str] = None

    def to_dict(self) -> dict:
        return _asdict(self)


@dataclass
class Review:
    accepted: list[Claim]
    rejected: list[Rejection]

    def to_dict(self) -> dict:
        return {
            "accepted": [c.to_dict() for c in self.accepted],
            "rejected": [r.to_dict() for r in self.rejected],
        }


@dataclass
class MemoryRecord:
    memory_id: str
    entity: str
    fact_key: str
    fact: str
    evidence_refs: list[str]
    confidence: float
    source_run_id: str
    created_at: str
    last_confirmed_at: str
    confirmations: int
    status: MemStatus
    superseded_by: Optional[str] = None
    supersede_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return _asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MemoryRecord":
        return MemoryRecord(
            memory_id=str(d["memory_id"]),
            entity=str(d["entity"]),
            fact_key=str(d["fact_key"]),
            fact=str(d["fact"]),
            evidence_refs=list(d.get("evidence_refs", [])),
            confidence=float(d.get("confidence", 0.0)),
            source_run_id=str(d.get("source_run_id", "")),
            created_at=str(d.get("created_at", "")),
            last_confirmed_at=str(d.get("last_confirmed_at", d.get("created_at", ""))),
            confirmations=int(d.get("confirmations", 1)),
            status=d.get("status", "confirmed"),
            superseded_by=d.get("superseded_by"),
            supersede_reason=d.get("supersede_reason"),
        )


@dataclass
class Answer:
    headline: str
    drivers: list[dict] = field(default_factory=list)
    ruled_out: list[dict] = field(default_factory=list)
    prior_knowledge: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict:
        return _asdict(self)
