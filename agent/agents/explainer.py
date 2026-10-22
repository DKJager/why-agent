"""Explainer agent (DESIGN section 9.4).

Builds the final ``Answer`` from accepted claims only. The narrative comes
from the ``TASK:narrate`` provider call and is verified by the Critic; on any
failure the deterministic renderer in ``agent/rules.py`` is used instead.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import rules
from ..errors import IterationLimit, RunTimeout
from ..models import Answer, Plan, Review
from ..sources import Agg, QuerySpec
from . import RunContext
from .critic import Critic

AGENT = "Explainer"


def _as_dict(obj: Any) -> Any:
    return obj.to_dict() if hasattr(obj, "to_dict") else obj


class Explainer:
    name = AGENT

    def __init__(self, critic: Optional[Critic] = None) -> None:
        self.critic = critic or Critic()

    # ---- helpers -----------------------------------------------------------
    def _data_as_of(self, ctx: RunContext) -> Optional[str]:
        try:
            spec = QuerySpec("orders", aggs=(Agg("max", "order_date", "data_as_of"),), limit=1)
            res = ctx.query_source(AGENT, spec, purpose="latest order date (data_as_of)")
            val = res.rows[0].get("data_as_of") if res.rows else None
            return str(val) if val else None
        except (RunTimeout, IterationLimit):
            raise
        except Exception:
            return None

    def _narrative(self, ctx: RunContext, payload: dict, accepted: list, evidence: list) -> tuple:
        """Return (text, used_fallback, reason)."""
        try:
            res = ctx.call_provider(AGENT, "narrate", payload, json_mode=False, max_tokens=1500)
        except (RunTimeout, IterationLimit):
            raise
        except Exception as e:
            return rules.render_answer(payload), True, f"provider error {type(e).__name__}"
        text = (res.text or "").strip()
        if not text:
            return rules.render_answer(payload), True, "empty narrative"
        rejections = self.critic.review_narrative(ctx, text, accepted, evidence)
        if rejections:
            return rules.render_answer(payload), True, f"{rejections[0].code}: {rejections[0].detail}"
        return text, False, ""

    # ---- public ------------------------------------------------------------
    def run(self, ctx: RunContext, plan: Plan, review: Review, evidence: list,
            unverified: list, recalled: list) -> Answer:
        accepted = list(review.accepted)
        recalled_d = [_as_dict(r) for r in (recalled or [])]
        payload = {
            "parsed": plan.parsed.to_dict(),
            "accepted": [c.to_dict() for c in accepted],
            "recalled": recalled_d,
        }
        parts = rules.answer_parts(payload)

        text, fell_back, reason = self._narrative(ctx, payload, accepted, evidence)
        if fell_back:
            ctx.trace.emit(AGENT, "fallback", "narrate", f"using rules.render_answer ({reason})",
                           status="fallback")

        caveats: list[str] = []
        n_unv = len({getattr(u, "claim_id", None) for u in (unverified or [])})
        if n_unv:
            caveats.append(f"{n_unv} claim(s) could not be verified by the Critic and are excluded "
                           f"from this answer (see output.unverified).")
        else:
            caveats.append("0 claims were left unverified.")
        as_of = self._data_as_of(ctx)
        if as_of:
            caveats.append(f"Data is current as of {as_of}.")
        if getattr(ctx.config, "sample_mode", False):
            caveats.append("Sample mode: all data is fictional synthetic data; no real company, "
                           "customer or carrier is represented.")
        if parts["prior_knowledge"]:
            caveats.append("Prior knowledge from H-MEM is a hypothesis, not evidence, and was not "
                           "used to support any finding.")

        answer = Answer(
            headline=parts["headline"],
            drivers=parts["drivers"],
            ruled_out=parts["ruled_out"],
            prior_knowledge=parts["prior_knowledge"],
            caveats=caveats,
            text=text,
        )
        confs = [d.get("confidence") for d in answer.drivers if d.get("confidence") is not None]
        ctx.emit(AGENT, "synthesize_answer",
                 f"{len(accepted)} accepted claim(s), {len(evidence)} evidence item(s)",
                 f"{len(answer.drivers)} driver(s), {len(answer.ruled_out)} ruled out; "
                 f"{'deterministic' if fell_back else 'model'} narrative",
                 confidence=max(confs) if confs else None,
                 status="fallback" if fell_back else "success")
        return answer


__all__ = ["Explainer"]
