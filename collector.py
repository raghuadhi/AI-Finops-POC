"""
Collector: derives cost.usd_computed and cost.inr_computed from telemetry
attributes + the versioned price book.

Implements the Telemetry Spec token-accounting rules (section 5):
  - Do not double-count cache-read / cache-write tokens.
  - Billable uncached input is normalised by the collector:
        billable_input = input_tokens - cache_read - cache_write
    (cache reads/writes are billed at their own rates, not the input rate)
  - Reasoning tokens are treated as an output subset (already inside
    output_tokens), so they are NOT added again.

Cost components map to the spec's cost.component enum. The collector returns a
per-call USD cost and the INR conversion using a configurable FX rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from price_book import get_price_book


# cost.component enum values (Telemetry Spec, required attribute "cost.component")
COST_COMPONENTS = {
    "model_tokens",
    "embedding",
    "rerank",
    "vector_db",
    "tool_call",
    "guardrail",
    "human_review",
    "shared_overhead",
}


@dataclass
class CallUsage:
    """Normalised usage for a single call, as emitted on the span."""
    cost_component: str
    price_book_version: str
    pricing_tier: str = "standard"
    # token fields (gen_ai.usage.*)
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_output_tokens: int = 0  # subset of output_tokens; not re-added
    # non-token quantities
    queries: int = 0            # rerank / vector_db
    tool_calls: int = 0
    guardrail_calls: int = 0
    human_review_minutes: float = 0.0


@dataclass
class CostResult:
    usd: float
    inr: float
    breakdown: dict = field(default_factory=dict)

    def billable_input_tokens(self) -> int:
        return self.breakdown.get("billable_input_tokens", 0)


def compute_cost(usage: CallUsage, usd_to_inr: float) -> CostResult:
    """Compute USD + INR cost for one call from the versioned price book."""
    pb = get_price_book(usage.price_book_version)
    if usage.cost_component not in COST_COMPONENTS:
        raise ValueError(f"unknown cost.component: {usage.cost_component}")

    tier_mult = pb["tier_multipliers"].get(usage.pricing_tier, 1.0)
    usd = 0.0
    breakdown: dict = {}

    comp = usage.cost_component

    if comp in ("model_tokens", "embedding"):
        if usage.model not in pb["models"]:
            raise KeyError(f"model '{usage.model}' not in price book {usage.price_book_version}")
        rates = pb["models"][usage.model]

        # Rule: do not double-count cache tokens. Billable uncached input is
        # the total input minus cache-read and cache-write portions.
        billable_input = max(
            0,
            usage.input_tokens
            - usage.cache_read_input_tokens
            - usage.cache_write_input_tokens,
        )
        # Reasoning tokens are a subset of output_tokens -> do not add again.
        billable_output = usage.output_tokens

        input_cost = billable_input / 1_000_000 * rates["input_per_mtok"]
        output_cost = billable_output / 1_000_000 * rates["output_per_mtok"]
        cache_read_cost = (
            usage.cache_read_input_tokens / 1_000_000 * rates["cache_read_per_mtok"]
        )
        cache_write_cost = (
            usage.cache_write_input_tokens / 1_000_000 * rates["cache_write_per_mtok"]
        )
        usd = input_cost + output_cost + cache_read_cost + cache_write_cost
        breakdown = {
            "billable_input_tokens": billable_input,
            "billable_output_tokens": billable_output,
            "input_cost_usd": round(input_cost, 8),
            "output_cost_usd": round(output_cost, 8),
            "cache_read_cost_usd": round(cache_read_cost, 8),
            "cache_write_cost_usd": round(cache_write_cost, 8),
        }

    elif comp == "rerank":
        usd = usage.queries * pb["rerank_per_query_usd"]
        breakdown = {"queries": usage.queries}

    elif comp == "vector_db":
        usd = usage.queries * (
            pb["vector_db_fixed_share_usd"] + pb["vector_db_per_query_usd"]
        )
        breakdown = {"queries": usage.queries}

    elif comp == "tool_call":
        usd = usage.tool_calls * pb["tool_call_usd"]
        breakdown = {"tool_calls": usage.tool_calls}

    elif comp == "guardrail":
        usd = usage.guardrail_calls * pb["guardrail_call_usd"]
        breakdown = {"guardrail_calls": usage.guardrail_calls}

    elif comp == "human_review":
        usd = usage.human_review_minutes * pb["human_review_per_min_usd"]
        breakdown = {"human_review_minutes": usage.human_review_minutes}

    elif comp == "shared_overhead":
        usd = pb["shared_overhead_usd"]
        breakdown = {}

    usd *= tier_mult
    breakdown["tier_multiplier"] = tier_mult
    inr = usd * usd_to_inr

    return CostResult(usd=round(usd, 8), inr=round(inr, 6), breakdown=breakdown)
