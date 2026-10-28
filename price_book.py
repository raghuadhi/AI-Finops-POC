"""
Versioned price book for AI FinOps cost computation.

Per the Telemetry Spec (section 6, design rule 2 & 3):
  - Never hard-code prices inside the application.
  - Compute cost in the collector/downstream job from the versioned price book.

Prices are keyed by price_book_version (effective date). The collector selects
the price book whose version matches cost.price_book_version recorded on the call.

All model prices are USD per 1,000,000 tokens unless noted.
These are illustrative demo figures, not official vendor prices.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Price books, keyed by version (effective date string, as per the spec's
# cost.price_book_version attribute, e.g. "2026-10-01").
# ---------------------------------------------------------------------------
PRICE_BOOKS: dict[str, dict] = {
    "2026-10-01": {
        "currency": "USD",
        # gen_ai.request.model -> per-million-token rates
        "models": {
            # Bedrock-style model ids
            "anthropic.claude-3-5-sonnet-20241022-v2:0": {
                "input_per_mtok": 3.00,
                "output_per_mtok": 15.00,
                "cache_read_per_mtok": 0.30,
                "cache_write_per_mtok": 3.75,
            },
            "anthropic.claude-3-haiku-20240307-v1:0": {
                "input_per_mtok": 0.25,
                "output_per_mtok": 1.25,
                "cache_read_per_mtok": 0.03,
                "cache_write_per_mtok": 0.30,
            },
            "amazon.titan-embed-text-v2:0": {
                "input_per_mtok": 0.02,
                "output_per_mtok": 0.0,
                "cache_read_per_mtok": 0.0,
                "cache_write_per_mtok": 0.0,
            },
        },
        # Non-token priced components
        "rerank_per_query_usd": 0.0020,          # reranker call
        "vector_db_fixed_share_usd": 0.0005,     # amortised fixed share per query
        "vector_db_per_query_usd": 0.0001,       # marginal per-query
        "tool_call_usd": 0.0010,                 # tool / API call
        "guardrail_call_usd": 0.0008,            # guardrail call
        "human_review_per_min_usd": 1.50,        # loaded reviewer rate per minute
        "shared_overhead_usd": 0.0003,           # shared overhead per task
        # pricing tier multipliers (pricing.tier attribute)
        "tier_multipliers": {
            "standard": 1.0,
            "priority": 1.25,
            "batch": 0.5,
        },
    }
}

DEFAULT_PRICE_BOOK_VERSION = "2026-10-01"


def get_price_book(version: str) -> dict:
    if version not in PRICE_BOOKS:
        raise KeyError(
            f"price_book_version '{version}' not found. "
            f"Available: {sorted(PRICE_BOOKS)}"
        )
    return PRICE_BOOKS[version]
