"""
Unit tests for the collector (Telemetry Spec acceptance criteria):
  - "Unit test validates the attribute set."
  - "Cost calculation can be reproduced from telemetry + price-book version."
Run: python -m pytest -q   (or) python test_collector.py
"""
from collector import CallUsage, compute_cost, COST_COMPONENTS
from price_book import get_price_book, DEFAULT_PRICE_BOOK_VERSION

PB = DEFAULT_PRICE_BOOK_VERSION
FX = 83.20


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_model_tokens_no_cache():
    u = CallUsage(cost_component="model_tokens", price_book_version=PB,
                  model="anthropic.claude-3-haiku-20240307-v1:0",
                  input_tokens=420, output_tokens=180)
    c = compute_cost(u, FX)
    # haiku: in 0.25/Mtok, out 1.25/Mtok
    expected = 420/1e6*0.25 + 180/1e6*1.25
    assert approx(c.usd, round(expected, 8)), (c.usd, expected)
    assert approx(c.inr, round(c.usd * FX, 6))
    assert c.breakdown["billable_input_tokens"] == 420


def test_cache_tokens_not_double_counted():
    # input includes cache portions; billable input must exclude them
    u = CallUsage(cost_component="model_tokens", price_book_version=PB,
                  model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                  input_tokens=1850, output_tokens=320,
                  cache_read_input_tokens=512)
    c = compute_cost(u, FX)
    r = get_price_book(PB)["models"]["anthropic.claude-3-5-sonnet-20241022-v2:0"]
    billable_in = 1850 - 512
    expected = (billable_in/1e6*r["input_per_mtok"]
                + 320/1e6*r["output_per_mtok"]
                + 512/1e6*r["cache_read_per_mtok"])
    assert approx(c.usd, round(expected, 8)), (c.usd, expected)
    assert c.breakdown["billable_input_tokens"] == billable_in


def test_reasoning_not_added_twice():
    # reasoning tokens are a subset of output; cost must match output-only calc
    base = CallUsage(cost_component="model_tokens", price_book_version=PB,
                     model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                     input_tokens=1000, output_tokens=500)
    with_reasoning = CallUsage(cost_component="model_tokens", price_book_version=PB,
                               model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                               input_tokens=1000, output_tokens=500,
                               reasoning_output_tokens=150)
    assert approx(compute_cost(base, FX).usd, compute_cost(with_reasoning, FX).usd)


def test_pricing_tier_multiplier():
    std = CallUsage(cost_component="model_tokens", price_book_version=PB,
                    model="anthropic.claude-3-haiku-20240307-v1:0",
                    input_tokens=1000, output_tokens=1000, pricing_tier="standard")
    pri = CallUsage(cost_component="model_tokens", price_book_version=PB,
                    model="anthropic.claude-3-haiku-20240307-v1:0",
                    input_tokens=1000, output_tokens=1000, pricing_tier="priority")
    assert approx(compute_cost(pri, FX).usd, round(compute_cost(std, FX).usd * 1.25, 8))


def test_non_token_components():
    for comp, kwargs, qty_field in [
        ("rerank", {"queries": 3}, "queries"),
        ("vector_db", {"queries": 3}, "queries"),
        ("tool_call", {"tool_calls": 2}, "tool_calls"),
        ("guardrail", {"guardrail_calls": 2}, "guardrail_calls"),
        ("human_review", {"human_review_minutes": 2.0}, "human_review_minutes"),
        ("shared_overhead", {}, None),
    ]:
        u = CallUsage(cost_component=comp, price_book_version=PB, **kwargs)
        c = compute_cost(u, FX)
        assert c.usd >= 0
        assert approx(c.inr, round(c.usd * FX, 6))


def test_all_components_known():
    assert "model_tokens" in COST_COMPONENTS and "human_review" in COST_COMPONENTS


def test_reproducible_from_version():
    # Same telemetry + same price-book version -> identical cost (determinism).
    u = CallUsage(cost_component="model_tokens", price_book_version=PB,
                  model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                  input_tokens=1850, output_tokens=320, cache_read_input_tokens=512)
    assert compute_cost(u, FX).usd == compute_cost(u, FX).usd


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
