"""
AI FinOps live demo -> local Langfuse (project: AI FinOps POC).

Emits the accounting hierarchy from the Task Taxonomy + Telemetry Spec:

    session / conversation
      └─ task        (unit of accounting; accepted or rejected)
           └─ step   (one agent-loop iteration)
                └─ call   (one model / tool / retrieval / rerank / guardrail call)

Each task becomes one Langfuse trace (joinable by task.id). Steps are spans,
calls are generations / typed observations. Cost is computed in the COLLECTOR
from the versioned price book (never hard-coded in the task flow) and attached
as cost_details (USD) plus metadata (INR + breakdown). Acceptance is recorded
as a late-arriving score joined back by task.id.

Required attributes (Telemetry Spec section 4) are stamped on every call:
  gen_ai.operation.name, gen_ai.provider.name, gen_ai.request.model,
  gen_ai.response.model, gen_ai.usage.*, error.type,
  tenant.id, task.id, task.type, task.attempt, task.parent_id,
  task.accepted, task.accepted_by, cost.component, cost.price_book_version,
  cost.usd_computed, cost.inr_computed, aws.bedrock.inference_profile_arn,
  pricing.tier
"""

from __future__ import annotations

import os
import uuid
import random
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langfuse import Langfuse

from price_book import DEFAULT_PRICE_BOOK_VERSION
from collector import CallUsage, compute_cost

load_dotenv()

USD_TO_INR = float(os.getenv("USD_TO_INR", "83.20"))
PRICE_BOOK_VERSION = DEFAULT_PRICE_BOOK_VERSION
TENANT_ID = "sandbox"
SEM_CONV_VERSION = "1.27.0"  # pinned OTel GenAI semantic-convention version

# OTel semantic-convention version is pinned per Telemetry Spec design rule 5.
SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"
HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
TITAN_EMBED = "amazon.titan-embed-text-v2:0"
BEDROCK_ARN = (
    "arn:aws:bedrock:ap-south-1:000000000000:"
    "inference-profile/apac.anthropic.claude-3-5-sonnet"
)

random.seed(7)  # deterministic demo

lf = Langfuse()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class TaskSpec:
    task_type: str
    accepted: bool
    accepted_by: str | None        # user | judge | policy | None(if rejected)
    pricing_tier: str = "standard"
    attempt: int = 1
    parent_id: str | None = None


def base_task_attrs(task_id: str, spec: TaskSpec) -> dict:
    """Custom task-level attributes shared by every call in the task."""
    return {
        "tenant.id": TENANT_ID,
        "task.id": task_id,
        "task.type": spec.task_type,
        "task.attempt": spec.attempt,
        "task.parent_id": spec.parent_id,
        "pricing.tier": spec.pricing_tier,
        "cost.price_book_version": PRICE_BOOK_VERSION,
        "otel.semconv.version": SEM_CONV_VERSION,
    }


def call_metadata(task_attrs: dict, usage: CallUsage, cost, extra: dict | None = None) -> dict:
    """Full attribute set stamped on a single call span (spec section 4)."""
    md = dict(task_attrs)
    md.update(
        {
            "cost.component": usage.cost_component,
            "cost.usd_computed": cost.usd,
            "cost.inr_computed": cost.inr,
            "cost.breakdown": cost.breakdown,
        }
    )
    if usage.model:
        md["gen_ai.request.model"] = usage.model
        md["gen_ai.response.model"] = usage.model
        md["gen_ai.provider.name"] = "aws.bedrock"
        md["aws.bedrock.inference_profile_arn"] = BEDROCK_ARN
        md["gen_ai.usage.input_tokens"] = usage.input_tokens
        md["gen_ai.usage.output_tokens"] = usage.output_tokens
        md["gen_ai.usage.cache_read.input_tokens"] = usage.cache_read_input_tokens
        md["gen_ai.usage.cache_write.input_tokens"] = usage.cache_write_input_tokens
        if usage.reasoning_output_tokens:
            md["gen_ai.usage.reasoning.output_tokens"] = usage.reasoning_output_tokens
    if extra:
        md.update(extra)
    return md


def emit_model_call(
    *, name, operation, model, task_attrs, input_tokens, output_tokens,
    cache_read=0, cache_write=0, reasoning=0, pricing_tier="standard",
    component="model_tokens", error_type=None,
):
    """Emit one model/embedding call as a Langfuse generation with full attrs."""
    usage = CallUsage(
        cost_component=component,
        price_book_version=PRICE_BOOK_VERSION,
        pricing_tier=pricing_tier,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        reasoning_output_tokens=reasoning,
    )
    cost = compute_cost(usage, USD_TO_INR)

    extra = {"gen_ai.operation.name": operation}
    if error_type:
        extra["error.type"] = error_type

    with lf.start_as_current_observation(as_type='generation', name=name,
    model=model,
    usage_details={
        "input": input_tokens,
        "output": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_write_input_tokens": cache_write,
    },
    # Native Langfuse cost (USD) comes from our collector, not model match.
    cost_details={"total": cost.usd},
    metadata=call_metadata(task_attrs, usage, cost, extra),
    level="ERROR" if error_type else "DEFAULT",
    status_message=error_type,) as gen:
        gen.update(output={"tokens_out": output_tokens, "error": error_type})
    return cost


def emit_typed_call(
    *, name, as_type, operation, component, task_attrs,
    queries=0, tool_calls=0, guardrail_calls=0, human_review_minutes=0.0,
    pricing_tier="standard",
):
    """Emit a non-model call (retriever/rerank/tool/guardrail/human review)."""
    usage = CallUsage(
        cost_component=component,
        price_book_version=PRICE_BOOK_VERSION,
        pricing_tier=pricing_tier,
        queries=queries,
        tool_calls=tool_calls,
        guardrail_calls=guardrail_calls,
        human_review_minutes=human_review_minutes,
    )
    cost = compute_cost(usage, USD_TO_INR)
    extra = {"gen_ai.operation.name": operation}
    obs = lf.start_observation(
        name=name,
        as_type=as_type,
        metadata=call_metadata(task_attrs, usage, cost, extra),
    )
    obs.update(output={"component": component, "usd": cost.usd})
    obs.end()
    return cost


# ---------------------------------------------------------------------------
# Task runners (one trace per task, joinable by task.id)
# ---------------------------------------------------------------------------
def run_task(session_id: str, user_id: str, spec: TaskSpec) -> dict:
    task_id = f"TASK-{uuid.uuid4().hex[:8].upper()}"
    task_attrs = base_task_attrs(task_id, spec)
    total_usd = 0.0
    total_inr = 0.0

    # Deterministic trace id derived from task_id -> lets 'accepted' be a
    # late-arriving score joined by task.id (spec design rule 4).
    trace_id = lf.create_trace_id(seed=task_id)

    with lf.start_as_current_span(
        name=f"{spec.task_type}:{task_id}",
        trace_context={"trace_id": trace_id},
        metadata=task_attrs,
        input={"task_type": spec.task_type, "attempt": spec.attempt},
    ) as task_span:
        lf.update_current_trace(
            name=f"{spec.task_type}",
            session_id=session_id,
            user_id=user_id,
            tags=[spec.task_type, f"tenant:{TENANT_ID}", f"tier:{spec.pricing_tier}"],
            metadata=task_attrs,
        )

        costs: list = []

        if spec.task_type == "SIMPLE_QA":
            with lf.start_as_current_span(name="step-1", metadata=task_attrs):
                costs.append(emit_model_call(
                    name="llm.answer", operation="chat", model=HAIKU,
                    task_attrs=task_attrs, input_tokens=420, output_tokens=180,
                    pricing_tier=spec.pricing_tier))

        elif spec.task_type == "RAG_QA":
            with lf.start_as_current_span(name="step-1-retrieve", metadata=task_attrs):
                # embedding call priced as model tokens on the embed model:
                costs.append(emit_model_call(
                    name="embed.query", operation="embeddings", model=TITAN_EMBED,
                    task_attrs=task_attrs, input_tokens=64, output_tokens=0,
                    component="embedding", pricing_tier=spec.pricing_tier))
                costs.append(emit_typed_call(
                    name="vector.search", as_type="retriever", operation="retrieve",
                    component="vector_db", task_attrs=task_attrs, queries=1,
                    pricing_tier=spec.pricing_tier))
                costs.append(emit_typed_call(
                    name="rerank", as_type="retriever", operation="rerank",
                    component="rerank", task_attrs=task_attrs, queries=1,
                    pricing_tier=spec.pricing_tier))
            with lf.start_as_current_span(name="step-2-generate", metadata=task_attrs):
                costs.append(emit_model_call(
                    name="llm.answer", operation="chat", model=SONNET,
                    task_attrs=task_attrs, input_tokens=1850, output_tokens=320,
                    cache_read=512, pricing_tier=spec.pricing_tier))

        elif spec.task_type == "AGENT_TOOL_TASK":
            # multi-step agent loop; tool defs re-enter context each step
            for step in range(1, 4):
                with lf.start_as_current_span(name=f"step-{step}", metadata=task_attrs):
                    # cost grows with loop length (bigger input each iteration)
                    costs.append(emit_model_call(
                        name="llm.plan", operation="chat", model=SONNET,
                        task_attrs=task_attrs,
                        input_tokens=1200 + step * 600, output_tokens=240,
                        cache_read=256 * step, reasoning=90,
                        pricing_tier=spec.pricing_tier))
                    costs.append(emit_typed_call(
                        name="tool.call", as_type="tool", operation="execute_tool",
                        component="tool_call", task_attrs=task_attrs, tool_calls=1,
                        pricing_tier=spec.pricing_tier))
            with lf.start_as_current_span(name="step-guardrail", metadata=task_attrs):
                costs.append(emit_typed_call(
                    name="guardrail.check", as_type="guardrail", operation="guardrail",
                    component="guardrail", task_attrs=task_attrs, guardrail_calls=1,
                    pricing_tier=spec.pricing_tier))

        elif spec.task_type == "CODE_GENERATION":
            with lf.start_as_current_span(name="step-1-generate", metadata=task_attrs):
                costs.append(emit_model_call(
                    name="llm.codegen", operation="chat", model=SONNET,
                    task_attrs=task_attrs, input_tokens=2200, output_tokens=640,
                    reasoning=150, pricing_tier=spec.pricing_tier))
            with lf.start_as_current_span(name="step-2-test", metadata=task_attrs):
                costs.append(emit_typed_call(
                    name="ci.run_tests", as_type="tool", operation="execute_tool",
                    component="tool_call", task_attrs=task_attrs, tool_calls=1,
                    pricing_tier=spec.pricing_tier))

        elif spec.task_type == "EVALUATION_JUDGE":
            # overhead task: cost attributed to parent via task.parent_id
            with lf.start_as_current_span(name="step-1-judge", metadata=task_attrs):
                costs.append(emit_model_call(
                    name="llm.judge", operation="chat", model=HAIKU,
                    task_attrs=task_attrs, input_tokens=900, output_tokens=60,
                    pricing_tier=spec.pricing_tier))

        # shared overhead on every task
        costs.append(emit_typed_call(
            name="overhead", as_type="span", operation="overhead",
            component="shared_overhead", task_attrs=task_attrs,
            pricing_tier=spec.pricing_tier))

        # human review minutes (only for user-accepted tasks here)
        if spec.accepted_by == "user":
            costs.append(emit_typed_call(
                name="human.review", as_type="span", operation="human_review",
                component="human_review", task_attrs=task_attrs,
                human_review_minutes=0.5, pricing_tier=spec.pricing_tier))

        total_usd = round(sum(c.usd for c in costs), 8)
        total_inr = round(sum(c.inr for c in costs), 6)

        task_span.update(
            output={
                "task.id": task_id,
                "task.accepted": spec.accepted,
                "task.accepted_by": spec.accepted_by,
                "cost.usd_computed": total_usd,
                "cost.inr_computed": total_inr,
            },
            metadata={
                **task_attrs,
                "task.accepted": spec.accepted,
                "task.accepted_by": spec.accepted_by,
                "cost.usd_computed": total_usd,
                "cost.inr_computed": total_inr,
            },
        )

    # Late-arriving acceptance event, joined by task.id via trace_id.
    lf.create_score(
        name="task.accepted",
        value=1 if spec.accepted else 0,
        data_type="BOOLEAN",
        trace_id=trace_id,
        comment=f"accepted_by={spec.accepted_by}",
        metadata={"task.id": task_id, "accepted_by": spec.accepted_by,
                  "task.type": spec.task_type},
    )
    # Also record the per-task cost as a numeric score for easy charting.
    lf.create_score(name="cost.usd_computed", value=total_usd,
                     data_type="NUMERIC", trace_id=trace_id,
                     metadata={"task.id": task_id})

    return {
        "task_id": task_id, "trace_id": trace_id, "type": spec.task_type,
        "accepted": spec.accepted, "accepted_by": spec.accepted_by,
        "usd": total_usd, "inr": total_inr,
    }


def main():
    print("Auth check:", lf.auth_check())

    session_id = f"session-{uuid.uuid4().hex[:8]}"
    user_id = "demo-user@codincity"

    # A representative mix of task types, acceptances, retries and tiers.
    specs = [
        TaskSpec("SIMPLE_QA", accepted=True, accepted_by="policy"),
        TaskSpec("RAG_QA", accepted=True, accepted_by="policy", pricing_tier="standard"),
        TaskSpec("RAG_QA", accepted=False, accepted_by=None, attempt=1),          # failed attempt (stays in numerator)
        TaskSpec("RAG_QA", accepted=True, accepted_by="user", attempt=2),         # retry accepted
        TaskSpec("AGENT_TOOL_TASK", accepted=True, accepted_by="user", pricing_tier="priority"),
        TaskSpec("CODE_GENERATION", accepted=True, accepted_by="policy"),
        TaskSpec("CODE_GENERATION", accepted=False, accepted_by=None),            # tests failed
    ]

    results = []
    for spec in specs:
        r = run_task(session_id, user_id, spec)
        results.append(r)
        print(f"  {r['type']:<16} {r['task_id']}  accepted={r['accepted']!s:<5} "
              f"by={str(r['accepted_by']):<6} USD={r['usd']:.6f} INR={r['inr']:.4f}")

    # EVALUATION_JUDGE as an overhead task attributed to the first code task.
    parent = next(r for r in results if r["type"] == "CODE_GENERATION")
    judge_spec = TaskSpec("EVALUATION_JUDGE", accepted=True, accepted_by="judge",
                          parent_id=parent["task_id"])
    jr = run_task(session_id, user_id, judge_spec)
    results.append(jr)
    print(f"  {jr['type']:<16} {jr['task_id']}  parent={parent['task_id']} "
          f"USD={jr['usd']:.6f} INR={jr['inr']:.4f}")

    # Session-level summary: cost per accepted task + acceptance rate.
    total_usd = round(sum(r["usd"] for r in results), 6)
    total_inr = round(sum(r["inr"] for r in results), 4)
    accepted = [r for r in results if r["accepted"]]
    n_accepted = len(accepted)
    cpat_usd = round(total_usd / n_accepted, 6) if n_accepted else float("nan")
    cpat_inr = round(total_inr / n_accepted, 4) if n_accepted else float("nan")
    acceptance_rate = round(n_accepted / len(results), 3)

    print("\n--- Session summary (cost per accepted task) ---")
    print(f"  session_id            : {session_id}")
    print(f"  tasks (attempts)      : {len(results)}")
    print(f"  accepted tasks        : {n_accepted}")
    print(f"  acceptance rate       : {acceptance_rate}")
    print(f"  total cost (all attempts) USD: {total_usd}  INR: {total_inr}")
    print(f"  COST PER ACCEPTED TASK    USD: {cpat_usd}  INR: {cpat_inr}")

    lf.flush()
    print("\nFlushed to Langfuse. Session:", session_id)
    return {"session_id": session_id, "results": results,
            "cpat_usd": cpat_usd, "acceptance_rate": acceptance_rate}


if __name__ == "__main__":
    main()
