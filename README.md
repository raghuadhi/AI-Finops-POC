# AI FinOps — Cost-per-Accepted-Task Langfuse Demo

A live demo that emits AI FinOps telemetry to a local Langfuse instance
(project **AI FinOps POC**, org **Codincity**), implementing the
**Task Taxonomy v1** and **Telemetry Specification v1** contracts.

## What it does

Emits the accounting hierarchy from the spec:

```
session / conversation
  └─ task        (unit of accounting; accepted or rejected)  -> one Langfuse trace
       └─ step   (one agent-loop iteration)                  -> span
            └─ call  (model / tool / retrieval / rerank / guardrail)  -> generation / typed obs
```

Covered task types: `SIMPLE_QA`, `RAG_QA`, `AGENT_TOOL_TASK`,
`CODE_GENERATION`, `EVALUATION_JUDGE` (overhead attributed to parent via
`task.parent_id`). Includes failed/retried attempts that remain in the
cost numerator, plus `standard`/`priority` pricing tiers.

## Files

| File | Purpose |
|------|---------|
| `price_book.py`  | Versioned price book (keyed by `cost.price_book_version`). Prices are **never** hard-coded in the app. |
| `collector.py`   | Collector that derives `cost.usd_computed` / `cost.inr_computed` from telemetry + price book. Implements token rules (no double-counting cache tokens; reasoning is an output subset). |
| `demo.py`        | Main demo. Builds session->task->step->call and stamps every required spec attribute; records `accepted` as a late-arriving score joined by `task.id`. |
| `test_collector.py` | Unit tests validating cost math + attribute set (spec acceptance criterion). |
| `verify.py`      | Verifies ingestion by querying Langfuse v4 `events_full` in ClickHouse; reconstructs a task from its id and reproduces cost-per-accepted-task. |

## Telemetry Spec attribute coverage (section 4)

Required OTel GenAI: `gen_ai.operation.name`, `gen_ai.provider.name`,
`gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`,
`gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`,
`gen_ai.usage.cache_write.input_tokens`, `gen_ai.usage.reasoning.output_tokens`,
`error.type`.

Custom: `tenant.id`, `task.id`, `task.type`, `task.attempt`, `task.parent_id`,
`task.accepted`, `task.accepted_by`, `cost.component`,
`cost.price_book_version`, `cost.usd_computed`, `cost.inr_computed`,
`aws.bedrock.inference_profile_arn`, `pricing.tier`.

## Run

```powershell
cd $env:USERPROFILE\langfuse-finops-demo
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # first time
.\.venv\Scripts\python.exe test_collector.py                    # unit tests
.\.venv\Scripts\python.exe demo.py                              # emit traces
.\.venv\Scripts\python.exe verify.py                            # verify in DB
```

Then open **http://localhost:3000** -> project **AI FinOps POC** -> Tracing /
Sessions to see the traces, and Scores for `task.accepted` + `cost.usd_computed`.

## Note on this Langfuse deployment

This local Langfuse runs in **v4 events_only mode**: the public read API
(`/api/public/traces`, `sessions`, `scores`, `metrics`) is disabled, and
traces/observations are stored in the unified `events_core` / `events_full`
ClickHouse tables (not the legacy `traces`/`observations` tables). `verify.py`
queries those tables directly. Data is fully visible in the Langfuse UI.

The `.env` holds the project API keys and the USD->INR FX rate. Do not commit it.
