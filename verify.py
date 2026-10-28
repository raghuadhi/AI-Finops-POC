"""
Verify demo telemetry in Langfuse v4 (events_only mode).

v4 stores spans/generations/scores in the unified events tables
(events_core / events_full), not the legacy traces/observations tables.
Custom attributes live in parallel arrays metadata_names / metadata_values.

Confirms Telemetry Spec acceptance criteria:
  - required attribute set present on a Bedrock-style call
  - a task can be reconstructed from its task.id
  - cost-per-accepted-task reproducible from stored telemetry + price book
"""
import io
import sys
import urllib.request

# force UTF-8 so Vertical output with unicode never crashes on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CH_USER = "clickhouse"
CH_PASS = "gLB1C9a8mdq1YzP2AnkZbVBFjREbEMLi"
T = "events_full"


def ch(sql: str) -> str:
    req = urllib.request.Request(
        "http://127.0.0.1:8123/?database=default",
        data=sql.encode(),
        headers={"X-ClickHouse-User": CH_USER, "X-ClickHouse-Key": CH_PASS},
    )
    return urllib.request.urlopen(req, timeout=30).read().decode()


# helper: pull a custom attribute out of the parallel metadata arrays
def md(attr: str) -> str:
    return (f"metadata_values[indexOf(metadata_names, '{attr}')]")


def main():
    print("events_full rows :", ch(f"SELECT count() FROM {T}").strip())

    print("\n--- events by type ---")
    print(ch(f"SELECT type, count() FROM {T} GROUP BY type ORDER BY type "
             "FORMAT TabSeparated"))

    print("--- distinct task types seen (from task.type attribute) ---")
    print(ch(f"SELECT DISTINCT {md('task.type')} AS task_type FROM {T} "
             f"WHERE has(metadata_names,'task.type') ORDER BY task_type "
             "FORMAT TabSeparated"))

    print("--- one RAG_QA task reconstructed from its task.id ---")
    print(ch(
        f"SELECT {md('task.id')} AS task_id, {md('task.type')} AS task_type, "
        f"{md('task.attempt')} AS attempt, {md('tenant.id')} AS tenant, "
        f"{md('cost.price_book_version')} AS price_book, "
        f"{md('cost.usd_computed')} AS usd, {md('cost.inr_computed')} AS inr, "
        f"session_id, user_id, name, type "
        f"FROM {T} WHERE {md('task.type')}='RAG_QA' AND type='SPAN' "
        f"AND {md('task.id')} != '' LIMIT 1 FORMAT Vertical"
    ))

    print("--- required spec attributes on a model call (gen llm.answer) ---")
    attrs = [
        "gen_ai.operation.name", "gen_ai.provider.name", "gen_ai.request.model",
        "gen_ai.response.model", "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens", "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.cache_write.input_tokens", "cost.component",
        "cost.price_book_version", "cost.usd_computed", "cost.inr_computed",
        "aws.bedrock.inference_profile_arn", "pricing.tier",
        "tenant.id", "task.id", "task.type", "task.attempt",
    ]
    sel = ", ".join(f"{md(a)} AS \"{a}\"" for a in attrs)
    print(ch(
        f"SELECT name, provided_model_name AS model, {sel} "
        f"FROM {T} WHERE type='GENERATION' AND name='llm.answer' "
        f"LIMIT 1 FORMAT Vertical"
    ))

    print("--- native Langfuse cost on that generation (cost_details) ---")
    print(ch(
        f"SELECT name, provided_model_name, "
        f"usage_details, provided_cost_details "
        f"FROM {T} WHERE type='GENERATION' AND name='llm.answer' "
        f"LIMIT 1 FORMAT Vertical"
    ))

    print("--- scores (acceptance + cost) ---")
    print(ch("SELECT name, count(), round(avg(value),4) AS avg_value "
             "FROM scores GROUP BY name FORMAT TabSeparated"))

    print("--- cost per accepted task (reproduced from stored scores) ---")
    print(ch(
        "WITH acc AS (SELECT trace_id, max(value) AS accepted FROM scores "
        "WHERE name='task.accepted' GROUP BY trace_id), "
        "cost AS (SELECT trace_id, max(value) AS usd FROM scores "
        "WHERE name='cost.usd_computed' GROUP BY trace_id) "
        "SELECT round(sum(cost.usd),6) AS total_usd, "
        "toInt32(sum(acc.accepted)) AS accepted_tasks, "
        "count() AS total_tasks, "
        "round(sum(cost.usd)/nullIf(sum(acc.accepted),0),6) AS cost_per_accepted_usd, "
        "round(sum(acc.accepted)/count(),3) AS acceptance_rate "
        "FROM acc INNER JOIN cost USING (trace_id) FORMAT Vertical"
    ))


def reconstruct_session(session_id: str):
    print(f"--- tasks in {session_id} (root task spans) ---")
    print(ch(
        f"SELECT {md('task.id')} AS task_id, trace_name, "
        f"{md('task.accepted')} AS accepted, {md('task.accepted_by')} AS by, "
        f"{md('cost.usd_computed')} AS usd, {md('cost.inr_computed')} AS inr "
        f"FROM {T} WHERE session_id='{session_id}' AND is_app_root=1 "
        f"ORDER BY start_time FORMAT TabSeparated"
    ))
    print(f"--- session rollup for {session_id} ---")
    print(ch(
        f"SELECT count() AS tasks, "
        f"countIf({md('task.accepted')}='True') AS accepted, "
        f"round(sum(toFloat64OrZero({md('cost.usd_computed')})),6) AS total_usd, "
        f"round(sum(toFloat64OrZero({md('cost.usd_computed')}))/"
        f"nullIf(countIf({md('task.accepted')}='True'),0),6) AS cost_per_accepted_usd "
        f"FROM {T} WHERE session_id='{session_id}' AND is_app_root=1 FORMAT Vertical"
    ))


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            reconstruct_session(sys.argv[1])
            sys.exit(0)
        main()
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:400])
        sys.exit(1)
