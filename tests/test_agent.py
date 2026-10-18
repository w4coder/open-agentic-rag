import pytest

from agentic_rag.tools import UnsafeSQL, validate_sql


def test_simple_docs_question_is_routed_and_cited(svc):
    out = svc.ask("What does error NB-4021 mean?")
    assert out["status"] == "ok" and out["citations"]
    assert "expired" in out["answer"]
    assert any(t.startswith("research:retrieve>grade") for t in out["trace"])


def test_multi_hop_question_is_decomposed_in_parallel(svc):
    out = svc.ask("What is the Business plan rate limit and what is the Enterprise availability SLA?")
    assert len(out["plan"]) == 2
    assert sum(t.startswith("research") for t in out["trace"]) == 2
    assert "3000" in out["answer"] and "99.99" in out["answer"]
    assert out["status"] == "ok"


def test_unanswerable_triggers_corrective_loop_then_refuses(svc):
    out = svc.ask("What is the CEO's favourite colour?")
    path = next(t for t in out["trace"] if t.startswith("research"))
    assert "rewrite" in path and "web_fallback" in path  # CRAG: rewrite, then fallback
    assert out["trace"].count("replan") == svc.settings.max_loops  # bounded reflection
    assert out["status"] in {"incomplete", "ungrounded"} and not out["citations"]


def test_web_route(svc):
    out = svc.ask("What is the latest Nimbus incident status?")
    assert out["status"] == "ok" and out["citations"][0]["kind"] == "web"


def test_sql_requires_human_approval_then_runs(svc):
    out = svc.ask("What was the total revenue in Q1 by region?")
    assert out["status"] == "awaiting_approval"
    assert out["request"]["sql"].lower().startswith("select")
    done = svc.resume(out["thread_id"], approved=True)
    assert done["status"] == "ok"
    assert "region=EMEA, revenue=200000" in done["answer"]  # Q1 EMEA = 120k + 80k
    assert "region=AMER, revenue=350000" in done["answer"]  # Q1 AMER = 200k + 150k


def test_sql_rejected_by_human(svc):
    out = svc.ask("What was the total revenue in Q2?")
    done = svc.resume(out["thread_id"], approved=False)
    assert done["status"] == "rejected" and not done["citations"]


def test_sql_without_approval_gate(svc_auto):
    out = svc_auto.ask("Total sales in Q1 for EMEA")
    assert out["status"] == "ok" and "200000" in out["answer"]


def test_memory_persists_across_threads(svc):
    svc.ask("Remember that I prefer answers in French", user_id="u1")
    items = svc.graph.store.search(("memory", svc.settings.tenant, "u1"))
    assert items and "French" in items[0].value["text"]


def test_chitchat_skips_retrieval(svc):
    out = svc.ask("Hello!")
    assert out["status"] == "direct" and not any(t.startswith("research") for t in out["trace"])


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales", "SELECT * FROM sales; DELETE FROM sales", "SELECT * FROM users",
    "UPDATE sales SET amount=0", "SELECT * FROM sales WHERE 1=1 UNION SELECT * FROM secrets",
])
def test_sql_guard_blocks_unsafe(sql):
    with pytest.raises(UnsafeSQL):
        validate_sql(sql)
