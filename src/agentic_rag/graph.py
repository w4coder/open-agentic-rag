"""Main agent graph: adaptive routing + planning + parallel research + self-reflection + HITL.

 START -> load_memory -> route --direct--> direct_answer -> END
                           |--memory--> remember -> END
                           |--sql-----> propose_sql -> approve_sql (interrupt) -> run_sql -+
                           |--docs/web-> plan ==Send x N==> research (CRAG subgraph) ======+
                                          ^                                                |
                                          |                                                v
                                          +-- replan (bounded) <-- verify <---------- synthesize
                                                                     |--ok / out of budget--> finalize -> END

Patterns (see docs/patterns.md): Adaptive RAG (router), query decomposition (plan + Send =
map-reduce), Corrective RAG (subgraph), Self-RAG-style reflection (verify -> replan),
human-in-the-loop (interrupt), short + long term memory (checkpointer + store), tool safety.

Thread semantics: a thread = a conversation. Each `ask` is a new *run* on that thread and
must start from a clean run state - the service sends `RUN_RESET` (every run-scoped key reset,
accumulating lists cleared through the `RESET` marker of their reducer). Without it, a second
question on the same thread inherits the previous run's status / evidence / missing parts.
"""
from __future__ import annotations

import hashlib
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Send, interrupt
from open_rag.generation.guardrails import GuardrailError, check_input, redact_pii

from agentic_rag.brain import Brain, Synthesis, Verdict
from agentic_rag.config import AgentSettings
from agentic_rag.crag import build_crag
from agentic_rag.tools import Tools, UnsafeSQL

REFUSAL = "I don't know based on the available sources."
RESET = "__reset__"


def extend_or_reset(left: list | None, right: list | None) -> list:
    """List reducer: parallel branches append; a list starting with RESET replaces."""
    right = right or []
    if right and right[0] == RESET:
        return list(right[1:])
    return (left or []) + right


class AgentState(TypedDict, total=False):
    question: str
    user_id: str
    tenant: str
    route: str
    plan: list[str]
    # Reducer: parallel `research` branches each *append* to the shared evidence list.
    evidence: Annotated[list[dict], extend_or_reset]
    trace: Annotated[list[str], extend_or_reset]  # nodes / crag paths, for trajectory evals
    memory: list[str]
    pending_sql: str
    synthesis: Synthesis | None
    verdict: Verdict | None
    missing: list[str]
    loops: int
    answer: str
    citations: list[dict]
    status: str  # ok | incomplete | ungrounded | rejected | blocked | direct | remembered


RUN_RESET: AgentState = {
    "route": "", "plan": [], "evidence": [RESET], "trace": [RESET], "memory": [],
    "pending_sql": "", "synthesis": None, "verdict": None, "missing": [], "loops": 0,
    "answer": "", "citations": [], "status": "",
}


def build_agent(settings: AgentSettings, brain: Brain, tools: Tools, checkpointer=None, store=None):
    crag = build_crag(brain, tools, settings)

    def _ns(state: AgentState) -> tuple[str, ...]:
        return ("memory", state.get("tenant", "default"), state.get("user_id", "anon"))

    # ------------------------------------------------------------------ memory
    def load_memory(state: AgentState, *, store: BaseStore) -> AgentState:
        items = store.search(_ns(state), limit=settings.memory_max_items)
        return {"memory": [i.value["text"] for i in items], "trace": ["load_memory"]}

    def remember(state: AgentState, *, store: BaseStore) -> AgentState:
        """Long-term memory is re-injected into future prompts: treat writes as untrusted
        input (injection screen, size cap, PII redaction) and bound the number of items."""
        text = state["question"].split(" ", 1)[-1].strip()
        try:
            text = check_input(text, settings.memory_max_chars)
        except GuardrailError as e:
            return {"answer": f"I can't store that ({e.code}).", "status": "blocked",
                    "trace": ["remember:blocked"]}
        text = redact_pii(text)
        ns = _ns(state)
        items = sorted(store.search(ns, limit=1000), key=lambda i: i.created_at)
        for old in items[: max(0, len(items) - settings.memory_max_items + 1)]:
            store.delete(ns, old.key)  # evict oldest beyond the cap
        key = hashlib.sha1(text.encode()).hexdigest()[:12]  # hash() is salted per process
        store.put(ns, key, {"text": text})
        return {"answer": "Noted, I will remember that.", "status": "remembered",
                "trace": ["remember"]}

    # ------------------------------------------------------------------ routing
    def route(state: AgentState) -> AgentState:
        return {"route": brain.route(state["question"]), "trace": ["route"]}

    def direct_answer(state: AgentState) -> AgentState:
        return {"answer": "Hello! Ask me about the Nimbus docs, sales figures or service status.",
                "status": "direct", "trace": ["direct_answer"]}

    # ------------------------------------------------------------------ planning + fan-out
    def plan(state: AgentState) -> AgentState:
        sub_questions = brain.plan(state["question"], state.get("missing", []))
        return {"plan": sub_questions, "trace": ["plan"]}

    def fan_out(state: AgentState) -> list[Send]:
        """Map step: one parallel `research` task per sub-question (LangGraph `Send`)."""
        mode = "web" if state.get("route") == "web" else "docs"
        return [Send("research", {"sub_question": q, "mode": mode, "tenant": state["tenant"]})
                for q in state["plan"]]

    def research(payload: dict) -> AgentState:
        out = crag.invoke({"sub_question": payload["sub_question"], "mode": payload["mode"],
                           "tenant": payload["tenant"]})
        return {"evidence": out["relevant"], "trace": ["research:" + ">".join(out["path"])]}

    # ------------------------------------------------------------------ SQL branch + HITL
    def propose_sql(state: AgentState) -> AgentState:
        sql = brain.text_to_sql(state["question"])
        return {"pending_sql": sql, "plan": [state["question"]], "trace": ["propose_sql"]}

    def approve_sql(state: AgentState) -> Command:
        """Human-in-the-loop gate. The graph pauses here (state is checkpointed) until the
        caller resumes with Command(resume=True/False). Kept in its own node because on resume
        LangGraph re-runs the interrupted node from its start: this node has no side effects."""
        if settings.require_approval and not interrupt(
            {"type": "approve_sql", "sql": state["pending_sql"]}
        ):
            return Command(goto="finalize", update={
                "status": "rejected", "answer": "Query rejected by the reviewer.",
                "trace": ["approve_sql:rejected"]})
        return Command(goto="run_sql", update={"trace": ["approve_sql:ok"]})

    def run_sql(state: AgentState) -> AgentState:
        try:
            ev = tools.sql.run(state["pending_sql"], state["question"])
        except UnsafeSQL as e:
            return {"status": "blocked", "answer": f"Unsafe SQL blocked: {e}",
                    "trace": ["run_sql:blocked"]}
        return {"evidence": [ev], "trace": ["run_sql"]}

    def after_sql(state: AgentState) -> str:
        return "finalize" if state.get("status") == "blocked" else "synthesize"

    # ------------------------------------------------------------------ synthesis + reflection
    def synthesize(state: AgentState) -> AgentState:
        unique = list({e["id"] + e["sub_question"]: e for e in state.get("evidence", [])}.values())
        syn = brain.synthesize(state["question"], state["plan"], unique, state.get("memory", []))
        return {"synthesis": syn, "trace": ["synthesize"]}

    def verify(state: AgentState) -> AgentState:
        verdict = brain.verify(state["question"], state["plan"], state["synthesis"],
                               state.get("evidence", []))
        return {"verdict": verdict, "missing": verdict.missing, "trace": ["verify"]}

    def after_verify(state: AgentState) -> str:
        v = state["verdict"]
        if v.grounded and v.complete:
            return "finalize"
        if state.get("route") != "sql" and state.get("loops", 0) < settings.max_loops:
            return "replan"
        return "finalize"

    def replan(state: AgentState) -> AgentState:
        return {"loops": state.get("loops", 0) + 1, "trace": ["replan"]}

    def finalize(state: AgentState) -> AgentState:
        if state.get("status") in {"rejected", "blocked"}:
            return {"citations": []}
        syn, v = state["synthesis"], state["verdict"]
        by_id = {e["id"]: e for e in state.get("evidence", [])}
        cites = [{"id": i, "source": by_id[i]["source"], "kind": by_id[i]["kind"]}
                 for i in syn.cited_ids if i in by_id]
        if not v.grounded:
            return {"status": "ungrounded", "answer": REFUSAL, "citations": []}
        if not v.complete:
            note = "" if not syn.cited_ids else f"{syn.answer} "
            return {"status": "incomplete", "citations": cites,
                    "answer": f"{note}(Partial answer - could not find: {'; '.join(v.missing)})"
                    if cites else REFUSAL}
        return {"status": "ok", "answer": syn.answer, "citations": cites}

    g = StateGraph(AgentState)
    nodes = {
        "load_memory": load_memory, "remember": remember, "route": route,
        "direct_answer": direct_answer, "plan": plan, "research": research,
        "propose_sql": propose_sql, "approve_sql": approve_sql, "run_sql": run_sql,
        "synthesize": synthesize, "verify": verify, "replan": replan, "finalize": finalize,
    }
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "route")
    g.add_conditional_edges(
        "route", lambda s: s["route"],
        {"direct": "direct_answer", "memory": "remember", "sql": "propose_sql",
         "docs": "plan", "web": "plan"},
    )
    g.add_edge("direct_answer", END)
    g.add_edge("remember", END)
    g.add_conditional_edges("plan", fan_out, ["research"])
    g.add_edge("research", "synthesize")
    g.add_edge("propose_sql", "approve_sql")  # approve_sql routes itself via Command(goto=...)
    g.add_conditional_edges("run_sql", after_sql, {"finalize": "finalize", "synthesize": "synthesize"})
    g.add_edge("synthesize", "verify")
    g.add_conditional_edges("verify", after_verify, {"finalize": "finalize", "replan": "replan"})
    g.add_edge("replan", "plan")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver(), store=store or InMemoryStore())
