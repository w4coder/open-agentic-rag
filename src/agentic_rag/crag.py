"""Corrective RAG (CRAG) subgraph: retrieve -> grade -> (rewrite & retry | web fallback).

    retrieve -> grade -+-> END                 (at least one relevant passage)
                       +-> rewrite -> retrieve (attempts left)
                       +-> web_fallback -> END (attempts exhausted)

WHY: plain RAG trusts whatever the retriever returns. CRAG *evaluates* retrieval quality and
reacts: reformulate the query, then fall back to another source. The loop is bounded
(`max_retrieval_attempts`) - unbounded agent loops are a cost/latency incident waiting to happen.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_rag.brain import Brain
from agentic_rag.config import AgentSettings
from agentic_rag.tools import Tools


class CragState(TypedDict, total=False):
    sub_question: str
    tenant: str
    mode: str  # "docs" | "web": the router may send us straight to the web
    query: str
    attempt: int
    candidates: list[dict]
    relevant: list[dict]
    path: list[str]  # nodes visited: exposed for trajectory evaluation


def build_crag(brain: Brain, tools: Tools, settings: AgentSettings):
    def retrieve(s: CragState) -> CragState:
        query = s.get("query") or s["sub_question"]
        return {"query": query,
                "candidates": tools.docs.search(query, s["sub_question"], s["tenant"]),
                "path": s.get("path", []) + ["retrieve"]}

    def grade(s: CragState) -> CragState:
        verdicts = brain.grade_many(s["sub_question"], [e["text"] for e in s["candidates"]])
        kept = [e for e, ok in zip(s["candidates"], verdicts) if ok]
        return {"relevant": kept, "path": s["path"] + ["grade"]}

    def rewrite(s: CragState) -> CragState:
        attempt = s.get("attempt", 0) + 1
        return {"attempt": attempt, "query": brain.rewrite(s["sub_question"], attempt),
                "path": s["path"] + ["rewrite"]}

    def web_fallback(s: CragState) -> CragState:
        found = tools.web.search(s["sub_question"], s["sub_question"])
        return {"relevant": found, "path": s.get("path", []) + ["web_fallback"]}

    def entry(s: CragState) -> str:
        return "web_fallback" if s.get("mode") == "web" else "retrieve"

    def after_grade(s: CragState) -> str:
        if s["relevant"]:
            return "done"
        return "rewrite" if s.get("attempt", 0) < settings.max_retrieval_attempts else "web_fallback"

    g = StateGraph(CragState)
    for name, fn in [("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                     ("web_fallback", web_fallback)]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, entry, {"retrieve": "retrieve", "web_fallback": "web_fallback"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade,
                            {"done": END, "rewrite": "rewrite", "web_fallback": "web_fallback"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("web_fallback", END)
    return g.compile()
