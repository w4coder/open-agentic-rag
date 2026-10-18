"""Agent evaluation: final answer + trajectory + cost, and a baseline comparison.

    python -m evals.run_eval                 # local, offline (heuristic brain)
    python -m evals.run_eval --langsmith     # also upload dataset + experiment

Three levels (you need all three to debug an agent):
1. FINAL ANSWER  - correct / correctly refused.
2. TRAJECTORY    - did it take the expected nodes (route, decompose, loop), nothing forbidden,
                   within the step budget?  A right answer through a wrong path is a time bomb.
3. COST          - number of graph steps (proxy for LLM calls/latency); an agent that is 2%
                   better but 6x costlier may not be worth shipping.

Baseline = single-shot RAG (retrieve once with the *whole* question, no planning/loops):
the honest question is "when does agentic beat plain RAG?" - answer: multi-hop, routing, refusal.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from agentic_rag.config import AgentSettings
from agentic_rag.service import AgentService

ROOT = Path(__file__).resolve().parent.parent
REFUSAL_MARKERS = ("don't know", "could not find", "rejected")


def load_dataset() -> list[dict]:
    lines = (ROOT / "evals" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def contains_all(answer: str, needles: list[str]) -> bool:
    return all(n.lower() in (answer or "").lower() for n in needles)


def refused(answer: str) -> bool:
    return any(m in (answer or "").lower() for m in REFUSAL_MARKERS)


def trajectory_ok(ex: dict, trace: list[str], budget: int) -> bool:
    joined = " ".join(trace)
    expected = all(n in joined for n in ex.get("expect_nodes", []))
    forbidden = not any(n in joined for n in ex.get("forbid_nodes", []))
    parallel = sum(t.startswith("research") for t in trace) >= ex.get("min_research", 0)
    return expected and forbidden and parallel and len(trace) <= budget


def score_agent(ex: dict, out: dict, budget: int) -> dict:
    ok = (contains_all(out["answer"], ex["must_contain"]) and not refused(out["answer"])
          if ex["answerable"] else refused(out["answer"]))
    return {"correct": float(ok), "trajectory": float(trajectory_ok(ex, out["trace"], budget)),
            "steps": len(out["trace"])}


def run_agent(svc: AgentService, data: list[dict]) -> list[dict]:
    rows = []
    for ex in data:
        out = svc.ask(ex["question"], user_id="eval")
        rows.append({"id": ex["id"], "kind": ex["kind"], **score_agent(ex, out, budget=40)})
    return rows


def run_baseline(svc: AgentService, data: list[dict]) -> list[dict]:
    """Single-shot RAG over documents only: no router, no decomposition, no loops, no tools."""
    rows = []
    for ex in data:
        ev = svc.tools.docs.search(ex["question"], ex["question"], svc.settings.tenant)
        syn = svc.brain.synthesize(ex["question"], [ex["question"]], ev, [])
        ok = (contains_all(syn.answer, ex["must_contain"]) and not refused(syn.answer)
              if ex["answerable"] else refused(syn.answer))
        rows.append({"id": ex["id"], "kind": ex["kind"], "correct": float(ok), "steps": 3})
    return rows


def summarize(rows: list[dict]) -> dict:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r["correct"])
    out = {"correct": round(statistics.mean(r["correct"] for r in rows), 3),
           "avg_steps": round(statistics.mean(r["steps"] for r in rows), 1),
           "by_kind": {k: round(statistics.mean(v), 2) for k, v in by_kind.items()}}
    if "trajectory" in rows[0]:
        out["trajectory"] = round(statistics.mean(r["trajectory"] for r in rows), 3)
    return out


def run_langsmith(svc: AgentService, data: list[dict]) -> None:
    from langsmith import Client, evaluate

    client = Client()
    name = "agentic-rag-golden"
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name, description="agentic-rag golden set incl. trajectories")
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"question": e["question"]} for e in data],
            outputs=[{k: v for k, v in e.items() if k != "question"} for e in data],
        )

    def target(inputs: dict) -> dict:
        out = svc.ask(inputs["question"], user_id="eval")
        return {"answer": out["answer"], "trace": out["trace"]}

    def correctness(outputs: dict, reference_outputs: dict) -> dict:
        return {"key": "correct", "score": score_agent(reference_outputs, outputs, 40)["correct"]}

    def trajectory(outputs: dict, reference_outputs: dict) -> dict:
        return {"key": "trajectory",
                "score": float(trajectory_ok(reference_outputs, outputs["trace"], 40))}

    def steps(outputs: dict) -> dict:
        return {"key": "n_steps", "score": float(len(outputs["trace"]))}

    evaluate(target, data=name, evaluators=[correctness, trajectory, steps],
             experiment_prefix=f"agent-{svc.brain.name}", max_concurrency=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langsmith", action="store_true")
    ap.add_argument("--min-correct", type=float, default=0.0, help="quality gate (exit 1 below)")
    ap.add_argument("--min-trajectory", type=float, default=0.0)
    args = ap.parse_args()

    data = load_dataset()
    # HITL gate off for batch evaluation (it is covered by tests/test_agent.py).
    # Provider comes from the environment (AGENT_LLM_PROVIDER): heuristic in PR CI, LLM on main.
    svc = AgentService(AgentSettings(require_approval=False,
                                     corpus_path=str(ROOT / "data" / "corpus")))
    results = {"agentic": summarize(run_agent(svc, data)),
               "baseline_single_shot_rag": summarize(run_baseline(svc, data))}
    out = ROOT / "evals" / "results"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if args.langsmith:
        run_langsmith(svc, data)
    agent = results["agentic"]
    if agent["correct"] < args.min_correct or agent["trajectory"] < args.min_trajectory:
        print(f"quality gate FAILED: correct={agent['correct']} trajectory={agent['trajectory']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
