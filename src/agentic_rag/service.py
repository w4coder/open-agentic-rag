"""Application service: identity, thread ownership, budgets, pause/resume (HITL), tracing.

Security model:
* a `Principal` (user, tenant, roles) comes from the API key - never from the request body;
* threads are registered to their owner (global registry in the store) and checkpoints are
  keyed `tenant:thread_id`: nobody can read, continue or resume someone else's thread;
* resuming a paused SQL run requires the `approver` role (and, with `four_eyes`, a different
  person than the requester). Every decision is audit-logged.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from open_rag.limits import Limits
from open_rag.observability import record_usage, request_id
from prometheus_client import Counter, Histogram

from agentic_rag.brain import get_brain
from agentic_rag.budget import BudgetExceeded, TokenBudget
from agentic_rag.config import AgentSettings, get_settings
from agentic_rag.graph import RUN_RESET, build_agent
from agentic_rag.persistence import build_persistence
from agentic_rag.tools import build_tools

log = logging.getLogger(__name__)
audit = logging.getLogger("agentic_rag.audit")

RUNS = Counter("agent_runs_total", "Agent runs by final status", ["status"])
LATENCY = Histogram("agent_run_seconds", "Agent run latency", buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 40, 80))
STEPS = Histogram("agent_run_steps", "Graph steps per run", buckets=(2, 4, 8, 12, 16, 24, 32, 40))
_THREADS = ("threads",)


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant: str
    roles: frozenset[str] = field(default_factory=frozenset)


class Forbidden(PermissionError):
    pass


class Conflict(RuntimeError):
    pass


class UpstreamError(RuntimeError):
    pass


class AgentService:
    def __init__(self, settings: AgentSettings | None = None, checkpointer=None, store=None,
                 brain=None, tools=None, limits: Limits | None = None):
        self.settings = s = settings or get_settings()
        self.brain = brain or get_brain(s)
        self.tools = tools or build_tools(s)
        if checkpointer is None and store is None:
            checkpointer, store = build_persistence(s.checkpoint_url)
        self.graph = build_agent(s, self.brain, self.tools, checkpointer, store)
        self.store = self.graph.store
        if limits is None:
            redis_client = None
            if s.redis_url:
                import redis

                redis_client = redis.Redis.from_url(s.redis_url, socket_timeout=0.5)
            limits = Limits(s.rate_limit_per_min, s.tenant_daily_tokens, redis_client)
        self.limits = limits

    # ---- identity / ownership -------------------------------------------------------------
    def local_principal(self, user_id: str = "anon") -> Principal:
        """CLI / tests / evals: the operator at the terminal is also the approver."""
        return Principal(user_id, self.settings.tenant, frozenset({"user", "approver"}))

    def _thread_owner(self, thread_id: str) -> dict | None:
        item = self.store.get(_THREADS, thread_id)
        return item.value if item else None

    def _claim_thread(self, thread_id: str, p: Principal) -> None:
        owner = self._thread_owner(thread_id)
        if owner is None:
            self.store.put(_THREADS, thread_id, {"tenant": p.tenant, "user_id": p.user_id})
        elif owner != {"tenant": p.tenant, "user_id": p.user_id}:
            raise Forbidden("thread belongs to another user")

    def _config(self, thread_id: str, p: Principal, budget: TokenBudget) -> dict:
        return {
            # thread = one conversation; tenant prefix = defence in depth on the checkpoint key
            "configurable": {"thread_id": f"{p.tenant}:{thread_id}"},
            "recursion_limit": self.settings.max_steps,  # hard cap on graph steps
            "callbacks": [budget],                        # hard cap on tokens
            "run_name": "agentic_rag",
            "tags": ["agentic-rag", f"brain:{self.brain.name}", f"tenant:{p.tenant}"],
            "metadata": {"user_id": p.user_id, "tenant": p.tenant, "request_id": request_id.get()},
        }

    def _pending(self, config: dict) -> bool:
        return bool(self.graph.get_state(config).interrupts)

    # ---- execution ---------------------------------------------------------------------------
    @staticmethod
    def _format(state: dict, thread_id: str) -> dict:
        if "__interrupt__" in state:
            payload = state["__interrupt__"][0].value
            return {"status": "awaiting_approval", "thread_id": thread_id, "request": payload}
        return {
            "status": state.get("status"), "answer": state.get("answer"),
            "citations": state.get("citations", []), "thread_id": thread_id,
            "trace": state.get("trace", []), "plan": state.get("plan", []),
        }

    def _run(self, inp, thread_id: str, p: Principal) -> dict:
        if not self.limits.has_token_budget(p.tenant):
            raise BudgetExceeded(f"daily token budget exhausted for tenant {p.tenant}")
        budget = TokenBudget(self.settings.max_tokens_per_run)
        config = self._config(thread_id, p, budget)
        started = time.perf_counter()
        try:
            out = self._format(self.graph.invoke(inp, config=config), thread_id)
        except BudgetExceeded:
            out = {"status": "budget_exceeded", "thread_id": thread_id, "citations": [],
                   "answer": "This request exceeded its cost budget; please narrow it down."}
        except GraphRecursionError:
            out = {"status": "step_budget_exceeded", "thread_id": thread_id, "citations": [],
                   "answer": "This request needed too many steps; please narrow it down."}
        except Exception as e:
            RUNS.labels(status="error").inc()
            log.exception("agent run failed thread=%s", thread_id)
            raise UpstreamError(str(e)) from e
        finally:
            self.limits.spend_tokens(p.tenant, record_usage(p.tenant, budget.usage_metadata))
        RUNS.labels(status=out["status"]).inc()
        LATENCY.observe(time.perf_counter() - started)
        STEPS.observe(len(out.get("trace", [])))
        return out

    def ask(self, question: str, user_id: str = "anon", thread_id: str | None = None,
            principal: Principal | None = None) -> dict:
        p = principal or self.local_principal(user_id)
        thread_id = thread_id or uuid.uuid4().hex
        self._claim_thread(thread_id, p)
        if self._pending(self._config(thread_id, p, TokenBudget(0))):
            raise Conflict("thread is awaiting approval: resume it or use a new thread")
        inp = {**RUN_RESET, "question": question, "user_id": p.user_id, "tenant": p.tenant}
        return self._run(inp, thread_id, p)

    def resume(self, thread_id: str, approved: bool, user_id: str = "anon",
               principal: Principal | None = None) -> dict:
        p = principal or self.local_principal(user_id)
        owner = self._thread_owner(thread_id)
        if owner is None or owner["tenant"] != p.tenant:
            raise Forbidden("unknown thread")
        if "approver" not in p.roles:
            raise Forbidden("approver role required")
        if self.settings.four_eyes and owner["user_id"] == p.user_id:
            raise Forbidden("four-eyes policy: requester cannot approve their own query")
        config = self._config(thread_id, p, TokenBudget(0))
        snapshot = self.graph.get_state(config)
        if not snapshot.interrupts:
            raise Conflict("thread is not awaiting approval")
        audit.info("sql_approval thread=%s tenant=%s requester=%s approver=%s approved=%s sql=%r",
                   thread_id, p.tenant, owner["user_id"], p.user_id, approved,
                   snapshot.interrupts[0].value.get("sql"))
        return self._run(Command(resume=approved), thread_id, p)
