"""Per-run LLM token budget, enforced by a LangChain callback.

`recursion_limit` caps graph *steps*; this caps *tokens* (= cost). The handler is attached to
the run config, sees every LLM call made anywhere in the graph (subgraphs, parallel `Send`
branches), and aborts the run as soon as the budget is crossed (`raise_error = True` makes
LangChain propagate the exception instead of swallowing it).
"""
from __future__ import annotations

from langchain_core.callbacks import UsageMetadataCallbackHandler


class BudgetExceeded(RuntimeError):
    pass


class TokenBudget(UsageMetadataCallbackHandler):
    raise_error = True

    def __init__(self, max_tokens: int) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.total = 0  # counted here too: the parent ignores responses without a model_name

    def on_llm_end(self, response, **kwargs) -> None:
        super().on_llm_end(response, **kwargs)  # per-model usage, for metrics
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.total += int(usage.get("total_tokens", 0) or 0)
        if self.max_tokens > 0 and self.total > self.max_tokens:
            raise BudgetExceeded(f"run used {self.total} tokens > budget {self.max_tokens}")
