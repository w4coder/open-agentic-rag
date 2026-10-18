"""Agent configuration."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    # Brain
    llm_provider: Literal["heuristic", "openai"] = "heuristic"
    llm_model: str = "gpt-4o-mini"
    llm_fallback_model: str = ""
    llm_timeout_s: float = 30.0
    min_grade_overlap: float = 0.34  # heuristic grader threshold

    # Budgets (hard caps: an unbounded agent is a cost/latency incident waiting to happen)
    max_loops: int = 2
    max_retrieval_attempts: int = 2
    max_steps: int = 40  # graph steps -> LangGraph recursion_limit
    max_tokens_per_run: int = 60_000  # LLM tokens per run (aborted beyond)
    tenant_daily_tokens: int = 0  # 0 = unlimited
    rate_limit_per_min: int = 30  # per tenant

    # Human-in-the-loop
    require_approval: bool = True
    four_eyes: bool = False  # an approver cannot approve their own request

    # Persistence: memory | sqlite:///path.db | postgresql://user:pw@host/db
    checkpoint_url: str = "memory"
    memory_max_items: int = 20
    memory_max_chars: int = 300

    # Tools
    corpus_path: str = "data/corpus"
    ingest_on_start: bool = True  # dev only: ignored when RAG_QDRANT_URL points to the shared index
    tenant: str = "tenant-a"  # default tenant for CLI / dev principal
    sql_timeout_s: float = 2.0
    web_provider: Literal["stub", "tavily"] = "stub"
    tavily_api_key: str = Field(default="", validation_alias=AliasChoices(
        "AGENT_TAVILY_API_KEY", "TAVILY_API_KEY"))

    # API: "key:user:tenant:role1|role2,..."  roles: user, approver
    api_keys: str = "dev-key:alice:tenant-a:user,dev-approver:bob:tenant-a:user|approver"
    redis_url: str = ""
    log_json: bool = False


@lru_cache
def get_settings() -> AgentSettings:
    return AgentSettings()
