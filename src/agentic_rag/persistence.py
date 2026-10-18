"""Checkpointer (short-term memory / HITL pause state) + store (long-term memory).

AGENT_CHECKPOINT_URL:
* `memory`                  - dev/tests only: lost on restart, not shared between replicas.
* `sqlite:///path/agent.db` - single node, durable (survives restarts).
* `postgresql://...`        - production: shared by every replica, so a run paused on replica
                              A (awaiting approval) can be resumed on replica B.
"""
from __future__ import annotations

import sqlite3

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


def _sqlite(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def build_persistence(url: str):
    if url == "memory":
        return InMemorySaver(), InMemoryStore()
    if url.startswith("sqlite:///"):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.store.sqlite import SqliteStore

        path = url[len("sqlite:///"):]
        saver, store = SqliteSaver(_sqlite(path)), SqliteStore(_sqlite(path))
        saver.setup()
        store.setup()
        return saver, store
    if url.startswith(("postgres://", "postgresql://")):
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(url, min_size=1, max_size=20, open=True, kwargs={
            "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
        saver, store = PostgresSaver(pool), PostgresStore(pool)
        saver.setup()
        store.setup()
        return saver, store
    raise ValueError(f"unsupported AGENT_CHECKPOINT_URL: {url!r}")
