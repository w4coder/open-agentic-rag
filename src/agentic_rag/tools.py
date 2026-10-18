"""Tools the agent can call. Each returns *evidence dicts* with a uniform shape so the
synthesizer and the verifier treat documents, SQL rows and web results the same way:

    {"id", "text", "source", "kind": "doc"|"sql"|"web", "sub_question"}

Security notes (agents act on the world - treat tools as an attack surface):
* SQL is **read-only by construction**, in depth: (1) parsed with sqlglot - one statement,
  a query, only allow-listed tables and functions; (2) the *normalised* SQL produced by the
  parser is what runs (no parser/engine differential); (3) the database enforces it again
  with SQLite's authorizer (read of allow-listed tables only - the moral equivalent of a
  read-only role with per-table GRANTs on a real warehouse) and `PRAGMA query_only`;
  (4) wall-clock timeout + row cap. Never rely on the LLM to write safe SQL.
* Web results are untrusted text: they enter the prompt as data, never as instructions.
* Every call is scoped to the caller's tenant.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from sqlglot import exp

from agentic_rag.config import AgentSettings

log = logging.getLogger(__name__)
Evidence = dict


# ------------------------------------------------------------------------------ documents
class DocsTool:
    """Hybrid retrieval over the knowledge base, reusing the open-rag stack as a library.

    * Production (`RAG_QDRANT_URL` set): connects to the *shared* open-rag index - the same
      collections, embeddings model and tenant isolation as the RAG API. Nothing is ingested.
    * Dev/tests (no URL): embedded in-memory Qdrant, corpus ingested at start-up.
    The embedding provider comes from the RAG_* environment (never forced).
    """

    def __init__(self, settings: AgentSettings, k: int = 4, rag_settings=None) -> None:
        from open_rag.config import Settings
        from open_rag.indexing.store import VectorIndex
        from open_rag.ingestion.pipeline import ingest_path
        from open_rag.providers import get_embeddings
        from open_rag.retrieval.hybrid import HybridRetriever
        from qdrant_client import QdrantClient

        self.k = k
        rs = rag_settings or Settings()
        client = None if rs.qdrant_url else QdrantClient(":memory:")
        self.index = VectorIndex(rs, get_embeddings(rs), client=client)
        if client is not None and settings.ingest_on_start:
            ingest_path(Path(settings.corpus_path), self.index, rs, settings.tenant)
        self.retriever = HybridRetriever(self.index, rs)

    def search(self, query: str, sub_question: str, tenant: str) -> list[Evidence]:
        return [
            {"id": h.chunk_id, "text": h.text, "source": h.source, "kind": "doc",
             "sub_question": sub_question}
            for h in self.retriever.retrieve(query, tenant)[: self.k]
        ]

    def healthy(self) -> bool:
        return self.index.healthy()


# ------------------------------------------------------------------------------------ SQL
_SALES = [
    ("EMEA", "Q1", "Compute", 120000), ("EMEA", "Q1", "Storage", 80000),
    ("EMEA", "Q2", "Compute", 140000), ("EMEA", "Q2", "Storage", 90000),
    ("AMER", "Q1", "Compute", 200000), ("AMER", "Q1", "Storage", 150000),
    ("AMER", "Q2", "Compute", 230000), ("AMER", "Q2", "Storage", 160000),
    ("APAC", "Q1", "Compute", 60000), ("APAC", "Q1", "Storage", 40000),
    ("APAC", "Q2", "Compute", 85000), ("APAC", "Q2", "Storage", 55000),
]
ALLOWED_TABLES = {"sales"}
ALLOWED_FUNCTIONS = {"sum", "count", "avg", "min", "max", "round", "abs", "upper", "lower",
                     "coalesce", "length", "cast", "case", "if"}
_WRITE_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
                exp.Command, exp.Merge, exp.TruncateTable)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum)\b", re.I
)
SCHEMA_HINT = "Table sales(region TEXT [EMEA|AMER|APAC], quarter TEXT [Q1|Q2], product TEXT, amount INTEGER EUR)"
MAX_ROWS = 50


class UnsafeSQL(ValueError):
    pass


def _func_name(f: exp.Func) -> str:
    return (f.name if isinstance(f, exp.Anonymous) else f.key).lower()


def validate_sql(sql: str) -> str:
    """Return the normalised, safe SQL to execute, or raise UnsafeSQL."""
    s = sql.strip().rstrip(";").strip()
    if _FORBIDDEN.search(s):  # cheap first screen, the parser below is the real check
        raise UnsafeSQL("forbidden keyword")
    try:
        statements = [t for t in sqlglot.parse(s, read="sqlite") if t is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQL("unparseable SQL") from e
    if len(statements) != 1:
        raise UnsafeSQL("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Query) or tree.find(*_WRITE_NODES):
        raise UnsafeSQL("only SELECT queries are allowed")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = {t.name.lower() for t in tree.find_all(exp.Table)} - ctes
    if not tables or not tables <= ALLOWED_TABLES:
        raise UnsafeSQL(f"table not allowed: {sorted(tables - ALLOWED_TABLES) or 'none'}")
    # AND/OR/XOR are modelled as Func by sqlglot: they are operators, not callable functions.
    funcs = {_func_name(f) for f in tree.find_all(exp.Func) if not isinstance(f, exp.Connector)}
    if not funcs <= ALLOWED_FUNCTIONS:
        raise UnsafeSQL(f"function not allowed: {sorted(funcs - ALLOWED_FUNCTIONS)}")
    return tree.sql(dialect="sqlite")


class SqlTool:
    def __init__(self, timeout_s: float = 2.0) -> None:
        self.timeout_s = timeout_s
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute("CREATE TABLE sales(region TEXT, quarter TEXT, product TEXT, amount INTEGER)")
        self.conn.executemany("INSERT INTO sales VALUES (?,?,?,?)", _SALES)
        self.conn.commit()
        self.conn.execute("PRAGMA query_only = ON")  # defence in depth: DB itself refuses writes
        self.conn.set_authorizer(self._authorize)
        self._lock = threading.Lock()  # one sqlite connection shared by parallel branches
        self._deadline = 0.0

    @staticmethod
    def _authorize(action, arg1, arg2, _db, _trigger) -> int:
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and (arg1 or "").lower() in ALLOWED_TABLES:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in ALLOWED_FUNCTIONS | {"iif"}:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def _progress(self) -> int:
        return 1 if time.perf_counter() >= self._deadline else 0  # non-zero aborts the query

    def run(self, sql: str, sub_question: str) -> Evidence:
        safe = validate_sql(sql)
        with self._lock:
            self._deadline = time.perf_counter() + self.timeout_s
            self.conn.set_progress_handler(self._progress, 1000)
            try:
                cur = self.conn.execute(f"SELECT * FROM ({safe}) LIMIT {MAX_ROWS}")
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
            except sqlite3.DatabaseError as e:
                raise UnsafeSQL(f"query refused or aborted: {e}") from e
            finally:
                self.conn.set_progress_handler(None, 0)
        table = "; ".join(", ".join(f"{c}={v}" for c, v in zip(cols, r)) for r in rows)
        return {
            "id": "sql-" + hashlib.sha1(safe.encode()).hexdigest()[:8],
            "text": f"SQL: {safe}\nRESULT: {table or 'no rows'}",
            "source": "sales_db", "kind": "sql", "sub_question": sub_question,
        }


# ------------------------------------------------------------------------------------ web
@dataclass
class WebPage:
    url: str
    text: str


_OFFLINE_WEB = [
    WebPage("https://status.nimbus.example/incidents",
            "Latest Nimbus status: no open incidents. The last resolved incident was a "
            "network degradation in eu-west lasting 42 minutes."),
    WebPage("https://news.example/cloud-storage",
            "Industry news: cloud storage prices fell by an average of 8 percent this year."),
]


def _web_evidence(url: str, text: str, sub_question: str) -> Evidence:
    return {"id": "web-" + hashlib.sha1(url.encode()).hexdigest()[:8], "text": text[:2000],
            "source": url, "kind": "web", "sub_question": sub_question}


class WebTool:
    """Offline stub (dev/tests): deterministic pages scored by token overlap."""

    def search(self, query: str, sub_question: str) -> list[Evidence]:
        from open_rag.providers import tokenize

        q = set(tokenize(query))
        scored = sorted(
            ((len(q & set(tokenize(p.text))), p) for p in _OFFLINE_WEB), key=lambda x: -x[0]
        )
        return [_web_evidence(p.url, p.text, sub_question) for score, p in scored[:2] if score > 1]


class TavilyWebTool:
    """Real web search (Tavily). Timeouts and errors degrade to "no evidence", never crash
    the run: the agent then answers from other sources or refuses honestly."""

    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout_s: float = 8.0, transport=None) -> None:
        import httpx

        self.api_key = api_key
        self.http = httpx.Client(timeout=timeout_s, transport=transport)

    def search(self, query: str, sub_question: str) -> list[Evidence]:
        try:
            r = self.http.post(self.URL, json={"query": query, "max_results": 3},
                               headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception:
            log.warning("web search failed (no web evidence)", exc_info=True)
            return []
        return [_web_evidence(x["url"], x.get("content", ""), sub_question)
                for x in results if x.get("url")]


@dataclass
class Tools:
    docs: DocsTool
    sql: SqlTool
    web: WebTool | TavilyWebTool


def build_tools(settings: AgentSettings) -> Tools:
    web = (TavilyWebTool(settings.tavily_api_key) if settings.web_provider == "tavily"
           else WebTool())
    return Tools(docs=DocsTool(settings), sql=SqlTool(settings.sql_timeout_s), web=web)
