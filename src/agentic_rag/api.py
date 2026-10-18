"""HTTP API with pause/resume for human-in-the-loop.

POST /v1/ask      {question, thread_id?}   -> answer  OR  {status: awaiting_approval, thread_id, request}
POST /v1/resume   {thread_id, approved}    -> answer        (requires the `approver` role)

Identity (user, tenant, roles) comes from the `X-API-Key` header, never from the body.
State lives in the LangGraph *checkpointer* keyed by thread: with AGENT_CHECKPOINT_URL pointing
to Postgres any replica can resume a run paused on another one.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from open_rag.observability import configure_logging, request_id
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from agentic_rag.budget import BudgetExceeded
from agentic_rag.config import get_settings
from agentic_rag.service import AgentService, Conflict, Forbidden, Principal, UpstreamError


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class ResumeRequest(BaseModel):
    thread_id: str = Field(max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    approved: bool


def parse_api_keys(spec: str) -> dict[str, Principal]:
    """'key:user:tenant:role1|role2,...' -> {key: Principal}"""
    out = {}
    for entry in filter(None, (e.strip() for e in spec.split(","))):
        key, user, tenant, roles = (entry.split(":") + ["", "", ""])[:4]
        out[key] = Principal(user or "anon", tenant or "default",
                             frozenset(filter(None, roles.split("|"))) or frozenset({"user"}))
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_json)
    app.state.svc = AgentService(settings)
    app.state.keys = parse_api_keys(settings.api_keys)
    yield


app = FastAPI(title="agentic-rag", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def correlate(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


_ERRORS = {Forbidden: 403, Conflict: 409, BudgetExceeded: 429, UpstreamError: 503}
for _exc, _code in _ERRORS.items():
    def _handler(_: Request, exc: Exception, code: int = _code):
        return JSONResponse({"detail": str(exc) if code != 503 else "upstream unavailable"},
                            status_code=code)
    app.add_exception_handler(_exc, _handler)


def principal(request: Request, x_api_key: str = Header(default="")) -> Principal:
    p = request.app.state.keys.get(x_api_key)
    if p is None:
        raise HTTPException(401, "Invalid API key")
    if not request.app.state.svc.limits.allow_request(p.tenant):
        raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})
    return p


@app.post("/v1/ask")
async def ask(req: AskRequest, request: Request, p: Principal = Depends(principal)):
    return await run_in_threadpool(request.app.state.svc.ask, req.question, p.user_id,
                                   req.thread_id, p)


@app.post("/v1/resume")
async def resume(req: ResumeRequest, request: Request, p: Principal = Depends(principal)):
    return await run_in_threadpool(request.app.state.svc.resume, req.thread_id, req.approved,
                                   p.user_id, p)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request):
    svc: AgentService = request.app.state.svc

    def check() -> dict:
        checks = {"docs_index": svc.tools.docs.healthy()}
        try:
            svc.store.get(("health",), "probe")  # checkpoint/store backend reachable
            checks["store"] = True
        except Exception:
            checks["store"] = False
        return checks

    checks = await run_in_threadpool(check)
    if not all(checks.values()):
        raise HTTPException(503, f"dependencies unavailable: {checks}")
    return {"status": "ready", **checks}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
