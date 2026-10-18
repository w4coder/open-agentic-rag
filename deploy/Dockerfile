# Build context = the repository root: docker build -f deploy/Dockerfile .
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml requirements.lock ./
COPY src src
# Pinned set first (it includes open-rag from its tagged GitHub release), then the package.
RUN python -m venv /venv \
 && /venv/bin/pip install -r requirements.lock \
 && /venv/bin/pip install --no-deps .

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH=/venv/bin:$PATH AGENT_LOG_JSON=true \
    AGENT_INGEST_ON_START=false
COPY --from=build /venv /venv
RUN useradd --uid 10001 --create-home app
WORKDIR /app
USER 10001
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/healthz', timeout=2)"
CMD ["uvicorn", "agentic_rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "20", "--no-server-header"]
