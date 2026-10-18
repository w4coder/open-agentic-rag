import pytest

from agentic_rag.config import AgentSettings
from agentic_rag.service import AgentService

CORPUS = "data/corpus"


@pytest.fixture(scope="module")
def svc() -> AgentService:
    return AgentService(AgentSettings(_env_file=None, corpus_path=CORPUS, require_approval=True))


@pytest.fixture
def svc_auto() -> AgentService:
    return AgentService(AgentSettings(_env_file=None, corpus_path=CORPUS, require_approval=False))
