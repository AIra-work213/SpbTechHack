"""Tests for the backend server and agent system."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TestServerImport:
    """Test that server module imports correctly."""

    def test_fastapi_app_exists(self):
        """Server module should have a FastAPI app."""
        from backend.server.server import app
        assert app is not None

    def test_agent_system_exists(self):
        """Agent system should be importable."""
        from backend.models.agents.system import AgentSystem
        assert AgentSystem is not None


class TestAgentSystemInit:
    """Test AgentSystem initialization."""

    def test_init_is_sync(self):
        """AgentSystem.__init__ should be synchronous."""
        from backend.models.agents.system import AgentSystem
        # Should not raise - init is sync
        agent = AgentSystem()
        assert agent.client is not None
        assert agent.server_params is not None
        assert agent.openai_tools == []
        assert agent._tools_loaded is False


class TestGraphModel:
    """Test Graph Pydantic model."""

    def test_graph_default_values(self):
        """Graph should have correct default values."""
        from backend.models.agents.system import Graph
        g = Graph(query="test query")
        assert g.query == "test query"
        assert g.ready_prompt is None
        assert g.agent_response is None
        assert g.judge_score == 0
        assert g.fail_cnt == 0
        assert g.urls == []
        assert g.called_tools == []


class TestPromptsModule:
    """Test Prompts module."""

    def test_prompts_have_json_format(self):
        """Prompts should instruct JSON format output."""
        from backend.models.prompts.prompts import Prompts
        p = Prompts()
        assert "json" in p.generate_prompt.lower() or "JSON" in p.generate_prompt
        assert "json" in p.agent_prompt.lower() or "JSON" in p.agent_prompt
        assert "json" in p.critical_prompt.lower() or "JSON" in p.critical_prompt


class TestKnowledgeBase:
    """Test KnowledgeBase module (requires Qdrant running)."""

    def test_knowledge_base_init(self):
        """KnowledgeBase should initialize with Qdrant."""
        from backend.data.Qdrant.qdrant_session import KnowledgeBase
        # This will connect to Qdrant at localhost:6333
        kb = KnowledgeBase()
        assert kb.client is not None
        assert kb.collection_name == "knowledge_base_ru_small"
        assert kb.model is not None


@pytest.mark.asyncio
class TestServerEndpoint:
    """Test server endpoint with mocked dependencies."""

    async def test_query_endpoint_structure(self):
        """Test that /query endpoint accepts correct structure."""
        from backend.server.server import app

        # Mock the agent_system.graph_invoke to avoid actual LLM calls
        with patch("backend.server.server.agent_system") as mock_agent:
            mock_agent.graph_invoke = AsyncMock(return_value={
                "query": "test",
                "agent_response": "mocked response",
                "called_tools": [],
                "urls": []
            })

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/query", json={"query": "Как получить паспорт?"})

            assert response.status_code == 200
            data = response.json()
            assert "response" in data
