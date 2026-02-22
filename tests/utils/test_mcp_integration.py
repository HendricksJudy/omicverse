"""Tests for BioContext MCP integration modules.

These tests run entirely offline — no MCP servers are contacted.  They verify
configuration, client construction, enricher logic, and AgentConfig wiring.
"""

import asyncio
import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: make ``omicverse.utils.*`` importable without full package
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "omicverse"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Minimal package stubs so relative imports in utils work
for name in [
    "omicverse",
    "omicverse.utils",
    "omicverse.utils.mcp_config",
    "omicverse.utils.mcp_client",
    "omicverse.utils.mcp_context_enricher",
    "omicverse.utils.agent_config",
]:
    sys.modules.pop(name, None)

omicverse_pkg = types.ModuleType("omicverse")
omicverse_pkg.__path__ = [str(PACKAGE_ROOT)]
omicverse_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse", loader=None, is_package=True
)
sys.modules["omicverse"] = omicverse_pkg

utils_pkg = types.ModuleType("omicverse.utils")
utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]
utils_pkg.__spec__ = importlib.machinery.ModuleSpec(
    "omicverse.utils", loader=None, is_package=True
)
sys.modules["omicverse.utils"] = utils_pkg
omicverse_pkg.utils = utils_pkg

# Load the modules under test
def _load_module(mod_name: str, file_name: str):
    fqn = f"omicverse.utils.{mod_name}"
    spec = importlib.util.spec_from_file_location(
        fqn, PACKAGE_ROOT / "utils" / file_name
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    setattr(utils_pkg, mod_name, mod)
    spec.loader.exec_module(mod)
    return mod


mcp_config_mod = _load_module("mcp_config", "mcp_config.py")
mcp_client_mod = _load_module("mcp_client", "mcp_client.py")
mcp_enricher_mod = _load_module("mcp_context_enricher", "mcp_context_enricher.py")

# agent_config imports mcp_config, so load it after mcp_config is available
agent_config_mod = _load_module("agent_config", "agent_config.py")


# ===================================================================
# mcp_config tests
# ===================================================================

class TestMCPConfig:

    def test_default_config_disabled(self):
        cfg = mcp_config_mod.MCPConfig()
        assert cfg.enabled is False
        assert cfg.max_tools_per_query == 3
        assert cfg.max_context_tokens == 2000

    def test_default_servers_is_biocontext_kb(self):
        cfg = mcp_config_mod.MCPConfig()
        assert len(cfg.servers) == 1
        assert cfg.servers[0].name == "BioContextAI-KB"

    def test_active_servers_filters_disabled(self):
        s1 = mcp_config_mod.MCPServerConfig(name="a", url="http://a", enabled=True)
        s2 = mcp_config_mod.MCPServerConfig(name="b", url="http://b", enabled=False)
        cfg = mcp_config_mod.MCPConfig(servers=[s1, s2])
        active = cfg.active_servers()
        assert len(active) == 1
        assert active[0].name == "a"

    def test_biocontext_registry_has_expected_keys(self):
        registry = mcp_config_mod.BIOCONTEXT_REGISTRY
        assert "biocontext-kb" in registry
        assert "string-db" in registry
        assert "biothings" in registry
        assert "gget" in registry
        assert len(registry) >= 10

    def test_server_config_defaults(self):
        s = mcp_config_mod.MCPServerConfig(name="test", url="http://test.com/mcp")
        assert s.transport == "streamable_http"
        assert s.enabled is True
        assert s.timeout == 30
        assert s.connect_timeout == 10


# ===================================================================
# mcp_client tests
# ===================================================================

class TestMCPToolInfo:

    def test_qualified_name(self):
        info = mcp_client_mod.MCPToolInfo(
            server_name="SRV",
            tool_name="search",
            description="Search tool",
        )
        assert info.qualified_name == "SRV/search"


class TestMCPToolResult:

    def test_error_result(self):
        result = mcp_client_mod.MCPToolResult(
            server_name="SRV",
            tool_name="search",
            is_error=True,
            error="timeout",
        )
        assert result.is_error
        assert result.error == "timeout"
        assert result.content is None


class TestBioContextMCPClient:

    def test_raises_import_error_when_mcp_unavailable(self):
        """When the mcp package is not installed, construction should fail."""
        original = mcp_client_mod._MCP_AVAILABLE
        try:
            mcp_client_mod._MCP_AVAILABLE = False
            with pytest.raises(ImportError, match="mcp"):
                mcp_client_mod.BioContextMCPClient([
                    mcp_config_mod.MCPServerConfig(name="test", url="http://x"),
                ])
        finally:
            mcp_client_mod._MCP_AVAILABLE = original

    def test_server_names_property(self):
        """Only enabled servers should appear in server_names."""
        if not mcp_client_mod._MCP_AVAILABLE:
            pytest.skip("mcp package not installed")
        s1 = mcp_config_mod.MCPServerConfig(name="A", url="http://a", enabled=True)
        s2 = mcp_config_mod.MCPServerConfig(name="B", url="http://b", enabled=False)
        client = mcp_client_mod.BioContextMCPClient([s1, s2])
        assert client.server_names == ["A"]

    def test_repr(self):
        if not mcp_client_mod._MCP_AVAILABLE:
            pytest.skip("mcp package not installed")
        s = mcp_config_mod.MCPServerConfig(name="KB", url="http://kb")
        client = mcp_client_mod.BioContextMCPClient([s])
        assert "KB" in repr(client)

    def test_call_tool_unknown_server(self):
        """Calling a tool on an unknown server returns an error result."""
        if not mcp_client_mod._MCP_AVAILABLE:
            pytest.skip("mcp package not installed")
        client = mcp_client_mod.BioContextMCPClient([
            mcp_config_mod.MCPServerConfig(name="X", url="http://x"),
        ])
        result = asyncio.run(
            client.call_tool("UNKNOWN", "some_tool", {})
        )
        assert result.is_error
        assert "No connection" in result.error


# ===================================================================
# mcp_context_enricher tests
# ===================================================================

class TestMCPContextEnricher:

    def _make_enricher(self, llm_response: str, tools=None):
        """Build an enricher with mock client and LLM."""
        mock_client = MagicMock()
        mock_client.server_names = ["TestSRV"]

        if tools is None:
            tools = [
                mcp_client_mod.MCPToolInfo(
                    server_name="TestSRV",
                    tool_name="gene_search",
                    description="Search genes",
                    input_schema={"properties": {"query": {"type": "string"}}},
                ),
            ]
        mock_client.list_all_tools = AsyncMock(return_value=tools)
        mock_client.call_tools_parallel = AsyncMock(return_value=[
            mcp_client_mod.MCPToolResult(
                server_name="TestSRV",
                tool_name="gene_search",
                content="TP53 is a tumor suppressor gene.",
            ),
        ])

        mock_llm = MagicMock()
        mock_llm.run = AsyncMock(return_value=llm_response)

        config = mcp_config_mod.MCPConfig(enabled=True)
        return mcp_enricher_mod.MCPContextEnricher(mock_client, mock_llm, config)

    def test_enrich_returns_context(self):
        """When LLM selects a tool, enrichment returns formatted context."""
        llm_resp = json.dumps([{
            "server_name": "TestSRV",
            "tool_name": "gene_search",
            "arguments": {"query": "TP53"},
        }])
        enricher = self._make_enricher(llm_resp)
        result = asyncio.run(
            enricher.enrich("find TP53 pathway", {"shape": (1000, 2000)})
        )
        assert "Biological Context" in result
        assert "TP53" in result

    def test_enrich_returns_empty_when_no_tools_needed(self):
        """When LLM returns empty list, no enrichment happens."""
        enricher = self._make_enricher("[]")
        result = asyncio.run(
            enricher.enrich("run PCA", {"shape": (1000, 2000)})
        )
        assert result == ""

    def test_enrich_returns_empty_on_invalid_json(self):
        """Graceful handling of malformed LLM response."""
        enricher = self._make_enricher("I cannot decide which tools to use")
        result = asyncio.run(
            enricher.enrich("analyze genes", {"shape": (1000, 2000)})
        )
        assert result == ""

    def test_enrich_returns_empty_when_no_tools_available(self):
        """When server has no tools, enrichment is skipped."""
        enricher = self._make_enricher("[]", tools=[])
        result = asyncio.run(
            enricher.enrich("query string db", {"shape": (1000, 2000)})
        )
        assert result == ""

    def test_enrich_respects_max_tools(self):
        """At most max_tools_per_query tools should be called."""
        calls = json.dumps([
            {"server_name": "TestSRV", "tool_name": "gene_search", "arguments": {"query": "TP53"}},
            {"server_name": "TestSRV", "tool_name": "gene_search", "arguments": {"query": "BRCA1"}},
            {"server_name": "TestSRV", "tool_name": "gene_search", "arguments": {"query": "EGFR"}},
            {"server_name": "TestSRV", "tool_name": "gene_search", "arguments": {"query": "MYC"}},
            {"server_name": "TestSRV", "tool_name": "gene_search", "arguments": {"query": "KRAS"}},
        ])
        enricher = self._make_enricher(calls)
        # Default max_tools_per_query is 3
        asyncio.run(
            enricher.enrich("find genes", {"shape": (1000, 2000)})
        )
        # call_tools_parallel should have been called with at most 3 items
        call_args = enricher._client.call_tools_parallel.call_args[0][0]
        assert len(call_args) <= 3

    def test_enrich_handles_tool_discovery_failure(self):
        """If tool discovery fails, enrichment returns empty string."""
        mock_client = MagicMock()
        mock_client.list_all_tools = AsyncMock(side_effect=Exception("connection refused"))
        mock_llm = MagicMock()
        config = mcp_config_mod.MCPConfig(enabled=True)
        enricher = mcp_enricher_mod.MCPContextEnricher(mock_client, mock_llm, config)

        result = asyncio.run(
            enricher.enrich("find TP53", {"shape": (1000, 2000)})
        )
        assert result == ""


# ===================================================================
# agent_config tests
# ===================================================================

class TestAgentConfigMCP:

    def test_default_mcp_disabled(self):
        """Default AgentConfig should have MCP disabled."""
        cfg = agent_config_mod.AgentConfig()
        assert cfg.mcp is not None
        assert cfg.mcp.enabled is False

    def test_from_flat_kwargs_enable_mcp(self):
        """from_flat_kwargs with enable_mcp=True should enable MCP."""
        cfg = agent_config_mod.AgentConfig.from_flat_kwargs(enable_mcp=True)
        assert cfg.mcp.enabled is True
        assert len(cfg.mcp.servers) == 1  # default KB server

    def test_from_flat_kwargs_custom_servers(self):
        """Custom servers passed via from_flat_kwargs."""
        s = mcp_config_mod.MCPServerConfig(name="Custom", url="http://custom/mcp")
        cfg = agent_config_mod.AgentConfig.from_flat_kwargs(
            enable_mcp=True,
            mcp_servers=[s],
        )
        assert cfg.mcp.enabled is True
        assert len(cfg.mcp.servers) == 1
        assert cfg.mcp.servers[0].name == "Custom"

    def test_from_flat_kwargs_mcp_limits(self):
        """Custom tool limits via flat kwargs."""
        cfg = agent_config_mod.AgentConfig.from_flat_kwargs(
            enable_mcp=True,
            mcp_max_tools=5,
            mcp_max_context_tokens=4000,
            mcp_tool_timeout=60,
        )
        assert cfg.mcp.max_tools_per_query == 5
        assert cfg.mcp.max_context_tokens == 4000
        assert cfg.mcp.tool_call_timeout == 60


# ===================================================================
# Approx token helper test
# ===================================================================

class TestApproxTokens:

    def test_token_estimation(self):
        assert mcp_enricher_mod._approx_tokens("") == 0
        # ~4 chars per token
        assert mcp_enricher_mod._approx_tokens("a" * 100) == 25
