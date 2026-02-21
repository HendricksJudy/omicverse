"""
Unit tests for the MCP Server Registry.

All tests are fully self-contained — no network access required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omicverse.utils.mcp_registry import (
    MCPServerDescriptor,
    MCPServerRegistry,
    _BUILTIN_SERVERS,
)


# ---------------------------------------------------------------------------
# Tests: Construction
# ---------------------------------------------------------------------------

class TestMCPServerRegistryConstruction:
    def test_includes_builtins_by_default(self):
        registry = MCPServerRegistry()
        assert "biocontext" in registry.server_names

    def test_no_builtins(self):
        registry = MCPServerRegistry(include_builtins=False)
        assert len(registry) == 0


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_http(self):
        registry = MCPServerRegistry(include_builtins=False)
        desc = registry.register(
            "test-server",
            url="https://example.com/mcp/",
            description="Test MCP server",
        )
        assert desc.name == "test-server"
        assert "test-server" in registry.server_names

    def test_register_stdio(self):
        registry = MCPServerRegistry(include_builtins=False)
        desc = registry.register(
            "local-server",
            command="uvx",
            args=["my_server@latest"],
            description="Local server",
        )
        assert desc.command == "uvx"

    def test_register_requires_url_or_command(self):
        registry = MCPServerRegistry(include_builtins=False)
        with pytest.raises(ValueError, match="Either 'url' or 'command'"):
            registry.register("bad-server", description="No transport")

    def test_unregister(self):
        registry = MCPServerRegistry(include_builtins=False)
        registry.register("srv", url="https://example.com/mcp/")
        assert "srv" in registry.server_names
        registry.unregister("srv")
        assert "srv" not in registry.server_names


# ---------------------------------------------------------------------------
# Tests: Connection management
# ---------------------------------------------------------------------------

class TestConnectionManagement:
    def test_connect(self):
        registry = MCPServerRegistry(include_builtins=False)
        registry.register("srv", url="https://example.com/mcp/", description="test")

        mock_manager = MagicMock()
        fake_info = MagicMock()
        fake_info.tools = [MagicMock(), MagicMock()]
        mock_manager.connect.return_value = fake_info

        info = registry.connect("srv", mock_manager)
        assert info is fake_info
        assert registry.is_connected("srv")
        mock_manager.connect.assert_called_once()

    def test_connect_unregistered_raises(self):
        registry = MCPServerRegistry(include_builtins=False)
        mock_manager = MagicMock()
        with pytest.raises(ValueError, match="not registered"):
            registry.connect("nonexistent", mock_manager)

    def test_connect_idempotent(self):
        registry = MCPServerRegistry(include_builtins=False)
        registry.register("srv", url="https://example.com/mcp/")

        mock_manager = MagicMock()
        fake_info = MagicMock()
        fake_info.tools = []
        mock_manager.connect.return_value = fake_info

        info1 = registry.connect("srv", mock_manager)
        info2 = registry.connect("srv", mock_manager)
        assert info1 is info2
        mock_manager.connect.assert_called_once()  # not called twice

    def test_disconnect(self):
        registry = MCPServerRegistry(include_builtins=False)
        registry.register("srv", url="https://example.com/mcp/")

        mock_manager = MagicMock()
        fake_info = MagicMock()
        fake_info.tools = []
        mock_manager.connect.return_value = fake_info

        registry.connect("srv", mock_manager)
        assert registry.is_connected("srv")
        registry.disconnect("srv", mock_manager)
        assert not registry.is_connected("srv")


# ---------------------------------------------------------------------------
# Tests: Tool previews
# ---------------------------------------------------------------------------

class TestToolPreviews:
    def test_previews_for_unconnected(self):
        registry = MCPServerRegistry()  # includes biocontext
        previews = registry.get_tool_previews()
        assert len(previews) > 0
        assert all(p["requires_connection"] for p in previews)

    def test_no_previews_for_connected(self):
        registry = MCPServerRegistry(include_builtins=False)
        registry.register(
            "srv", url="https://example.com/mcp/",
            tool_previews=[{"name": "tool1", "description": "test"}],
        )

        mock_manager = MagicMock()
        fake_info = MagicMock()
        fake_info.tools = []
        mock_manager.connect.return_value = fake_info
        registry.connect("srv", mock_manager)

        previews = registry.get_tool_previews()
        assert len(previews) == 0  # connected servers don't show previews


# ---------------------------------------------------------------------------
# Tests: from_config_list
# ---------------------------------------------------------------------------

class TestFromConfigList:
    def test_from_config_list(self):
        registry = MCPServerRegistry(include_builtins=False)
        registry.from_config_list([
            {"name": "srv1", "url": "https://a.com/mcp/", "description": "Server 1"},
            {"name": "srv2", "command": "uvx", "args": ["pkg@latest"]},
        ])
        assert len(registry) == 2
        assert "srv1" in registry.server_names
        assert "srv2" in registry.server_names

    def test_from_config_list_skips_nameless(self):
        registry = MCPServerRegistry(include_builtins=False)
        registry.from_config_list([
            {"url": "https://a.com/mcp/"},  # no name
        ])
        assert len(registry) == 0


# ---------------------------------------------------------------------------
# Tests: repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr(self):
        registry = MCPServerRegistry()
        r = repr(registry)
        assert "registered=" in r
        assert "connected=0" in r
