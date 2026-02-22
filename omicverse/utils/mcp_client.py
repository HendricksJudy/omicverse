"""
BioContext MCP Client for OmicVerse Agent.

Manages connections to one or more remote MCP servers, discovers their tools,
and executes tool calls.  Connections are **lazy** — the first ``call_tool``
or ``list_all_tools`` invocation opens the transport.

Transport
---------
Remote servers default to **Streamable HTTP** (``mcp.client.streamable_http``).
Legacy SSE transport is supported as a fallback.

Dependencies
------------
Requires the ``mcp`` package (``pip install 'mcp>=1.8.0'``).  When the
package is missing, importing this module still succeeds but
``BioContextMCPClient`` raises a clear ``ImportError`` on instantiation.
"""

from __future__ import annotations

import asyncio
import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .mcp_config import MCPServerConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft dependency check
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
try:
    from mcp import ClientSession  # type: ignore[import-untyped]
    from mcp.client.streamable_http import streamablehttp_client  # type: ignore[import-untyped]
    _MCP_AVAILABLE = True
except ImportError:
    pass

_MCP_SSE_AVAILABLE = False
try:
    from mcp.client.sse import sse_client  # type: ignore[import-untyped]
    _MCP_SSE_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Lightweight descriptors
# ---------------------------------------------------------------------------

@dataclass
class MCPToolInfo:
    """Metadata for a single tool discovered on an MCP server."""

    server_name: str
    tool_name: str
    description: str
    input_schema: dict = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """Return ``server_name/tool_name``."""
        return f"{self.server_name}/{self.tool_name}"


@dataclass
class MCPToolResult:
    """Result of executing an MCP tool."""

    server_name: str
    tool_name: str
    content: Any = None
    error: Optional[str] = None
    is_error: bool = False


# ---------------------------------------------------------------------------
# Per-server connection handle
# ---------------------------------------------------------------------------

class _ServerConnection:
    """Manages a single MCP server connection and its lifecycle."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._session: Optional[ClientSession] = None
        self._context_stack: Optional[Any] = None  # contextlib.AsyncExitStack
        self._tools: Optional[List[MCPToolInfo]] = None
        self._connected = False

    async def connect(self) -> None:
        """Open transport and initialise the client session."""
        if self._connected:
            return

        import contextlib

        stack = contextlib.AsyncExitStack()
        try:
            if self.config.transport == "sse":
                if not _MCP_SSE_AVAILABLE:
                    raise ImportError(
                        "SSE transport requested but mcp.client.sse is not available. "
                        "Install mcp with SSE support: pip install 'mcp[sse]>=1.8.0'"
                    )
                transport = sse_client(self.config.url)
            else:
                # Default: Streamable HTTP
                transport = streamablehttp_client(self.config.url)

            read_stream, write_stream, _ = await stack.enter_async_context(transport)
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()

            self._session = session
            self._context_stack = stack
            self._connected = True
            logger.info("Connected to MCP server %s at %s", self.config.name, self.config.url)

        except Exception:
            await stack.aclose()
            raise

    async def disconnect(self) -> None:
        """Close transport gracefully."""
        if self._context_stack is not None:
            try:
                await self._context_stack.aclose()
            except Exception as exc:
                logger.warning("Error disconnecting from %s: %s", self.config.name, exc)
            finally:
                self._session = None
                self._context_stack = None
                self._connected = False
                self._tools = None

    async def list_tools(self) -> List[MCPToolInfo]:
        """Discover tools on this server (cached after first call)."""
        if self._tools is not None:
            return self._tools

        if not self._connected:
            await self.connect()

        assert self._session is not None
        response = await self._session.list_tools()

        self._tools = [
            MCPToolInfo(
                server_name=self.config.name,
                tool_name=tool.name,
                description=getattr(tool, "description", "") or "",
                input_schema=getattr(tool, "inputSchema", {}) or {},
            )
            for tool in response.tools
        ]
        logger.info("Discovered %d tools on %s", len(self._tools), self.config.name)
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> MCPToolResult:
        """Execute a single tool call and return the result."""
        if not self._connected:
            await self.connect()

        assert self._session is not None
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=self.config.timeout,
            )
            # Extract text content from the result
            content_parts = []
            for block in (result.content or []):
                if hasattr(block, "text"):
                    content_parts.append(block.text)
                elif hasattr(block, "data"):
                    content_parts.append(str(block.data))

            text_content = "\n".join(content_parts) if content_parts else str(result.content)
            is_error = getattr(result, "isError", False)

            return MCPToolResult(
                server_name=self.config.name,
                tool_name=tool_name,
                content=text_content,
                is_error=is_error,
                error=text_content if is_error else None,
            )
        except asyncio.TimeoutError:
            msg = f"Tool call {tool_name} on {self.config.name} timed out after {self.config.timeout}s"
            logger.warning(msg)
            return MCPToolResult(
                server_name=self.config.name,
                tool_name=tool_name,
                is_error=True,
                error=msg,
            )
        except Exception as exc:
            msg = f"Tool call {tool_name} on {self.config.name} failed: {exc}"
            logger.warning(msg)
            return MCPToolResult(
                server_name=self.config.name,
                tool_name=tool_name,
                is_error=True,
                error=msg,
            )


# ---------------------------------------------------------------------------
# Multi-server client
# ---------------------------------------------------------------------------

class BioContextMCPClient:
    """Manages connections to multiple MCP servers and proxies tool calls.

    Parameters
    ----------
    servers : list[MCPServerConfig]
        Server configurations.  Only servers with ``enabled=True`` are used.

    Raises
    ------
    ImportError
        If the ``mcp`` package is not installed.

    Examples
    --------
    >>> from omicverse.utils.mcp_config import BIOCONTEXT_KB
    >>> client = BioContextMCPClient([BIOCONTEXT_KB])
    >>> tools = await client.list_all_tools()
    >>> result = await client.call_tool("BioContextAI-KB", "string_search", {"query": "TP53"})
    """

    def __init__(self, servers: List[MCPServerConfig]) -> None:
        if not _MCP_AVAILABLE:
            raise ImportError(
                "The 'mcp' package is required for BioContext MCP integration.\n"
                "Install it with: pip install 'mcp>=1.8.0'\n"
                "Or install omicverse with MCP support: pip install 'omicverse[mcp]'"
            )

        self._connections: Dict[str, _ServerConnection] = {}
        for server in servers:
            if server.enabled:
                self._connections[server.name] = _ServerConnection(server)

    # -- lifecycle ----------------------------------------------------------

    async def connect_all(self) -> Dict[str, Optional[str]]:
        """Connect to all configured servers.

        Returns a dict mapping server names to ``None`` (success) or an error
        message string.  Failures are non-fatal — other servers remain usable.
        """
        results: Dict[str, Optional[str]] = {}
        for name, conn in self._connections.items():
            try:
                await conn.connect()
                results[name] = None
            except Exception as exc:
                logger.warning("Failed to connect to %s: %s", name, exc)
                results[name] = str(exc)
        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for conn in self._connections.values():
            await conn.disconnect()

    # -- discovery ----------------------------------------------------------

    async def list_all_tools(self) -> List[MCPToolInfo]:
        """Discover tools across all connected servers.

        Servers that fail to respond are silently skipped.
        """
        all_tools: List[MCPToolInfo] = []
        for name, conn in self._connections.items():
            try:
                tools = await conn.list_tools()
                all_tools.extend(tools)
            except Exception as exc:
                logger.warning("Tool discovery failed for %s: %s", name, exc)
        return all_tools

    async def search_tools(self, query: str) -> List[MCPToolInfo]:
        """Keyword search across all discovered tool names and descriptions."""
        query_lower = query.lower()
        all_tools = await self.list_all_tools()
        return [
            t for t in all_tools
            if query_lower in t.tool_name.lower() or query_lower in t.description.lower()
        ]

    # -- execution ----------------------------------------------------------

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[dict] = None,
    ) -> MCPToolResult:
        """Call a tool on the specified server.

        Parameters
        ----------
        server_name : str
            Name of the server (must match ``MCPServerConfig.name``).
        tool_name : str
            Name of the tool as returned by ``list_tools``.
        arguments : dict, optional
            JSON-serialisable arguments for the tool.

        Returns
        -------
        MCPToolResult
            Result including content or error details.
        """
        conn = self._connections.get(server_name)
        if conn is None:
            return MCPToolResult(
                server_name=server_name,
                tool_name=tool_name,
                is_error=True,
                error=f"No connection for server '{server_name}'",
            )
        return await conn.call_tool(tool_name, arguments or {})

    async def call_tools_parallel(
        self,
        calls: List[dict],
    ) -> List[MCPToolResult]:
        """Execute multiple tool calls concurrently.

        Parameters
        ----------
        calls : list[dict]
            Each dict must have keys ``server_name``, ``tool_name``, and
            optionally ``arguments``.

        Returns
        -------
        list[MCPToolResult]
            Results in the same order as *calls*.
        """
        coros = [
            self.call_tool(
                c["server_name"],
                c["tool_name"],
                c.get("arguments", {}),
            )
            for c in calls
        ]
        return await asyncio.gather(*coros)

    # -- introspection ------------------------------------------------------

    @property
    def server_names(self) -> List[str]:
        """Names of all configured (enabled) servers."""
        return list(self._connections.keys())

    def __repr__(self) -> str:
        servers = ", ".join(self._connections.keys())
        return f"BioContextMCPClient(servers=[{servers}])"
