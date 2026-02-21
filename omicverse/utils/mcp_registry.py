"""
MCP Server Registry for OmicVerse Agent.

Enables users to discover, register, and manage MCP servers.  Servers are
stored as lightweight descriptors and only connected on demand (lazy
connection).

Design
------
* ``MCPServerDescriptor`` – metadata for a registered server (not yet connected).
* ``MCPServerRegistry`` – manages server descriptors and provides tool previews
  for the :class:`ToolCatalog`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPServerDescriptor:
    """Metadata for a registered (but not necessarily connected) MCP server."""

    name: str
    description: str = ""
    url: Optional[str] = None           # HTTP endpoint
    command: Optional[str] = None       # stdio executable
    args: Optional[List[str]] = None    # stdio arguments
    category: str = "general"           # "knowledgebase" | "tool" | "general"
    tool_previews: List[Dict[str, str]] = field(default_factory=list)
    # tool_previews: [{"name": "tool_name", "description": "..."}]


# Built-in server descriptors (not connected by default)
_BUILTIN_SERVERS: Dict[str, MCPServerDescriptor] = {
    "biocontext": MCPServerDescriptor(
        name="biocontext",
        description="STRING, UniProt, KEGG, Reactome, PanglaoDB, Open Targets, EuropePMC",
        url="https://mcp.biocontext.ai/mcp/",
        category="knowledgebase",
        tool_previews=[
            {"name": "string_interaction_partners", "description": "Protein-protein interactions from STRING"},
            {"name": "string_functional_enrichment", "description": "Functional enrichment via STRING"},
            {"name": "uniprot_protein_lookup", "description": "UniProt protein details"},
            {"name": "kegg_pathway_info", "description": "KEGG pathway annotation"},
            {"name": "reactome_pathway_info", "description": "Reactome pathway information"},
            {"name": "panglao_cell_markers", "description": "Cell-type markers from PanglaoDB"},
            {"name": "europepmc_search", "description": "Literature search via EuropePMC"},
            {"name": "opentargets_target_info", "description": "Drug target info from Open Targets"},
        ],
    ),
}


class MCPServerRegistry:
    """Registry of available MCP servers (lazy - not connected by default).

    Provides:
    - Built-in server descriptors (BioContext, etc.)
    - User-registered custom servers
    - Tool previews for the unified catalog
    - Connection management
    """

    def __init__(self, *, include_builtins: bool = True):
        self._servers: Dict[str, MCPServerDescriptor] = {}
        self._connected: Dict[str, Any] = {}  # name -> MCPServerInfo
        if include_builtins:
            self._servers.update(_BUILTIN_SERVERS)

    # -- registration --------------------------------------------------------

    def register(
        self,
        name: str,
        *,
        url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        description: str = "",
        category: str = "general",
        tool_previews: Optional[List[Dict[str, str]]] = None,
    ) -> MCPServerDescriptor:
        """Register an MCP server for lazy connection.

        Parameters
        ----------
        name : str
            Unique name for the server.
        url : str, optional
            HTTP endpoint (Streamable HTTP transport).
        command : str, optional
            Executable for stdio transport.
        args : list[str], optional
            Arguments for the stdio command.
        description : str
            Human-readable description.
        category : str
            Server category ("knowledgebase", "tool", "general").
        tool_previews : list[dict], optional
            Preview tool descriptors: [{"name": "...", "description": "..."}]

        Returns
        -------
        MCPServerDescriptor
        """
        if not url and not command:
            raise ValueError("Either 'url' or 'command' must be provided")

        desc = MCPServerDescriptor(
            name=name,
            description=description,
            url=url,
            command=command,
            args=args,
            category=category,
            tool_previews=tool_previews or [],
        )
        self._servers[name] = desc
        logger.info("Registered MCP server '%s'", name)
        return desc

    def unregister(self, name: str) -> None:
        """Remove a server from the registry."""
        self._servers.pop(name, None)
        self._connected.pop(name, None)

    # -- introspection -------------------------------------------------------

    @property
    def server_names(self) -> List[str]:
        """Return names of all registered servers."""
        return list(self._servers.keys())

    @property
    def connected_names(self) -> List[str]:
        """Return names of currently connected servers."""
        return list(self._connected.keys())

    def get_descriptor(self, name: str) -> Optional[MCPServerDescriptor]:
        """Return the descriptor for a server, or None."""
        return self._servers.get(name)

    def is_connected(self, name: str) -> bool:
        """Check if a server is currently connected."""
        return name in self._connected

    # -- connection management -----------------------------------------------

    def connect(self, name: str, mcp_manager: Any) -> Any:
        """Connect a registered server using the given MCPClientManager.

        Parameters
        ----------
        name : str
            Name of the registered server.
        mcp_manager : MCPClientManager
            The client manager to use for the connection.

        Returns
        -------
        MCPServerInfo
            Server info with discovered tools.

        Raises
        ------
        ValueError
            If the server is not registered.
        """
        desc = self._servers.get(name)
        if desc is None:
            raise ValueError(f"Server '{name}' is not registered")

        if name in self._connected:
            return self._connected[name]

        info = mcp_manager.connect(
            name,
            url=desc.url,
            command=desc.command,
            args=desc.args,
        )
        self._connected[name] = info
        logger.info("Connected MCP server '%s': %d tools", name, len(info.tools))
        return info

    def disconnect(self, name: str, mcp_manager: Any) -> None:
        """Disconnect a server."""
        if name in self._connected:
            mcp_manager.disconnect(name)
            del self._connected[name]

    # -- catalog integration -------------------------------------------------

    def get_tool_previews(self) -> List[Dict[str, str]]:
        """Return tool previews for all registered (but not connected) servers.

        These previews are used by :class:`ToolCatalog` to show the LLM what
        tools *could* become available if an MCP server is connected.

        Returns
        -------
        list[dict]
            Each dict has keys: name, description, server_name, requires_connection.
        """
        previews = []
        for name, desc in self._servers.items():
            if name in self._connected:
                continue  # already connected, real tools are in the catalog
            for tp in desc.tool_previews:
                previews.append({
                    "name": tp["name"],
                    "description": tp.get("description", ""),
                    "server_name": name,
                    "requires_connection": True,
                })
        return previews

    def from_config_list(self, servers: List[Dict[str, Any]]) -> None:
        """Register multiple servers from a configuration list.

        Parameters
        ----------
        servers : list[dict]
            Each dict should have 'name' and either 'url' or 'command'.
        """
        for cfg in servers:
            name = cfg.get("name")
            if not name:
                continue
            self.register(
                name,
                url=cfg.get("url"),
                command=cfg.get("command"),
                args=cfg.get("args"),
                description=cfg.get("description", ""),
                category=cfg.get("category", "general"),
                tool_previews=cfg.get("tool_previews", []),
            )

    def __len__(self) -> int:
        return len(self._servers)

    def __repr__(self) -> str:
        connected = len(self._connected)
        total = len(self._servers)
        return f"MCPServerRegistry(registered={total}, connected={connected})"
