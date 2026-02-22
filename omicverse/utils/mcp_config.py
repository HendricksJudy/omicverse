"""
BioContext MCP Configuration for OmicVerse Agent.

Defines MCP server configurations and a pre-built registry of BioContext
community servers.  Users enable MCP via ``enable_mcp=True`` on the Agent
constructor and may supply additional servers through ``mcp_servers``.

Transport
---------
Remote servers use **Streamable HTTP** (the current MCP standard, superseding
SSE).  The ``transport`` field on each ``MCPServerConfig`` selects the wire
protocol; ``"streamable_http"`` is the default.

References
----------
- BioContext registry: https://biocontext.ai/registry
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------

@dataclass
class MCPServerConfig:
    """Connection details for a single MCP server."""

    name: str
    url: str  # e.g. "https://biocontext-kb.fastmcp.app/mcp"
    transport: str = "streamable_http"  # "streamable_http" | "sse"
    enabled: bool = True
    timeout: int = 30  # seconds per tool call
    connect_timeout: int = 10  # seconds for initial connection
    description: str = ""


# ---------------------------------------------------------------------------
# Pre-configured BioContext servers
# ---------------------------------------------------------------------------

BIOCONTEXT_KB = MCPServerConfig(
    name="BioContextAI-KB",
    url="https://biocontext-kb.fastmcp.app/mcp",
    description=(
        "BioContextAI Knowledgebase – 52 tools covering STRING, PanglaoDb, "
        "Human Protein Atlas, Antibody Registry, EuropePMC, Reactome, "
        "Open Targets, Grants.gov and 15+ more biological databases."
    ),
)

BIOCONTEXT_REGISTRY: Dict[str, MCPServerConfig] = {
    "biocontext-kb": BIOCONTEXT_KB,
    "string-db": MCPServerConfig(
        name="STRING-DB",
        url="https://mcp.string-db.org/",
        description="Protein-protein interaction networks and functional annotations (16 tools).",
    ),
    "biothings": MCPServerConfig(
        name="BioThings",
        url="https://biothings-mcp.longevity-genie.info/mcp",
        description="BioThings.io ecosystem – gene, variant, chemical, and disease annotations (19 tools).",
    ),
    "gget": MCPServerConfig(
        name="gget",
        url="https://gget-mcp.longevity-genie.info/mcp",
        description="Genomics data retrieval via gget – NCBI, Ensembl, UniProt lookups (20 tools).",
    ),
    "ebi-ols": MCPServerConfig(
        name="EMBL-EBI-OLS",
        url="https://www.ebi.ac.uk/ols4/api/mcp",
        description="Biomedical ontologies from the EMBL-EBI Ontology Lookup Service (6 tools).",
    ),
    "biocypher": MCPServerConfig(
        name="BioCypher-KG",
        url="https://mcp.biocypher.org/mcp",
        description="BioCypher knowledge graph – pipeline creation and knowledge harmonisation (9 tools).",
    ),
    "nucleotide-archive": MCPServerConfig(
        name="Nucleotide-Archive",
        url="https://nucleotide-archive-mcp.fastmcp.app/mcp",
        description="RNA-seq dataset search from the European Nucleotide Archive (11 tools).",
    ),
    "cellosaurus": MCPServerConfig(
        name="Cellosaurus",
        url="https://unofficial-cellosaurus-mcp.fastmcp.app/mcp",
        description="Cell line knowledge resource – provenance, cross-references, species (6 tools).",
    ),
    "open-targets": MCPServerConfig(
        name="Open-Targets",
        url="https://mcp.platform.opentargets.org/mcp",
        description="Drug target validation and disease association analysis.",
    ),
    "omnipath": MCPServerConfig(
        name="OmniPath",
        url="https://explore.omnipathdb.org/api/mcp",
        description="Molecular interactions and signalling pathway data via SQL queries.",
    ),
    "opengenes": MCPServerConfig(
        name="OpenGenes",
        url="https://opengenes-mcp.longevity-genie.info/mcp",
        description="Aging research and lifespan intervention data (3 tools).",
    ),
    "synergy-age": MCPServerConfig(
        name="SynergyAge",
        url="https://synergy-age-mcp.longevity-genie.info/mcp",
        description="Genetic interventions affecting lifespan across organisms (3 tools).",
    ),
}


def _default_servers() -> List[MCPServerConfig]:
    """Return the default server list (BioContextAI Knowledgebase only)."""
    return [BIOCONTEXT_KB]


# ---------------------------------------------------------------------------
# Aggregated MCP config
# ---------------------------------------------------------------------------

@dataclass
class MCPConfig:
    """MCP integration settings for the OmicVerse Agent."""

    enabled: bool = False  # opt-in
    servers: List[MCPServerConfig] = field(default_factory=_default_servers)
    max_tools_per_query: int = 3  # max MCP tool calls per user request
    max_context_tokens: int = 2000  # approximate token budget for MCP context
    tool_call_timeout: int = 30  # seconds

    # --- helpers -----------------------------------------------------------

    def active_servers(self) -> List[MCPServerConfig]:
        """Return only the enabled servers."""
        return [s for s in self.servers if s.enabled]
