# BioContext MCP Integration Plan for OmicVerse Agent

## Overview

Integrate BioContext MCP servers as an **MCP client** within the OmicVerse Agent, enabling the agent to query biological knowledge databases (STRING, Reactome, Open Targets, UniProt, PanglaoDb, etc.) **before code generation** to enrich analysis pipelines with real biological context.

**Mode**: Augment code generation — the agent queries MCP tools to gather biological context (gene info, pathways, interactions), then uses that context when generating analysis code.

**Scope**: Start with the BioContextAI Knowledgebase MCP (52 tools) as default, with a config system so users can add any BioContext or MCP-compatible server.

---

## Architecture

```
User Request ──→ OmicVerseAgent.run_async()
                      │
                      ├─ 1. Analyze task complexity
                      ├─ 2. [NEW] MCP Context Enrichment
                      │      ├─ Determine if biological context is needed
                      │      ├─ Connect to BioContext MCP servers
                      │      ├─ Call relevant MCP tools (gene lookup, pathway query, etc.)
                      │      └─ Format results as context for code generation
                      ├─ 3. Priority 1 or Priority 2 workflow (with enriched context)
                      └─ 4. Execute generated code
```

---

## New Files

### 1. `omicverse/utils/mcp_client.py` — MCP Client Manager (~350 lines)

Core MCP client module that manages connections to remote MCP servers.

**Key classes:**
- `MCPServerConfig`: Dataclass holding server URL, name, transport type, optional auth
- `MCPToolInfo`: Lightweight descriptor of a discovered tool (name, description, input schema)
- `BioContextMCPClient`: Main client class that:
  - Connects to one or more MCP servers via Streamable HTTP transport
  - Discovers available tools on each server (`list_tools()`)
  - Calls tools and returns results (`call_tool(server, tool_name, args)`)
  - Manages connection lifecycle (connect, disconnect, reconnect)
  - Caches tool listings to avoid repeated discovery
  - Handles timeouts and retries gracefully

**Dependencies:** `mcp` Python SDK (pip install `mcp`), `httpx` (transitive dep of mcp)

**Transport:** Streamable HTTP (`mcp.client.streamable_http.streamablehttp_client`) — the current MCP standard. Falls back to SSE for legacy servers.

**Key implementation details:**
```python
@dataclass
class MCPServerConfig:
    name: str
    url: str                          # e.g., "https://biocontext-kb.fastmcp.app/mcp"
    transport: str = "streamable_http"  # "streamable_http" | "sse"
    enabled: bool = True
    timeout: int = 30                 # seconds per tool call
    description: str = ""

@dataclass
class MCPToolInfo:
    server_name: str
    tool_name: str
    description: str
    input_schema: dict

class BioContextMCPClient:
    def __init__(self, servers: List[MCPServerConfig])
    async def connect_all(self) -> None
    async def disconnect_all(self) -> None
    async def list_all_tools(self) -> List[MCPToolInfo]
    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any
    async def search_tools(self, query: str) -> List[MCPToolInfo]  # keyword search
```

### 2. `omicverse/utils/mcp_context_enricher.py` — Context Enrichment Engine (~300 lines)

Bridges the MCP client with the agent's LLM workflow. Determines what biological context is needed and fetches it.

**Key class:**
- `MCPContextEnricher`:
  - Takes a user request + adata metadata
  - Uses the LLM to determine which MCP tools to call (tool selection prompt)
  - Executes the selected MCP tool calls
  - Formats results into a context block for prompt injection
  - Respects token budgets (truncates large results)

**Key methods:**
```python
class MCPContextEnricher:
    def __init__(self, mcp_client: BioContextMCPClient, llm: OmicVerseLLMBackend)
    async def enrich(self, request: str, adata_meta: dict) -> str
        """Returns formatted context string to inject into code generation prompt."""
    async def _select_tools(self, request: str, available_tools: List[MCPToolInfo]) -> List[dict]
        """LLM selects which tools to call and with what arguments."""
    def _format_context(self, tool_results: List[dict]) -> str
        """Format MCP results into a context block for the prompt."""
```

### 3. `omicverse/utils/mcp_config.py` — MCP Configuration (~80 lines)

Configuration dataclass and default server definitions.

```python
@dataclass
class MCPConfig:
    enabled: bool = False              # Opt-in by default
    servers: List[MCPServerConfig] = field(default_factory=_default_servers)
    max_tools_per_query: int = 3       # Max MCP tool calls per user request
    max_context_tokens: int = 2000     # Token budget for MCP context in prompt
    tool_call_timeout: int = 30        # Seconds

# Default: BioContextAI Knowledgebase
BIOCONTEXT_KB = MCPServerConfig(
    name="BioContextAI-KB",
    url="https://biocontext-kb.fastmcp.app/mcp",
    description="52 tools: STRING, PanglaoDb, HPA, Reactome, Open Targets, UniProt, EuropePMC, etc."
)

# Pre-configured catalog (users enable via config)
BIOCONTEXT_REGISTRY = {
    "biocontext-kb": BIOCONTEXT_KB,
    "string-db": MCPServerConfig(name="STRING-DB", url="https://mcp.string-db.org/", description="PPI networks"),
    "biothings": MCPServerConfig(name="BioThings", url="https://biothings-mcp.longevity-genie.info/mcp", ...),
    "gget": MCPServerConfig(name="gget", url="https://gget-mcp.longevity-genie.info/mcp", ...),
    "ebi-ols": MCPServerConfig(name="EMBL-EBI-OLS", url="https://www.ebi.ac.uk/ols4/api/mcp", ...),
    "biocypher": MCPServerConfig(name="BioCypher-KG", url="https://mcp.biocypher.org/mcp", ...),
    "nucleotide-archive": MCPServerConfig(name="Nucleotide-Archive", url="https://nucleotide-archive-mcp.fastmcp.app/mcp", ...),
    "cellosaurus": MCPServerConfig(name="Cellosaurus", url="https://unofficial-cellosaurus-mcp.fastmcp.app/mcp", ...),
    "open-targets": MCPServerConfig(name="Open-Targets", url="https://mcp.platform.opentargets.org/mcp", ...),
    "omnipath": MCPServerConfig(name="OmniPath", url="https://explore.omnipathdb.org/api/mcp", ...),
}
```

---

## Modified Files

### 4. `omicverse/utils/agent_config.py` — Add MCPConfig

Add `MCPConfig` to the `AgentConfig` dataclass:

```python
from .mcp_config import MCPConfig

@dataclass
class AgentConfig:
    llm: LLMConfig = ...
    reflection: ReflectionConfig = ...
    execution: ExecutionConfig = ...
    context: ContextConfig = ...
    mcp: MCPConfig = field(default_factory=MCPConfig)  # NEW
    ...
```

Update `from_flat_kwargs` to accept `enable_mcp`, `mcp_servers`, etc.

### 5. `omicverse/utils/smart_agent.py` — Integrate MCP into Workflow

**Changes:**

a. **`__init__`** (~line 259): Add `enable_mcp: bool = False` and `mcp_servers: list = None` parameters. Initialize `BioContextMCPClient` and `MCPContextEnricher` when enabled.

```python
# In __init__, after filesystem context initialization:
if self._config.mcp.enabled:
    from .mcp_client import BioContextMCPClient
    from .mcp_context_enricher import MCPContextEnricher
    self._mcp_client = BioContextMCPClient(self._config.mcp.servers)
    self._mcp_enricher = MCPContextEnricher(self._mcp_client, self._llm)
    print(f"   🌐 BioContext MCP enabled: {len(self._config.mcp.servers)} server(s)")
```

b. **`run_async`** (~line 2924): After complexity analysis, add MCP context enrichment step before code generation.

```python
# After complexity analysis, before Priority 1/2:
mcp_context = ""
if self._config.mcp.enabled and self._mcp_enricher:
    print(f"🌐 Querying biological context from MCP servers...")
    try:
        mcp_context = await self._mcp_enricher.enrich(request, {
            "shape": adata.shape,
            "var_names": list(adata.var_names[:20]),
            "obs_columns": list(adata.obs.columns),
        })
        if mcp_context:
            print(f"   ✅ Retrieved biological context ({len(mcp_context)} chars)")
    except Exception as e:
        print(f"   ⚠️ MCP context enrichment failed (non-fatal): {e}")
```

c. **`_run_registry_workflow`** and **`_run_skills_workflow`**: Inject `mcp_context` into the code generation prompts as an additional section:

```
Biological Context (from external knowledge databases):
{mcp_context}
```

d. **Cleanup**: Add `async def close(self)` method to disconnect MCP clients on agent shutdown.

### 6. `omicverse/__init__.py` — No changes needed

The `Agent` alias already points to `OmicVerseAgent`. The new MCP parameters are optional with defaults.

---

## User-Facing API

```python
import omicverse as ov

# Basic usage with BioContext enabled
agent = ov.Agent(
    model="gemini-2.5-flash",
    enable_mcp=True,  # Enables BioContextAI Knowledgebase by default
)
result = agent.run("find pathway enrichment for DE genes related to TP53 signaling", adata)

# Advanced: Add additional MCP servers
from omicverse.utils.mcp_config import MCPServerConfig, BIOCONTEXT_REGISTRY

agent = ov.Agent(
    model="gemini-2.5-flash",
    enable_mcp=True,
    mcp_servers=[
        BIOCONTEXT_REGISTRY["biocontext-kb"],
        BIOCONTEXT_REGISTRY["string-db"],
        MCPServerConfig(name="my-server", url="https://my-mcp-server.com/mcp"),
    ]
)

# Or via AgentConfig
from omicverse.utils.agent_config import AgentConfig
from omicverse.utils.mcp_config import MCPConfig

config = AgentConfig(
    mcp=MCPConfig(
        enabled=True,
        max_tools_per_query=5,
        max_context_tokens=3000,
    )
)
agent = ov.Agent(config=config)
```

---

## Implementation Order

1. **`omicverse/utils/mcp_config.py`** — Configuration and server registry
2. **`omicverse/utils/mcp_client.py`** — MCP client with Streamable HTTP transport
3. **`omicverse/utils/mcp_context_enricher.py`** — LLM-driven context enrichment
4. **`omicverse/utils/agent_config.py`** — Add MCPConfig to AgentConfig
5. **`omicverse/utils/smart_agent.py`** — Wire MCP into the agent workflow
6. **Tests** — Unit tests for MCP client and enricher
7. **Dependencies** — Add `mcp>=1.8` to optional dependencies

---

## Error Handling & Resilience

- MCP is **non-blocking**: if servers are unreachable, the agent proceeds without biological context (warning only)
- Connection timeouts: 10s for connect, 30s for tool calls
- Tool call failures are caught per-tool; partial results still used
- MCP context is **optional enrichment**, never required for code generation
- Lazy connection: MCP servers connected on first use, not at agent init
- Graceful degradation: if `mcp` package not installed, `enable_mcp=True` raises a clear ImportError with install instructions

---

## Dependencies

Add to `pyproject.toml` as optional:
```toml
[project.optional-dependencies]
mcp = ["mcp>=1.8.0"]
```

Or install: `pip install omicverse[mcp]`
